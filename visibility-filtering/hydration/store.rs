use crate::clients::socialgraph_client::EdgeQuery;
use crate::hydration::decode::author::DecodedAuthor;
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::execute::Reply;
use crate::hydration::fetcher::{AnyFetcher, Fetcher};
use crate::hydration::metrics::record_unasked_keys;
use crate::hydration::plan::{Edge, Group, KeyOrigin, Source};
use crate::hydration::{
    candidate_count_by_key, HydratedTweet, Hydration, HydrationPlan, HydrationRequest, Hydrator,
    Hydrators,
};
use crate::models::{
    AuthorId, ConversationControlFeatures, HydratedTweetCandidate, PureCore, RawCandidate,
    SafetyLabelMap, TweetFeatures, TweetId, Viewer, ViewerFeatures,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use strum::VariantArray;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};
use xai_visibility_filtering_proto as vf_pb;

struct RequestTweet {
    tweet_id: TweetId,
    request_author: Option<AuthorId>,
    author: Option<AuthorId>,
}

pub(super) struct CallRequest<'p> {
    pub(super) group: &'p Group,
    pub(super) is_first: bool,
    pub(super) keys: Vec<u64>,
    pub(super) key_count: usize,
    pub(super) queries: Vec<EdgeQuery>,
    pub(super) viewer_id: Option<u64>,
    pub(super) batch_size: Option<usize>,
    pub(super) counts: HashMap<u64, usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Landing {
    Answered,
    SourcesJoined,
}

#[derive(Default)]
pub(super) struct Store {
    viewer_id: Option<u64>,
    request_tweets: Vec<RequestTweet>,
    requested: usize,
    is_expanding_retweet_sources: bool,
    callable: Hydrators,
    pure_cores: Fetcher<PureCore>,
    tweets: Fetcher<TweetFeatures>,
    controls: Fetcher<ConversationControl>,
    labels: Fetcher<Arc<vf_pb::SafetyLabelMap>>,
    viewer: Fetcher<DecodedViewer>,
    authors: Fetcher<DecodedAuthor>,
    edges: [Fetcher<bool>; Edge::VARIANTS.len()],
    has_called: Vec<bool>,
    viewer_country: Fetcher<Arc<str>>,
    pub(super) core_elapsed: Duration,
    pub(super) tweets_elapsed: Option<Duration>,
}

impl Store {
    pub(super) fn new(
        plan: &HydrationPlan,
        viewer_id: Option<u64>,
        raw: &[RawCandidate],
        is_expanding_retweet_sources: bool,
    ) -> Self {
        Self {
            viewer_id,
            requested: raw.len(),
            is_expanding_retweet_sources,
            request_tweets: raw
                .iter()
                .map(|candidate| RequestTweet {
                    tweet_id: candidate.tweet_id,
                    request_author: candidate.request_author_id.map(AuthorId),
                    author: None,
                })
                .collect(),
            callable: plan.callable(viewer_id),
            has_called: vec![false; plan.groups().count()],
            ..Self::default()
        }
    }

    fn candidate_count(&self) -> usize {
        self.request_tweets
            .iter()
            .take(self.requested)
            .filter(|request_tweet| request_tweet.author.is_some())
            .count()
    }

    fn has_joined_sources(&self) -> bool {
        self.request_tweets.len() > self.requested
    }

    fn controls(&self) -> impl Iterator<Item = &ConversationControl> {
        self.request_tweets
            .iter()
            .filter_map(|request_tweet| self.control(request_tweet))
    }

    fn control(&self, request_tweet: &RequestTweet) -> Option<&ConversationControl> {
        self.controls.get(request_tweet.tweet_id.0)
    }

    fn key(&self, origin: KeyOrigin, request_tweet: &RequestTweet) -> Option<u64> {
        match origin {
            KeyOrigin::RequestTweets => Some(request_tweet.tweet_id.0),
            KeyOrigin::Viewer => self.viewer_id,
            KeyOrigin::ViewerForCoAllowedList => self
                .viewer_id
                .filter(|_| self.control(request_tweet).is_some_and(lists_countries)),
            KeyOrigin::PureCoreAuthor => request_tweet.author.map(AuthorId::get),
            KeyOrigin::PureCoreReplyRoot => self
                .pure_cores
                .get(request_tweet.tweet_id.0)?
                .direct_reply_root_author_id
                .map(AuthorId::get),
            KeyOrigin::ExclusiveConversationAuthor => {
                self.tweets
                    .get(request_tweet.tweet_id.0)?
                    .exclusive_conversation_author_id
            }
            KeyOrigin::ConversationRoot(arms) => self
                .control(request_tweet)
                .filter(|control| arms.contains(&control.arm))
                .map(|control| control.conversation_tweet_author_id),
            KeyOrigin::MyNetworkRootNotFollowingViewer => self
                .control(request_tweet)
                .and_then(|control| self.root_not_following_viewer(control)),
        }
    }

    fn keys(&self, nodes: Hydrators) -> Vec<u64> {
        let mut keys = Vec::new();
        for (position, node) in nodes.iter().enumerate() {
            let origin = node.spec().key;
            if nodes
                .iter()
                .take(position)
                .any(|earlier| earlier.spec().key == origin)
            {
                continue;
            }
            match origin {
                KeyOrigin::Viewer => keys.extend(self.viewer_id),
                _ => keys.extend(
                    self.request_tweets
                        .iter()
                        .filter_map(|request_tweet| self.key(origin, request_tweet)),
                ),
            }
        }
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    fn claim(&mut self, nodes: Hydrators) -> Vec<u64> {
        let keys = self.keys(nodes);
        match nodes.iter().next().and_then(|node| self.fetcher_mut(node)) {
            Some(fetcher) => fetcher.claim(keys),
            None => Vec::new(),
        }
    }

    fn root_not_following_viewer(&self, control: &ConversationControl) -> Option<u64> {
        if control.arm != ConversationControlArm::MyNetwork {
            return None;
        }
        let root = control.conversation_tweet_author_id;
        let holds = *self.edge_fetcher(Edge::FollowedBy)?.get(root)?;
        (!holds).then_some(root)
    }

    fn edge_fetcher(&self, edge: Edge) -> Option<&Fetcher<bool>> {
        self.edges.get(usize::from(edge as u8))
    }

    fn edge_fetcher_mut(&mut self, edge: Edge) -> Option<&mut Fetcher<bool>> {
        self.edges.get_mut(usize::from(edge as u8))
    }

    fn fetcher(&self, node: Hydrator) -> Option<&dyn AnyFetcher> {
        let fetcher: &dyn AnyFetcher = match node.spec().source {
            Source::TesPureCore => &self.pure_cores,
            Source::TesTweet => &self.tweets,
            Source::TesConversationControl => &self.controls,
            Source::SafetyLabels => &self.labels,
            Source::GizmoduckViewer => &self.viewer,
            Source::GizmoduckAuthor => &self.authors,
            Source::ViewerCountry => &self.viewer_country,
            Source::Flock | Source::Wingman => self.edge_fetcher(node.edge()?)?,
        };
        Some(fetcher)
    }

    fn fetcher_mut(&mut self, node: Hydrator) -> Option<&mut dyn AnyFetcher> {
        let fetcher: &mut dyn AnyFetcher = match node.spec().source {
            Source::TesPureCore => &mut self.pure_cores,
            Source::TesTweet => &mut self.tweets,
            Source::TesConversationControl => &mut self.controls,
            Source::SafetyLabels => &mut self.labels,
            Source::GizmoduckViewer => &mut self.viewer,
            Source::GizmoduckAuthor => &mut self.authors,
            Source::ViewerCountry => &mut self.viewer_country,
            Source::Flock | Source::Wingman => self.edge_fetcher_mut(node.edge()?)?,
        };
        Some(fetcher)
    }

    fn candidate_count_by_key(&self, nodes: Hydrators) -> HashMap<u64, usize> {
        match nodes.iter().next().map(|node| node.spec().key) {
            Some(KeyOrigin::RequestTweets) => {
                return candidate_count_by_key(
                    self.request_tweets
                        .iter()
                        .map(|request_tweet| request_tweet.tweet_id.0),
                );
            }
            Some(KeyOrigin::Viewer | KeyOrigin::ViewerForCoAllowedList) => {
                return self
                    .keys(nodes)
                    .into_iter()
                    .map(|viewer| (viewer, 1))
                    .collect();
            }
            _ => {}
        }
        let mut counts = HashMap::new();
        for request_tweet in self
            .request_tweets
            .iter()
            .filter(|request_tweet| request_tweet.author.is_some())
        {
            for (position, node) in nodes.iter().enumerate() {
                let Some(key) = self.key(node.spec().key, request_tweet) else {
                    continue;
                };
                let counted = nodes
                    .iter()
                    .take(position)
                    .any(|earlier| self.key(earlier.spec().key, request_tweet) == Some(key));
                if !counted {
                    *counts.entry(key).or_default() += 1;
                }
            }
        }
        counts
    }

    pub(super) fn offer<'p>(&mut self, group: &'p Group) -> Option<CallRequest<'p>> {
        if self.viewer_id.is_none() && group.nodes.iter().all(Hydrator::needs_viewer) {
            return None;
        }
        let is_first = self.has_called.get(group.position) != Some(&true);
        let queries: Vec<EdgeQuery> = group
            .edges()
            .iter()
            .map(|&(_, graph, direction, nodes)| EdgeQuery {
                graph,
                direction,
                destination_ids: self.claim(nodes),
            })
            .collect();
        let keys = if queries.is_empty() {
            self.claim(group.nodes)
        } else {
            Vec::new()
        };
        let call_keys = || {
            let destinations = queries.iter().flat_map(|query| &query.destination_ids);
            keys.iter().chain(destinations).copied()
        };
        let key_count = call_keys().count();
        if key_count == 0 {
            return None;
        }
        if let Some(has_called) = self.has_called.get_mut(group.position) {
            *has_called = true;
        }
        let mut counts = self.candidate_count_by_key(group.nodes);
        let call_keys: HashSet<u64> = call_keys().collect();
        counts.retain(|key, _| call_keys.contains(key));
        Some(CallRequest {
            group,
            is_first,
            batch_size: self.batch_size(group, is_first, key_count),
            counts,
            viewer_id: self.viewer_id,
            keys,
            key_count,
            queries,
        })
    }

    fn batch_size(&self, group: &Group, is_first: bool, key_count: usize) -> Option<usize> {
        let size = match group.source {
            Source::TesTweet | Source::GizmoduckViewer | Source::ViewerCountry => return None,
            Source::TesPureCore | Source::TesConversationControl | Source::GizmoduckAuthor => {
                key_count
            }
            _ if !is_first => key_count,
            Source::SafetyLabels | Source::Wingman => self.requested,
            Source::Flock if group.input == Some(Hydrator::PureCore) => self.candidate_count(),
            Source::Flock => self.requested,
        };
        Some(size)
    }

    pub(super) fn is_country_lookup_skipped(&self) -> bool {
        self.callable.contains(Hydrator::ViewerCountry)
            && !self.viewer_country.has_claimed()
            && self
                .controls()
                .any(|control| control.arm == ConversationControlArm::Co)
    }

    pub(super) fn land(
        &mut self,
        call: CallRequest<'_>,
        reply: Reply,
        elapsed: Duration,
    ) -> Landing {
        let keys = &call.keys;
        match reply {
            Reply::PureCores(pure_cores) => {
                if call.is_first {
                    self.core_elapsed = elapsed;
                }
                self.pure_cores.land(keys, pure_cores);
                let pure_cores = &self.pure_cores;
                for request_tweet in &mut self.request_tweets {
                    request_tweet.author = request_tweet
                        .request_author
                        .or_else(|| Some(pure_cores.get(request_tweet.tweet_id.0)?.author_id));
                }
                if call.is_first && self.is_expanding_retweet_sources {
                    self.join_retweet_sources();
                    if self.has_joined_sources() {
                        return Landing::SourcesJoined;
                    }
                }
            }
            Reply::Tweets(tweets) => {
                if call.is_first {
                    self.tweets_elapsed = Some(elapsed);
                }
                self.tweets.land(keys, tweets);
            }
            Reply::Controls(controls) => self.controls.land(keys, controls),
            Reply::Labels(labels) => self.labels.land(keys, labels),
            Reply::Viewer(viewer) => self.viewer.land(keys, viewer),
            Reply::Authors(authors) => self.authors.land(keys, authors),
            Reply::Select(answers) => {
                let queries = call.group.edges().iter().zip(&call.queries);
                for ((&(edge, ..), query), answer) in queries.zip(answers) {
                    if let Some(fetcher) = self.edge_fetcher_mut(edge) {
                        fetcher.land(&query.destination_ids, answer);
                    }
                }
            }
            Reply::SecondDegree(answers) => {
                if let Some(fetcher) = self.edge_fetcher_mut(Edge::SecondDegree) {
                    fetcher.land(keys, answers);
                }
            }
            Reply::ViewerCountry(country) => self.viewer_country.land(keys, country),
        }
        Landing::Answered
    }

    fn join_retweet_sources(&mut self) {
        let mut known: HashSet<TweetId> = self
            .request_tweets
            .iter()
            .map(|request_tweet| request_tweet.tweet_id)
            .collect();
        let sources: Vec<RequestTweet> = self
            .request_tweets
            .iter()
            .filter_map(|request_tweet| {
                let core = self.pure_cores.get(request_tweet.tweet_id.0)?;
                let source = core
                    .source_tweet_id
                    .filter(|&source| known.insert(source))?;
                Some(RequestTweet {
                    tweet_id: source,
                    request_author: None,
                    author: core.source_author_id,
                })
            })
            .collect();
        self.request_tweets.extend(sources);
    }

    pub(super) fn assemble(mut self, request: HydrationRequest<'_>) -> Hydration {
        let incomplete = self
            .callable
            .iter()
            .filter(|&node| {
                self.fetcher(node)
                    .is_some_and(|fetcher| fetcher.has_incomplete())
            })
            .fold(Hydrators::empty(), Hydrators::with);
        let mut unclaimed = Vec::new();
        let mut tweets: HashMap<TweetId, HydratedTweet> =
            HashMap::with_capacity(self.request_tweets.len());
        for request_tweet in &self.request_tweets {
            let id = request_tweet.tweet_id;
            let tweet = tweets.entry(id).or_insert_with(|| HydratedTweet {
                candidate: None,
                has_failed_node: false,
                source_tweet_id: self
                    .pure_cores
                    .get(id.0)
                    .and_then(|core| core.source_tweet_id),
                safety_labels: self.labels.get(id.0).cloned(),
            });
            if let Some(author_id) = request_tweet.author {
                let candidate =
                    self.candidate(request_tweet, author_id, incomplete, &mut unclaimed);
                tweet.has_failed_node |= !candidate.failed.is_empty();
                tweet.candidate = Some(candidate);
            }
        }
        for ((client, method), keys) in unclaimed_keys_by_label(unclaimed) {
            record_unasked_keys(client, method, keys);
        }
        for (id, tweet) in &mut tweets {
            if let Some(candidate) = &mut tweet.candidate {
                candidate.tweet_features = self.tweets.take(id.0).unwrap_or_default();
            }
        }
        let viewer = match request.viewer_id {
            None => Viewer::LoggedOut,
            Some(id) => {
                let DecodedViewer {
                    profile,
                    has_age_verified_18_label,
                } = self.viewer.take(id).unwrap_or_default();
                Viewer::LoggedIn {
                    id,
                    profile,
                    has_age_verified_18_label,
                }
            }
        };
        Hydration {
            viewer: ViewerFeatures::from_request(
                viewer,
                request.country_code,
                request.client_capability,
            ),
            tweets,
            has_fetched_sources: self.has_joined_sources(),
        }
    }

    fn failed(
        &self,
        request_tweet: &RequestTweet,
        incomplete: Hydrators,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> Hydrators {
        let has_joined_sources = self.has_joined_sources();
        let checked = if has_joined_sources {
            self.callable
        } else {
            incomplete
        };
        let mut answered_incompletely = Hydrators::empty();
        for node in checked.iter() {
            let Some(key) = self.key(node.spec().key, request_tweet) else {
                continue;
            };
            let Some(fetcher) = self.fetcher(node) else {
                continue;
            };
            if has_joined_sources && !fetcher.is_claimed(key) {
                unclaimed.push((node, key));
                answered_incompletely = answered_incompletely.with(node);
            } else if fetcher.is_incomplete(key) {
                answered_incompletely = answered_incompletely.with(node);
            }
        }
        if answered_incompletely.is_empty() {
            return answered_incompletely;
        }
        self.callable
            .iter()
            .fold(answered_incompletely, |failed, node| match node.input() {
                Some(input_node)
                    if failed.contains(input_node)
                        && self.key(node.spec().key, request_tweet).is_none() =>
                {
                    failed.with(node)
                }
                _ => failed,
            })
    }

    fn candidate(
        &self,
        request_tweet: &RequestTweet,
        author_id: AuthorId,
        incomplete: Hydrators,
        unclaimed: &mut Vec<(Hydrator, u64)>,
    ) -> HydratedTweetCandidate {
        let id = request_tweet.tweet_id.0;
        let mut candidate = HydratedTweetCandidate {
            tweet_id: id,
            author_id: author_id.get(),
            edges: self
                .callable
                .iter()
                .filter(|&node| {
                    node.edge()
                        .and_then(|edge| {
                            let key = self.key(node.spec().key, request_tweet)?;
                            self.edge_fetcher(edge)?.get(key).copied()
                        })
                        .unwrap_or(false)
                })
                .fold(Hydrators::empty(), Hydrators::with),
            failed: self.failed(request_tweet, incomplete, unclaimed),
            ..Default::default()
        };
        if let Some(labels) = self.labels.get(id) {
            candidate.safety_labels = SafetyLabelMap::from_proto_label_types(labels);
        }
        if let Some(author) = self.authors.get(author_id.get()) {
            (candidate.author_features, candidate.author_labels) = *author;
        }
        candidate.conversation_control =
            self.control(request_tweet)
                .cloned()
                .map(|control| ConversationControlFeatures {
                    viewer_country: self
                        .viewer_id
                        .filter(|_| control.arm == ConversationControlArm::Co)
                        .and_then(|viewer_id| self.viewer_country.get(viewer_id))
                        .cloned(),
                    control,
                });
        candidate
    }
}

fn lists_countries(control: &ConversationControl) -> bool {
    control.arm == ConversationControlArm::Co && !control.allowed_country_codes.is_empty()
}

fn unclaimed_keys_by_label(
    mut unclaimed: Vec<(Hydrator, u64)>,
) -> HashMap<(&'static str, &'static str), usize> {
    unclaimed.sort_unstable_by_key(|&(node, key)| (node as u8, key));
    unclaimed.dedup();
    let mut by_label = HashMap::new();
    for (node, _) in &unclaimed {
        *by_label.entry(node.spec().label).or_default() += 1;
    }
    by_label
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::HydrationBatch;
    use crate::rules::SafetyLevel;

    #[test]
    fn authors_prefer_the_request_author_and_leave_unresolved_tweets_without_one() {
        let core = |author| PureCore {
            author_id: AuthorId(author),
            source_tweet_id: None,
            source_author_id: None,
            direct_reply_root_author_id: None,
        };
        let raw =
            [(1, Some(10)), (2, None), (3, None), (4, Some(41))].map(|(id, author)| RawCandidate {
                tweet_id: TweetId(id),
                request_author_id: author,
            });
        let plan = HydrationPlan::new(SafetyLevel::FilterAll, Hydrators::empty());
        let mut store = Store::new(&plan, None, &raw, false);
        let group = plan.groups().next().unwrap();
        let call = store
            .offer(group)
            .expect("pure core claims the request tweets");
        let pure_cores = HydrationBatch::from_results(
            [2, 3, 4],
            HashMap::from([(2, Ok::<_, &str>(Some(core(20)))), (4, Ok(Some(core(40))))]),
        );
        store.land(call, Reply::PureCores(pure_cores), Duration::ZERO);
        let resolved: Vec<(TweetId, u64)> = store
            .request_tweets
            .iter()
            .filter_map(|request_tweet| Some((request_tweet.tweet_id, request_tweet.author?.get())))
            .collect();
        assert_eq!(
            resolved,
            vec![(TweetId(1), 10), (TweetId(2), 20), (TweetId(4), 41)]
        );
    }

    #[test]
    fn a_key_no_call_claimed_fails_its_candidate_once_sources_joined() {
        let raw = [RawCandidate {
            tweet_id: TweetId(1),
            request_author_id: None,
        }];
        let plan = HydrationPlan::new(SafetyLevel::FilterAll, Hydrators::empty());
        let mut store = Store::new(&plan, None, &raw, true);
        let call = store.offer(plan.groups().next().unwrap()).unwrap();
        let retweet = PureCore {
            author_id: AuthorId(10),
            source_tweet_id: Some(TweetId(5)),
            source_author_id: Some(AuthorId(20)),
            direct_reply_root_author_id: None,
        };
        let cores =
            HydrationBatch::from_results([1], HashMap::from([(1, Ok::<_, ()>(Some(retweet)))]));
        assert_eq!(
            store.land(call, Reply::PureCores(cores), Duration::ZERO),
            Landing::SourcesJoined
        );
        let mut unclaimed = Vec::new();
        let source = &store.request_tweets[1];
        for _ in 0..2 {
            let candidate =
                store.candidate(source, AuthorId(20), Hydrators::empty(), &mut unclaimed);
            assert_eq!(candidate.failed, Hydrators::of(Hydrator::PureCore));
        }
        assert_eq!(
            unclaimed_keys_by_label(unclaimed),
            HashMap::from([(Hydrator::PureCore.spec().label, 1)])
        );
    }
}
