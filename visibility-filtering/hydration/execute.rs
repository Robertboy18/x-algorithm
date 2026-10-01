use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::decode::author::DecodedAuthor;
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::fallback_cache::FallbackCache;
use crate::hydration::metrics::{
    self, record_batch_size, record_expanded_batch, record_flock_missing_keys,
    record_viewer_country, record_wingman_second_degree, timed_results,
};
use crate::hydration::plan::{Group, Source};
use crate::hydration::sources::Sources;
use crate::hydration::store::{CallRequest, Landing, Store};
use crate::hydration::{Hydration, HydrationPlan, HydrationRequest, HYDRATION_TIMEOUT};
use crate::models::{PureCore, TweetFeatures};
use crate::rules::SafetyLevel;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use std::collections::{HashMap, HashSet};
use std::convert::identity;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use xai_core_entities::entities::ConversationControl;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Debug, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum ViewerCountry {
    NoAllowedList,
    Found,
    NoRow,
    Failed,
}

pub(super) enum Reply {
    PureCores(RawHydrationBatch<PureCore>),
    Tweets(RawHydrationBatch<TweetFeatures>),
    Controls(RawHydrationBatch<ConversationControl>),
    Labels(RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>),
    Viewer(RawHydrationBatch<DecodedViewer>),
    Authors(RawHydrationBatch<DecodedAuthor>),
    Select(Vec<RawHydrationBatch<bool>>),
    SecondDegree(RawHydrationBatch<bool>),
    ViewerCountry(RawHydrationBatch<Arc<str>>),
}

type InFlight<'p> = Pin<Box<dyn Future<Output = (CallRequest<'p>, Reply)> + Send + 'p>>;

struct Timed {
    client: String,
    method: String,
    level: SafetyLevel,
    counts: HashMap<u64, usize>,
}

impl Timed {
    async fn run<V, A: AsRef<[RawHydrationBatch<V>]>>(
        &self,
        call: impl Future<Output = A>,
        timed_out: impl FnOnce(RawHydrationBatch<V>) -> A,
    ) -> A {
        timed_results(
            &self.client,
            &self.method,
            self.level,
            &self.counts,
            HYDRATION_TIMEOUT,
            call,
            timed_out,
        )
        .await
    }

    fn keyed<'p, V, G>(
        self,
        call: CallRequest<'p>,
        get: G,
        after: impl FnOnce(RawHydrationBatch<V>) -> RawHydrationBatch<V> + Send + 'p,
        reply: fn(RawHydrationBatch<V>) -> Reply,
    ) -> InFlight<'p>
    where
        V: Send + 'p,
        G: for<'c> FnOnce(&'c CallRequest<'p>) -> BoxFuture<'c, RawHydrationBatch<V>> + Send + 'p,
    {
        Box::pin(async move {
            let answer = self.run(get(&call), identity).await;
            (call, reply(after(answer)))
        })
    }
}

impl HydrationPlan {
    pub(crate) async fn hydrate(
        &self,
        sources: &dyn Sources,
        request: HydrationRequest<'_>,
    ) -> Hydration {
        let started = Instant::now();
        let mut store = Store::new(
            self,
            request.viewer_id,
            request.raw_candidates,
            request.is_expanding_retweet_sources,
        );
        let mut running: FuturesUnordered<InFlight<'_>> = FuturesUnordered::new();
        let inputless = || self.groups().filter(|g| g.input.is_none());
        let mut ready: Vec<&Group> = inputless().collect();
        loop {
            while let Some(group) = ready.pop() {
                if let Some(call) = store.offer(group) {
                    running.extend(self.send(call, sources));
                }
            }
            let Some((call, reply)) = running.next().await else {
                break;
            };
            let group = call.group;
            if store.land(call, reply, started.elapsed()) == Landing::SourcesJoined {
                ready.extend(inputless());
            }
            ready.extend(self.readers(group));
        }
        if store.is_country_lookup_skipped() {
            record_viewer_country(ViewerCountry::NoAllowedList.into(), self.level());
        }
        metrics::record_tes_join_latency(
            self.level(),
            store
                .tweets_elapsed
                .map_or(store.core_elapsed, |tweets| tweets.max(store.core_elapsed)),
        );
        store.assemble(request)
    }

    fn send<'p>(
        &'p self,
        mut call: CallRequest<'p>,
        sources: &'p dyn Sources,
    ) -> Option<InFlight<'p>> {
        let level = self.level();
        let group = call.group;
        let (client, method) = group.label();
        if let Some(size) = call.batch_size {
            record_batch_size(&client, size);
        }
        if !call.is_first {
            record_expanded_batch(&client, &method, call.key_count);
        }
        let timed = Timed {
            client,
            method,
            level,
            counts: mem::take(&mut call.counts),
        };
        let in_flight: InFlight<'p> = match group.source {
            Source::TesPureCore => timed.keyed(
                call,
                |call| sources.pure_cores(&call.keys),
                move |pure_cores| fall_back(sources.pure_core_cache(), pure_cores),
                Reply::PureCores,
            ),
            Source::TesTweet => timed.keyed(
                call,
                |call| sources.tweets(&call.keys),
                identity,
                Reply::Tweets,
            ),
            Source::TesConversationControl => timed.keyed(
                call,
                |call| sources.conversation_controls(&call.keys),
                identity,
                Reply::Controls,
            ),
            Source::SafetyLabels => timed.keyed(
                call,
                |call| sources.safety_labels(&call.keys),
                identity,
                Reply::Labels,
            ),
            Source::GizmoduckViewer => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |_| sources.viewer(viewer_id, group.fields()),
                    identity,
                    Reply::Viewer,
                )
            }
            Source::GizmoduckAuthor => timed.keyed(
                call,
                |call| sources.users(&call.keys, group.fields()),
                move |authors| fall_back(sources.author_cache(), authors),
                Reply::Authors,
            ),
            Source::Flock => {
                let viewer_id = call.viewer_id?;
                Box::pin(async move {
                    let select = async {
                        let edges = sources.select_edges(viewer_id, &call.queries).await;
                        let (edges, missing) = missing_sets_read_no_edge(edges);
                        record_flock_missing_keys(&timed.client, &timed.method, level, missing);
                        edges
                    };
                    let per_query = |timed_out| vec![timed_out; call.queries.len()];
                    let edges = timed.run(select, per_query).await;
                    (call, Reply::Select(edges))
                })
            }
            Source::ViewerCountry => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |_| sources.viewer_country(viewer_id),
                    move |country| {
                        let result = match country.hydrated(&viewer_id) {
                            Some(Hydrated::Found(_)) => ViewerCountry::Found,
                            Some(Hydrated::NotFound) => ViewerCountry::NoRow,
                            _ => ViewerCountry::Failed,
                        };
                        record_viewer_country(result.into(), level);
                        country
                    },
                    Reply::ViewerCountry,
                )
            }
            Source::Wingman => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.second_degree(viewer_id, &call.keys),
                    move |answers| second_degree_fails_open(answers, level),
                    Reply::SecondDegree,
                )
            }
        };
        Some(in_flight)
    }
}

fn fall_back<V: Clone>(
    cache: Option<&FallbackCache<V>>,
    batch: RawHydrationBatch<V>,
) -> RawHydrationBatch<V> {
    match cache {
        Some(cache) => cache.resolve_hydration_batch(batch),
        None => batch,
    }
}

fn missing_sets_read_no_edge(
    edges: Vec<RawHydrationBatch<bool>>,
) -> (Vec<RawHydrationBatch<bool>>, usize) {
    let mut missing = HashSet::new();
    let edges = edges
        .into_iter()
        .map(|edge| {
            let mut answers = edge.into_hydrated();
            for (destination, answer) in &mut answers {
                if let Hydrated::Partial(holds) = *answer {
                    *answer = Hydrated::Found(holds);
                    missing.insert(*destination);
                }
            }
            HydrationBatch::from_hydrated(answers)
        })
        .collect();
    (edges, missing.len())
}

fn second_degree_fails_open(
    answers: RawHydrationBatch<bool>,
    level: SafetyLevel,
) -> RawHydrationBatch<bool> {
    let answers = answers.into_hydrated();
    let answered = |in_network: bool| {
        answers
            .values()
            .filter(|answer| answer.value() == Some(&in_network))
            .count()
    };
    record_wingman_second_degree(answered(true), answered(false), level);
    HydrationBatch::from_hydrated(
        answers
            .into_iter()
            .map(|(root, answer)| (root, Hydrated::Found(answer.into_value().unwrap_or(false))))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, EdgeQuery, Graph};
    use crate::hydration::decode::author::fallback_cache;
    use crate::hydration::decode::tweet::pure_core_fallback_cache;
    use crate::hydration::sources::{control, suspended, Fault, InMemorySources};
    use crate::hydration::{Hydrator, Hydrators};
    use crate::models::{
        ClientCapability, HydratedTweetCandidate, RawCandidate, TweetId, Viewer, ViewerFeatures,
        ViewerProfile,
    };
    use crate::rules::{RuleEngine, SafetyLevel};
    use xai_core_entities::entities::{
        ConversationControlArm, ExtendedProfile, GizmoduckUser, GizmoduckUserResult, PureCoreData,
        UserResponseState,
    };
    use xai_core_entities::gizmoduck_client::QueryFields;

    const VIEWER: u64 = 50;

    fn raw(tweet_id: u64, request_author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id,
        }
    }

    struct InRequestOrder {
        viewer_features: ViewerFeatures,
        candidates: Vec<HydratedTweetCandidate>,
        safety_labels: HashMap<TweetId, Arc<vf_pb::SafetyLabelMap>>,
        failed_ids: HashSet<TweetId>,
    }

    fn in_request_order(hydration: &Hydration, raw: &[RawCandidate]) -> InRequestOrder {
        let tweets = || {
            raw.iter()
                .filter_map(|c| Some((c.tweet_id, hydration.tweet(c.tweet_id)?)))
        };
        InRequestOrder {
            viewer_features: hydration.viewer().clone(),
            candidates: tweets()
                .filter_map(|(_, tweet)| tweet.candidate().cloned())
                .collect(),
            safety_labels: tweets()
                .filter_map(|(id, tweet)| Some((id, tweet.safety_labels()?.clone())))
                .collect(),
            failed_ids: tweets()
                .filter(|(_, tweet)| tweet.has_failed_node())
                .map(|(id, _)| id)
                .collect(),
        }
    }

    async fn hydrate(
        sources: &InMemorySources,
        level: SafetyLevel,
        viewer_id: Option<u64>,
        raw: &[RawCandidate],
    ) -> InRequestOrder {
        let hydration = RuleEngine::for_tests()
            .plan(level)
            .hydrate(
                sources,
                HydrationRequest::new(
                    viewer_id,
                    Some("US".into()),
                    ClientCapability::default(),
                    raw,
                ),
            )
            .await;
        in_request_order(&hydration, raw)
    }

    fn ids(ids: &[u64]) -> HashSet<TweetId> {
        ids.iter().copied().map(TweetId).collect()
    }

    fn exclusive_tweet() -> TweetFeatures {
        TweetFeatures {
            exclusive_conversation_author_id: Some(30),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn failed_ids_reports_exactly_the_candidates_each_node_flags() {
        use ConversationControlArm::{Co, Subscribers};
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || InMemorySources::default().tweet(1, 10).tweet(2, 20);
        let rows = [
            ("healthy home", TimelineHome, world(), vec![]),
            (
                "healthy home hydration",
                TimelineHomeHydration,
                world(),
                vec![],
            ),
            (
                "failed pure core",
                TimelineHome,
                world().fault(Source::TesPureCore, Fault::Fails),
                vec![1, 2],
            ),
            (
                "incomplete author",
                TimelineHome,
                world().user(
                    10,
                    GizmoduckUserResult {
                        response_state: Some(UserResponseState::Failed),
                        ..suspended()
                    },
                ),
                vec![1],
            ),
            (
                "failed author-keyed select",
                TimelineHome,
                world().fail_graph(Graph::Mutes),
                vec![1, 2],
            ),
            (
                "failed blocked-by select",
                TimelineHomeHydration,
                world().fail_graph(Graph::Blocks),
                vec![1, 2],
            ),
            (
                "failed tweet row",
                TimelineHome,
                world().fail_key(Source::TesTweet, 1),
                vec![1],
            ),
            (
                "failed label lookup",
                TimelineHome,
                world().fail_key(Source::SafetyLabels, 1),
                vec![1],
            ),
            (
                "failed exclusive select",
                TimelineHome,
                world()
                    .tweet_features(1, exclusive_tweet())
                    .fail_graph(Graph::SuperFollows),
                vec![1],
            ),
            (
                "failed root-edge select",
                TimelineHomeHydration,
                world()
                    .control(1, control(Subscribers, 30, &[]))
                    .fail_graph(Graph::SuperFollows),
                vec![1],
            ),
            (
                "failed conversation-control row",
                TimelineHomeHydration,
                world().fail_key(Source::TesConversationControl, 1),
                vec![1],
            ),
            (
                "failed country lookup",
                TimelineHomeHydration,
                world()
                    .control(1, control(Co, 30, &["us"]))
                    .control(2, control(Co, 30, &[]))
                    .fault(Source::ViewerCountry, Fault::Fails),
                vec![1],
            ),
            (
                "failed viewer",
                TimelineHome,
                world().fault(Source::GizmoduckViewer, Fault::Fails),
                vec![1, 2],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, sources, expected) in rows {
            let hydrated = hydrate(&sources, level, Some(VIEWER), &raw).await;
            assert_eq!(hydrated.failed_ids, ids(&expected), "{name}");
        }
    }

    #[tokio::test]
    async fn each_candidate_carries_the_nodes_that_failed_for_it() {
        use ConversationControlArm::MyNetwork;
        use Hydrator::{
            BlockedByAuthor, BlockedByReplyRoot, Follows, PureCore, RootFollowsViewer,
            RootFollowsViewerSecondDegree, SuperFollowsExclusive, Tweet,
        };
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || InMemorySources::default().tweet(1, 10).tweet(2, 20);
        let rows = [
            (
                "failed tweet row",
                TimelineHome,
                Some(VIEWER),
                world()
                    .tweet_features(2, exclusive_tweet())
                    .fail_key(Source::TesTweet, 1),
                [
                    Hydrators::of(Tweet).with(SuperFollowsExclusive),
                    Hydrators::empty(),
                ],
            ),
            (
                "failed root edge on a MyNetwork reply",
                TimelineHomeHydration,
                Some(VIEWER),
                world()
                    .control(1, control(MyNetwork, 30, &[]))
                    .fail_graph(Graph::Follows),
                [
                    Hydrators::of(RootFollowsViewer)
                        .with(RootFollowsViewerSecondDegree)
                        .with(Follows)
                        .with(BlockedByAuthor),
                    Hydrators::of(Follows).with(BlockedByAuthor),
                ],
            ),
            (
                "failed pure core, authors from the request",
                TimelineHomeHydration,
                Some(VIEWER),
                world().fault(Source::TesPureCore, Fault::Fails),
                [Hydrators::of(PureCore).with(BlockedByReplyRoot); 2],
            ),
            (
                "failed pure core, logged out",
                TimelineHomeHydration,
                None,
                world().fault(Source::TesPureCore, Fault::Fails),
                [Hydrators::of(PureCore); 2],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, viewer_id, sources, expected) in rows {
            let hydrated = hydrate(&sources, level, viewer_id, &raw).await;
            let failed: Vec<Hydrators> = hydrated.candidates.iter().map(|c| c.failed).collect();
            assert_eq!(failed, expected, "{name}");
        }
    }

    #[tokio::test]
    async fn safety_labels_carry_only_the_tweets_whose_labels_were_found() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .tweet(3, 30)
            .labels(2, Default::default())
            .fail_key(Source::SafetyLabels, 1);
        let raw = [raw(1, None), raw(2, None), raw(3, None)];
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(
            hydrated
                .safety_labels
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            ids(&[2])
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_timed_out_label_lookup_fails_every_tweet() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .fault(Source::SafetyLabels, Fault::Hangs);
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &[raw(1, None)]).await;
        assert_eq!(hydrated.failed_ids, ids(&[1]));
        assert!(hydrated.safety_labels.is_empty());
    }

    fn author_keyed(sources: &InMemorySources) -> bool {
        sources.calls().contains(&Source::GizmoduckAuthor) || !sources.selects().is_empty()
    }

    #[tokio::test(start_paused = true)]
    async fn author_keyed_calls_wait_for_pure_core() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .fault(Source::TesPureCore, Fault::Hangs);
        let raw = [raw(1, None)];
        let hydration = hydrate(&sources, SafetyLevel::TimelineHome, Some(VIEWER), &raw);
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
        assert!(early.is_err());
        assert!(!author_keyed(&sources));
        let hydrated = hydration.await;
        assert!(!author_keyed(&sources));
        assert!(hydrated.candidates.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn author_calls_do_not_wait_for_tweets() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended())
            .edge(Graph::Follows, VIEWER, 10)
            .fault(Source::TesTweet, Fault::Hangs);
        let raw = [raw(1, None)];
        let started = tokio::time::Instant::now();
        let hydration = hydrate(&sources, SafetyLevel::TimelineHome, Some(VIEWER), &raw);
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
        assert!(early.is_err());
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert_eq!(sources.selects().len(), 1);

        let hydrated = hydration.await;
        assert_eq!(started.elapsed(), HYDRATION_TIMEOUT);
        let candidate = &hydrated.candidates[0];
        assert!(candidate.author_features.is_suspended);
        assert!(candidate.edges.contains(Hydrator::Follows));
        assert_eq!(candidate.tweet_features, Default::default());
        assert_eq!(hydrated.failed_ids, ids(&[1]));
    }

    fn sorted(mut calls: Vec<Source>) -> Vec<String> {
        let mut names: Vec<String> = calls.drain(..).map(|s| format!("{s:?}")).collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn empty_key_sets_and_logged_out_viewers_send_no_call() {
        let raw = [raw(1, None)];
        let logged_out = InMemorySources::default().tweet(1, 10);
        hydrate(&logged_out, SafetyLevel::TimelineHomeHydration, None, &raw).await;
        assert_eq!(
            sorted(logged_out.calls()),
            [
                "GizmoduckAuthor",
                "SafetyLabels",
                "TesConversationControl",
                "TesPureCore",
                "TesTweet"
            ]
        );

        let logged_in = InMemorySources::default().tweet(1, 10);
        hydrate(
            &logged_in,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert!(!logged_in.calls().contains(&Source::ViewerCountry));
        assert_eq!(
            logged_in.selects(),
            [vec![
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Forward,
                    destination_ids: vec![10],
                },
                EdgeQuery {
                    graph: Graph::Blocks,
                    direction: EdgeDirection::Reverse,
                    destination_ids: vec![10],
                },
            ]]
        );
    }

    #[tokio::test]
    async fn filter_all_calls_pure_core_only_and_keeps_the_request_side_viewer() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet_features(1, exclusive_tweet());
        let hydrated = hydrate(
            &sources,
            SafetyLevel::FilterAll,
            Some(VIEWER),
            &[raw(1, None)],
        )
        .await;
        assert_eq!(sources.calls(), [Source::TesPureCore]);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
        assert_eq!(hydrated.viewer_features.country_code.as_deref(), Some("us"));
        assert_eq!(hydrated.candidates[0].author_id, 10);
        assert!(hydrated.safety_labels.is_empty());
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_failed_viewer_call_fails_both_viewer_nodes_unverified() {
        use crate::hydration::plan::HydrationPlan;
        use xai_core_entities::entities::{Label, Labels};
        use xai_x_thrift::user_labels::LabelValue;
        let viewer_nodes = Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels);
        let labeled = GizmoduckUser {
            labels: Labels {
                labels: vec![Label {
                    label_value: LabelValue::AGE_VERIFIED_18.0,
                    created_at_msec: 0,
                }],
            },
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .viewer(VIEWER, labeled)
            .fault(Source::GizmoduckViewer, Fault::Fails);
        let raw = [raw(1, None)];
        let hydration = HydrationPlan::new(SafetyLevel::TimelineHomeHydration, viewer_nodes)
            .hydrate(
                &sources,
                HydrationRequest::new(Some(VIEWER), None, ClientCapability::default(), &raw),
            )
            .await;
        let hydrated = in_request_order(&hydration, &raw);
        assert_eq!(hydrated.candidates[0].failed, viewer_nodes);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
    }

    #[tokio::test]
    async fn a_level_that_plans_the_viewer_profile_decodes_it() {
        let sources = InMemorySources::default().viewer(
            VIEWER,
            GizmoduckUser {
                extended_profile: Some(ExtendedProfile {
                    age_in_years: Some(30),
                }),
                ..Default::default()
            },
        );
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, Some(VIEWER), &[]).await;
        let Viewer::LoggedIn { profile, .. } = hydrated.viewer_features.viewer else {
            panic!("a logged-in request stays logged in")
        };
        assert_ne!(profile, ViewerProfile::default());
    }

    #[tokio::test]
    async fn exclusive_edges_dedup_conversation_authors() {
        let sources = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .tweet_features(1, exclusive_tweet())
                .tweet_features(2, exclusive_tweet())
                .edge(Graph::SuperFollows, VIEWER, 30)
        };
        let raw = [raw(1, None), raw(2, None), raw(1, None), raw(3, Some(40))];
        for viewer_id in [Some(VIEWER), None] {
            let sources = sources();
            let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, viewer_id, &raw).await;
            assert_eq!(sources.keys(Source::TesPureCore), [vec![1, 2, 3]]);
            assert_eq!(sources.keys(Source::TesTweet), [vec![1, 2, 3]]);
            let exclusive = (Some(30), viewer_id.is_some());
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| (
                        c.tweet_features.exclusive_conversation_author_id,
                        c.edges.contains(Hydrator::SuperFollowsExclusive)
                    ))
                    .collect::<Vec<_>>(),
                [exclusive, exclusive, exclusive, (None, false)]
            );
            let super_follows: Vec<Vec<u64>> = sources
                .selects()
                .into_iter()
                .flatten()
                .filter(|query| query.graph == Graph::SuperFollows)
                .map(|query| query.destination_ids)
                .collect();
            let expected: &[Vec<u64>] = if viewer_id.is_some() {
                &[vec![30]]
            } else {
                &[]
            };
            assert_eq!(super_follows, expected);
        }
    }

    #[tokio::test]
    async fn a_super_follows_key_two_groups_read_is_asked_once() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .tweet_features(1, exclusive_tweet())
            .control(2, control(ConversationControlArm::Subscribers, 30, &[]))
            .edge(Graph::SuperFollows, VIEWER, 30);
        let raw = [raw(1, None), raw(2, None)];
        let hydrated = hydrate(
            &sources,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        let super_follows: Vec<Vec<u64>> = sources
            .selects()
            .into_iter()
            .flatten()
            .filter(|query| query.graph == Graph::SuperFollows)
            .map(|query| query.destination_ids)
            .collect();
        assert_eq!(super_follows, [vec![30]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.edges.contains(Hydrator::SuperFollowsExclusive),
                    c.edges.contains(Hydrator::SuperFollowsRoot)
                ))
                .collect::<Vec<_>>(),
            [(true, false), (false, true)]
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    fn root_edges(sources: &InMemorySources) -> Vec<Vec<EdgeQuery>> {
        sources
            .selects()
            .into_iter()
            .filter(|queries| {
                queries.iter().any(|query| {
                    query.graph == Graph::Follows && query.direction == EdgeDirection::Reverse
                })
            })
            .collect()
    }

    #[tokio::test]
    async fn one_select_carries_both_root_edges_and_a_failure_fails_every_tweet_it_keyed() {
        use ConversationControlArm::{ByInvitation, Community, MyNetwork, Subscribers};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(Subscribers, 40, &[]))
                .control(4, control(ByInvitation, 40, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .edge(Graph::SuperFollows, VIEWER, 40)
        };
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let facts = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    (
                        c.edges.contains(Hydrator::RootFollowsViewer),
                        c.edges.contains(Hydrator::SuperFollowsRoot),
                    )
                })
                .collect::<Vec<_>>()
        };

        let healthy = world();
        let hydrated = hydrate(
            &healthy,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert_eq!(
            root_edges(&healthy),
            [vec![
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Reverse,
                    destination_ids: vec![30],
                },
                EdgeQuery {
                    graph: Graph::SuperFollows,
                    direction: EdgeDirection::Forward,
                    destination_ids: vec![40],
                },
            ]]
        );
        assert_eq!(
            facts(&hydrated),
            [(true, false), (true, false), (false, true), (false, false)]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fail_graph(Graph::SuperFollows);
        let hydrated = hydrate(
            &failed,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert_eq!(facts(&hydrated), [(false, false); 4]);
        assert_eq!(hydrated.failed_ids, ids(&[1, 2, 3]));

        let logged_out = world();
        let hydrated = hydrate(&logged_out, SafetyLevel::TimelineHomeHydration, None, &raw).await;
        assert!(root_edges(&logged_out).is_empty());
        assert_eq!(facts(&hydrated), [(false, false); 4]);
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_query_missing_from_the_select_answer_reads_no_edge_and_fails_no_candidate() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .edge(Graph::Follows, VIEWER, 10)
            .edge(Graph::Mutes, VIEWER, 10)
            .miss_graph(Graph::Mutes);
        let hydrated = hydrate(
            &sources,
            SafetyLevel::TimelineHome,
            Some(VIEWER),
            &[raw(1, None)],
        )
        .await;
        assert_eq!(
            hydrated.candidates[0].edges,
            Hydrators::of(Hydrator::Follows)
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn wingman_asks_once_for_my_network_roots_the_followed_by_edge_answered_no() {
        use ConversationControlArm::{Community, MyNetwork};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .tweet(5, 10)
                .tweet(6, 10)
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(MyNetwork, 60, &[]))
                .control(4, control(MyNetwork, 60, &[]))
                .control(5, control(MyNetwork, 70, &[]))
                .control(6, control(Community, 80, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .second_degree_path(60, VIEWER)
        };
        let raw = [1, 2, 3, 4, 5, 6].map(|id| raw(id, None));
        let second_degree = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| c.edges.contains(Hydrator::RootFollowsViewerSecondDegree))
                .collect::<Vec<_>>()
        };
        let level = SafetyLevel::TimelineHomeHydration;

        let healthy = world();
        let hydrated = hydrate(&healthy, level, Some(VIEWER), &raw).await;
        assert_eq!(healthy.keys(Source::Wingman), [vec![60, 70]]);
        assert_eq!(
            second_degree(&hydrated),
            [false, false, true, true, false, false]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fault(Source::Wingman, Fault::Fails);
        let hydrated = hydrate(&failed, level, Some(VIEWER), &raw).await;
        assert_eq!(second_degree(&hydrated), [false; 6]);
        assert!(hydrated.failed_ids.is_empty());

        for (sources, viewer_id, raw) in [
            (world().fail_graph(Graph::Follows), Some(VIEWER), &raw[..]),
            (world(), None, &raw[..]),
            (world(), Some(VIEWER), &raw[..2]),
        ] {
            let hydrated = hydrate(&sources, level, viewer_id, raw).await;
            assert!(sources.keys(Source::Wingman).is_empty());
            assert_eq!(second_degree(&hydrated), vec![false; raw.len()]);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_root_edge_call_started_before_pure_core_lands_fails_no_candidate_when_it_times_out()
    {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .control(1, control(ConversationControlArm::Subscribers, 40, &[]))
            .edge(Graph::SuperFollows, VIEWER, 40)
            .fault(Source::TesPureCore, Fault::Delays(HYDRATION_TIMEOUT / 2))
            .hang_graph(Graph::SuperFollows);
        let raw = [raw(1, None)];
        let hydration = hydrate(
            &sources,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        );
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 4, &mut hydration).await;
        assert!(early.is_err());
        assert!(sources
            .selects()
            .concat()
            .contains(&EdgeQuery::forward(Graph::SuperFollows, vec![40])));
        assert!(!sources.calls().contains(&Source::GizmoduckAuthor));

        let hydrated = hydration.await;
        assert!(!hydrated.candidates[0]
            .edges
            .contains(Hydrator::SuperFollowsRoot));
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn one_country_lookup_reaches_every_co_tweet_that_needs_it() {
        use ConversationControlArm::Co;
        let raw = [raw(1, None), raw(2, None)];
        let world = |countries: &[&str]| {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .control(1, control(Co, 30, countries))
                .control(2, control(Co, 30, countries))
                .country(VIEWER, "us")
        };
        let country = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    c.conversation_control
                        .as_ref()
                        .unwrap()
                        .viewer_country
                        .as_deref()
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        };

        let listed = world(&["us"]);
        let hydrated = hydrate(
            &listed,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert_eq!(listed.keys(Source::ViewerCountry), [vec![VIEWER]]);
        assert_eq!(
            country(&hydrated),
            [Some("us".to_owned()), Some("us".to_owned())]
        );

        for (sources, viewer_id) in [(world(&[]), Some(VIEWER)), (world(&["us"]), None)] {
            let hydrated = hydrate(
                &sources,
                SafetyLevel::TimelineHomeHydration,
                viewer_id,
                &raw,
            )
            .await;
            assert!(sources.keys(Source::ViewerCountry).is_empty());
            assert_eq!(country(&hydrated), [None, None]);
        }
    }

    #[tokio::test]
    async fn authors_share_one_key_and_a_missing_user_is_complete() {
        let sources = InMemorySources::default().tweet(1, 10).tweet(2, 10);
        let raw = [raw(1, None), raw(2, None)];
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert!(hydrated
            .candidates
            .iter()
            .all(|c| !c.author_features.is_suspended));
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn repeated_ids_keep_the_last_resolved_candidate_and_fail_if_any_occurrence_failed() {
        let sources = InMemorySources::default().user(
            10,
            GizmoduckUserResult {
                response_state: Some(UserResponseState::Failed),
                ..suspended()
            },
        );
        let raw = [raw(1, Some(10)), raw(1, Some(20)), raw(1, None)];
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10, 20]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (c.author_id, c.failed))
                .collect::<Vec<_>>(),
            [(20, Hydrators::empty()); 3]
        );
        assert_eq!(hydrated.failed_ids, ids(&[1]));
    }

    #[tokio::test]
    async fn the_author_cache_serves_the_last_known_author_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended())
            .with_author_cache(fallback_cache());
        let raw = [raw(1, None)];
        let first = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert!(first.candidates[0].author_features.is_suspended);

        sources.break_source(Source::GizmoduckAuthor, Fault::Fails);
        let second = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert!(second.candidates[0].author_features.is_suspended);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn the_pure_core_cache_serves_the_last_known_core_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .with_pure_core_cache(pure_core_fallback_cache(8));
        let raw = [raw(1, None)];
        let first = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(first.candidates[0].author_id, 10);

        sources.break_source(Source::TesPureCore, Fault::Fails);
        let second = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(second.candidates[0].author_id, 10);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_pure_core_not_found_is_not_served_when_the_call_fails() {
        let sources = InMemorySources::default().with_pure_core_cache(pure_core_fallback_cache(8));
        let raw = [raw(1, Some(10))];
        let first = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(first.candidates[0].author_id, 10);
        assert!(first.failed_ids.is_empty());

        sources.break_source(Source::TesPureCore, Fault::Fails);
        let second = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(second.failed_ids, ids(&[1]));
    }

    #[tokio::test]
    async fn gizmoduck_calls_ask_for_every_field_any_level_reads() {
        use QueryFields::{ACCOUNT, EXTENDED_PROFILE, LABELS, SAFETY};
        for (level, viewer_fields) in [
            (
                SafetyLevel::TimelineHome,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY],
            ),
            (
                SafetyLevel::TimelineHomeHydration,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY, LABELS],
            ),
        ] {
            let sources = InMemorySources::default().tweet(1, 10);
            hydrate(&sources, level, Some(VIEWER), &[raw(1, None)]).await;
            assert_eq!(
                sources.fields(Source::GizmoduckViewer),
                [viewer_fields],
                "{level:?}"
            );
            assert_eq!(
                sources.fields(Source::GizmoduckAuthor),
                [vec![SAFETY, LABELS]],
                "{level:?}"
            );
        }
    }

    #[test]
    fn a_first_call_sent_after_sources_join_records_the_requests_own_batch_size() {
        use crate::models::AuthorId;
        let engine = RuleEngine::for_tests();
        let plan = engine.plan(SafetyLevel::TimelineHomeHydration);
        let raw = [raw(1, None)];
        let mut store = Store::new(plan, Some(VIEWER), &raw, true);
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
            store.land(call, Reply::PureCores(cores), std::time::Duration::ZERO),
            Landing::SourcesJoined
        );
        for source in [Source::SafetyLabels, Source::Flock] {
            let group = plan.groups().find(|group| group.source == source).unwrap();
            let call = store.offer(group).unwrap();
            assert_eq!(
                (call.is_first, call.batch_size),
                (true, Some(1)),
                "{source:?}"
            );
        }
    }

    fn call_labels(sources: &InMemorySources) -> Vec<String> {
        let mut labels: Vec<String> = sources
            .calls()
            .into_iter()
            .filter(|source| *source != Source::Flock)
            .map(|source| format!("{source:?}"))
            .chain(sources.selects().into_iter().map(|queries| {
                let graphs: Vec<&str> = queries.iter().map(|q| <&str>::from(q.graph)).collect();
                format!("Flock {}", graphs.join(","))
            }))
            .collect();
        labels.sort();
        labels
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_source_delays_only_the_calls_waiting_on_it() {
        use ConversationControlArm::{Co, Community, MyNetwork};
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .pure_core(
                    1,
                    PureCoreData {
                        author_id: 10,
                        conversation_id: Some(100),
                        in_reply_to_tweet_id: Some(100),
                        in_reply_to_user_id: Some(30),
                        ..Default::default()
                    },
                )
                .tweet(2, 20)
                .tweet_features(1, exclusive_tweet())
                .control(1, control(Community, 30, &[]))
                .control(2, control(Co, 30, &["us"]))
                .tweet(3, 20)
                .control(3, control(MyNetwork, 40, &[]))
        };
        let rows: [(SafetyLevel, Source, &[&str]); 15] = [
            (
                TimelineHome,
                Source::TesPureCore,
                &[
                    "Flock follows,blocks,mutes,mute_retweets",
                    "GizmoduckAuthor",
                ],
            ),
            (TimelineHome, Source::TesTweet, &["Flock super_follows"]),
            (TimelineHome, Source::SafetyLabels, &[]),
            (TimelineHome, Source::GizmoduckViewer, &[]),
            (TimelineHome, Source::GizmoduckAuthor, &[]),
            (TimelineHome, Source::Flock, &[]),
            (
                TimelineHomeHydration,
                Source::TesPureCore,
                &["Flock follows,blocks", "GizmoduckAuthor"],
            ),
            (
                TimelineHomeHydration,
                Source::TesTweet,
                &["Flock super_follows"],
            ),
            (
                TimelineHomeHydration,
                Source::TesConversationControl,
                &["Flock follows,super_follows", "ViewerCountry", "Wingman"],
            ),
            (TimelineHomeHydration, Source::SafetyLabels, &[]),
            (TimelineHomeHydration, Source::GizmoduckViewer, &[]),
            (TimelineHomeHydration, Source::GizmoduckAuthor, &[]),
            (TimelineHomeHydration, Source::Flock, &["Wingman"]),
            (TimelineHomeHydration, Source::ViewerCountry, &[]),
            (TimelineHomeHydration, Source::Wingman, &[]),
        ];
        let raw = [raw(1, None), raw(2, None), raw(3, None)];
        for (level, hung, waiting) in rows {
            let healthy = world();
            hydrate(&healthy, level, Some(VIEWER), &raw).await;
            let every_call = call_labels(&healthy);
            let groups = RuleEngine::for_tests().plan(level).groups().count();
            assert_eq!(every_call.len(), groups, "{level:?}: {every_call:?}");
            assert!(
                waiting
                    .iter()
                    .all(|call| every_call.iter().any(|c| c == call)),
                "{level:?} {hung:?}: {every_call:?}"
            );

            let sources = world().fault(hung, Fault::Hangs);
            let started = tokio::time::Instant::now();
            let hydration = hydrate(&sources, level, Some(VIEWER), &raw);
            tokio::pin!(hydration);
            let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
            assert!(early.is_err(), "{level:?} {hung:?}");
            let expected: Vec<String> = every_call
                .iter()
                .filter(|call| !waiting.contains(&call.as_str()))
                .cloned()
                .collect();
            assert_eq!(call_labels(&sources), expected, "{level:?} {hung:?}");
            hydration.await;
            assert_eq!(started.elapsed(), HYDRATION_TIMEOUT, "{level:?} {hung:?}");
        }
    }
}
