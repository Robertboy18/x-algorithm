use crate::hydration::sources::Sources;
use crate::hydration::{HydratedTweet, Hydration, HydrationRequest, Hydrators};
use crate::models::{ClientCapability, HydratedTweetCandidate, RawCandidate, TweetId, Verdict};
use crate::rules::metrics::{self as ft_metrics, RetweetSources, Rpc};
use crate::rules::{RuleEngine, SafetyLevel};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterRequest {
    pub viewer_id: Option<u64>,
    pub country_code: Option<String>,
    pub client_capability: ClientCapability,
    pub safety_level: SafetyLevel,
    pub candidates: Vec<RawCandidate>,
    pub rpc: Rpc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationStatus {
    Evaluated,
    UnresolvedAuthor,
    Failed,
}

#[derive(Clone)]
pub struct FilterOutcome {
    pub tweet_id: TweetId,
    pub source_tweet_id: Option<TweetId>,
    pub verdict: Verdict,
    pub rested_on: Hydrators,
    pub status: EvaluationStatus,
    pub safety_labels: Option<vf_pb::SafetyLabelMap>,
}

pub struct FilterResponse {
    pub outcomes: Vec<FilterOutcome>,
}

pub struct FilterTweets {
    sources: Arc<dyn Sources>,
    rule_engine: RuleEngine,
    #[cfg(test)]
    pub(crate) client_capabilities: std::sync::Mutex<Vec<ClientCapability>>,
}

impl FilterTweets {
    pub(crate) fn new(sources: Arc<dyn Sources>, rule_engine: RuleEngine) -> Self {
        Self {
            sources,
            rule_engine,
            #[cfg(test)]
            client_capabilities: std::sync::Mutex::default(),
        }
    }

    pub async fn run(&self, request: FilterRequest) -> FilterResponse {
        let hydrated = self.hydrate(request).await;
        FilterResponse {
            outcomes: self.evaluate_in_request_order(&hydrated),
        }
    }

    pub(crate) async fn hydrate(&self, request: FilterRequest) -> HydratedRequest {
        self.hydrate_request(request, false).await
    }

    pub(crate) async fn hydrate_with_retweet_sources(
        &self,
        request: FilterRequest,
    ) -> HydratedRequest {
        self.hydrate_request(request, true).await
    }

    async fn hydrate_request(
        &self,
        request: FilterRequest,
        is_expanding_retweet_sources: bool,
    ) -> HydratedRequest {
        #[cfg(test)]
        self.client_capabilities
            .lock()
            .unwrap()
            .push(request.client_capability);
        let started = Instant::now();
        let hydration = self
            .rule_engine
            .plan(request.safety_level)
            .hydrate(
                &*self.sources,
                HydrationRequest::new(
                    request.viewer_id,
                    request.country_code,
                    request.client_capability,
                    &request.candidates,
                )
                .with_retweet_sources(is_expanding_retweet_sources),
            )
            .await;
        let hydrated_at = Instant::now();
        let retweet_sources = if !is_expanding_retweet_sources {
            RetweetSources::NoSource
        } else if hydration.has_fetched_sources() {
            RetweetSources::Fetched
        } else if request.candidates.iter().any(|candidate| {
            hydration
                .tweet(candidate.tweet_id)
                .is_some_and(|tweet| tweet.is_evaluable() && tweet.source_tweet_id().is_some())
        }) {
            RetweetSources::InBatch
        } else {
            RetweetSources::NoSource
        };
        ft_metrics::record_phase(
            request.rpc,
            "hydration",
            retweet_sources,
            hydrated_at - started,
        );
        HydratedRequest {
            safety_level: request.safety_level,
            rpc: request.rpc,
            retweet_sources,
            candidates: request.candidates,
            hydration,
        }
    }

    pub(crate) fn evaluate_in_request_order(
        &self,
        hydrated: &HydratedRequest,
    ) -> Vec<FilterOutcome> {
        let evaluating = Instant::now();
        let outcomes = hydrated
            .requested_ids()
            .map(|tweet_id| self.outcome(hydrated, tweet_id, None))
            .collect();
        ft_metrics::record_phase(
            hydrated.rpc,
            "post_hydration",
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        outcomes
    }

    pub(crate) fn evaluate(
        &self,
        hydrated: &HydratedRequest,
        ids: impl IntoIterator<Item = TweetId>,
        copied_retweets: &HashMap<TweetId, HydratedTweetCandidate>,
    ) -> HashMap<TweetId, FilterOutcome> {
        let evaluating = Instant::now();
        let mut outcomes = HashMap::new();
        for tweet_id in ids {
            outcomes.entry(tweet_id).or_insert_with(|| {
                self.outcome(hydrated, tweet_id, copied_retweets.get(&tweet_id))
            });
        }
        ft_metrics::record_phase(
            hydrated.rpc,
            "post_hydration",
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        outcomes
    }

    fn outcome(
        &self,
        hydrated: &HydratedRequest,
        tweet_id: TweetId,
        copied_retweet: Option<&HydratedTweetCandidate>,
    ) -> FilterOutcome {
        let tweet = hydrated.hydration.tweet(tweet_id);
        let (verdict, rested_on, status) = match copied_retweet.or_else(|| tweet?.candidate()) {
            None => (
                Verdict::unresolved_author(),
                Hydrators::empty(),
                EvaluationStatus::UnresolvedAuthor,
            ),
            Some(candidate) => {
                let evaluation = self.rule_engine.evaluate(
                    hydrated.safety_level,
                    hydrated.hydration.viewer(),
                    candidate,
                );
                let status = if tweet.is_some_and(HydratedTweet::has_failed_node) {
                    EvaluationStatus::Failed
                } else {
                    EvaluationStatus::Evaluated
                };
                (evaluation.verdict, evaluation.rested_on, status)
            }
        };
        FilterOutcome {
            tweet_id,
            source_tweet_id: tweet.and_then(HydratedTweet::source_tweet_id),
            verdict,
            rested_on,
            status,
            safety_labels: tweet
                .and_then(HydratedTweet::safety_labels)
                .map(|labels| vf_pb::SafetyLabelMap::clone(labels)),
        }
    }
}

pub(crate) struct HydratedRequest {
    safety_level: SafetyLevel,
    rpc: Rpc,
    retweet_sources: RetweetSources,
    candidates: Vec<RawCandidate>,
    hydration: Hydration,
}

impl HydratedRequest {
    pub(crate) fn hydration(&self) -> &Hydration {
        &self.hydration
    }

    #[cfg(test)]
    pub(crate) fn retweet_sources(&self) -> RetweetSources {
        self.retweet_sources
    }

    pub(crate) fn requested_ids(&self) -> impl ExactSizeIterator<Item = TweetId> + '_ {
        self.candidates.iter().map(|candidate| candidate.tweet_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeQuery, Graph};
    use crate::hydration::plan::Source;
    use crate::hydration::sources::{Fault, InMemorySources};
    use crate::models::LimitedEngagementReason;
    use crate::rules::fixtures::{allow, limited};
    use xai_core_entities::entities::PureCoreData;

    fn candidate(tweet_id: u64, author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id: author_id,
        }
    }

    fn service(sources: &Arc<InMemorySources>) -> FilterTweets {
        FilterTweets::new(sources.clone(), RuleEngine::for_tests())
    }

    #[tokio::test(start_paused = true)]
    async fn pure_core_timeout_fails_every_candidate_at_hydration_timeout() {
        let sources = Arc::new(InMemorySources::default().fault(Source::TesPureCore, Fault::Hangs));
        let started = tokio::time::Instant::now();
        let response = tokio::time::timeout(
            crate::hydration::HYDRATION_TIMEOUT * 2,
            service(&sources).run(FilterRequest {
                viewer_id: Some(50),
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![candidate(1, None), candidate(2, Some(20))],
                rpc: Rpc::FilterTweets,
            }),
        )
        .await
        .unwrap();
        assert_eq!(started.elapsed(), crate::hydration::HYDRATION_TIMEOUT);
        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| (outcome.verdict.clone(), outcome.status))
                .collect::<Vec<_>>(),
            vec![
                (
                    Verdict::unresolved_author(),
                    EvaluationStatus::UnresolvedAuthor
                ),
                (allow(), EvaluationStatus::Failed),
            ]
        );
    }

    #[tokio::test]
    async fn home_hydration_limits_posts_whose_author_or_direct_reply_root_blocks_the_viewer() {
        let reply = |author_id, in_reply_to_tweet_id, in_reply_to_user_id| PureCoreData {
            author_id,
            conversation_id: Some(100),
            in_reply_to_tweet_id: Some(in_reply_to_tweet_id),
            in_reply_to_user_id: Some(in_reply_to_user_id),
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(1, reply(10, 100, 30))
                .pure_core(2, reply(20, 101, 40))
                .tweet(3, 10)
                .edge(Graph::Blocks, 20, 50)
                .edge(Graph::Blocks, 30, 50),
        );
        let service = &service(&sources);
        let verdicts = |viewer_id| async move {
            service
                .run(FilterRequest {
                    viewer_id,
                    country_code: None,
                    client_capability: ClientCapability::default(),
                    safety_level: SafetyLevel::TimelineHomeHydration,
                    candidates: vec![candidate(1, None), candidate(2, None), candidate(3, None)],
                    rpc: Rpc::FilterTweets,
                })
                .await
                .outcomes
                .into_iter()
                .map(|outcome| (outcome.status, outcome.verdict))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            verdicts(Some(50)).await,
            vec![
                (
                    EvaluationStatus::Evaluated,
                    limited(
                        LimitedEngagementReason::RootAuthorBlockedViewer,
                        "blocked_viewer/limited_engagement/root_author_blocked_viewer",
                    ),
                ),
                (
                    EvaluationStatus::Evaluated,
                    limited(
                        LimitedEngagementReason::BlockedViewer,
                        "blocked_viewer/limited_engagement",
                    ),
                ),
                (EvaluationStatus::Evaluated, allow()),
            ]
        );
        assert_eq!(
            verdicts(None).await,
            vec![(EvaluationStatus::Evaluated, allow()); 3]
        );
        assert_eq!(
            sources.selects(),
            [vec![
                EdgeQuery::forward(Graph::Follows, vec![10, 20]),
                EdgeQuery::reverse(Graph::Blocks, vec![10, 20, 30]),
            ]]
        );
    }
    #[tokio::test]
    async fn run_preserves_order_duplicates_unresolved_authors_and_labels() {
        let labels = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(999_999, vf_pb::SafetyLabel::default())]),
        };
        let sources = Arc::new(
            InMemorySources::default()
                .labels(1, Default::default())
                .labels(2, labels),
        );
        let response = service(&sources)
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![
                    candidate(2, Some(20)),
                    candidate(1, None),
                    candidate(2, Some(20)),
                ],
                rpc: Rpc::FilterTweets,
            })
            .await;

        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.tweet_id)
                .collect::<Vec<_>>(),
            vec![TweetId(2), TweetId(1), TweetId(2)]
        );
        assert_eq!(response.outcomes[0].verdict, allow());
        assert_eq!(response.outcomes[1].verdict, Verdict::unresolved_author());
        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.status)
                .collect::<Vec<_>>(),
            vec![
                EvaluationStatus::Evaluated,
                EvaluationStatus::UnresolvedAuthor,
                EvaluationStatus::Evaluated
            ]
        );
        assert_eq!(response.outcomes[2].verdict, allow());
        assert!(response
            .outcomes
            .iter()
            .all(|outcome| outcome.safety_labels.is_some()));
        assert!(!response.outcomes[0]
            .safety_labels
            .as_ref()
            .unwrap()
            .labels
            .is_empty());
        assert_eq!(
            response.outcomes[0].safety_labels,
            response.outcomes[2].safety_labels
        );
    }
}
