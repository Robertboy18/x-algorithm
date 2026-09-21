use crate::hydration::{HydrationOutput, HydrationPipeline, HydrationRequest};
use crate::models::{RawCandidate, TweetId};
use crate::rules::metrics as ft_metrics;
use crate::rules::{RuleEngine, SafetyLevel, Verdict};
use std::collections::HashMap;
use std::time::Instant;
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterRequest {
    pub viewer_id: Option<u64>,
    pub country_code: Option<String>,
    pub safety_level: SafetyLevel,
    pub candidates: Vec<RawCandidate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationStatus {
    Evaluated,
    UnresolvedAuthor,
    Failed,
}

pub struct FilterOutcome {
    pub tweet_id: TweetId,
    pub verdict: Verdict,
    pub status: EvaluationStatus,
    pub safety_labels: Option<vf_pb::SafetyLabelMap>,
}

pub struct FilterResponse {
    pub outcomes: Vec<FilterOutcome>,
}

pub struct FilterTweets {
    hydration_pipeline: HydrationPipeline,
    rule_engine: RuleEngine,
}

impl FilterTweets {
    pub(crate) fn new(hydration_pipeline: HydrationPipeline, rule_engine: RuleEngine) -> Self {
        Self {
            hydration_pipeline,
            rule_engine,
        }
    }

    pub async fn run(&self, request: FilterRequest) -> FilterResponse {
        let started = Instant::now();
        let hydration = self
            .hydration_pipeline
            .hydrate(HydrationRequest::new(
                request.viewer_id,
                request.country_code,
                &request.candidates,
                request.safety_level,
            ))
            .await;
        let hydrated_at = Instant::now();
        ft_metrics::record_phase("hydration", hydrated_at - started);
        let HydrationOutput {
            viewer_features,
            candidates: hydrated_candidates,
            safety_labels,
            failed_ids,
        } = hydration;
        let evaluated: HashMap<TweetId, Verdict> = hydrated_candidates
            .iter()
            .map(|candidate| {
                (
                    TweetId(candidate.tweet_id),
                    self.rule_engine
                        .evaluate(request.safety_level, &viewer_features, candidate),
                )
            })
            .collect();

        let outcomes: Vec<FilterOutcome> = request
            .candidates
            .iter()
            .map(|candidate| {
                let (verdict, status) = match evaluated.get(&candidate.tweet_id) {
                    None => (
                        Verdict::unresolved_author(),
                        EvaluationStatus::UnresolvedAuthor,
                    ),
                    Some(verdict) if failed_ids.contains(&candidate.tweet_id) => {
                        (verdict.clone(), EvaluationStatus::Failed)
                    }
                    Some(verdict) => (verdict.clone(), EvaluationStatus::Evaluated),
                };
                FilterOutcome {
                    tweet_id: candidate.tweet_id,
                    verdict,
                    status,
                    safety_labels: safety_labels
                        .get(&candidate.tweet_id)
                        .map(|labels| vf_pb::SafetyLabelMap::clone(labels)),
                }
            })
            .collect();

        ft_metrics::record_verdicts(
            request.safety_level,
            outcomes.iter().map(|outcome| &outcome.verdict),
        );
        ft_metrics::record_phase("post_hydration", hydrated_at.elapsed());

        FilterResponse { outcomes }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::clients::socialgraph_client::FakeSocialgraphClient;
    use crate::hydration::tes_composite::MockTweetForVisibilitySource;
    use crate::safety_label_source::lookup::{ManhattanLookup, RemoteSource, TwemcacheLookup};
    use crate::safety_label_source::types::{ManhattanOutcome, TwemcacheOutcome};
    use crate::safety_label_source::SafetyLabelSource;
    use std::sync::Arc;
    use tonic::async_trait;
    use xai_core_entities::gizmoduck_client::{GizmoduckClient, MockGizmoduckClient};
    use xai_core_entities::tweet_entity_service_client::{MockTESClient, TESClient};

    fn full_label_map() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(999_999, vf_pb::SafetyLabel::default())]),
        }
    }

    struct FakeTwemcache;

    #[async_trait]
    impl TwemcacheLookup for FakeTwemcache {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, TwemcacheOutcome> {
            ids.iter()
                .copied()
                .map(|id| {
                    let outcome = if id == 2 {
                        TwemcacheOutcome::Hit(full_label_map())
                    } else {
                        TwemcacheOutcome::Miss
                    };
                    (id, outcome)
                })
                .collect()
        }
    }

    struct FakeManhattan;

    #[async_trait]
    impl ManhattanLookup for FakeManhattan {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, ManhattanOutcome> {
            ids.iter()
                .copied()
                .map(|id| (id, ManhattanOutcome::Resolved(full_label_map())))
                .collect()
        }
    }

    pub(crate) fn filter_tweets() -> FilterTweets {
        filter_tweets_with_gizmoduck(Arc::new(MockGizmoduckClient::default()))
    }

    pub(crate) fn filter_tweets_with_gizmoduck(
        gizmoduck: Arc<dyn GizmoduckClient + Send + Sync>,
    ) -> FilterTweets {
        filter_tweets_with_clients(Arc::new(MockTESClient::default()), gizmoduck)
    }

    pub(crate) fn safety_labels() -> Arc<SafetyLabelSource> {
        Arc::new(SafetyLabelSource::new(Arc::new(RemoteSource::new(
            Arc::new(FakeTwemcache),
            Arc::new(FakeManhattan),
        ))))
    }

    pub(crate) fn filter_tweets_with_clients(
        tes: Arc<dyn TESClient + Send + Sync>,
        gizmoduck: Arc<dyn GizmoduckClient + Send + Sync>,
    ) -> FilterTweets {
        let socialgraph = Arc::new(FakeSocialgraphClient);
        FilterTweets::new(
            HydrationPipeline::new(
                tes,
                Arc::new(MockTweetForVisibilitySource::default()),
                gizmoduck,
                socialgraph,
                safety_labels(),
                None,
                None,
            ),
            RuleEngine::for_tests(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::SocialgraphClient;
    use crate::filter::test_support::filter_tweets;
    use crate::hydration::tes_composite::{
        MockTweetForVisibilitySource, TweetForVisibility, TweetForVisibilitySource,
    };
    use crate::models::{
        CoreFeature, ExclusiveContentFeatures, TweetFeatures, VfAction, ViewerAuthorRelationship,
    };
    use std::sync::{Arc, Mutex};
    use tonic::async_trait;
    use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, PureCoreData, Safety};
    use xai_core_entities::gizmoduck_client::MockGizmoduckClient;
    use xai_core_entities::tweet_entity_service_client::MockTESClient;

    #[derive(Default)]
    struct RecordingSocialgraph {
        relationships: Mutex<Vec<Vec<u64>>>,
        super_follows: Mutex<Vec<Vec<u64>>>,
    }

    #[async_trait]
    impl SocialgraphClient for RecordingSocialgraph {
        async fn batch_check_relationships(
            &self,
            _: u64,
            authors: &[u64],
        ) -> HashMap<u64, ViewerAuthorRelationship> {
            self.relationships.lock().unwrap().push(authors.to_vec());
            authors
                .iter()
                .map(|&id| {
                    (
                        id,
                        ViewerAuthorRelationship {
                            viewer_follows_author: true,
                            ..Default::default()
                        },
                    )
                })
                .collect()
        }

        async fn batch_check_super_follows(
            &self,
            _: u64,
            authors: &[u64],
        ) -> Option<HashMap<u64, bool>> {
            self.super_follows.lock().unwrap().push(authors.to_vec());
            Some(authors.iter().map(|&id| (id, true)).collect())
        }
    }

    struct PendingComposite;

    #[async_trait]
    impl TweetForVisibilitySource for PendingComposite {
        async fn get_tweets_for_visibility(
            &self,
            _: &[u64],
        ) -> HashMap<u64, anyhow::Result<Option<TweetForVisibility>>> {
            std::future::pending().await
        }
    }

    fn core_client() -> Arc<MockTESClient> {
        Arc::new(MockTESClient {
            core_data: HashMap::from([
                (
                    1,
                    Some(PureCoreData {
                        author_id: 10,
                        text: "pure core text".into(),
                        source_tweet_id: Some(999),
                        ..Default::default()
                    }),
                ),
                (
                    2,
                    Some(PureCoreData {
                        author_id: 20,
                        ..Default::default()
                    }),
                ),
            ]),
            ..Default::default()
        })
    }

    fn exclusive_tweet() -> TweetForVisibility {
        TweetForVisibility {
            author_id: 900,
            source_tweet_id: None,
            is_nullcast: false,
            nsfw_user: false,
            nsfw_admin: false,
            has_takedown: false,
            takedown_reasons: vec![],
            media: Default::default(),
            is_community_tweet: false,
            edit_control: None,
            exclusive_conversation_author_id: Some(30),
        }
    }

    fn candidate(tweet_id: u64, author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id: author_id,
        }
    }

    #[tokio::test]
    async fn run_uses_only_core_and_composite_tes_calls() {
        let tes = core_client();
        let composite = Arc::new(MockTweetForVisibilitySource {
            tweets: HashMap::from([(1, Some(exclusive_tweet()))]),
            ..Default::default()
        });
        let sg = Arc::new(RecordingSocialgraph::default());
        let service = FilterTweets::new(
            HydrationPipeline::new(
                tes.clone(),
                composite.clone(),
                Arc::new(MockGizmoduckClient::default()),
                sg.clone(),
                test_support::safety_labels(),
                None,
                None,
            ),
            RuleEngine::for_tests(),
        );
        let result = service
            .run(FilterRequest {
                viewer_id: Some(50),
                country_code: None,
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![candidate(1, None), candidate(1, None)],
            })
            .await;
        assert_eq!(tes.call_count(), 1);
        assert_eq!(*composite.requests.lock().unwrap(), vec![vec![1]]);
        assert_eq!(*sg.relationships.lock().unwrap(), vec![vec![10]]);
        assert_eq!(*sg.super_follows.lock().unwrap(), vec![vec![30]]);
        assert_eq!(result.outcomes.len(), 2);
        assert!(result
            .outcomes
            .iter()
            .all(|outcome| matches!(outcome.verdict.action, VfAction::Allow)));
    }

    #[tokio::test(start_paused = true)]
    async fn composite_timeout_preserves_pure_core_author_and_relationship_features() {
        let tes = core_client();
        let gizmoduck = Arc::new(MockGizmoduckClient {
            users: HashMap::from([(
                10,
                Some(GizmoduckUserResult {
                    user: Some(GizmoduckUser {
                        safety: Safety {
                            suspended: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            )]),
            ..Default::default()
        });
        let sg = Arc::new(RecordingSocialgraph::default());
        let pipeline = HydrationPipeline::new(
            tes.clone(),
            Arc::new(PendingComposite),
            gizmoduck,
            sg.clone(),
            test_support::safety_labels(),
            None,
            None,
        );
        let raw = [candidate(1, None)];
        let started = tokio::time::Instant::now();
        let hydration = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            pipeline.hydrate(HydrationRequest::new(
                Some(50),
                None,
                &raw,
                SafetyLevel::TimelineHome,
            )),
        );
        tokio::pin!(hydration);
        assert!(futures::poll!(&mut hydration).is_pending());
        assert_eq!(*sg.relationships.lock().unwrap(), vec![vec![10]]);
        let result = hydration.await.unwrap();
        assert_eq!(started.elapsed(), crate::hydration::HYDRATION_TIMEOUT);
        let tweet = &result.candidates[0];
        assert_eq!(tweet.author_id, 10);
        assert_eq!(
            tweet.tweet_features,
            TweetFeatures {
                core: CoreFeature {
                    text: "pure core text".into(),
                    source_tweet_id: None
                },
                ..Default::default()
            }
        );
        assert!(tweet.author_features.is_suspended);
        assert!(tweet.relationship.viewer_follows_author);
        assert_eq!(tweet.exclusive_content, None);
        assert_eq!(tes.call_count(), 1);
        assert!(sg.super_follows.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn exclusive_content_deduplicates_tweets_and_conversation_authors() {
        let composite = Arc::new(MockTweetForVisibilitySource {
            tweets: [1, 2]
                .into_iter()
                .map(|id| (id, Some(exclusive_tweet())))
                .collect(),
            ..Default::default()
        });
        for viewer_id in [Some(50), None] {
            let sg = Arc::new(RecordingSocialgraph::default());
            let pipeline = HydrationPipeline::new(
                core_client(),
                composite.clone(),
                Arc::new(MockGizmoduckClient::default()),
                sg.clone(),
                test_support::safety_labels(),
                None,
                None,
            );
            let raw = [
                candidate(1, None),
                candidate(2, None),
                candidate(1, None),
                candidate(3, Some(40)),
            ];
            let result = pipeline
                .hydrate(HydrationRequest::new(
                    viewer_id,
                    None,
                    &raw,
                    SafetyLevel::TimelineHome,
                ))
                .await;
            let expected = Some(ExclusiveContentFeatures {
                conversation_author_id: 30,
                viewer_super_follows_author: viewer_id.is_some(),
            });
            assert_eq!(
                result
                    .candidates
                    .iter()
                    .map(|c| c.exclusive_content.clone())
                    .collect::<Vec<_>>(),
                vec![expected.clone(), expected.clone(), expected, None]
            );
            let expected_calls = if viewer_id.is_some() {
                vec![vec![30]]
            } else {
                vec![]
            };
            assert_eq!(*sg.super_follows.lock().unwrap(), expected_calls);
        }
    }

    #[tokio::test]
    async fn run_preserves_order_duplicates_unresolved_authors_and_labels() {
        let response = filter_tweets()
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![
                    candidate(2, Some(20)),
                    candidate(1, None),
                    candidate(2, Some(20)),
                ],
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
        assert!(matches!(
            response.outcomes[0].verdict.action,
            VfAction::Allow
        ));
        assert!(matches!(
            response.outcomes[1].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            response.outcomes[1].verdict.decided_by,
            Some("unresolved_author_id")
        );
        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.status)
                .collect::<Vec<_>>(),
            vec![
                EvaluationStatus::Failed,
                EvaluationStatus::UnresolvedAuthor,
                EvaluationStatus::Failed
            ]
        );
        assert!(matches!(
            response.outcomes[2].verdict.action,
            VfAction::Allow
        ));
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

    #[tokio::test]
    async fn run_labels_both_occurrences_of_a_cold_id() {
        let response = filter_tweets()
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![candidate(3, Some(30)), candidate(3, Some(30))],
            })
            .await;

        let labels = response
            .outcomes
            .iter()
            .map(|outcome| outcome.safety_labels.as_ref().unwrap())
            .collect::<Vec<_>>();
        assert!(!labels[0].labels.is_empty());
        assert_eq!(labels[0], labels[1]);
    }

    #[tokio::test]
    async fn run_selects_policy_from_safety_level() {
        let service = filter_tweets();
        let request = |safety_level| FilterRequest {
            viewer_id: None,
            country_code: None,
            safety_level,
            candidates: vec![candidate(1, Some(10))],
        };

        let home = service.run(request(SafetyLevel::TimelineHome)).await;
        let filter_all = service.run(request(SafetyLevel::FilterAll)).await;

        assert!(matches!(home.outcomes[0].verdict.action, VfAction::Allow));
        assert!(matches!(
            filter_all.outcomes[0].verdict.action,
            VfAction::Drop(_)
        ));
        assert_eq!(
            filter_all.outcomes[0].verdict.decided_by,
            Some("FilterAllRule")
        );
    }
}
