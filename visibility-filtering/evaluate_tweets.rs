use crate::filter::{EvaluationStatus, FilterOutcome, FilterRequest, FilterTweets};
use crate::models::{RawCandidate, TweetId, VfAction};
use crate::rules::metrics::{self as ft_metrics, RequestMetricsGuard};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use vf_pb::tweet_evaluation::Outcome;
use xai_visibility_filtering::models::FilteredReason;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::action::{self, Action, DropReason};
use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

const REQUESTS: &str = "evaluate_tweets_requests";
const LATENCY_MS: &str = "evaluate_tweets_latency_ms";
const BATCH_SIZE: &str = "evaluate_tweets_batch_size";

pub struct EvaluateTweetsEndpoint {
    filter_tweets: Arc<FilterTweets>,
}

impl EvaluateTweetsEndpoint {
    pub(crate) fn new(filter_tweets: Arc<FilterTweets>) -> Self {
        Self { filter_tweets }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
        let request_metrics = RequestMetricsGuard::named(REQUESTS, LATENCY_MS);
        match self.handle_inner(request.into_inner()).await {
            Ok(response) => {
                request_metrics.mark_success();
                Ok(Response::new(response))
            }
            Err(status) => {
                request_metrics.mark_failure();
                Err(status)
            }
        }
    }

    async fn handle_inner(
        &self,
        req: vf_pb::EvaluateTweetsRequest,
    ) -> Result<vf_pb::EvaluateTweetsResponse, Status> {
        let level = ThriftLevel(req.safety_level);
        if !ThriftLevel::ENUM_VALUES.contains(&level) {
            return Err(Status::invalid_argument("unknown safety level"));
        }
        let safety_level = match level {
            ThriftLevel::FILTER_ALL => SafetyLevel::FilterAll,
            ThriftLevel::TIMELINE_HOME => SafetyLevel::TimelineHome,
            ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS => SafetyLevel::TimelineHomeRecommendations,
            _ => return Err(Status::unimplemented("safety level has no Rust policy")),
        };
        ft_metrics::record_batch_size(BATCH_SIZE, req.tweets.len());
        let candidates = req
            .tweets
            .iter()
            .filter(|o| o.quote_context.is_none())
            .map(|o| RawCandidate {
                tweet_id: TweetId(o.tweet_id),
                request_author_id: None,
            })
            .collect();
        let response = self
            .filter_tweets
            .run(FilterRequest {
                viewer_id: crate::filter_tweets::normalize_viewer_id(req.viewer_id),
                country_code: req.country_code,
                safety_level,
                candidates,
            })
            .await;
        let outcomes: HashMap<TweetId, FilterOutcome> = response
            .outcomes
            .into_iter()
            .map(|outcome| (outcome.tweet_id, outcome))
            .collect();
        let results = req
            .tweets
            .into_iter()
            .map(|tweet| {
                let outcome = if tweet.quote_context.is_some() {
                    Outcome::NotEvaluated(vf_pb::NotEvaluated {})
                } else {
                    match outcomes.get(&TweetId(tweet.tweet_id)) {
                        Some(FilterOutcome {
                            status: EvaluationStatus::Evaluated,
                            verdict,
                            ..
                        }) => match canonical_action(&verdict.action, safety_level) {
                            Some(action) => match xai_x_thrift::serialize_compact(&action) {
                                Ok(bytes) => Outcome::ActionThriftCompact(bytes.into()),
                                Err(_) => Outcome::Failed(vf_pb::Failed {}),
                            },
                            None => Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                        },
                        _ => Outcome::Failed(vf_pb::Failed {}),
                    }
                };
                vf_pb::TweetEvaluation {
                    tweet: Some(tweet),
                    outcome: Some(outcome),
                }
            })
            .collect();
        Ok(vf_pb::EvaluateTweetsResponse { results })
    }
}

fn canonical_action(verdict: &VfAction, level: SafetyLevel) -> Option<Action> {
    let reason = match verdict {
        VfAction::Allow => return Some(Action::Allow(action::Allow::new())),
        VfAction::Interstitial(_) => return None,
        VfAction::Drop(reason) => reason,
    };
    let drop_reason = match reason {
        FilteredReason::AuthorIsProtected => DropReason::ProtectedAuthor(true),
        FilteredReason::AuthorIsSuspended => DropReason::SuspendedAuthor(true),
        FilteredReason::AuthorBlockViewer => DropReason::AuthorBlocksViewer(true),
        FilteredReason::ViewerBlocksAuthor => DropReason::ViewerBlocksAuthor(true),
        FilteredReason::ViewerMutesAuthor => DropReason::ViewerMutesAuthor(true),
        FilteredReason::ExclusiveTweet => DropReason::ExclusiveTweet(true),
        FilteredReason::UnspecifiedReason if level == SafetyLevel::FilterAll => {
            DropReason::Unspecified(true)
        }
        FilteredReason::UnspecifiedReason
        | FilteredReason::ContainNsfwMedia
        | FilteredReason::PossiblyUndesirable
        | FilteredReason::AuthorAccountIsInactive
        | FilteredReason::AuthorIsUnsafe
        | FilteredReason::ReportedTweet
        | FilteredReason::TweetMatchesViewerMutedKeyword(_)
        | FilteredReason::TweetIsBounced
        | FilteredReason::SafetyResult(_)
        | FilteredReason::AuthorIsDeactivated
        | FilteredReason::TweetIsNullcast => return None,
    };
    Some(Action::Drop(action::Drop::new(Some(drop_reason), None)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::PureCoreData;
    use xai_core_entities::gizmoduck_client::MockGizmoduckClient;
    use xai_core_entities::tweet_entity_service_client::MockTESClient;

    #[tokio::test]
    async fn evaluate_tweets_gates_levels_and_maps_outcomes_in_order() {
        let tes = MockTESClient {
            core_data: [(
                3,
                Some(PureCoreData {
                    author_id: 30,
                    ..Default::default()
                }),
            )]
            .into(),
            ..Default::default()
        };
        let endpoint = EvaluateTweetsEndpoint::new(Arc::new(
            crate::filter::test_support::filter_tweets_with_clients(
                Arc::new(tes),
                Arc::new(MockGizmoduckClient::default()),
            ),
        ));
        for (level, code) in [
            (0, tonic::Code::Unimplemented),
            (82, tonic::Code::Unimplemented),
            (9999, tonic::Code::InvalidArgument),
        ] {
            let error = endpoint
                .handle(Request::new(vf_pb::EvaluateTweetsRequest {
                    safety_level: level,
                    ..Default::default()
                }))
                .await
                .unwrap_err();
            assert_eq!(error.code(), code);
        }
        let tweet = |tweet_id, outer_tweet_id: Option<u64>| vf_pb::TweetData {
            tweet_id,
            quote_context: outer_tweet_id.map(|outer_tweet_id| vf_pb::QuoteContext {
                outer_tweet_id,
                outer_author_id: None,
            }),
        };
        let tweets = vec![
            tweet(1, None),
            tweet(1, Some(2)),
            tweet(3, None),
            tweet(3, None),
        ];
        let response = endpoint
            .handle(Request::new(vf_pb::EvaluateTweetsRequest {
                safety_level: 16,
                tweets: tweets.clone(),
                ..Default::default()
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            response
                .results
                .iter()
                .map(|r| r.tweet.unwrap())
                .collect::<Vec<_>>(),
            tweets
        );
        let filter_all_drop = xai_x_thrift::serialize_compact(&Action::Drop(action::Drop::new(
            Some(DropReason::Unspecified(true)),
            None,
        )))
        .unwrap();
        assert_eq!(
            response
                .results
                .into_iter()
                .map(|r| r.outcome.unwrap())
                .collect::<Vec<_>>(),
            vec![
                Outcome::Failed(vf_pb::Failed {}),
                Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                Outcome::ActionThriftCompact(filter_all_drop.clone().into()),
                Outcome::ActionThriftCompact(filter_all_drop.into()),
            ]
        );
    }

    #[test]
    fn canonical_action_never_substitutes_lossy_treatments() {
        assert!(canonical_action(
            &VfAction::Interstitial(FilteredReason::ContainNsfwMedia),
            SafetyLevel::TimelineHome
        )
        .is_none());
        assert!(canonical_action(
            &VfAction::Drop(FilteredReason::UnspecifiedReason),
            SafetyLevel::TimelineHome
        )
        .is_none());
        assert_eq!(
            canonical_action(
                &VfAction::Drop(FilteredReason::AuthorIsProtected),
                SafetyLevel::TimelineHome
            ),
            Some(Action::Drop(action::Drop::new(
                Some(DropReason::ProtectedAuthor(true)),
                None
            )))
        );
    }
}
