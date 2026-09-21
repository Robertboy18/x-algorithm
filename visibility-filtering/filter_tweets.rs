use crate::filter::{FilterOutcome, FilterRequest, FilterTweets};
use crate::models::{RawCandidate, TweetId, VfAction};
use crate::reference_compare::{ReferenceCompareHarness, TweetVerdict};
use crate::rules::metrics::{self as ft_metrics, RequestMetricsGuard};
use crate::rules::SafetyLevel;
use std::sync::Arc;
use std::time::Duration;
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterTweetsEndpoint {
    filter_tweets: Arc<FilterTweets>,
    reference_compare: Option<Arc<ReferenceCompareHarness>>,
}

impl FilterTweetsEndpoint {
    pub(crate) fn new(
        filter_tweets: Arc<FilterTweets>,
        reference_compare: Option<Arc<ReferenceCompareHarness>>,
    ) -> Self {
        Self {
            filter_tweets,
            reference_compare,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        let request_metrics = RequestMetricsGuard::new();
        let grpc_timeout = parse_grpc_timeout(request.metadata());
        let req = request.into_inner();
        ft_metrics::record_batch_size(ft_metrics::BATCH_SIZE, req.tweets.len());
        let viewer_id = normalize_viewer_id(req.viewer_id);
        ft_metrics::record_viewer_state(req.viewer_id, viewer_id);

        let safety_level = match req.safety_level() {
            vf_pb::SafetyLevel::FilterAll => SafetyLevel::FilterAll,
            vf_pb::SafetyLevel::TimelineHome => SafetyLevel::TimelineHome,
            vf_pb::SafetyLevel::TimelineHomeRecommendations => {
                SafetyLevel::TimelineHomeRecommendations
            }
        };
        let candidates: Vec<RawCandidate> = req
            .tweets
            .iter()
            .map(|t| RawCandidate {
                tweet_id: TweetId(t.tweet_id),
                request_author_id: t.author_id,
            })
            .collect();

        let reference_compare = self.reference_compare.as_ref().and_then(|harness| {
            harness.begin_compare(
                req.viewer_id,
                req.country_code.clone(),
                safety_level,
                req.tweets.iter().map(|t| t.tweet_id).collect(),
            )
        });

        let response = self
            .filter_tweets
            .run(FilterRequest {
                viewer_id,
                country_code: req.country_code,
                safety_level,
                candidates,
            })
            .await;

        if let Some(verdicts) = reference_compare {
            verdicts.send(
                response
                    .outcomes
                    .iter()
                    .map(|outcome| TweetVerdict {
                        tweet_id: outcome.tweet_id.0,
                        verdict: outcome.verdict.clone(),
                    })
                    .collect(),
            );
        }

        let results = response
            .outcomes
            .into_iter()
            .map(to_visibility_result)
            .collect();

        request_metrics.record_deadline(grpc_timeout);
        request_metrics.mark_success();
        Ok(Response::new(vf_pb::VisibilityFilterResponse { results }))
    }
}

pub(crate) fn normalize_viewer_id(raw: Option<u64>) -> Option<u64> {
    raw.filter(|&id| id as i64 > 0)
}

fn parse_grpc_timeout(metadata: &MetadataMap) -> Option<Duration> {
    let raw = metadata.get("grpc-timeout")?.to_str().ok()?;
    let (digits, unit) = raw.split_at(raw.len().checked_sub(1)?);
    if digits.is_empty() || digits.len() > 8 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    let nanos_per_unit = match unit {
        "H" => 3_600_000_000_000,
        "M" => 60_000_000_000,
        "S" => 1_000_000_000,
        "m" => 1_000_000,
        "u" => 1_000,
        "n" => 1,
        _ => return None,
    };
    Some(Duration::from_nanos(value.checked_mul(nanos_per_unit)?))
}

fn to_visibility_result(outcome: FilterOutcome) -> vf_pb::TweetVisibilityResult {
    let (kind, filtered_reason) = match outcome.verdict.action {
        VfAction::Allow => (vf_pb::action::Kind::Allow(true), None),
        VfAction::Drop(reason) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(reason.into()),
        ),
        VfAction::Interstitial(reason) => {
            (vf_pb::action::Kind::Interstitial(true), Some(reason.into()))
        }
    };
    vf_pb::TweetVisibilityResult {
        tweet_id: outcome.tweet_id.0,
        action: Some(vf_pb::Action { kind: Some(kind) }),
        filtered_reason,
        safety_labels: outcome.safety_labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Verdict;
    use xai_visibility_filtering::models::FilteredReason;

    fn outcome(tweet_id: u64, action: VfAction) -> FilterOutcome {
        FilterOutcome {
            tweet_id: TweetId(tweet_id),
            verdict: Verdict {
                action,
                decided_by: Some("test"),
            },
            status: crate::filter::EvaluationStatus::Evaluated,
            safety_labels: None,
        }
    }

    async fn gizmoduck_calls(viewer_id: Option<u64>) -> usize {
        let gizmoduck = std::sync::Arc::new(
            xai_core_entities::gizmoduck_client::MockGizmoduckClient::default(),
        );
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(crate::filter::test_support::filter_tweets_with_gizmoduck(
                gizmoduck.clone(),
            )),
            None,
        );
        let response = endpoint
            .handle(Request::new(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: vec![vf_pb::TweetInput {
                    tweet_id: 2,
                    author_id: Some(20),
                }],
                viewer_id,
                country_code: None,
            }))
            .await
            .unwrap();
        assert_eq!(response.into_inner().results.len(), 1);
        gizmoduck.call_count()
    }

    #[tokio::test]
    async fn endpoint_treats_zero_viewer_id_as_logged_out() {
        let logged_out = gizmoduck_calls(None).await;
        assert_eq!(gizmoduck_calls(Some(0)).await, logged_out);
        assert_eq!(gizmoduck_calls(Some(42)).await, logged_out + 1);
    }

    #[test]
    fn parse_grpc_timeout_units_and_garbage() {
        let parsed = |value: &str| {
            let mut metadata = MetadataMap::new();
            metadata.insert("grpc-timeout", value.parse().unwrap());
            parse_grpc_timeout(&metadata)
        };
        assert_eq!(parsed("400m"), Some(Duration::from_millis(400)));
        assert_eq!(parsed("1S"), Some(Duration::from_secs(1)));
        assert_eq!(parsed("2M"), Some(Duration::from_secs(120)));
        assert_eq!(parsed("500u"), Some(Duration::from_micros(500)));
        assert_eq!(parsed("400"), None);
        assert_eq!(parsed("m"), None);
        assert_eq!(parsed("400x"), None);
        assert_eq!(parsed("+400m"), None);
        assert_eq!(parsed("123456789m"), None);
        assert_eq!(parse_grpc_timeout(&MetadataMap::new()), None);
    }

    #[test]
    fn normalize_viewer_id_cases() {
        assert_eq!(normalize_viewer_id(Some(0)), None);
        assert_eq!(normalize_viewer_id(Some(u64::MAX)), None);
        assert_eq!(normalize_viewer_id(Some(42)), Some(42));
        assert_eq!(normalize_viewer_id(None), None);
    }

    #[test]
    fn actions_map_to_proto_kinds() {
        let cases = [
            (VfAction::Allow, vf_pb::action::Kind::Allow(true)),
            (
                VfAction::Drop(FilteredReason::ContainNsfwMedia),
                vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            ),
            (
                VfAction::Interstitial(FilteredReason::ContainNsfwMedia),
                vf_pb::action::Kind::Interstitial(true),
            ),
        ];

        for (action, expected) in cases {
            let result = to_visibility_result(outcome(1, action));
            assert_eq!(result.action.and_then(|action| action.kind), Some(expected));
        }
    }
}
