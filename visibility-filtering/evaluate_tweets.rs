use crate::caller_identity::{self, Endpoint};
use crate::filter::{EvaluationStatus, FilterOutcome, FilterRequest, FilterTweets};
use crate::filter_tweets::normalize_viewer_id;
use crate::limited_actions_copy::LimitedActionsCopy;
use crate::models::{RawCandidate, TweetId, Verdict};
use crate::params::{ClientSwitches, LimitedActionsPolicies};
use crate::retweet;
use crate::rules::metrics::{self as ft_metrics, RequestMetricsGuard, Rpc};
use crate::rules::SafetyLevel;
use crate::treatment;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use vf_pb::tweet_evaluation::Outcome;
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

const REQUESTS: &str = "evaluate_tweets_requests";
const LATENCY_MS: &str = "evaluate_tweets_latency_ms";
const BATCH_SIZE: &str = "evaluate_tweets_batch_size";

pub struct EvaluateTweetsEndpoint {
    filter_tweets: Arc<FilterTweets>,
    client_switches: ClientSwitches,
    limited_actions_copy: LimitedActionsCopy,
}

impl EvaluateTweetsEndpoint {
    pub(crate) fn new(
        filter_tweets: Arc<FilterTweets>,
        client_switches: ClientSwitches,
        limited_actions_copy: LimitedActionsCopy,
    ) -> Self {
        Self {
            filter_tweets,
            client_switches,
            limited_actions_copy,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
        let entered = tokio::time::Instant::now();
        let request_metrics = RequestMetricsGuard::named(REQUESTS, LATENCY_MS);
        caller_identity::record(Endpoint::EvaluateTweets, &request);
        let context = crate::hydration::request_context(
            entered,
            crate::filter_tweets::parse_grpc_timeout(request.metadata()),
        );
        let twitter_context = xai_twittercontext::extract_twitter_context(request.metadata());
        match context
            .scope(self.handle_inner(request.into_inner(), twitter_context))
            .await
        {
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
        twitter_context: Option<TwitterContextViewer>,
    ) -> Result<vf_pb::EvaluateTweetsResponse, Status> {
        let level = ThriftLevel(req.safety_level);
        if !ThriftLevel::ENUM_VALUES.contains(&level) {
            return Err(Status::invalid_argument("unknown safety level"));
        }
        let safety_level = match level {
            ThriftLevel::FILTER_ALL => SafetyLevel::FilterAll,
            ThriftLevel::TIMELINE_HOME => SafetyLevel::TimelineHome,
            ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS => SafetyLevel::TimelineHomeRecommendations,
            ThriftLevel::TIMELINE_HOME_HYDRATION => SafetyLevel::TimelineHomeHydration,
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
        let viewer_id = normalize_viewer_id(req.viewer_id);
        let client_capability = self.client_switches.resolve(
            twitter_context.as_ref(),
            viewer_id,
            req.country_code.as_deref(),
        );
        let outcomes = retweet::evaluate_merging_sources(
            &self.filter_tweets,
            FilterRequest {
                viewer_id,
                country_code: req.country_code.clone(),
                client_capability,
                safety_level,
                candidates,
                rpc: Rpc::EvaluateTweets,
            },
        )
        .await;
        ft_metrics::record_verdicts(
            Rpc::EvaluateTweets,
            safety_level,
            outcomes.iter().map(|outcome| &outcome.verdict),
        );
        ft_metrics::record_rested_on(
            Rpc::EvaluateTweets,
            safety_level,
            outcomes.iter().map(|outcome| outcome.rested_on),
        );
        let policies = self.client_switches.limited_actions_policies(
            twitter_context.as_ref(),
            viewer_id,
            req.country_code.as_deref(),
            outcomes
                .iter()
                .filter_map(|outcome| match &outcome.verdict {
                    Verdict::Shown {
                        engagement: Some(limit),
                        ..
                    } => Some(limit.value.reasons()),
                    Verdict::Shown { .. } | Verdict::Withheld(_) => None,
                })
                .flatten(),
            &self.limited_actions_copy,
            xai_stats_receiver::global_stats_receiver().as_deref(),
        );
        let outcomes: HashMap<TweetId, FilterOutcome> = outcomes
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
                        }) => evaluated_outcome(verdict, safety_level, &policies),
                        _ => Outcome::Failed(vf_pb::Failed {}),
                    }
                };
                vf_pb::TweetEvaluation {
                    outcome: Some(outcome),
                }
            })
            .collect();
        Ok(vf_pb::EvaluateTweetsResponse { results })
    }
}

fn evaluated_outcome(
    verdict: &Verdict,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Outcome {
    let state = treatment::thrift_result_state(verdict, level, policies);
    match xai_x_thrift::serialize_compact(&state) {
        Ok(bytes) => Outcome::ResultStateThriftCompact(bytes.into()),
        Err(_) => Outcome::Failed(vf_pb::Failed {}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::plan::Source;
    use crate::hydration::sources::InMemorySources;
    use crate::models::VerifyBlurSupport;
    use crate::rules::RuleEngine;
    use xai_core_entities::entities::{
        GizmoduckUser, GizmoduckUserResult, PureCoreData, Safety, UserResponseState,
    };
    use xai_x_thrift::action::{self, Action, DropReason};
    use xai_x_thrift::safety_result::{FilteredReason as ThriftFilteredReason, SafetyResult};
    use xai_x_thrift::tweet_service::{
        TweetFieldsResultFiltered, TweetFieldsResultFound, TweetFieldsResultState,
    };

    fn encoded(state: TweetFieldsResultState) -> Outcome {
        Outcome::ResultStateThriftCompact(xai_x_thrift::serialize_compact(&state).unwrap().into())
    }

    fn filtered(reason: ThriftFilteredReason) -> Outcome {
        encoded(TweetFieldsResultState::Filtered(
            TweetFieldsResultFiltered::new(reason),
        ))
    }

    #[tokio::test]
    async fn evaluate_tweets_gates_levels_maps_outcomes_and_merges_retweet_sources() {
        let core = |author_id, source_tweet_id| PureCoreData {
            author_id,
            source_tweet_id,
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(3, core(30, None))
                .pure_core(4, core(40, Some(6)))
                .pure_core(5, core(50, Some(6)))
                .pure_core(6, core(60, None))
                .pure_core(7, core(70, Some(8)))
                .user(
                    60,
                    GizmoduckUserResult {
                        user: Some(GizmoduckUser {
                            safety: Safety {
                                suspended: true,
                                ..Default::default()
                            },
                            ..Default::default()
                        }),
                        response_state: Some(UserResponseState::Found),
                    },
                ),
        );
        let endpoint = EvaluateTweetsEndpoint::new(
            Arc::new(FilterTweets::new(sources.clone(), RuleEngine::for_tests())),
            ClientSwitches::for_tests(),
            LimitedActionsCopy::from_json("[]"),
        );
        for (level, code) in [
            (0, tonic::Code::Unimplemented),
            (4, tonic::Code::Unimplemented),
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
        for (level, evaluated) in [
            (
                ThriftLevel::FILTER_ALL.0,
                filtered(ThriftFilteredReason::SafetyResult(SafetyResult::new(
                    None,
                    Action::Drop(action::Drop::new(Some(DropReason::Unspecified(true)), None)),
                ))),
            ),
            (
                ThriftLevel::TIMELINE_HOME_HYDRATION.0,
                encoded(TweetFieldsResultState::Found(TweetFieldsResultFound::new(
                    None,
                ))),
            ),
        ] {
            let response = endpoint
                .handle(Request::new(vf_pb::EvaluateTweetsRequest {
                    safety_level: level,
                    tweets: tweets.clone(),
                    ..Default::default()
                }))
                .await
                .unwrap()
                .into_inner();
            assert_eq!(
                response
                    .results
                    .into_iter()
                    .map(|r| r.outcome.unwrap())
                    .collect::<Vec<_>>(),
                vec![
                    Outcome::Failed(vf_pb::Failed {}),
                    Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                    evaluated.clone(),
                    evaluated,
                ]
            );
        }
        let suspended = filtered(ThriftFilteredReason::AuthorIsSuspended(true));
        for (tweet_ids, core_data_calls, outcomes) in [
            (vec![4, 6], 1, vec![suspended.clone(), suspended.clone()]),
            (
                vec![5, 7],
                2,
                vec![suspended, Outcome::Failed(vf_pb::Failed {})],
            ),
        ] {
            let calls_before = sources.keys(Source::TesPureCore).len();
            let response = endpoint
                .handle(Request::new(vf_pb::EvaluateTweetsRequest {
                    safety_level: 8,
                    tweets: tweet_ids.into_iter().map(|id| tweet(id, None)).collect(),
                    ..Default::default()
                }))
                .await
                .unwrap()
                .into_inner();
            assert_eq!(
                sources.keys(Source::TesPureCore).len() - calls_before,
                core_data_calls
            );
            assert_eq!(
                response
                    .results
                    .into_iter()
                    .map(|r| r.outcome.unwrap())
                    .collect::<Vec<_>>(),
                outcomes
            );
        }
    }

    #[test]
    fn a_conversation_control_limit_carries_its_prompt_in_the_request_language() {
        use crate::limited_actions_copy::tests::BUNDLE;
        use crate::models::LimitedEngagementReason::ConversationControl;
        use crate::models::{Decided, LimitedEngagement, MediaRestriction};
        use crate::rules::fixtures::limited;
        use xai_x_thrift::action::{
            AnyInterstitial, CtaLimitedActionPrompt, Interstitial, InterstitialReason,
            LimitedAction, LimitedActionCtaType, LimitedActionPrompt, LimitedActionsPolicy,
            LimitedEngagements, TweetInterstitial,
        };
        let copy = LimitedActionsCopy::from_json(BUNDLE);
        let limit = limited(ConversationControl, "rule");
        let composite = Verdict::Shown {
            media: Some(Decided {
                value: MediaRestriction::NsfwInterstitial,
                by: "nsfw_rule",
            }),
            engagement: Some(Decided {
                value: LimitedEngagement::new(ConversationControl),
                by: "rule",
            }),
        };
        for (language, subtext) in [
            ("", "Only some accounts can reply."),
            ("ja", "一部のアカウントのみが返信できます。"),
        ] {
            let context = TwitterContextViewer {
                request_language_code: language.to_string(),
                ..TwitterContextViewer::default()
            };
            let policies = ClientSwitches::for_tests().limited_actions_policies(
                Some(&context),
                None,
                None,
                [ConversationControl],
                &copy,
                None,
            );
            let thrift_limit = LimitedEngagements::new(
                action::LimitedEngagementReason::ConversationControl(
                    action::ConversationControl::new(),
                ),
                LimitedActionsPolicy::new(vec![LimitedAction::new(
                    action::LimitedActionType::REPLY,
                    LimitedActionPrompt::CtaLimitedActionPrompt(CtaLimitedActionPrompt::new(
                        "Who can reply?".to_string(),
                        subtext.to_string(),
                        LimitedActionCtaType::SEE_CONVERSATION,
                    )),
                )]),
                "limited_replies".to_string(),
            );
            let found = |action| {
                encoded(TweetFieldsResultState::Found(TweetFieldsResultFound::new(
                    ThriftFilteredReason::SafetyResult(SafetyResult::new(None, action)),
                )))
            };
            for (verdict, action) in [
                (&limit, Action::LimitedEngagements(thrift_limit.clone())),
                (
                    &composite,
                    Action::TweetInterstitial(TweetInterstitial {
                        interstitial: Some(AnyInterstitial::Interstitial(Interstitial::new(
                            InterstitialReason::ContainsNsfwMedia(true),
                            None,
                        ))),
                        limited_engagements: Some(thrift_limit.clone()),
                        ..TweetInterstitial::default()
                    }),
                ),
            ] {
                assert_eq!(
                    evaluated_outcome(verdict, SafetyLevel::TimelineHomeHydration, &policies),
                    found(action),
                    "{language:?} {verdict:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn the_forwarded_client_reaches_the_fetched_source_and_no_header_gets_the_defaults() {
        use crate::models::ClientCapability;
        use crate::rules::fixtures::CLIENT_CLASSES;
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(
                    4,
                    PureCoreData {
                        author_id: 40,
                        source_tweet_id: Some(6),
                        ..Default::default()
                    },
                )
                .tweet(6, 60),
        );
        let filter_tweets = Arc::new(FilterTweets::new(sources, RuleEngine::for_tests()));
        let endpoint = EvaluateTweetsEndpoint::new(
            Arc::clone(&filter_tweets),
            ClientSwitches::for_tests(),
            LimitedActionsCopy::from_json("[]"),
        );
        let class = CLIENT_CLASSES
            .iter()
            .find(|class| {
                class.capability.verify_blur_support == Some(VerifyBlurSupport::IosNeedsUpdate)
            })
            .unwrap();
        let header = xai_twittercontext::hydrate_twitter_context(&TwitterContextViewer {
            client_application_id: class.app_id,
            user_agent: class.user_agent.into(),
            ..TwitterContextViewer::default()
        })
        .unwrap();
        for (header, expected) in [
            (Some(header), class.capability),
            (None, ClientCapability::default()),
        ] {
            let mut request = Request::new(vf_pb::EvaluateTweetsRequest {
                safety_level: ThriftLevel::TIMELINE_HOME_HYDRATION.0,
                viewer_id: Some(1),
                tweets: vec![vf_pb::TweetData {
                    tweet_id: 4,
                    quote_context: None,
                }],
                ..Default::default()
            });
            if let Some(header) = header {
                request.metadata_mut().insert("twittercontext", header);
            }
            endpoint.handle(request).await.unwrap();
            assert_eq!(
                std::mem::take(&mut *filter_tweets.client_capabilities.lock().unwrap()),
                [expected]
            );
        }
    }
}
