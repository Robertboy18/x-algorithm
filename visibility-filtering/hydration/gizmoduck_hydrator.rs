use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::hydration::batch::{
    AuthorHydrationBatch, Completeness, HydrationBatch, TweetHydrationBatch,
};
use crate::hydration::fallback_cache::FallbackCache;
use crate::hydration::metrics::{record_author_labels, record_batch_size, timed_results};
use crate::hydration::{keyed_by_author, tweets_per_author};
use crate::models::{AuthorFeatures, AuthorId, AuthorLabel, AuthorLabelSet, TweetCandidateInput};
use crate::rules::SafetyLevel;
use std::time::Duration;
use xai_core_entities::entities::{GizmoduckUserResult, UserResponseState};
use xai_core_entities::gizmoduck_client::QueryFields;
use xai_x_thrift::user_labels::LabelValue;

const CLIENT_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const CLIENT: &str = "gizmoduck";
const CACHE_CAPACITY: usize = 1_000_000;

pub struct GizmoduckAuthorHydrator {
    pub gizmoduck_client: GizmoduckLookup,
    fallback_cache: Option<FallbackCache<AuthorId, Completeness<AuthorFeatures>>>,
}

impl GizmoduckAuthorHydrator {
    pub(crate) fn new(
        gizmoduck_client: GizmoduckLookup,
        fallback_cache: Option<FallbackCache<AuthorId, Completeness<AuthorFeatures>>>,
    ) -> Self {
        Self {
            gizmoduck_client,
            fallback_cache,
        }
    }

    pub(crate) fn fallback_cache() -> FallbackCache<AuthorId, Completeness<AuthorFeatures>> {
        FallbackCache::new("author", CACHE_CAPACITY)
    }

    pub(crate) async fn hydrate(
        &self,
        candidates: &[TweetCandidateInput],
        safety_level: SafetyLevel,
    ) -> TweetHydrationBatch<Completeness<AuthorFeatures>> {
        let cache_request = self
            .fallback_cache
            .as_ref()
            .map(|cache| (cache, cache.begin_request()));
        let candidate_count_by_key = tweets_per_author(candidates);
        let author_ids: Vec<u64> = candidate_count_by_key.keys().map(|a| a.get()).collect();

        let user_results: AuthorHydrationBatch<GizmoduckUserResult> = if author_ids.is_empty() {
            HydrationBatch::empty()
        } else {
            record_batch_size(CLIENT, author_ids.len());
            timed_results(
                CLIENT,
                "get_users",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                async {
                    let response = self
                        .gizmoduck_client
                        .get_users(author_ids, &[QueryFields::SAFETY, QueryFields::LABELS])
                        .await;
                    keyed_by_author(&candidate_count_by_key, response)
                },
            )
            .await
        };

        let mut label_counts = LabelCounts::default();
        let author_features =
            user_results.map(|result| evaluable_author_features(result, &mut label_counts));
        record_author_labels(label_counts.mapped, label_counts.unmapped);
        let author_features = if let Some((cache, generation)) = cache_request {
            cache.resolve_hydration_batch(generation, author_features)
        } else {
            author_features
        };
        author_features.project(candidates.iter().map(|c| (c.tweet_id, c.author_id)))
    }
}

#[derive(Default)]
struct LabelCounts {
    mapped: usize,
    unmapped: usize,
}

fn evaluable_author_features(
    result: GizmoduckUserResult,
    counts: &mut LabelCounts,
) -> Completeness<AuthorFeatures> {
    let complete = !matches!(
        result.response_state,
        None | Some(UserResponseState::Failed) | Some(UserResponseState::Partial)
    );
    Completeness::new(complete, author_features(result, counts))
}

fn author_features(user_result: GizmoduckUserResult, counts: &mut LabelCounts) -> AuthorFeatures {
    user_result
        .user
        .map(|user| {
            let mut user_labels = AuthorLabelSet::default();
            for label in &user.labels.labels {
                match author_label(LabelValue(label.label_value)) {
                    Some(modeled) => {
                        user_labels.insert(modeled);
                        counts.mapped += 1;
                    }
                    None => counts.unmapped += 1,
                }
            }
            AuthorFeatures {
                is_suspended: user.safety.suspended,
                is_deactivated: user.safety.deactivated,
                is_protected: user.safety.is_protected,
                is_nsfw_user: user.safety.nsfw_user,
                is_nsfw_admin: user.safety.nsfw_admin,
                is_erased: user.safety.erased,
                is_offboarded: user.safety.offboarded,
                user_labels,
            }
        })
        .unwrap_or_default()
}

fn author_label(value: LabelValue) -> Option<AuthorLabel> {
    match value {
        LabelValue::NSFW_HIGH_RECALL => Some(AuthorLabel::NsfwHighRecall),
        LabelValue::NSFW_HIGH_PRECISION => Some(AuthorLabel::NsfwHighPrecision),
        LabelValue::NSFW_NEAR_PERFECT => Some(AuthorLabel::NsfwNearPerfect),
        LabelValue::NSFW_AVATAR_IMAGE => Some(AuthorLabel::NsfwAvatarImage),
        LabelValue::NSFW_BANNER_IMAGE => Some(AuthorLabel::NsfwBannerImage),
        LabelValue::SPAM_HIGH_RECALL => Some(AuthorLabel::SpamHighRecall),
        LabelValue::ABUSIVE_HIGH_RECALL => Some(AuthorLabel::AbusiveHighRecall),
        LabelValue::COMPROMISED => Some(AuthorLabel::Compromised),
        LabelValue::READ_ONLY => Some(AuthorLabel::ReadOnly),
        LabelValue::IMPERSONATION_HIGH_PRECISION => Some(AuthorLabel::ImpersonationHighPrecision),
        LabelValue::DO_NOT_AMPLIFY => Some(AuthorLabel::DoNotAmplify),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::Hydrated;
    use crate::models::{resolve_candidate, RawCandidate, TweetId};
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use xai_core_entities::entities::{
        GizmoduckUser, Label, Labels, PCFLabel, PureCoreData, Safety,
    };
    use xai_core_entities::gizmoduck_client::{
        GizmoduckClient, MockGizmoduckClient, UserFields, ViewerData,
    };

    struct FailingAfterFirstClient {
        calls: AtomicUsize,
    }

    #[tonic::async_trait]
    impl GizmoduckClient for FailingAfterFirstClient {
        async fn get_users(
            &self,
            user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<GizmoduckUserResult>>> {
            let succeeds = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
            user_ids
                .into_iter()
                .map(|id| {
                    let result = if succeeds {
                        Ok(Some(GizmoduckUserResult {
                            user: Some(GizmoduckUser {
                                user_id: id as u64,
                                safety: Safety {
                                    suspended: true,
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                            ..Default::default()
                        }))
                    } else {
                        Err(anyhow::anyhow!("gizmoduck unavailable"))
                    };
                    (id, result)
                })
                .collect()
        }

        async fn get_users_with_perspective(
            &self,
            _viewer_id: i64,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<GizmoduckUserResult>>> {
            unreachable!()
        }

        async fn get_viewer_roles(&self, _user_id: u64) -> Result<Vec<String>> {
            unreachable!()
        }

        async fn get_viewer_data(&self, _user_id: u64) -> Result<ViewerData> {
            unreachable!()
        }

        async fn get_viewer_data_with_fields(
            &self,
            _user_id: u64,
            _query_fields: &[QueryFields],
        ) -> Result<ViewerData> {
            unreachable!()
        }

        async fn get_pcf_labels(&self, _user_ids: Vec<i64>) -> HashMap<i64, Result<PCFLabel>> {
            unreachable!()
        }

        async fn get_profile_description_languages(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, Result<Option<String>>> {
            unreachable!()
        }

        async fn get_user_fields(&self, _user_ids: Vec<i64>) -> HashMap<i64, Result<UserFields>> {
            unreachable!()
        }

        async fn get_by_screen_name(
            &self,
            _screen_name: &str,
        ) -> Result<Option<GizmoduckUserResult>> {
            unreachable!()
        }
    }

    fn candidate(tweet_id: u64, author_id: u64) -> TweetCandidateInput {
        resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(tweet_id),
                request_author_id: Some(author_id),
            },
            &HashMap::<TweetId, PureCoreData>::new(),
            &HashMap::new(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn not_found_authors_default_features_and_share_one_backend_key() {
        let client = Arc::new(MockGizmoduckClient::default());
        let hydrator = GizmoduckAuthorHydrator::new(
            GizmoduckLookup::new(client.clone()),
            Some(FallbackCache::with_test_capacity("author")),
        );
        let candidates = vec![candidate(1, 10), candidate(2, 10)];

        let features = hydrator
            .hydrate(&candidates, SafetyLevel::TimelineHome)
            .await;

        assert_eq!(client.call_count(), 1);
        for tweet_id in [TweetId(1), TweetId(2)] {
            assert!(matches!(
                features.hydrated(&tweet_id),
                Some(Hydrated::NotFound)
            ));
            let feature = features.get_or_default(&tweet_id).into_value();
            assert!(!feature.is_suspended);
            assert!(!feature.is_deactivated);
            assert!(!feature.is_protected);
            assert!(!feature.is_nsfw_user);
            assert!(!feature.is_nsfw_admin);
            assert!(!feature.is_erased);
            assert!(!feature.is_offboarded);
        }
    }

    #[tokio::test]
    async fn stale_recovery_uses_resident_value() {
        let candidates = vec![candidate(1, 10)];
        let hydrator = GizmoduckAuthorHydrator::new(
            GizmoduckLookup::new(Arc::new(FailingAfterFirstClient {
                calls: AtomicUsize::new(0),
            })),
            Some(FallbackCache::with_test_capacity("author")),
        );
        let first = hydrator
            .hydrate(&candidates, SafetyLevel::TimelineHome)
            .await;
        assert!(first.get_or_default(&TweetId(1)).into_value().is_suspended);

        let second = hydrator
            .hydrate(&candidates, SafetyLevel::TimelineHome)
            .await;
        assert!(second.get_or_default(&TweetId(1)).into_value().is_suspended);
    }

    #[test]
    fn only_failed_partial_or_missing_response_states_are_incomplete() {
        let suspended = GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety: Safety {
                    suspended: true,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        for (state, complete) in [
            (Some(UserResponseState::Found), true),
            (Some(UserResponseState::NotFound), true),
            (Some(UserResponseState::DeactivatedUser), true),
            (Some(UserResponseState::SuspendedUser), true),
            (Some(UserResponseState::ProtectedUser), true),
            (Some(UserResponseState::ErasedUser), true),
            (Some(UserResponseState::OffboardedUser), true),
            (Some(UserResponseState::UnsafeUser), true),
            (Some(UserResponseState::Partial), false),
            (Some(UserResponseState::Failed), false),
            (None, false),
        ] {
            let features = evaluable_author_features(
                GizmoduckUserResult {
                    response_state: state,
                    ..suspended.clone()
                },
                &mut LabelCounts::default(),
            );
            assert_eq!(features.is_complete(), complete, "{state:?}");
            assert!(features.value().is_suspended, "{state:?}");
        }
    }

    fn user_with_labels(label_values: &[i32]) -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                user_id: 1,
                labels: Labels {
                    labels: label_values
                        .iter()
                        .map(|&label_value| Label {
                            label_value,
                            created_at_msec: 0,
                        })
                        .collect(),
                },
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn every_author_label_variant_round_trips_from_its_thrift_constant() {
        for (thrift, variant) in [
            (LabelValue::NSFW_HIGH_RECALL, AuthorLabel::NsfwHighRecall),
            (
                LabelValue::NSFW_HIGH_PRECISION,
                AuthorLabel::NsfwHighPrecision,
            ),
            (LabelValue::NSFW_NEAR_PERFECT, AuthorLabel::NsfwNearPerfect),
            (LabelValue::NSFW_AVATAR_IMAGE, AuthorLabel::NsfwAvatarImage),
            (LabelValue::NSFW_BANNER_IMAGE, AuthorLabel::NsfwBannerImage),
            (LabelValue::SPAM_HIGH_RECALL, AuthorLabel::SpamHighRecall),
            (
                LabelValue::ABUSIVE_HIGH_RECALL,
                AuthorLabel::AbusiveHighRecall,
            ),
            (LabelValue::COMPROMISED, AuthorLabel::Compromised),
            (LabelValue::READ_ONLY, AuthorLabel::ReadOnly),
            (
                LabelValue::IMPERSONATION_HIGH_PRECISION,
                AuthorLabel::ImpersonationHighPrecision,
            ),
            (LabelValue::DO_NOT_AMPLIFY, AuthorLabel::DoNotAmplify),
        ] {
            let mut counts = LabelCounts::default();
            let features = author_features(user_with_labels(&[thrift.0]), &mut counts);
            assert!(features.user_labels.has_label(variant), "{thrift:?}");
            assert_eq!((counts.mapped, counts.unmapped), (1, 0), "{thrift:?}");
        }
    }

    #[test]
    fn unmodelled_thrift_labels_drop_at_conversion() {
        for unmodelled in [
            LabelValue::EGREGIOUS_NSFW,
            LabelValue::RECOMMENDATIONS_BLACKLIST,
        ] {
            let mut counts = LabelCounts::default();
            let features = author_features(
                user_with_labels(&[unmodelled.0, LabelValue::SPAM_HIGH_RECALL.0]),
                &mut counts,
            );
            let mut expected = AuthorLabelSet::default();
            expected.insert(AuthorLabel::SpamHighRecall);
            assert_eq!(features.user_labels, expected, "{unmodelled:?}");
            assert_eq!((counts.mapped, counts.unmapped), (1, 1), "{unmodelled:?}");
        }
    }

    #[test]
    fn error_and_missing_author_results_fail_open_but_stay_failed() {
        let (a10, a20) = (candidate(1, 10).author_id, candidate(2, 20).author_id);
        let user_results: AuthorHydrationBatch<GizmoduckUserResult> = HydrationBatch::from_results(
            [a10, a20],
            HashMap::from([(a10, Err(anyhow::anyhow!("gizmoduck unavailable")))]),
        );

        let mut counts = LabelCounts::default();
        let by_tweet = user_results
            .map(|user_result| author_features(user_result, &mut counts))
            .project([(TweetId(1), a10), (TweetId(2), a20)]);

        for tweet_id in [TweetId(1), TweetId(2)] {
            assert!(matches!(
                by_tweet.hydrated(&tweet_id),
                Some(Hydrated::Failed(_))
            ));
            let feature = by_tweet.get_or_default(&tweet_id);
            assert!(!feature.is_suspended);
            assert!(!feature.is_deactivated);
        }
    }
}
