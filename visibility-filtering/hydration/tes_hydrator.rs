use crate::hydration::batch::{Hydrated, HydrationBatch, TweetHydrationBatch};
use crate::hydration::fallback_cache::FallbackCache;
use crate::hydration::metrics::{record_batch_size, timed_results};
use crate::hydration::tes_composite::{TweetForVisibility, TweetForVisibilitySource};
use crate::models::{
    CoreFeature, MediaFeature, NsfwFeature, TweetCandidateInput, TweetFeatures, TweetId,
};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use xai_core_entities::entities::{MediaEntities, PureCoreData};
use xai_core_entities::tweet_entity_service_client::TESClient;

const CLIENT_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const CLIENT: &str = "tes";

pub(crate) type AuthorIdFallbackCache = FallbackCache<TweetId, u64>;

pub struct TesHydrator {
    tes_client: Arc<dyn TESClient + Send + Sync>,
    tweet_source: Arc<dyn TweetForVisibilitySource>,
    author_id_cache: Option<AuthorIdFallbackCache>,
}

#[derive(Default)]
pub(crate) struct PureCoreHydration {
    pub(crate) core: HashMap<TweetId, PureCoreData>,
    pub(crate) recovered_authors: HashMap<TweetId, u64>,
}

impl TesHydrator {
    pub(crate) fn new(
        tes_client: Arc<dyn TESClient + Send + Sync>,
        tweet_source: Arc<dyn TweetForVisibilitySource>,
        author_id_cache: Option<AuthorIdFallbackCache>,
    ) -> Self {
        Self {
            tes_client,
            tweet_source,
            author_id_cache,
        }
    }

    pub(crate) fn author_id_fallback_cache(capacity: usize) -> AuthorIdFallbackCache {
        FallbackCache::new("author_id", capacity)
    }

    pub(crate) async fn fetch_pure_core(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> PureCoreHydration {
        if tweet_ids.is_empty() {
            return PureCoreHydration::default();
        }
        let cache_request = self
            .author_id_cache
            .as_ref()
            .map(|cache| (cache, cache.begin_request()));
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();
        record_batch_size(CLIENT, candidate_count_by_key.len());
        let fetched = timed_results(
            CLIENT,
            "get_tweet_core_datas",
            safety_level,
            &candidate_count_by_key,
            CLIENT_TIMEOUT,
            self.tes_client.get_tweet_core_datas(raw_ids),
        )
        .await;
        resolve_pure_core(cache_request, fetched.map_keys(TweetId))
    }

    pub(crate) async fn hydrate_tweets(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> TweetHydrationBatch<TweetForVisibility> {
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();
        timed_results(
            CLIENT,
            "get_tweets_for_visibility",
            safety_level,
            &candidate_count_by_key,
            CLIENT_TIMEOUT,
            self.tweet_source.get_tweets_for_visibility(&raw_ids),
        )
        .await
        .map_keys(TweetId)
    }

    pub(crate) fn assemble_tweet_features(
        &self,
        candidates: &[TweetCandidateInput],
        core_datas: &HashMap<TweetId, PureCoreData>,
        tweet_keyed: &TweetHydrationBatch<TweetForVisibility>,
    ) -> HashMap<TweetId, TweetFeatures> {
        candidates
            .iter()
            .map(|c| {
                (
                    c.tweet_id,
                    build_tweet_features(c.tweet_id, core_datas, tweet_keyed),
                )
            })
            .collect()
    }
}

fn candidates_per_tweet(tweet_ids: &[TweetId]) -> HashMap<u64, usize> {
    let mut candidate_count_by_key = HashMap::with_capacity(tweet_ids.len());
    for tweet_id in tweet_ids {
        *candidate_count_by_key.entry(tweet_id.0).or_default() += 1;
    }
    candidate_count_by_key
}

fn resolve_pure_core(
    cache_request: Option<(&AuthorIdFallbackCache, u64)>,
    fetched: TweetHydrationBatch<PureCoreData>,
) -> PureCoreHydration {
    let mut core = HashMap::new();
    let mut author_ids = HashMap::new();
    for (tweet_id, hydrated) in fetched.into_hydrated() {
        let author_id = match hydrated {
            Hydrated::Found(core_data) => {
                let author_id = core_data.author_id;
                core.insert(tweet_id, core_data);
                Hydrated::Found(author_id)
            }
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed(error) => Hydrated::Failed(error),
        };
        author_ids.insert(tweet_id, author_id);
    }
    let Some((cache, generation)) = cache_request else {
        return PureCoreHydration {
            core,
            recovered_authors: HashMap::new(),
        };
    };
    let recovered_authors = cache
        .resolve_hydration_batch(generation, HydrationBatch::from_hydrated(author_ids))
        .into_hydrated()
        .into_iter()
        .filter_map(|(tweet_id, hydrated)| match hydrated {
            Hydrated::Found(author_id) if !core.contains_key(&tweet_id) => {
                Some((tweet_id, author_id))
            }
            _ => None,
        })
        .collect();
    PureCoreHydration {
        core,
        recovered_authors,
    }
}

fn build_tweet_features(
    tweet_id: TweetId,
    core_datas: &HashMap<TweetId, PureCoreData>,
    tweet_keyed: &TweetHydrationBatch<TweetForVisibility>,
) -> TweetFeatures {
    let tweet = tweet_keyed.get(&tweet_id);
    let core = CoreFeature {
        text: core_datas
            .get(&tweet_id)
            .map(|core| core.text.clone())
            .unwrap_or_default(),
        source_tweet_id: tweet.and_then(|tweet| tweet.source_tweet_id),
    };
    match tweet {
        Some(tweet) => TweetFeatures {
            core,
            media: tweet.media.clone(),
            takedown_reasons: tweet.takedown_reasons.clone(),
            nsfw: NsfwFeature {
                user: tweet.nsfw_user,
                admin: tweet.nsfw_admin,
            },
            is_nullcast: tweet.is_nullcast,
            is_community_tweet: tweet.is_community_tweet,
            edit_control: tweet.edit_control.clone(),
        },
        None => TweetFeatures {
            core,
            ..Default::default()
        },
    }
}

pub(crate) fn media_feature(entities: MediaEntities) -> MediaFeature {
    let mut feature = MediaFeature {
        has_media: !entities.is_empty(),
        ..Default::default()
    };

    for restrictions in entities
        .iter()
        .filter(|e| e.media_key.is_some())
        .filter_map(|e| e.additional_metadata.as_ref())
        .filter_map(|metadata| metadata.restrictions.as_ref())
    {
        feature.has_dmca_media |= restrictions.is_dmca == Some(true);
        if let Some(geo) = &restrictions.geo_restrictions {
            feature
                .geo_allow_list
                .extend(geo.whitelisted_country_codes.iter().flatten().cloned());
            feature
                .geo_deny_list
                .extend(geo.blacklisted_country_codes.iter().flatten().cloned());
        }
    }

    feature
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::HydrationError;
    use crate::hydration::tes_composite::MockTweetForVisibilitySource;
    use crate::models::{resolve_candidate, RawCandidate};
    use xai_core_entities::entities::{EditControl, PureCoreData, TakedownReason};
    use xai_core_entities::tweet_entity_service_client::MockTESClient;

    fn found<V>(id: u64, value: V) -> TweetHydrationBatch<V> {
        TweetHydrationBatch::from_results(
            [TweetId(id)],
            HashMap::from([(TweetId(id), Ok::<_, anyhow::Error>(Some(value)))]),
        )
    }

    fn candidate(tweet_id: u64, author_id: u64) -> TweetCandidateInput {
        let core = HashMap::from([(
            TweetId(tweet_id),
            PureCoreData {
                author_id,
                ..Default::default()
            },
        )]);
        resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(tweet_id),
                request_author_id: None,
            },
            &core,
            &HashMap::new(),
        )
        .unwrap()
    }

    fn hydrator() -> TesHydrator {
        TesHydrator::new(
            Arc::new(MockTESClient::default()),
            Arc::new(MockTweetForVisibilitySource::default()),
            None,
        )
    }

    fn tweet() -> TweetForVisibility {
        TweetForVisibility {
            author_id: 100,
            source_tweet_id: None,
            is_nullcast: false,
            nsfw_user: false,
            nsfw_admin: false,
            has_takedown: false,
            takedown_reasons: Vec::new(),
            media: MediaFeature::default(),
            is_community_tweet: false,
            edit_control: None,
            exclusive_conversation_author_id: None,
        }
    }

    #[test]
    fn tweet_candidate_counts_deduplicate_backend_keys() {
        let tweet_ids = vec![TweetId(1), TweetId(1), TweetId(2)];

        assert_eq!(
            candidates_per_tweet(&tweet_ids),
            HashMap::from([(1, 2), (2, 1)])
        );
    }

    fn core_batch(
        entries: impl IntoIterator<Item = (u64, Hydrated<u64>)>,
    ) -> TweetHydrationBatch<PureCoreData> {
        HydrationBatch::from_hydrated(
            entries
                .into_iter()
                .map(|(id, hydrated)| {
                    let hydrated = match hydrated {
                        Hydrated::Found(author_id) => Hydrated::Found(PureCoreData {
                            author_id,
                            ..Default::default()
                        }),
                        Hydrated::NotFound => Hydrated::NotFound,
                        Hydrated::Failed(e) => Hydrated::Failed(e),
                    };
                    (TweetId(id), hydrated)
                })
                .collect(),
        )
    }

    fn failed() -> Hydrated<u64> {
        Hydrated::Failed(HydrationError::Timeout)
    }

    fn author_id_cache() -> AuthorIdFallbackCache {
        TesHydrator::author_id_fallback_cache(8)
    }

    #[test]
    fn failed_pure_core_recovers_only_previously_found_author_ids() {
        let cache = author_id_cache();
        let first = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(1, Hydrated::Found(10)), (2, Hydrated::NotFound)]),
        );
        assert!(first.recovered_authors.is_empty());

        let second = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(1, failed()), (2, failed()), (3, failed())]),
        );

        assert!(second.core.is_empty());
        assert_eq!(second.recovered_authors, HashMap::from([(TweetId(1), 10)]));

        let fresh_again = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(1, Hydrated::Found(10))]),
        );
        assert!(fresh_again.recovered_authors.is_empty());
    }

    #[test]
    fn without_cache_failed_pure_core_recovers_nothing() {
        resolve_pure_core(None, core_batch([(1, Hydrated::Found(10))]));

        let second = resolve_pure_core(None, core_batch([(1, failed())]));

        assert!(second.core.is_empty());
        assert!(second.recovered_authors.is_empty());
    }

    #[test]
    fn assemble_preserves_composite_features_when_core_missing() {
        let candidates = vec![candidate(10, 100)];
        let tweet_keyed = found(
            10,
            TweetForVisibility {
                source_tweet_id: Some(9),
                is_nullcast: true,
                is_community_tweet: true,
                nsfw_user: true,
                nsfw_admin: true,
                has_takedown: true,
                takedown_reasons: vec![TakedownReason::Dmca],
                edit_control: Some(EditControl::Initial(Default::default())),
                media: MediaFeature {
                    has_media: true,
                    has_dmca_media: true,
                    ..Default::default()
                },
                ..tweet()
            },
        );

        let features =
            hydrator().assemble_tweet_features(&candidates, &HashMap::new(), &tweet_keyed);

        assert_eq!(
            features[&TweetId(10)],
            TweetFeatures {
                core: CoreFeature {
                    text: String::new(),
                    source_tweet_id: Some(9),
                },
                media: MediaFeature {
                    has_media: true,
                    has_dmca_media: true,
                    geo_allow_list: Vec::new(),
                    geo_deny_list: Vec::new(),
                },
                takedown_reasons: vec![TakedownReason::Dmca],
                nsfw: NsfwFeature {
                    user: true,
                    admin: true
                },
                is_nullcast: true,
                is_community_tweet: true,
                edit_control: Some(EditControl::Initial(Default::default())),
            }
        );
    }

    #[test]
    fn missing_core_keeps_nullcast_drop_for_supplied_and_recovered_authors() {
        use crate::models::{HydratedTweetCandidate, VfAction, ViewerFeatures};
        use crate::rules::RuleEngine;

        let cache = author_id_cache();
        resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(10, Hydrated::Found(100))]),
        );
        let pure_core = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(10, failed())]),
        );
        let engine = RuleEngine::for_tests();
        for request_author_id in [None, Some(100)] {
            let candidate = resolve_candidate(
                &RawCandidate {
                    tweet_id: TweetId(10),
                    request_author_id,
                },
                &pure_core.core,
                &pure_core.recovered_authors,
            )
            .unwrap();
            for is_nullcast in [false, true] {
                let features = hydrator().assemble_tweet_features(
                    &[candidate],
                    &pure_core.core,
                    &found(
                        10,
                        TweetForVisibility {
                            is_nullcast,
                            ..tweet()
                        },
                    ),
                );
                let hydrated = HydratedTweetCandidate {
                    tweet_id: candidate.tweet_id.0,
                    author_id: candidate.author_id.get(),
                    tweet_features: features[&candidate.tweet_id].clone(),
                    ..Default::default()
                };
                for level in [
                    SafetyLevel::TimelineHome,
                    SafetyLevel::TimelineHomeRecommendations,
                ] {
                    let verdict = engine.evaluate(level, &ViewerFeatures::default(), &hydrated);
                    if is_nullcast {
                        assert!(matches!(verdict.action, VfAction::Drop(_)));
                        assert_eq!(verdict.decided_by, Some("NullcastedTweetDropRule"));
                    } else {
                        assert!(matches!(verdict.action, VfAction::Allow));
                    }
                }
            }
        }
    }
}
