use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::clients::simclusters_ann_client::SimClustersAnnClient;
use crate::models::candidate::{PostCandidate, RetrievalSource};
use crate::models::engagement_signals::EngagementSignal;
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableSimclustersSource, SimclustersMaxCandidateAgeHours};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{
    build_moka_cache, MokaCache, MokaCacheConfig,
};
use xai_candidate_pipeline::hydrator::{CacheStore, Hydrator};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_stats_receiver::global_stats_receiver;
use xai_x_thrift::simclusters_ann::{
    EmbeddingType, InternalId, ModelVersion, Query, ScoringAlgorithm, SimClustersANNConfig,
    SimClustersANNTweetCandidate, SimClustersEmbeddingId,
};

const MAX_SANN_CANDIDATES: usize = 10_000;
const MAX_RESULTS: usize = 800;
const SOURCE_EMBEDDING_TYPE: EmbeddingType = EmbeddingType::LOG_FAV_LONGEST_L2_EMBEDDING_TWEET;
const CANDIDATE_EMBEDDING_TYPE: EmbeddingType = EmbeddingType::LOG_FAV_BASED_TWEET;
const MODEL_VERSION: ModelVersion = ModelVersion::MODEL_20M_145K_2020;
const ANN_MAX_NUM_RESULTS: i32 = 200;
const ANN_MIN_SCORE: f64 = 0.0;
const ANN_MAX_TOP_POSTS_PER_CLUSTER: i32 = 800;
const ANN_MAX_SCAN_CLUSTERS: i32 = 50;
const ANN_MIN_POST_CANDIDATE_AGE_HOURS: i32 = 0;
const POST_ANN_MIN_SCORE: f64 = 0.5;
const CACHE_METRIC: &str = "SimclustersSource.cache";

pub struct SimclustersSource {
    client: Arc<dyn SimClustersAnnClient + Send + Sync>,
    cache: MokaCache<(i64, i32), Vec<SimClustersANNTweetCandidate>>,
    core_data_hydrator: CoreDataCandidateHydrator,
}

impl SimclustersSource {
    pub fn new(
        client: Arc<dyn SimClustersAnnClient + Send + Sync>,
        core_data_hydrator: CoreDataCandidateHydrator,
    ) -> Self {
        Self {
            client,
            cache: build_moka_cache(MokaCacheConfig {
                size: 2_000_000,
                ttl: Duration::from_secs(600),
            }),
            core_data_hydrator,
        }
    }

    async fn get_post_candidates(
        &self,
        signal_id: i64,
        max_candidate_age_hours: i32,
    ) -> Result<Vec<SimClustersANNTweetCandidate>, String> {
        let cache_key = (signal_id, max_candidate_age_hours);
        if let Some(cached) = self.cache.get(&cache_key).await {
            Self::stat_cache("cache_hit");
            return Ok(cached);
        }
        Self::stat_cache("cache_miss");

        let candidates = self
            .client
            .get_tweet_candidates(build_query(signal_id, max_candidate_age_hours))
            .await
            .map_err(|e| format!("SimclustersSource: {e}"))?;

        self.cache.insert(cache_key, candidates.clone()).await;
        Ok(candidates)
    }

    fn stat_cache(result: &'static str) {
        if let Some(receiver) = global_stats_receiver() {
            receiver.incr(CACHE_METRIC, &[("requests", result)], 1);
        }
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for SimclustersSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableSimclustersSource)
            && !query.in_network_only
            && !query.has_cached_posts
            && has_post_signals(query)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let signal_ids = post_signal_ids(query);
        if signal_ids.is_empty() {
            return Ok(vec![]);
        }

        let max_per_query = (MAX_SANN_CANDIDATES / signal_ids.len()).max(1);
        let max_candidate_age_hours: i32 = query.params.get(SimclustersMaxCandidateAgeHours);

        let futures = signal_ids.into_iter().map(|signal_id| async move {
            let candidates = self
                .get_post_candidates(signal_id, max_candidate_age_hours)
                .await?;
            Ok::<_, String>(candidates)
        });

        let results = futures::future::join_all(futures).await;
        let mut per_query_results = Vec::with_capacity(results.len());
        for result in results {
            let candidates = result?;
            let filtered: Vec<SimClustersANNTweetCandidate> = candidates
                .into_iter()
                .filter(|c| *c.score > POST_ANN_MIN_SCORE)
                .take(max_per_query)
                .collect();
            per_query_results.push(filtered);
        }

        let mut interleaved = interleave_by_post_id(per_query_results);
        interleaved.truncate(MAX_RESULTS);
        let mut candidates: Vec<PostCandidate> = interleaved
            .into_iter()
            .enumerate()
            .map(|(index, c)| PostCandidate {
                tweet_id: c.tweet_id as u64,
                served_type: Some(pb::ServedType::ForYouSimclusters),
                retrieval_sources: vec![RetrievalSource {
                    served_type: pb::ServedType::ForYouSimclusters,
                    cluster: None,
                    score: Some(*c.score as f32),
                    position: Some(index as u32 + 1),
                }],
                ..Default::default()
            })
            .collect();

        let hydrated = self.core_data_hydrator.hydrate(query, &candidates).await;
        self.core_data_hydrator
            .update_all(&mut candidates, hydrated);
        candidates.retain(|c| c.author_id != 0);
        Ok(candidates)
    }
}

fn has_post_signals(query: &ScoredPostsQuery) -> bool {
    [
        query.explicit_engagement_signals.as_ref(),
        query.implicit_engagement_signals.as_ref(),
    ]
    .into_iter()
    .flatten()
    .any(|by_type| by_type.values().any(|list| !list.is_empty()))
}

fn post_signal_ids(query: &ScoredPostsQuery) -> Vec<i64> {
    let mut signals: Vec<&EngagementSignal> = Vec::new();
    for by_type in [
        query.explicit_engagement_signals.as_ref(),
        query.implicit_engagement_signals.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for list in by_type.values() {
            signals.extend(list.iter());
        }
    }

    signals.sort_by_key(|b| std::cmp::Reverse(b.engaged_at_ms));

    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    for signal in signals {
        if seen.insert(signal.tweet_id) {
            ids.push(signal.tweet_id);
        }
    }
    ids
}

fn build_query(signal_id: i64, max_candidate_age_hours: i32) -> Query {
    Query {
        source_embedding_id: SimClustersEmbeddingId {
            embedding_type: SOURCE_EMBEDDING_TYPE,
            model_version: MODEL_VERSION,
            internal_id: InternalId::TweetId(signal_id),
        },
        config: SimClustersANNConfig {
            max_num_results: ANN_MAX_NUM_RESULTS,
            min_score: OrderedFloat(ANN_MIN_SCORE),
            candidate_embedding_type: CANDIDATE_EMBEDDING_TYPE,
            max_top_tweets_per_cluster: ANN_MAX_TOP_POSTS_PER_CLUSTER,
            max_scan_clusters: ANN_MAX_SCAN_CLUSTERS,
            max_tweet_candidate_age_hours: max_candidate_age_hours,
            min_tweet_candidate_age_hours: ANN_MIN_POST_CANDIDATE_AGE_HOURS,
            ann_algorithm: ScoringAlgorithm::COSINE_SIMILARITY,
            engagement_threshold: None,
            is_cluster_detail_based_filtering_enabled: None,
            cluster_detail_based_threshold: None,
        },
    }
}

fn interleave_by_post_id(
    candidates: Vec<Vec<SimClustersANNTweetCandidate>>,
) -> Vec<SimClustersANNTweetCandidate> {
    let mut queues: Vec<VecDeque<_>> = candidates.into_iter().map(VecDeque::from).collect();
    let mut active: VecDeque<usize> = (0..queues.len()).collect();
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    while let Some(idx) = active.pop_front() {
        let Some(candidate) = queues[idx].pop_front() else {
            continue;
        };
        if seen.insert(candidate.tweet_id) {
            result.push(candidate);
            if !queues[idx].is_empty() {
                active.push_back(idx);
            }
        } else if !queues[idx].is_empty() {
            active.push_front(idx);
        }
    }

    result
}
