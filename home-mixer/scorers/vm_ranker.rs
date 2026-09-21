use crate::clients::vm_ranker_client::{VMRankerClient, VMRankerCluster};
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_vm_ranker_proto::{DppParams, RankCandidate, RankRequest};

const DPP_VALUE_MODEL_ID: &str = "dpp";

pub struct VMRanker {
    pub client: Arc<dyn VMRankerClient>,
    pub xds_client: Option<Arc<dyn VMRankerClient>>,
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for VMRanker {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableVMRanker)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let cluster = VMRankerCluster::parse(&query.params.get(VMRankerClusterId));
        let request = build_request(query, candidates);

        let use_xds = self.xds_client.is_some()
            && crate::util::xds::use_xds_for_vm_ranker_cluster(query, &cluster.gate_name());

        let response = if use_xds {
            let xds = self.xds_client.as_ref().expect("checked is_some above");
            match xds.rank(cluster, request.clone()).await {
                Ok(resp) => resp,
                Err(e) => {
                    let enable_fallback = query.params.get(VMRankerEnableFallback);
                    if !enable_fallback {
                        let msg = format!("VMRanker xDS gRPC call failed (fallback disabled): {e}");
                        return vec![Err(msg); candidates.len()];
                    }
                    tracing::warn!(cluster = ?cluster, error = %e, "VMRanker xDS rank failed; falling back to DNS");
                    match self.client.rank(cluster, request).await {
                        Ok(resp) => resp,
                        Err(e) => {
                            let msg = format!("VMRanker gRPC call failed: {e}");
                            return vec![Err(msg); candidates.len()];
                        }
                    }
                }
            }
        } else {
            match self.client.rank(cluster, request).await {
                Ok(resp) => resp,
                Err(e) => {
                    let msg = format!("VMRanker gRPC call failed: {e}");
                    return vec![Err(msg); candidates.len()];
                }
            }
        };

        let score_map: FxHashMap<u64, f64> = response
            .candidates
            .iter()
            .map(|sc| (sc.tweet_id, sc.score))
            .collect();

        candidates
            .iter()
            .map(|c| {
                Ok(PostCandidate {
                    score: score_map.get(&c.tweet_id).copied().or(c.score),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}

fn build_request(query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> RankRequest {
    let proto_candidates: Vec<RankCandidate> = candidates
        .iter()
        .map(|c| RankCandidate {
            tweet_id: c.tweet_id,
            retweeted_tweet_id: c.retweeted_tweet_id.unwrap_or(0),
            score: c.score,
            ..Default::default()
        })
        .collect();

    let dpp_theta = query.params.get(VMRankerDppTheta);
    let dpp_max_selected_rank = query.params.get(VMRankerDppMaxSelectedRank);

    let dpp_params = if dpp_theta > 0.0 || dpp_max_selected_rank > 0 {
        Some(DppParams {
            theta: dpp_theta,
            max_selected_rank: dpp_max_selected_rank,
        })
    } else {
        None
    };

    RankRequest {
        viewer_id: query.user_id,
        candidates: proto_candidates,
        value_model_id: DPP_VALUE_MODEL_ID.to_string(),
        dpp_params,
        ..Default::default()
    }
}
