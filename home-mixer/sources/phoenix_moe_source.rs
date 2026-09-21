use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixMOESource, EnablePhoenixRetrievalFallback, PhoenixMOEMaxResults,
    PhoenixRetrievalMOEInferenceClusterId, PhoenixXdsRetrievalMaxRetries,
};
use crate::util::egress::RetrievalDispatch;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub struct PhoenixMOESource {
    pub dispatch: RetrievalDispatch,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixMOESource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePhoenixMOESource)
            && (!query.is_topic_request() || query.is_bulk_topic_request())
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixMOESource: missing retrieval_sequence".to_string())?;

        let cluster = PhoenixRetrievalCluster::parse(
            &query.params.get(PhoenixRetrievalMOEInferenceClusterId),
        );

        let response = self
            .dispatch
            .retrieve_with_fallback(
                query,
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                quality_factor::apply(query.params.get(PhoenixMOEMaxResults)),
                0,
                vec![],
                None,
                None,
                None,
                query.params.get(PhoenixXdsRetrievalMaxRetries),
                query.params.get(EnablePhoenixRetrievalFallback),
            )
            .await
            .map_err(|e| format!("PhoenixMOESource: {e}"))?;

        let candidates: Vec<PostCandidate> = response
            .top_k_candidates
            .into_iter()
            .flat_map(|scored_candidates| scored_candidates.candidates)
            .filter_map(|scored_candidate| scored_candidate.candidate)
            .map(|tweet_info| PostCandidate {
                tweet_id: tweet_info.tweet_id,
                author_id: tweet_info.author_id,
                in_reply_to_tweet_id: (tweet_info.in_reply_to_tweet_id != 0)
                    .then_some(tweet_info.in_reply_to_tweet_id),
                retweeted_tweet_id: (tweet_info.retweeted_tweet_id != 0)
                    .then_some(tweet_info.retweeted_tweet_id),
                served_type: Some(pb::ServedType::ForYouPhoenixRetrievalMoe),
                ..Default::default()
            })
            .collect();

        Ok(candidates)
    }
}
