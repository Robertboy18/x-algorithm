use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{PhoenixExperimentOverrides, RerankerHeadTag};
use rustc_hash::FxHashSet;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::TOP_LOG_PROBS_NUM;
use xai_geo_ip::zip_to_dma_code;
use xai_recsys_proto::{
    country_code_string_to_enum, CandidateSet, ClientContext, DeviceFeature,
    PredictNextActionsRequest, ProductSurface, UserContext,
};

const PHOENIX_CLIENT_MAX_CANDIDATES: usize = 2800;

pub fn build_client_context(query: &ScoredPostsQuery) -> Option<ClientContext> {
    if query.user_id == 0 {
        return None;
    }

    Some(ClientContext {
        user_id: query.user_id as i64,
        app_id: query.client_app_id as i64,
        country_code: query.country_code.clone(),
        language_code: query.language_code.clone(),
        user_roles: query.user_roles.clone(),
        ip_address: query.ip_address.clone(),
        client_version: query.client_version.clone(),
        ..Default::default()
    })
}

pub fn build_user_context(query: &ScoredPostsQuery) -> Option<UserContext> {
    Some(UserContext {
        user_age_bracket: query
            .user_demographics
            .as_ref()
            .map(|d| d.user_age_bracket)
            .unwrap_or(0),
        user_gender: query
            .user_demographics
            .as_ref()
            .map(|d| d.user_gender)
            .unwrap_or(0),
        user_state: query
            .user_demographics
            .as_ref()
            .map(|d| d.user_state)
            .unwrap_or(0),
        user_age_in_years: query.user_age_in_years.unwrap_or(0),
        user_inferred_gender: query.user_inferred_gender.map(|v| v as i32).unwrap_or(0),
        user_inferred_gender_score: query.user_inferred_gender_score.unwrap_or(0.0),
        followed_grok_topics: query
            .followed_grok_topics
            .map(|t| t.to_vec())
            .unwrap_or_default(),
        followed_starter_packs: query
            .followed_starter_packs
            .map(|p| p.to_vec())
            .unwrap_or_default(),
        user_longitude: query
            .ip_location
            .as_ref()
            .and_then(|l| l.longitude)
            .unwrap_or(0.0) as f32,
        user_latitude: query
            .ip_location
            .as_ref()
            .and_then(|l| l.latitude)
            .unwrap_or(0.0) as f32,
        user_dma_code: zip_to_dma_code(
            query
                .ip_location
                .as_ref()
                .and_then(|l| l.zip_code.as_deref()),
        ) as i32,
        user_installed_apps: query.user_installed_apps.clone().unwrap_or_default(),
        ..Default::default()
    })
}

pub fn build_device_feature(query: &ScoredPostsQuery) -> DeviceFeature {
    DeviceFeature {
        engaging_ip_country_code: country_code_string_to_enum(query.country_code.as_ref()) as i32,
        device_network_type: query.device_network_type as i32,
        timezone: query.time_zone as i32,
        ip_address: query.ip_address.clone(),
    }
}

fn followed_user_ids(query: &ScoredPostsQuery) -> FxHashSet<u64> {
    query
        .user_features
        .followed_user_ids
        .iter()
        .copied()
        .map(|id| id as u64)
        .collect()
}

fn candidate_to_tweet_info(
    c: &PostCandidate,
    query: &ScoredPostsQuery,
    followed_ids: &FxHashSet<u64>,
) -> xai_recsys_proto::TweetInfo {
    let is_followed = followed_ids.contains(&c.get_original_author_id())
        || c.get_original_author_id() == query.user_id;
    c.as_tweet_info(is_followed)
}

pub fn build_tweet_infos(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
) -> Vec<xai_recsys_proto::TweetInfo> {
    let followed_ids = followed_user_ids(query);
    candidates
        .iter()
        .take(PHOENIX_CLIENT_MAX_CANDIDATES)
        .map(|c| candidate_to_tweet_info(c, query, &followed_ids))
        .collect()
}

pub fn build_request_without_sequence_and_candidates(
    query: &ScoredPostsQuery,
    product_surface: ProductSurface,
) -> PredictNextActionsRequest {
    let candidate_set = CandidateSet {
        user_id: query.user_id,
        product_surface: product_surface as i32,
        device_feature: Some(build_device_feature(query)),
        ..Default::default()
    };
    PredictNextActionsRequest {
        candidate_sets: vec![candidate_set],
        return_logprob: true,
        top_logprobs_num: TOP_LOG_PROBS_NUM,
        return_backbone_scores: query.return_backbone_scores,
        client_context: build_client_context(query),
        user_context: build_user_context(query),
        metadata: query.request_id.to_string(),
        experiment_overrides: parse_experiment_overrides(
            &query.params.get(PhoenixExperimentOverrides),
        ),
        ..Default::default()
    }
}

pub fn parse_experiment_overrides(spec: &str) -> std::collections::HashMap<String, String> {
    spec.split(';')
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            let (k, v) = (k.trim(), v.trim());
            (!k.is_empty()).then(|| (k.to_string(), v.to_string()))
        })
        .collect()
}

pub fn build_prediction_request(
    query: &ScoredPostsQuery,
    candidates: &[PostCandidate],
    product_surface: ProductSurface,
) -> PredictNextActionsRequest {
    let mut request = build_request_without_sequence_and_candidates(query, product_surface);
    request.candidate_sets[0].candidates = build_tweet_infos(query, candidates);
    let mut sequence = query.scoring_sequence.clone().unwrap_or_default();
    sequence.reranker_head_tag = Some(query.params.get(RerankerHeadTag) as u32);
    request.sequences = vec![sequence];
    request.columnar_sequences = query.columnar_scoring_sequence.iter().cloned().collect();
    request
}
