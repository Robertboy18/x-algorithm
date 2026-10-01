use crate::models::candidate::PostCandidate;
use crate::models::content_features;
use crate::models::query::ScoredPostsQuery;
use prost::Message;
use xai_recsys_proto::{ContentFeatures, ProductSurface};
use xai_vm_ranker_proto as pb;
use xai_vm_ranker_proto::experiment::{CandidatePayload, RequestPayload};

pub(crate) fn request_payload(query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> Vec<u8> {
    RequestPayload {
        engine_time_hint_ms: phoenix_scored_at_ms(candidates),
        user_sequence_arrow_ipc: query.columnar_scoring_sequence.clone().unwrap_or_default(),
        product_surface: if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        } as i32,
        client_app_id: i64::from(query.client_app_id),
    }
    .encode_to_vec()
}

pub(crate) fn candidate_payload(candidate: &PostCandidate) -> Vec<u8> {
    CandidatePayload {
        content_features: Some(content_features_proto(&content_features::build(candidate))),
        quoted_content_features: content_features::build_quoted(candidate)
            .map(|f| content_features_proto(&f)),
        quoted_tweet_id: candidate.quoted_tweet_id.unwrap_or(0),
        ..Default::default()
    }
    .encode_to_vec()
}

fn phoenix_scored_at_ms(candidates: &[PostCandidate]) -> Option<i64> {
    candidates
        .iter()
        .filter_map(|c| c.last_scored_at_ms)
        .max()
        .map(|ms| ms as i64)
}

fn content_features_proto(f: &ContentFeatures) -> pb::experiment::ContentFeatures {
    pb::experiment::ContentFeatures {
        has_video: f.has_video,
        max_video_duration_ms: f.max_video_duration_ms,
        has_photo: f.has_photo,
        media_count: f.media_count,
        weighted_text_len: f.weighted_text_len,
        newline_count: f.newline_count,
        has_url: f.has_url,
    }
}
