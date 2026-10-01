use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;
use xai_feature_switches::Params;
use xai_value_model::{
    compute_value_scores, CandidateScoringInputs, QueryScoringContext, ValueModelWeights,
};
use xai_vm_ranker_proto::RankRequest;

use crate::metrics::{VALUE_MODEL_FALLBACK, VALUE_MODEL_REQUESTS, VALUE_MODEL_STAGE};
use crate::params::*;
use crate::ranking_config::snowflake_creation_ms;

const NEW_USER_MIN_FOLLOWING: u32 = 5;
const FALLBACK_WARN_INTERVAL_MS: u64 = 10_000;

static LAST_FALLBACK_WARN_MS: AtomicU64 = AtomicU64::new(0);

pub struct ValueModelOutput {
    pub weighted: Vec<f64>,
    pub scores: Vec<f64>,
}

struct Fallback {
    reason: &'static str,
    detail: String,
}

impl Fallback {
    fn new(reason: &'static str, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

pub fn compute(
    req: &RankRequest,
    params: Option<&Params>,
    compute_value_model: bool,
) -> Option<ValueModelOutput> {
    if !compute_value_model {
        record_request("passthrough");
        return None;
    }
    match compute_or_fallback(req, params) {
        Ok(output) => {
            record_request("value_model");
            Some(output)
        }
        Err(fallback) => {
            record_request("fallback");
            record_fallback(&fallback, req.viewer_id);
            None
        }
    }
}

pub fn weights_from_params(params: &Params, viewer_id: u64) -> ValueModelWeights {
    ValueModelWeights {
        favorite: params.get(FavoriteWeight),
        reply: params.get(ReplyWeight),
        retweet: params.get(RetweetWeight),
        photo_expand: params.get(PhotoExpandWeight),
        video_open: params.get(VideoOpenWeight),
        click: params.get(ClickWeight),
        open_link: params.get(OpenLinkWeight),
        profile_click: params.get(ProfileClickWeight),
        vqv: params.get(VqvWeight),
        share: params.get(ShareWeight),
        share_via_dm: params.get(ShareViaDmWeight),
        share_via_copy_link: params.get(ShareViaCopyLinkWeight),
        dwell: params.get(DwellWeight),
        quote: params.get(QuoteWeight),
        quoted_click: params.get(QuotedClickWeight),
        quoted_vqv: params.get(QuotedVqvWeight),
        follow_author: params.get(FollowAuthorWeight),
        post_unexplored: params.get(PostUnexploredWeight),
        not_interested: params.get(NotInterestedWeight),
        block_author: params.get(BlockAuthorWeight),
        mute_author: params.get(MuteAuthorWeight),
        report: params.get(ReportWeight),
        not_dwelled: params.get(NotDwelledWeight),
        cont_dwell_time: params.get(ContDwellTimeWeight),
        cont_click_dwell_time: params.get(ContClickDwellTimeWeight),
        min_video_duration_ms: params.get(MinVideoDurationMs),
        enable_quoted_vqv_duration_check: params.get(EnableQuotedVqvDurationCheck),
        bidirectional_follow_reply_weight_boost: params.get(BidirectionalFollowReplyWeightBoost),
        bidirectional_follow_dwell_weight_boost: params.get(BidirectionalFollowDwellWeightBoost),
        enable_author_diversity: params.get(EnableAuthorDiversity),
        author_diversity_decay: params.get(AuthorDiversityDecay),
        author_diversity_floor: params.get(AuthorDiversityFloor),
        oon_rescore_in_network_replies_retweets: params
            .get(EnableOonRescoreForInNetworkRepliesRetweets),
        multiplier_pre_offset: params.get(MultiplierPreOffset),
    }
    .perturbed(
        params.get(WeightPerturbationSigma),
        &params.get(WeightPerturbationSalt),
        viewer_id,
    )
}

pub fn scoring_context(req: &RankRequest, params: &Params) -> QueryScoringContext {
    if req.topic_request {
        return QueryScoringContext {
            effective_oon_weight: params.get(TopicOonWeightFactor),
        };
    }
    let now_ms = req
        .viewer
        .as_ref()
        .map_or_else(|| now_ms() as i64, |v| v.now_ms);
    let account_age_secs = snowflake_creation_ms(req.viewer_id)
        .map(|created| now_ms - created)
        .filter(|&age| age >= 0)
        .map(|age| (age / 1000) as u64);
    let is_eligible_new_user = account_age_secs
        .is_some_and(|age| age < params.get(NewUserAgeThresholdSecs))
        && req.viewer_following_count >= NEW_USER_MIN_FOLLOWING;
    QueryScoringContext {
        effective_oon_weight: if is_eligible_new_user {
            params.get(NewUserOonWeightFactor)
        } else {
            params.get(OonWeightFactor)
        },
    }
}

pub fn candidate_inputs(
    req: &RankRequest,
    weights: &ValueModelWeights,
) -> Vec<CandidateScoringInputs> {
    req.candidates
        .iter()
        .map(|c| CandidateScoringInputs::from_rank_candidate(c, weights.min_video_duration_ms))
        .collect()
}

fn compute_or_fallback(
    req: &RankRequest,
    params: Option<&Params>,
) -> Result<ValueModelOutput, Fallback> {
    let params = params.ok_or_else(|| Fallback::new("no_config", ""))?;
    let weights = weights_from_params(params, req.viewer_id);
    let ctx = scoring_context(req, params);
    let inputs = candidate_inputs(req, &weights);
    let raw = compute_value_scores(&weights, &ctx, &inputs);
    let cached_weighted = !inputs.is_empty() && inputs.iter().all(|c| c.weighted_score.is_some());
    let stage = if cached_weighted {
        "cached_weighted"
    } else {
        "heads"
    };
    VALUE_MODEL_STAGE.with_label_values(&[stage]).inc();
    Ok(ValueModelOutput {
        weighted: raw.weighted,
        scores: raw.scores,
    })
}

fn record_request(mode: &str) {
    VALUE_MODEL_REQUESTS.with_label_values(&[mode]).inc();
}

fn record_fallback(fallback: &Fallback, viewer_id: u64) {
    VALUE_MODEL_FALLBACK
        .with_label_values(&[fallback.reason])
        .inc();
    let now = now_ms();
    let last = LAST_FALLBACK_WARN_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= FALLBACK_WARN_INTERVAL_MS
        && LAST_FALLBACK_WARN_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        warn!(
            "value model fell back to upstream scores: viewer={viewer_id} reason={} {}",
            fallback.reason, fallback.detail
        );
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
