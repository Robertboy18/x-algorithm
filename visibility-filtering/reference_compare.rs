use crate::models::VfAction;
use crate::rules::{SafetyLevel, Verdict};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;
use xai_stats_receiver::global_stats_receiver;
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering::models::{Action, FilteredReason};
use xai_visibility_filtering::vf_client::{SafetyLevel as ReferenceSafetyLevel, VfClient};

const COMPARED: &str = "vf_reference_compared";
const EXACT_MATCH: &str = "vf_reference_exact_match";
const DIFFERED: &str = "vf_reference_differed";
const ERROR: &str = "vf_reference_error";
const SKIPPED: &str = "vf_reference_skipped";
const ENABLED: &str = "vf_reference_enabled";

const HARNESS_LINE_MARKER: &str = "vf_reference_compare";
const SCHEMA_VERSION: u32 = 1;
const LINE_BUDGET_BYTES: usize = 12 * 1024;

const REFERENCE_TIMEOUT: Duration = Duration::from_millis(1500);

pub(crate) fn should_build_harness(
    flag_enabled: bool,
    app_env: Option<&str>,
) -> Result<bool, &'static str> {
    match (flag_enabled, app_env) {
        (false, _) => Ok(false),
        (true, Some("prod")) => Err(
            "VF_DUAL_CALL_HARNESS_ENABLED is set but APP_ENV=prod; the reference comparator is staging-only",
        ),
        (true, _) => Ok(true),
    }
}

fn service_pair(action: &VfAction) -> (&'static str, Option<&FilteredReason>) {
    match action {
        VfAction::Allow => ("allow", None),
        VfAction::Drop(reason) => ("drop", Some(reason)),
        VfAction::Interstitial(reason) => ("interstitial", Some(reason)),
    }
}

fn reference_action_label(reason: &Option<FilteredReason>) -> &'static str {
    match reason {
        None => "allow",
        Some(FilteredReason::SafetyResult(safety_result)) => match safety_result.action {
            Action::NotEvaluated => "not_evaluated",
            Action::Allow => "allow",
            Action::Drop(_) => "drop",
            Action::Interstitial => "interstitial",
            Action::Downrank => "downrank",
            Action::Tombstone => "tombstone",
            Action::Avoid => "avoid",
        },
        Some(_) => "drop",
    }
}

fn reason_token(reason: &FilteredReason) -> String {
    match reason {
        FilteredReason::SafetyResult(safety_result) => match &safety_result.reason {
            Some(inner) => format!("{inner:?}"),
            None => "SafetyResult".to_string(),
        },
        FilteredReason::TweetMatchesViewerMutedKeyword(_) => {
            "TweetMatchesViewerMutedKeyword".to_string()
        }
        other => format!("{other:?}"),
    }
}

fn service_verdict_str(verdict: &Verdict) -> String {
    let (action, reason) = service_pair(&verdict.action);
    let mut out = action.to_string();
    if let Some(reason) = reason {
        out.push(':');
        out.push_str(&reason_token(reason));
    }
    if let Some(rule) = verdict.decided_by {
        out.push('@');
        out.push_str(rule);
    }
    out
}

fn reference_verdict_str(reference: &Option<FilteredReason>) -> String {
    match reference {
        None => "allow".to_string(),
        Some(reason) => format!(
            "{}:{}",
            reference_action_label(reference),
            reason_token(reason)
        ),
    }
}

pub(crate) fn is_exact_match(service: &VfAction, reference: &Option<FilteredReason>) -> bool {
    let (service_action, service_reason) = service_pair(service);
    service_action == reference_action_label(reference) && service_reason == reference.as_ref()
}

pub(crate) struct TweetVerdict {
    pub tweet_id: u64,
    pub verdict: Verdict,
}

pub(crate) struct VerdictSender(tokio::sync::oneshot::Sender<Vec<TweetVerdict>>);

impl VerdictSender {
    pub(crate) fn send(self, verdicts: Vec<TweetVerdict>) {
        let _ = self.0.send(verdicts);
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct CompareCounts {
    pub compared: u64,
    pub exact_match: u64,
    pub differed: u64,
    pub errors: HashMap<&'static str, u64>,
}

pub(crate) struct CompareContext<'a> {
    pub viewer_id: u64,
    pub safety_level: SafetyLevel,
    pub dc: &'a str,
    pub build_sha: &'a str,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Diff {
    pub tweet_id: u64,
    pub service: String,
    pub reference: String,
}

pub(crate) fn compare_batch(
    verdicts: &[TweetVerdict],
    reference_results: &HashMap<u64, anyhow::Result<Option<FilteredReason>>>,
) -> (CompareCounts, Vec<Diff>) {
    let mut counts = CompareCounts::default();
    let mut diffs = Vec::new();
    for TweetVerdict { tweet_id, verdict } in verdicts {
        let reference = match reference_results.get(tweet_id) {
            None => {
                *counts.errors.entry("missing_result").or_default() += 1;
                continue;
            }
            Some(Err(_)) => {
                *counts.errors.entry("reference_item").or_default() += 1;
                continue;
            }
            Some(Ok(reason)) => reason,
        };
        counts.compared += 1;
        if is_exact_match(&verdict.action, reference) {
            counts.exact_match += 1;
        } else {
            counts.differed += 1;
            diffs.push(Diff {
                tweet_id: *tweet_id,
                service: service_verdict_str(verdict),
                reference: reference_verdict_str(reference),
            });
        }
    }
    (counts, diffs)
}

fn batch_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    format!("{nanos:x}-{:x}", SEQ.fetch_add(1, Ordering::Relaxed))
}

struct Group<'a> {
    service: &'a str,
    reference: &'a str,
    tweet_ids: Vec<u64>,
}

fn group_diffs(diffs: &[Diff]) -> Vec<Group<'_>> {
    let mut index: HashMap<(&str, &str), usize> = HashMap::new();
    let mut groups: Vec<Group<'_>> = Vec::new();
    for diff in diffs {
        let at = *index
            .entry((diff.service.as_str(), diff.reference.as_str()))
            .or_insert_with(|| {
                groups.push(Group {
                    service: &diff.service,
                    reference: &diff.reference,
                    tweet_ids: Vec::new(),
                });
                groups.len() - 1
            });
        #[expect(clippy::indexing_slicing, reason = "at indexes a group already pushed")]
        groups[at].tweet_ids.push(diff.tweet_id);
    }
    groups
}

fn group_slices(group: &Group<'_>, budget: usize) -> Vec<serde_json::Value> {
    let whole = serde_json::json!([group.service, group.reference, group.tweet_ids]);
    if whole.to_string().len() < budget {
        return vec![whole];
    }
    let fixed = serde_json::json!([group.service, group.reference, []])
        .to_string()
        .len()
        + 1;
    let ids_per_slice = budget.saturating_sub(fixed).div_euclid(21).max(1);
    group
        .tweet_ids
        .chunks(ids_per_slice)
        .map(|ids| serde_json::json!([group.service, group.reference, ids]))
        .collect()
}

fn line_json(
    context: &CompareContext<'_>,
    batch: &str,
    chunk: [usize; 2],
    diffs: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "h": HARNESS_LINE_MARKER,
        "v": SCHEMA_VERSION,
        "batch": batch,
        "chunk": chunk,
        "build": context.build_sha,
        "dc": context.dc,
        "level": <&str>::from(context.safety_level),
        "viewer": context.viewer_id,
        "diffs": diffs,
    })
}

pub(crate) fn chunk_lines(
    context: &CompareContext<'_>,
    batch: &str,
    diffs: &[Diff],
) -> Vec<serde_json::Value> {
    if diffs.is_empty() {
        return Vec::new();
    }
    let header_len = line_json(context, batch, [1, 1], Vec::new())
        .to_string()
        .len();
    let budget = LINE_BUDGET_BYTES.saturating_sub(header_len);
    let mut pages: Vec<Vec<serde_json::Value>> = Vec::new();
    let mut current: Vec<serde_json::Value> = Vec::new();
    let mut used = 0;
    for group in group_diffs(diffs) {
        for slice in group_slices(&group, budget) {
            let cost = slice.to_string().len() + 1;
            if used + cost > budget && !current.is_empty() {
                pages.push(std::mem::take(&mut current));
                used = 0;
            }
            used += cost;
            current.push(slice);
        }
    }
    pages.push(current);
    let total = pages.len();
    pages
        .into_iter()
        .enumerate()
        .map(|(index, page)| line_json(context, batch, [index + 1, total], page))
        .collect()
}

pub(crate) fn comparable_request(
    safety_level: SafetyLevel,
    viewer_id: Option<u64>,
) -> Result<(ReferenceSafetyLevel, u64), &'static str> {
    let level = match safety_level {
        SafetyLevel::TimelineHome => ReferenceSafetyLevel::TimelineHome,
        SafetyLevel::TimelineHomeRecommendations => {
            ReferenceSafetyLevel::TimelineHomeRecommendations
        }
        SafetyLevel::FilterAll => return Err("level_unmapped"),
    };
    match viewer_id {
        Some(viewer_id) => Ok((level, viewer_id)),
        None => Err("logged_out_viewer"),
    }
}

const BUILD_SHA_LEN: usize = 12;
const VF_IMAGE_ENV: &str = "VF_IMAGE";

fn resolve_build_sha(compiled: &str, image: Option<&str>) -> String {
    if let Some(sha) = sha_prefix(compiled) {
        return sha.to_owned();
    }
    if let Some(image) = image
        && let Some(tag) = image.rsplit(':').next()
        && let Some(sha) = sha_prefix(tag)
    {
        return sha.to_owned();
    }
    let mut fallback = compiled.to_owned();
    fallback.truncate(BUILD_SHA_LEN);
    fallback
}

fn sha_prefix(s: &str) -> Option<&str> {
    let n = s.bytes().take_while(u8::is_ascii_hexdigit).count();
    (n >= BUILD_SHA_LEN).then(|| &s[..BUILD_SHA_LEN])
}

pub struct ReferenceCompareHarness {
    reference: Arc<dyn VfClient + Send + Sync>,
    dc: String,
    build_sha: String,
}

impl ReferenceCompareHarness {
    pub(crate) fn new(reference: Arc<dyn VfClient + Send + Sync>, datacenter: &str) -> Self {
        let compiled = xai_build_version::current_build_information().git_commit_sha;
        let image = std::env::var(VF_IMAGE_ENV).ok();
        let build_sha = resolve_build_sha(&compiled, image.as_deref());
        let harness = Self {
            reference,
            dc: datacenter.to_string(),
            build_sha,
        };
        info!(
            build_sha = %harness.build_sha,
            "reference_compare: harness enabled"
        );
        harness.incr(ENABLED, &[]);
        harness
    }

    pub(crate) fn begin_compare(
        self: &Arc<Self>,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
        tweet_ids: Vec<u64>,
    ) -> Option<VerdictSender> {
        let (reference_level, viewer_id) = match comparable_request(safety_level, viewer_id) {
            Ok(comparable) => comparable,
            Err(reason) => {
                self.incr(SKIPPED, &[("reason", reason)]);
                return None;
            }
        };
        if tweet_ids.is_empty() {
            return None;
        }
        let (tx, rx) = tokio::sync::oneshot::channel::<Vec<TweetVerdict>>();
        let harness = Arc::clone(self);
        tokio::spawn(async move {
            let viewer = TwitterContextViewer {
                user_id: viewer_id as i64,
                request_country_code: country_code.unwrap_or_default(),
                ..Default::default()
            };
            let reference_fut =
                harness
                    .reference
                    .get_result(tweet_ids, reference_level, viewer_id, Some(viewer));
            let (reference_outcome, verdicts) =
                futures::future::join(tokio::time::timeout(REFERENCE_TIMEOUT, reference_fut), rx)
                    .await;
            let reference_results: HashMap<u64, anyhow::Result<Option<FilteredReason>>> =
                match reference_outcome {
                    Ok(results) => results
                        .into_iter()
                        .map(|(id, r)| (id, r.map(|t| t.reason)))
                        .collect(),
                    Err(_) => {
                        harness.incr(ERROR, &[("kind", "timeout")]);
                        return;
                    }
                };
            let Ok(verdicts) = verdicts else { return };
            let context = CompareContext {
                viewer_id,
                safety_level,
                dc: &harness.dc,
                build_sha: &harness.build_sha,
            };
            let (counts, diffs) = compare_batch(&verdicts, &reference_results);
            harness.emit(safety_level, &counts);
            if !diffs.is_empty() {
                for line in chunk_lines(&context, &batch_id(), &diffs) {
                    #[expect(clippy::print_stdout, reason = "stdout is the diff sink")]
                    {
                        println!("{line}");
                    }
                }
            }
        });
        Some(VerdictSender(tx))
    }

    fn emit(&self, safety_level: SafetyLevel, counts: &CompareCounts) {
        let level = <&str>::from(safety_level);
        self.incr_nonzero(COMPARED, &[("safety_level", level)], counts.compared);
        self.incr_nonzero(EXACT_MATCH, &[("safety_level", level)], counts.exact_match);
        self.incr_nonzero(DIFFERED, &[("safety_level", level)], counts.differed);
        for (kind, count) in &counts.errors {
            self.incr_nonzero(ERROR, &[("kind", kind)], *count);
        }
    }

    fn incr(&self, metric: &str, labels: &[(&str, &str)]) {
        self.incr_nonzero(metric, labels, 1);
    }

    fn incr_nonzero(&self, metric: &str, labels: &[(&str, &str)], count: u64) {
        if count == 0 {
            return;
        }
        if let Some(sr) = global_stats_receiver() {
            let mut stamped: Vec<(&str, &str)> = labels.to_vec();
            stamped.push(("build_sha", &self.build_sha));
            sr.incr(metric, &stamped, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_visibility_filtering::models::{
        Action, DropReason, KeywordMatch, SafetyResult as ReferenceSafetyResult,
    };
    use xai_visibility_filtering::tweet_safety_label::SafetyLabelFailure;
    use xai_visibility_filtering::vf_client::TweetVisibility;

    fn reference_allow() -> Option<FilteredReason> {
        None
    }

    fn reference_bare_drop() -> Option<FilteredReason> {
        Some(FilteredReason::AuthorIsSuspended)
    }

    fn reference_safety_result(action: Action) -> Option<FilteredReason> {
        Some(FilteredReason::SafetyResult(ReferenceSafetyResult {
            reason: None,
            action,
        }))
    }

    fn reference_muted_keyword() -> Option<FilteredReason> {
        Some(FilteredReason::TweetMatchesViewerMutedKeyword(
            KeywordMatch {
                keyword: "spoilers".to_string(),
            },
        ))
    }

    fn service_allow() -> VfAction {
        VfAction::Allow
    }

    fn service_drop() -> VfAction {
        VfAction::Drop(FilteredReason::AuthorIsSuspended)
    }

    fn service_interstitial() -> VfAction {
        VfAction::Interstitial(FilteredReason::ContainNsfwMedia)
    }

    #[test]
    fn strict_equality_no_normalization() {
        assert!(is_exact_match(&service_allow(), &reference_allow()));
        assert!(is_exact_match(&service_drop(), &reference_bare_drop()));
        assert!(!is_exact_match(
            &VfAction::Drop(FilteredReason::AuthorIsUnsafe),
            &reference_bare_drop()
        ));
        assert!(!is_exact_match(&service_interstitial(), &reference_allow()));
        assert!(!is_exact_match(
            &service_allow(),
            &reference_safety_result(Action::Avoid)
        ));
        assert!(!is_exact_match(
            &service_drop(),
            &reference_safety_result(Action::Drop(DropReason {}))
        ));
        assert!(!is_exact_match(
            &service_allow(),
            &reference_muted_keyword()
        ));
        assert!(!is_exact_match(&service_drop(), &reference_muted_keyword()));
    }

    #[test]
    fn comparable_request_maps_home_levels_and_skips_the_rest() {
        assert_eq!(
            comparable_request(SafetyLevel::TimelineHome, Some(7)),
            Ok((ReferenceSafetyLevel::TimelineHome, 7))
        );
        assert_eq!(
            comparable_request(SafetyLevel::TimelineHomeRecommendations, Some(7)),
            Ok((ReferenceSafetyLevel::TimelineHomeRecommendations, 7))
        );
        assert_eq!(
            comparable_request(SafetyLevel::FilterAll, Some(7)),
            Err("level_unmapped")
        );
        assert_eq!(
            comparable_request(SafetyLevel::TimelineHome, None),
            Err("logged_out_viewer")
        );
    }

    #[test]
    fn should_build_harness_requires_flag_and_rejects_prod() {
        assert_eq!(should_build_harness(false, Some("prod")), Ok(false));
        assert_eq!(should_build_harness(false, Some("staging")), Ok(false));
        assert_eq!(should_build_harness(false, None), Ok(false));
        assert_eq!(should_build_harness(true, Some("staging")), Ok(true));
        assert_eq!(should_build_harness(true, None), Ok(true));
        assert!(should_build_harness(true, Some("prod")).is_err());
    }

    fn verdict(tweet_id: u64, action: VfAction, decided_by: Option<&'static str>) -> TweetVerdict {
        TweetVerdict {
            tweet_id,
            verdict: Verdict { action, decided_by },
        }
    }

    fn context() -> CompareContext<'static> {
        CompareContext {
            viewer_id: 99,
            safety_level: SafetyLevel::TimelineHomeRecommendations,
            dc: "atla",
            build_sha: "abc123def456",
        }
    }

    #[test]
    fn compare_batch_counts_policy_free_and_collects_differing_pairs_only() {
        let verdicts = vec![
            verdict(1, service_allow(), None),
            verdict(2, service_drop(), Some("DropSuspendedAuthorRule")),
            verdict(3, service_allow(), None),
            verdict(4, service_allow(), None),
            verdict(5, service_allow(), None),
        ];
        let reference_results: HashMap<u64, anyhow::Result<Option<FilteredReason>>> =
            HashMap::from([
                (1, Ok(reference_allow())),
                (2, Ok(reference_allow())),
                (3, Ok(reference_bare_drop())),
                (4, Err(anyhow::anyhow!("reference error"))),
            ]);

        let (counts, diffs) = compare_batch(&verdicts, &reference_results);

        assert_eq!(counts.compared, 3);
        assert_eq!(counts.exact_match, 1);
        assert_eq!(counts.differed, 2);
        assert_eq!(
            counts.errors,
            HashMap::from([("reference_item", 1), ("missing_result", 1)])
        );
        assert_eq!(diffs.len(), 2, "differing pairs only: {diffs:?}");
    }

    #[test]
    fn verdict_grammar_encodes_action_reason_and_rule() {
        let cases = [
            (verdict(0, service_allow(), None), "allow"),
            (
                verdict(0, service_drop(), Some("drop_suspended_author")),
                "drop:AuthorIsSuspended@drop_suspended_author",
            ),
            (verdict(0, service_drop(), None), "drop:AuthorIsSuspended"),
            (
                verdict(0, service_interstitial(), Some("nsfw_media")),
                "interstitial:ContainNsfwMedia@nsfw_media",
            ),
        ];
        for (v, expected) in &cases {
            assert_eq!(service_verdict_str(&v.verdict), *expected);
        }

        assert_eq!(reference_verdict_str(&reference_allow()), "allow");
        assert_eq!(
            reference_verdict_str(&reference_bare_drop()),
            "drop:AuthorIsSuspended"
        );
        assert_eq!(
            reference_verdict_str(&Some(FilteredReason::SafetyResult(ReferenceSafetyResult {
                reason: Some(
                    xai_visibility_filtering::models::SafetyResultReason::NsfwHighPrecision
                ),
                action: Action::Avoid,
            }))),
            "avoid:NsfwHighPrecision"
        );
        for (action, label) in [
            (Action::NotEvaluated, "not_evaluated"),
            (Action::Allow, "allow"),
            (Action::Drop(DropReason {}), "drop"),
            (Action::Interstitial, "interstitial"),
            (Action::Downrank, "downrank"),
            (Action::Tombstone, "tombstone"),
            (Action::Avoid, "avoid"),
        ] {
            assert_eq!(
                reference_verdict_str(&reference_safety_result(action)),
                format!("{label}:SafetyResult")
            );
        }
    }

    #[test]
    fn muted_keyword_payload_never_reaches_the_line() {
        let encoded = reference_verdict_str(&reference_muted_keyword());
        assert_eq!(encoded, "drop:TweetMatchesViewerMutedKeyword");
        assert!(!encoded.contains("spoilers"), "viewer content leaked");
    }

    fn diff(tweet_id: u64, service: &str, reference: &str) -> Diff {
        Diff {
            tweet_id,
            service: service.to_string(),
            reference: reference.to_string(),
        }
    }

    const ID: u64 = 1_000_000_000_000_000_000;

    #[test]
    fn identical_pairs_group_into_one_diffs_entry() {
        let diffs = vec![
            diff(1, "allow", "avoid:SafetyResult"),
            diff(2, "drop:ContainNsfwMedia@nsfw_media", "allow"),
            diff(3, "allow", "avoid:SafetyResult"),
            diff(4, "allow", "avoid:SafetyResult"),
        ];

        let lines = chunk_lines(&context(), "b1", &diffs);

        assert_eq!(lines.len(), 1, "a full request fits one line");
        assert_eq!(lines[0]["chunk"], serde_json::json!([1, 1]));
        assert_eq!(
            lines[0]["diffs"],
            serde_json::json!([
                ["allow", "avoid:SafetyResult", [1, 3, 4]],
                ["drop:ContainNsfwMedia@nsfw_media", "allow", [2]],
            ])
        );

        let repeated: Vec<Diff> = (0..150)
            .map(|i| diff(ID + i, "allow", "avoid:SafetyResult"))
            .collect();
        let lines = chunk_lines(&context(), "b2", &repeated);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].to_string().len() <= LINE_BUDGET_BYTES);
        assert_eq!(lines[0]["diffs"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn lines_split_at_the_byte_budget_and_stay_self_contained() {
        let diffs: Vec<Diff> = (0..300)
            .map(|i| {
                diff(
                    ID + i,
                    &format!("interstitial:ContainNsfwMedia@NsfwAuthorInterstitialRule{i}"),
                    "avoid:SafetyResult",
                )
            })
            .collect();

        let lines = chunk_lines(&context(), "b3", &diffs);

        assert!(lines.len() > 1, "300 distinct pairs exceed one budget");
        let total = lines.len();
        let mut seen = 0;
        for (i, line) in lines.iter().enumerate() {
            assert!(line.to_string().len() <= LINE_BUDGET_BYTES);
            assert_eq!(line["h"], "vf_reference_compare");
            assert_eq!(line["v"], 1);
            assert_eq!(line["batch"], "b3");
            assert_eq!(line["chunk"], serde_json::json!([i + 1, total]));
            assert_eq!(line["build"], "abc123def456");
            assert_eq!(line["dc"], "atla");
            assert_eq!(line["level"], "timeline_home_recommendations");
            assert_eq!(line["viewer"], 99);
            seen += line["diffs"].as_array().unwrap().len();
        }
        assert_eq!(seen, 300, "splitting is lossless");
    }

    #[test]
    fn oversized_single_group_splits_its_id_list() {
        let diffs: Vec<Diff> = (0..1000)
            .map(|i| diff(ID + i, "allow", "avoid:SafetyResult"))
            .collect();

        let lines = chunk_lines(&context(), "b4", &diffs);

        assert!(lines.len() > 1, "1000 ids exceed one budget");
        let ids: Vec<u64> = lines
            .iter()
            .flat_map(|line| line["diffs"].as_array().unwrap().iter())
            .flat_map(|group| group[2].as_array().unwrap().iter())
            .map(|id| id.as_u64().unwrap())
            .collect();
        assert_eq!(ids.len(), 1000, "splitting is lossless");
        assert_eq!(ids[0], ID);
        assert_eq!(ids[999], ID + 999);
        for line in &lines {
            assert!(line.to_string().len() <= LINE_BUDGET_BYTES);
            assert_eq!(line["diffs"][0][0], "allow");
            assert_eq!(line["diffs"][0][1], "avoid:SafetyResult");
        }
    }

    type RecordedCall = (Vec<u64>, ReferenceSafetyLevel, u64, Option<String>);

    struct FakeReference {
        calls: std::sync::Mutex<Vec<RecordedCall>>,
    }

    #[tonic::async_trait]
    impl VfClient for FakeReference {
        async fn get_result(
            &self,
            tweet_ids: Vec<u64>,
            safety_level: ReferenceSafetyLevel,
            for_user_id: u64,
            context: Option<TwitterContextViewer>,
        ) -> HashMap<u64, anyhow::Result<TweetVisibility>> {
            let results = tweet_ids
                .iter()
                .map(|&id| {
                    (
                        id,
                        Ok(TweetVisibility {
                            action: Action::Allow,
                            reason: None,
                            safety_labels: Err(SafetyLabelFailure::LookupFailed),
                        }),
                    )
                })
                .collect();
            self.calls.lock().unwrap().push((
                tweet_ids,
                safety_level,
                for_user_id,
                context.map(|c| c.request_country_code),
            ));
            results
        }
    }

    fn fake_harness() -> (Arc<ReferenceCompareHarness>, Arc<FakeReference>) {
        let fake = Arc::new(FakeReference {
            calls: std::sync::Mutex::new(Vec::new()),
        });
        (
            Arc::new(ReferenceCompareHarness::new(fake.clone(), "atla")),
            fake,
        )
    }

    #[tokio::test]
    async fn begin_compare_skips_unmappable_requests_without_calling_reference() {
        let (harness, fake) = fake_harness();

        assert!(
            harness
                .begin_compare(Some(7), None, SafetyLevel::FilterAll, vec![1])
                .is_none(),
            "FilterAll has no reference level"
        );
        assert!(
            harness
                .begin_compare(Some(7), None, SafetyLevel::TimelineHome, vec![])
                .is_none(),
            "empty requests have nothing to compare"
        );
        tokio::task::yield_now().await;
        assert!(fake.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn begin_compare_fetches_reference_concurrently_with_request_context() {
        let (harness, fake) = fake_harness();

        let sender = harness
            .begin_compare(
                Some(99),
                Some("de".to_string()),
                SafetyLevel::TimelineHomeRecommendations,
                vec![1, 2],
            )
            .expect("comparable request");
        sender.send(vec![
            verdict(1, service_allow(), None),
            verdict(2, service_allow(), None),
        ]);

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if !fake.calls.lock().unwrap().is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "reference never called"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let calls = fake.calls.lock().unwrap();
        assert_eq!(
            *calls,
            vec![(
                vec![1, 2],
                ReferenceSafetyLevel::TimelineHomeRecommendations,
                99,
                Some("de".to_string()),
            )]
        );
    }
}
