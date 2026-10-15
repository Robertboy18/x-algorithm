use std::sync::Mutex;

use serde_json::Value;
use tracing::error;
use xai_abuse_proto::enforcement as abuse_proto;

pub use xai_abuse_enforcement_types::action_control::{
    ConfidenceEvidence, ConfidenceLevel, EvidenceOverride, LabelMapEntry, EVIDENCE_OVERRIDE_MAX_MS,
};

pub const CONFIG_KEY: &str = "confidence";
pub const DEFAULT_EVIDENCE_MAX_AGE_DAYS: i64 = 30;
const DAY_MS: i64 = 86_400_000;
const MAX_EVIDENCE_AGE_DAYS: f64 = 365.0;
pub const CLOCK_SKEW_MS: i64 = 300_000;
const CONFIG_KEYS: &[&str] = &["label_map", "evidence_max_age_days"];
const MAX_READS: u32 = 100_000;
const ALPHA: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Producer,
    LabelMap,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Producer => "producer",
            Source::LabelMap => "label_map",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    pub level: ConfidenceLevel,
    pub source: Source,
    pub evidence: Option<ConfidenceEvidence>,
    pub now_ms: i64,
}

pub fn level_from_proto(value: i32) -> Option<ConfidenceLevel> {
    match abuse_proto::Confidence::try_from(value) {
        Ok(abuse_proto::Confidence::Unspecified) => None,
        Ok(abuse_proto::Confidence::Low) | Err(_) => Some(ConfidenceLevel::Low),
        Ok(abuse_proto::Confidence::Mid) => Some(ConfidenceLevel::Mid),
        Ok(abuse_proto::Confidence::High) => Some(ConfidenceLevel::High),
    }
}

pub fn level_to_proto(level: ConfidenceLevel) -> abuse_proto::Confidence {
    match level {
        ConfidenceLevel::Low => abuse_proto::Confidence::Low,
        ConfidenceLevel::Mid => abuse_proto::Confidence::Mid,
        ConfidenceLevel::High => abuse_proto::Confidence::High,
    }
}

pub fn evidence_from_proto(e: &abuse_proto::ConfidenceEvidence) -> ConfidenceEvidence {
    ConfidenceEvidence {
        precision_lower_bound: e.precision_lower_bound,
        reads: e.reads,
        humans: e.humans,
        read_at_ms: e.read_at_ms,
        link: e.link.clone(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfidenceConfig {
    pub label_map: Vec<LabelMapEntry>,
    pub evidence_max_age_ms: i64,
    pub unreadable: Option<String>,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            label_map: Vec::new(),
            evidence_max_age_ms: DEFAULT_EVIDENCE_MAX_AGE_DAYS * DAY_MS,
            unreadable: None,
        }
    }
}

impl ConfidenceConfig {
    pub fn from_config(config: Option<&Value>) -> Self {
        Self::from_value(
            config
                .and_then(|c| c.get(crate::action_control::CONFIG_KEY))
                .and_then(|a| a.get(CONFIG_KEY)),
        )
    }

    pub fn from_value(v: Option<&Value>) -> Self {
        static LAST_BAD: Mutex<Option<String>> = Mutex::new(None);
        let mut out = Self::default();
        let Some(v) = v.filter(|v| !v.is_null()) else {
            return out;
        };
        let mut bad = Vec::new();
        match v.as_object() {
            Some(o) => bad.extend(
                o.keys()
                    .filter(|k| !CONFIG_KEYS.contains(&k.as_str()))
                    .map(|k| format!("unknown key {k}")),
            ),
            None => bad.push(v.to_string()),
        }
        match v.get("label_map") {
            None | Some(Value::Null) => {}
            Some(Value::Object(map)) => {
                let parsed: Option<Vec<LabelMapEntry>> = map
                    .iter()
                    .map(|(pattern, level)| {
                        let level = ConfidenceLevel::parse(level.as_str()?)?;
                        let ok = !pattern.is_empty()
                            && pattern.len() <= 256
                            && pattern
                                .bytes()
                                .all(|b| b.is_ascii_alphanumeric() || b"_.:@-*".contains(&b));
                        ok.then(|| LabelMapEntry {
                            pattern: pattern.clone(),
                            level,
                        })
                    })
                    .collect();
                match parsed {
                    Some(entries) => out.label_map = entries,
                    None => bad.push(format!("label_map={}", Value::Object(map.clone()))),
                }
            }
            Some(other) => bad.push(format!("label_map={other}")),
        }
        match v.get("evidence_max_age_days") {
            None | Some(Value::Null) => {}
            Some(days) => match days
                .as_f64()
                .filter(|d| d.is_finite() && *d > 0.0 && *d <= MAX_EVIDENCE_AGE_DAYS)
            {
                Some(d) => out.evidence_max_age_ms = (d * DAY_MS as f64) as i64,
                None => bad.push(format!("evidence_max_age_days={days}")),
            },
        }
        if bad.is_empty() {
            return out;
        }
        let bad = bad.join(", ");
        let mut last = LAST_BAD.lock().unwrap_or_else(|p| p.into_inner());
        if last.as_deref() != Some(bad.as_str()) {
            error!(
                value = %bad,
                "GrowthBook action_control.{CONFIG_KEY} is malformed; every level-driven decision records only"
            );
            *last = Some(bad.clone());
        }
        Self {
            label_map: Vec::new(),
            evidence_max_age_ms: 0,
            unreadable: Some(bad),
        }
    }

    pub fn translate(&self, labels: &[&str]) -> Option<ConfidenceLevel> {
        labels
            .iter()
            .flat_map(|l| {
                self.label_map
                    .iter()
                    .filter(move |e| glob(&e.pattern, l))
                    .map(|e| e.level)
            })
            .min()
    }
}

fn glob(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !text.starts_with(first) || text.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    if !text.ends_with(last) {
        return false;
    }
    for mid in &parts[1..parts.len() - 1] {
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

pub fn claim(
    score: &abuse_proto::ScoreResult,
    labels: &[&str],
    cfg: &ConfidenceConfig,
    now_ms: i64,
) -> Option<Claim> {
    let (level, source) = match level_from_proto(score.confidence) {
        Some(l) => (l, Source::Producer),
        None => (cfg.translate(labels)?, Source::LabelMap),
    };
    Some(Claim {
        level,
        source,
        evidence: score.confidence_evidence.as_ref().map(evidence_from_proto),
        now_ms,
    })
}

pub fn message_claim(
    score: &abuse_proto::ScoreResult,
    cfg: &ConfidenceConfig,
    now_ms: i64,
) -> Option<Claim> {
    let labels: Vec<&str> = if cfg.label_map.is_empty() {
        Vec::new()
    } else {
        score.summary.as_ref().map_or_else(Vec::new, |s| {
            s.labels
                .iter()
                .map(|l| crate::action_control::strip_dry_run_prefix(l))
                .collect()
        })
    };
    claim(score, &labels, cfg, now_ms)
}

fn upper_tail(s: u32, n: u32, p: f64) -> f64 {
    if s == 0 {
        return 1.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let mut ln_fact = Vec::with_capacity(n as usize + 1);
    ln_fact.push(0.0f64);
    for i in 1..=n {
        ln_fact.push(ln_fact[i as usize - 1] + f64::from(i).ln());
    }
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    (s..=n)
        .map(|k| {
            let ln_c = ln_fact[n as usize] - ln_fact[k as usize] - ln_fact[(n - k) as usize];
            (ln_c + f64::from(k) * lp + f64::from(n - k) * lq).exp()
        })
        .sum()
}

fn read_meets(reads: u32, humans: u32, floor: f64) -> bool {
    upper_tail(reads - humans, reads, floor) <= ALPHA
}

pub fn supported_level(
    ev: &ConfidenceEvidence,
    now_ms: i64,
    max_age_ms: i64,
) -> Option<ConfidenceLevel> {
    let lb = ev.precision_lower_bound;
    if !(lb.is_finite() && (0.0..=1.0).contains(&lb))
        || ev.reads == 0
        || ev.reads > MAX_READS
        || ev.humans > ev.reads
        || ev.read_at_ms <= 0
        || ev.read_at_ms > now_ms.saturating_add(CLOCK_SKEW_MS)
        || now_ms.saturating_sub(ev.read_at_ms) > max_age_ms
    {
        return None;
    }
    ConfidenceLevel::ALL
        .into_iter()
        .rev()
        .find(|l| lb >= l.floor() && read_meets(ev.reads, ev.humans, l.floor()))
}

pub fn evidence_problem(ev: &ConfidenceEvidence, now_ms: i64) -> Option<String> {
    let lb = ev.precision_lower_bound;
    if !(lb.is_finite() && (0.0..=1.0).contains(&lb)) {
        return Some("precision_lower_bound must be a number from 0 to 1".into());
    }
    if ev.reads == 0 || ev.reads > MAX_READS {
        return Some(format!("reads must be 1 to {MAX_READS}"));
    }
    if ev.humans > ev.reads {
        return Some("humans can't exceed reads".into());
    }
    if ev.read_at_ms <= 0 || ev.read_at_ms > now_ms.saturating_add(CLOCK_SKEW_MS) {
        return Some("read_at_ms must be the read's finish time, not in the future".into());
    }
    if ev.link.len() > 1_000 {
        return Some("link is longer than 1000 bytes".into());
    }
    None
}

pub fn override_active(ov: &EvidenceOverride, record_updated_at_ms: i64, now_ms: i64) -> bool {
    let written = record_updated_at_ms;
    !ov.by.trim().is_empty()
        && !ov.reason.trim().is_empty()
        && now_ms < ov.expires_at_ms
        && written < ov.expires_at_ms
        && written <= now_ms.saturating_add(CLOCK_SKEW_MS)
        && now_ms >= written.saturating_sub(CLOCK_SKEW_MS)
        && ov.expires_at_ms <= written.min(now_ms).saturating_add(EVIDENCE_OVERRIDE_MAX_MS)
}

pub fn override_problem(ov: &EvidenceOverride, now_ms: i64) -> Option<String> {
    if ov.by.trim().is_empty() || ov.by.len() > 256 {
        return Some("evidence_override.by must name the person (1-256 bytes)".into());
    }
    if ov.reason.trim().is_empty() || ov.reason.len() > 1_000 {
        return Some("evidence_override.reason must say why (1-1000 bytes)".into());
    }
    if ov.expires_at_ms <= now_ms || ov.expires_at_ms > now_ms + EVIDENCE_OVERRIDE_MAX_MS {
        return Some("evidence_override.expires_at_ms must be in the next 7 days".into());
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub struct LevelOutcome {
    pub claimed: ConfidenceLevel,
    pub source: Source,
    pub path: &'static str,
    pub acted: Option<ConfidenceLevel>,
    pub evidence_supports: Option<Option<ConfidenceLevel>>,
    pub override_by: Option<String>,
}

impl LevelOutcome {
    pub fn routing_label(claim: &Claim) -> Self {
        Self {
            claimed: claim.level,
            source: claim.source,
            path: "routing_label",
            acted: None,
            evidence_supports: None,
            override_by: None,
        }
    }

    pub fn info_keys(&self) -> Vec<(String, String)> {
        let mut keys = vec![
            (
                "confidence_level".to_owned(),
                self.claimed.as_str().to_owned(),
            ),
            (
                "confidence_source".to_owned(),
                self.source.as_str().to_owned(),
            ),
            ("confidence_path".to_owned(), self.path.to_owned()),
        ];
        if let Some(l) = self.acted {
            keys.push(("confidence_acted_level".to_owned(), l.as_str().to_owned()));
        }
        if let Some(s) = self.evidence_supports {
            keys.push((
                "confidence_evidence_supports".to_owned(),
                s.map_or("none", ConfidenceLevel::as_str).to_owned(),
            ));
        }
        if let Some(by) = &self.override_by {
            keys.push(("confidence_override_by".to_owned(), by.clone()));
        }
        keys
    }
}

pub const INFO_KEYS: &[&str] = &[
    "confidence_level",
    "confidence_source",
    "confidence_path",
    "confidence_acted_level",
    "confidence_evidence_supports",
    "confidence_override_by",
];

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const NOW: i64 = 1_791_000_000_000;

    fn ev(lb: f64, reads: u32, humans: u32) -> ConfidenceEvidence {
        ConfidenceEvidence {
            precision_lower_bound: lb,
            reads,
            humans,
            read_at_ms: NOW - DAY_MS,
            link: "documents/x".into(),
        }
    }

    fn supported(e: &ConfidenceEvidence) -> Option<ConfidenceLevel> {
        supported_level(e, NOW, DEFAULT_EVIDENCE_MAX_AGE_DAYS * DAY_MS)
    }

    #[test]
    fn the_memo_read_counts_meet_their_floors_and_one_fewer_does_not() {
        use ConfidenceLevel::*;
        assert_eq!(supported(&ev(0.95, 59, 0)), Some(High));
        assert_eq!(supported(&ev(0.95, 93, 1)), Some(High));
        assert_eq!(supported(&ev(0.95, 124, 2)), Some(High));
        assert_eq!(
            supported(&ev(0.95, 58, 0)),
            Some(Mid),
            "0 of 58 is under 0.95"
        );
        assert_eq!(supported(&ev(0.85, 19, 0)), Some(Mid));
        assert_eq!(supported(&ev(0.85, 50, 3)), Some(Mid));
        assert_eq!(supported(&ev(0.85, 68, 5)), Some(Mid));
        assert_eq!(supported(&ev(0.85, 18, 0)), Some(Low));
        assert_eq!(supported(&ev(0.70, 9, 0)), Some(Low));
        assert_eq!(supported(&ev(0.70, 8, 0)), None);
        assert_eq!(supported(&ev(0.70, 50, 25)), None);
    }

    #[test]
    fn a_typed_bound_cannot_claim_more_than_its_reads_show() {
        assert_eq!(
            supported(&ev(0.99, 10, 0)),
            Some(ConfidenceLevel::Low),
            "10 reads can't certify high or mid"
        );
        assert_eq!(
            supported(&ev(0.80, 200, 0)),
            Some(ConfidenceLevel::Low),
            "the stored bound also caps it"
        );
    }

    #[test]
    fn old_future_or_malformed_evidence_supports_nothing() {
        let mut old = ev(0.95, 59, 0);
        old.read_at_ms = NOW - 31 * DAY_MS;
        assert_eq!(supported(&old), None);
        let mut future = ev(0.95, 59, 0);
        future.read_at_ms = NOW + 2 * CLOCK_SKEW_MS;
        assert_eq!(supported(&future), None);
        for bad in [
            ev(f64::NAN, 59, 0),
            ev(1.5, 59, 0),
            ev(0.95, 0, 0),
            ev(0.95, 5, 6),
            ev(0.95, MAX_READS + 1, 0),
        ] {
            assert_eq!(supported(&bad), None, "{bad:?}");
        }
        let mut edge = ev(0.95, 59, 0);
        edge.read_at_ms = NOW + CLOCK_SKEW_MS;
        assert_eq!(
            supported(&edge),
            Some(ConfidenceLevel::High),
            "exactly the skew"
        );
        edge.read_at_ms += 1;
        assert_eq!(supported(&edge), None, "a millisecond past the skew");
        let mut never = ev(0.95, 59, 0);
        never.read_at_ms = 0;
        assert_eq!(supported(&never), None);
        assert_eq!(supported_level(&ev(0.95, 59, 0), NOW, 0), None, "max age 0");
    }

    #[test]
    fn unknown_proto_values_read_as_low() {
        assert_eq!(level_from_proto(0), None);
        assert_eq!(level_from_proto(1), Some(ConfidenceLevel::Low));
        assert_eq!(level_from_proto(2), Some(ConfidenceLevel::Mid));
        assert_eq!(level_from_proto(3), Some(ConfidenceLevel::High));
        assert_eq!(level_from_proto(4), Some(ConfidenceLevel::Low));
        assert_eq!(level_from_proto(-1), Some(ConfidenceLevel::Low));
    }

    #[test]
    fn label_map_translates_with_globs_and_picks_the_lowest() {
        let cfg = ConfidenceConfig::from_value(Some(&json!({
            "label_map": {"*_suspend": "high", "*_bounce": "mid", "bb1_*_spam_bot_*": "low"}
        })));
        assert_eq!(
            cfg.translate(&["anchor_campaign_suspend"]),
            Some(ConfidenceLevel::High)
        );
        assert_eq!(
            cfg.translate(&["bb1_user_bounce"]),
            Some(ConfidenceLevel::Mid)
        );
        assert_eq!(
            cfg.translate(&["bb1_post_spam_bot_suspend"]),
            Some(ConfidenceLevel::Low),
            "two entries match; the lower wins"
        );
        assert_eq!(
            cfg.translate(&["anchor_campaign_suspend", "bb1_user_bounce"]),
            Some(ConfidenceLevel::Mid)
        );
        assert_eq!(cfg.translate(&["llm_slop_user"]), None);
        assert_eq!(cfg.translate(&[]), None);
        assert!(glob("a*b*c", "a-b-c"));
        assert!(glob("*", "x"));
        assert!(!glob("a*a", "a"));
        assert!(!glob("ab", "abc"));
    }

    #[test]
    fn the_map_ships_empty_and_anything_malformed_makes_the_config_unreadable() {
        let shipped = ConfidenceConfig::from_value(None);
        assert!(shipped.label_map.is_empty());
        assert_eq!(shipped.unreadable, None);
        assert_eq!(ConfidenceConfig::default().evidence_max_age_ms, 30 * DAY_MS);
        let fine = ConfidenceConfig::from_value(Some(&json!({
            "label_map": {"*_suspend": "high"}, "evidence_max_age_days": 7
        })));
        assert_eq!(fine.unreadable, None);
        assert_eq!(fine.evidence_max_age_ms, 7 * DAY_MS);
        assert_eq!(fine.label_map.len(), 1);
        for bad in [
            json!({"label_map": {"*_suspend": "high", "x": "very_high"}}),
            json!({"label_map": ["*_suspend"]}),
            json!({"label_map": {"bad label": "high"}}),
            json!({"evidence_max_age_days": "30"}),
            json!({"evidence_max_age_days": -1}),
            json!({"evidence_max_age_days": 0}),
            json!({"evidence_max_age_days": 3650}),
            json!({"floors": {"high": 0.5}}),
            json!({"label_map": {"*_suspend": "high"}, "evidence_max_age": 30}),
            json!("on"),
            json!([]),
        ] {
            let cfg = ConfidenceConfig::from_value(Some(&bad));
            assert!(cfg.unreadable.is_some(), "{bad}");
            assert!(cfg.label_map.is_empty(), "{bad}");
            assert_eq!(cfg.evidence_max_age_ms, 0, "{bad}");
        }
        assert!(
            ConfidenceConfig::from_value(Some(&json!({"floors": {"high": 0.5}})))
                .unreadable
                .unwrap()
                .contains("unknown key floors")
        );
    }

    #[test]
    fn overrides_need_a_person_a_reason_and_an_expiry_within_seven_days() {
        let ov = |by: &str, reason: &str, expires: i64| EvidenceOverride {
            by: by.into(),
            reason: reason.into(),
            expires_at_ms: expires,
        };
        let written = NOW - DAY_MS;
        assert!(override_active(
            &ov("operator@example.com", "seeded", NOW + DAY_MS),
            written,
            NOW
        ));
        assert!(
            !override_active(&ov("operator@example.com", "seeded", NOW), written, NOW),
            "expired"
        );
        assert!(!override_active(
            &ov(" ", "seeded", NOW + DAY_MS),
            written,
            NOW
        ));
        assert!(!override_active(
            &ov("operator@example.com", "", NOW + DAY_MS),
            written,
            NOW
        ));
        assert!(
            !override_active(
                &ov("operator@example.com", "seeded", written + 8 * DAY_MS),
                written,
                NOW
            ),
            "a stored expiry past 7 days from the write is refused"
        );
        let active = |expires: i64, written: i64| {
            override_active(&ov("operator@example.com", "seeded", expires), written, NOW)
        };
        assert!(
            !active(NOW + 30 * DAY_MS, NOW + 30 * DAY_MS),
            "written past the skew"
        );
        assert!(
            !active(NOW + 7 * DAY_MS, NOW + 30 * DAY_MS),
            "written past the skew"
        );
        assert!(active(written + 7 * DAY_MS, written));
        assert!(!active(written + 7 * DAY_MS + 1, written));
        let ahead = NOW + CLOCK_SKEW_MS - 1;
        assert!(active(NOW + 7 * DAY_MS, ahead));
        assert!(!active(NOW + 7 * DAY_MS + 1, ahead));
        assert!(active(NOW + CLOCK_SKEW_MS + 60_000, NOW + CLOCK_SKEW_MS));
        assert!(!active(
            NOW + CLOCK_SKEW_MS + 60_000,
            NOW + CLOCK_SKEW_MS + 1
        ));
        assert!(
            !active(NOW + 60_000, NOW + CLOCK_SKEW_MS - 1),
            "expires before its write"
        );
        assert!(!active(ahead, ahead), "expires at its write");
        assert!(active(ahead + 1, ahead));
        assert!(override_problem(&ov("a", "b", NOW + 7 * DAY_MS), NOW).is_none());
        assert!(override_problem(&ov("a", "b", NOW + 7 * DAY_MS + 1), NOW).is_some());
        assert!(override_problem(&ov("a", "b", NOW), NOW).is_some());
        assert!(override_problem(&ov("", "b", NOW + DAY_MS), NOW).is_some());
    }
}
