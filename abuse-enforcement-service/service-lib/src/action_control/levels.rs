use super::*;
use crate::confidence::{self, Claim, ConfidenceLevel, LevelOutcome};

pub fn level_lkey(
    entity: ControlEntity,
    head: &str,
    classifier: Option<&str>,
    level: ConfidenceLevel,
) -> String {
    let e = entity.as_str();
    let l = level.as_str();
    match classifier {
        None => format!("{e}/l/{head}/{l}"),
        Some(c) => format!("{e}/cl/{head}/{c}/{l}"),
    }
}

impl Snapshot {
        fn level_hit(
        &self,
        entity: ControlEntity,
        names: &[&str],
        classifier: Option<&str>,
        level: ConfidenceLevel,
    ) -> Option<Hit<'_>> {
        let slot = |c: Option<&str>| -> Vec<String> {
            names
                .iter()
                .map(|n| level_lkey(entity, n, c, level))
                .collect()
        };
        let mut levels = Vec::with_capacity(2);
        if let Some(c) = classifier {
            levels.push(vec![slot(Some(c))]);
        }
        levels.push(vec![slot(None)]);
        self.find(&levels)
    }

            fn opt_in_hit(
        &self,
        entity: ControlEntity,
        names: &[&str],
        classifier: Option<&str>,
    ) -> Option<Hit<'_>> {
        let slot = |c: Option<&str>| -> Vec<String> {
            names.iter().map(|n| lkey_for(entity, n, c, None)).collect()
        };
        let mut levels = Vec::with_capacity(2);
        if let Some(c) = classifier {
            levels.push(vec![slot(Some(c))]);
        }
        levels.push(vec![slot(None)]);
        self.find(&levels)
    }

                        pub(super) fn level_decider(
        &self,
        entity: ControlEntity,
        names: &[&str],
        classifier: Option<&str>,
        level: ConfidenceLevel,
    ) -> LevelDecider<'_> {
        match self.level_hit(entity, names, classifier, level) {
            Some(Hit::Record(k, r)) if r.mode != ControlMode::Rules => LevelDecider::Record {
                lkey: k,
                record: r,
                level_record: true,
            },
            Some(Hit::Unreadable) => LevelDecider::Unreadable("unreadable"),
            Some(Hit::Ambiguous) => LevelDecider::Unreadable("band_ambiguous"),
            _ => match self.opt_in_hit(entity, names, classifier) {
                Some(Hit::Record(k, r)) if r.mode == ControlMode::ConfidenceDefaults => {
                    LevelDecider::Record {
                        lkey: k,
                        record: r,
                        level_record: false,
                    }
                }
                Some(Hit::Unreadable | Hit::Ambiguous) => LevelDecider::Unreadable("unreadable"),
                _ => LevelDecider::Nothing,
            },
        }
    }
}

pub(super) enum LevelDecider<'a> {
    Record {
        lkey: &'a str,
        record: &'a ActionControlRecord,
                level_record: bool,
    },
        Unreadable(&'static str),
        Nothing,
}

pub(super) struct LevelContext<'a> {
    pub entity: EntityType,
    pub head: &'a str,
    pub alias: Option<&'a str>,
    pub classifier: &'a str,
    pub factory: bool,
    pub cse: bool,
    pub stale: bool,
}

fn recording(
    matched: Option<Matched>,
    actions: Vec<PageAction>,
    forced: &'static str,
    outcome: LevelOutcome,
) -> Resolved {
    Resolved {
        matched,
        mode: ControlMode::Recording,
        page_actions: actions,
        forced: Some(forced),
        confidence: Some(outcome),
        ..Default::default()
    }
}

pub(super) fn resolve_level(
    snap: &Snapshot,
    m: &LevelContext<'_>,
    claim: &Claim,
    cfg: &ControlConfig,
    safety_ok: &mut dyn FnMut(EntityType) -> bool,
) -> Option<Resolved> {
    let entity = entity_of(m.entity);
    let names: Vec<&str> = std::iter::once(m.head).chain(m.alias).collect();
    let classifier = (!m.classifier.is_empty() && valid_name(m.classifier)).then_some(m.classifier);
    let mut outcome = LevelOutcome::routing_label(claim);
    let mut level = claim.level;
    let mut carried: Option<ConfidenceEvidence> = None;
    let mut matched: Option<Matched> = None;
    let mut last_actions: Vec<PageAction> = Vec::new();
    loop {
        let decider = match snap.level_decider(entity, &names, classifier, level) {
            LevelDecider::Record {
                lkey,
                record,
                level_record,
            } => Some((lkey, record, level_record)),
            LevelDecider::Unreadable(why) => {
                return Some(recording(matched, last_actions, why, outcome));
            }
            LevelDecider::Nothing => None,
        };
        let Some((lkey, record, level_record)) = decider else {
            matched.as_ref()?;
            return Some(recording(
                matched,
                last_actions,
                "evidence_below_floor",
                outcome,
            ));
        };
        outcome.path = if level_record && record.mode != ControlMode::ConfidenceDefaults {
            "level_record"
        } else {
            "defaults"
        };
        matched = Some(Matched {
            lkey: lkey.to_owned(),
            record: record.clone(),
        });
        let actions = match record.mode {
            ControlMode::ConfidenceDefaults => PageAction::confidence_defaults(entity, level),
            _ => record.actions.clone(),
        };
        let entity_ok =
            record.entity_type == entity && actions.iter().all(|a| a.entity() == entity);
        if m.factory {
            return Some(recording(matched, Vec::new(), "factory_managed", outcome));
        }
        if !entity_ok {
            return Some(recording(matched, Vec::new(), "wrong_entity", outcome));
        }
        if record.mode == ControlMode::Recording {
            return Some(Resolved {
                matched,
                mode: ControlMode::Recording,
                page_actions: actions,
                confidence: Some(outcome),
                ..Default::default()
            });
        }
        if actions.is_empty() {
            return Some(recording(matched, actions, "no_default_actions", outcome));
        }
        let forced = if m.cse {
            Some("cse_lock")
        } else if !cfg.enabled {
            Some("kill_switch")
        } else if m.stale {
            Some("stale_snapshot")
        } else if cfg.confidence.unreadable.is_some() {
            Some("confidence_config_unreadable")
        } else if !safety_ok(m.entity) {
            Some("safety_checks_changed")
        } else {
            None
        };
        if let Some(f) = forced {
            return Some(recording(matched, actions, f, outcome));
        }
        let own = if level_record {
            record.evidence.as_deref().cloned()
        } else {
            None
        };
        let evidence = own
            .or_else(|| carried.clone())
            .or_else(|| claim.evidence.clone());
        let supports = evidence.as_ref().and_then(|e| {
            confidence::supported_level(e, claim.now_ms, cfg.confidence.evidence_max_age_ms)
        });
        outcome.evidence_supports = Some(supports);
        let overridden = record
            .evidence_override
            .as_ref()
            .filter(|o| confidence::override_active(o, record.updated_at_ms, claim.now_ms));
        if supports.is_some_and(|s| s >= level) || overridden.is_some() {
            if supports.is_none_or(|s| s < level) {
                outcome.override_by = overridden.map(|o| o.by.clone());
            }
            outcome.acted = Some(level);
            return Some(Resolved {
                matched,
                mode: ControlMode::Live,
                page_actions: actions,
                confidence: Some(outcome),
                ..Default::default()
            });
        }
        match supports {
            Some(lower) => {
                carried = evidence;
                level = lower;
                last_actions = actions;
            }
            None => {
                let why = if evidence.is_some() {
                    "evidence_below_floor"
                } else {
                    "evidence_missing"
                };
                return Some(recording(matched, actions, why, outcome));
            }
        }
    }
}

pub(super) fn for_compare(rec: &ActionControlRecord) -> ActionControlRecord {
    match (rec.mode, rec.confidence) {
        (ControlMode::ConfidenceDefaults, Some(level)) => ActionControlRecord {
            mode: ControlMode::Live,
            actions: PageAction::confidence_defaults(rec.entity_type, level),
            ..rec.clone()
        },
        _ => rec.clone(),
    }
}

pub(super) fn defaults_harsher(
    before: Option<&ActionControlRecord>,
    after: &ActionControlRecord,
) -> Vec<String> {
    if after.confidence.is_some() {
        return Vec::new();
    }
    let was = before.is_some_and(|b| b.mode == ControlMode::ConfidenceDefaults);
    match (was, after.mode) {
        (false, ControlMode::ConfidenceDefaults) => {
            vec!["lets decisions with a confidence level act with the default table".into()]
        }
        (true, ControlMode::Rules) => vec![
            "hands decisions without a confidence level back to the GrowthBook rules, which act"
                .into(),
        ],
        _ => Vec::new(),
    }
}

pub(super) fn evidence_harsher(
    before: Option<&ActionControlRecord>,
    after: &ActionControlRecord,
    now_ms: i64,
    max_age_ms: i64,
) -> Vec<String> {
    let mut why = Vec::new();
    if !matches!(
        after.mode,
        ControlMode::Live | ControlMode::ConfidenceDefaults
    ) {
        return why;
    }
    let supports = |r: Option<&ActionControlRecord>| {
        r.and_then(|r| r.evidence.as_deref())
            .and_then(|e| confidence::supported_level(e, now_ms, max_age_ms))
    };
    let (was, now) = (supports(before), supports(Some(after)));
    if now > was
        && let Some(l) = now
    {
        why.push(format!(
            "new evidence lets the record act at {}",
            l.as_str()
        ));
    }
    let expiry = |r: Option<&ActionControlRecord>| {
        r.and_then(|r| {
            r.evidence_override
                .as_ref()
                .filter(|o| confidence::override_active(o, r.updated_at_ms, now_ms))
                .map(|o| o.expires_at_ms)
        })
    };
    if let Some(until) = expiry(Some(after))
        && expiry(before).is_none_or(|was| until > was)
    {
        why.push(format!(
            "an override lets the level act past its evidence floor until {until} (epoch ms)"
        ));
    }
    why
}

pub(super) fn validate_confidence(
    req: &ActionControlPutRequest,
    now_ms: i64,
) -> Result<(), SetError> {
    let bad = |code: &'static str, msg: String| Err(SetError::Invalid(code, msg));
    if req.confidence.is_some() && req.band.is_some() {
        return bad(
            "invalid_scope",
            "a record is keyed by band or by confidence level, not both".into(),
        );
    }
    if req.mode == ControlMode::ConfidenceDefaults {
        if req.band.is_some() {
            return bad(
                "invalid_scope",
                "confidence defaults are set on a head, classifier or confidence scope".into(),
            );
        }
        if !req.actions.is_empty() {
            return bad(
                "defaults_take_no_actions",
                "mode confidence_defaults acts with the service's default table; send actions: []"
                    .into(),
            );
        }
    }
    if let Some(ev) = &req.evidence {
        if req.confidence.is_none() {
            return bad(
                "invalid_evidence",
                "evidence belongs on a confidence scope".into(),
            );
        }
        if let Some(p) = confidence::evidence_problem(ev, now_ms) {
            return bad("invalid_evidence", p);
        }
    }
    if let Some(ov) = &req.evidence_override {
        let acts = matches!(
            req.mode,
            ControlMode::Live | ControlMode::ConfidenceDefaults
        );
        let scope_ok = req.confidence.is_some() || req.mode == ControlMode::ConfidenceDefaults;
        if !acts || !scope_ok {
            return bad(
                "invalid_override",
                "an evidence override belongs on a confidence scope that acts, or on a head or classifier following the confidence defaults"
                    .into(),
            );
        }
        if let Some(p) = confidence::override_problem(ov, now_ms) {
            return bad("invalid_override", p);
        }
    }
    Ok(())
}

pub(super) fn record_problem(rec: &ActionControlRecord, now_ms: i64) -> Option<&'static str> {
    let ahead = now_ms.saturating_add(confidence::CLOCK_SKEW_MS);
    if rec.updated_at_ms > ahead {
        return Some("written later than this pod's clock");
    }
    if rec
        .evidence
        .as_ref()
        .is_some_and(|ev| ev.read_at_ms > ahead)
    {
        return Some("evidence read later than this pod's clock");
    }
    if let Some(ov) = &rec.evidence_override {
        if ov.by.trim().is_empty() || ov.reason.trim().is_empty() {
            return Some("an evidence override must name a person and a reason");
        }
        if ov.expires_at_ms <= rec.updated_at_ms {
            return Some("an evidence override can't expire before its own write");
        }
        if ov.expires_at_ms
            > rec
                .updated_at_ms
                .saturating_add(confidence::EVIDENCE_OVERRIDE_MAX_MS)
        {
            return Some("an evidence override can't last past 7 days from its write");
        }
    }
    if rec.confidence.is_some() && rec.band.is_some() {
        return Some("a record can't be keyed by both band and confidence");
    }
    if rec.mode == ControlMode::ConfidenceDefaults
        && (rec.band.is_some() || !rec.actions.is_empty())
    {
        return Some("confidence_defaults takes no band and no actions");
    }
    if let Some(ev) = &rec.evidence {
        if rec.confidence.is_none() {
            return Some("evidence on a record without a confidence level");
        }
        let lb = ev.precision_lower_bound;
        if !(lb.is_finite() && (0.0..=1.0).contains(&lb)) {
            return Some("evidence lower bound is not a number from 0 to 1");
        }
    }
    None
}

pub(super) fn view(cfg: &ControlConfig) -> ConfidenceView {
    ConfidenceView {
        levels: ConfidenceLevel::ALL
            .into_iter()
            .map(|level| ConfidenceLevelView {
                level,
                floor: level.floor(),
                user_defaults: PageAction::confidence_defaults(ControlEntity::User, level),
                post_defaults: PageAction::confidence_defaults(ControlEntity::Post, level),
            })
            .collect(),
        evidence_max_age_ms: cfg.confidence.evidence_max_age_ms,
        label_map: cfg.confidence.label_map.clone(),
        config_error: cfg.confidence.unreadable.clone(),
    }
}
