use std::sync::Arc;

use xai_abuse_proto::enforcement::{self as proto, ScoreResult};

use super::*;
use crate::action_control::test_support::record;
use crate::action_control::{
    ActionControlRecord, ConfidenceEvidence, ControlEntity, ControlMode, EvidenceOverride,
    Lifetime, PageAction,
};
use crate::confidence::ConfidenceLevel;
use crate::overturn_hold::{FakeHoldStore, Hold, HoldGate, HOLD_KIND_SUSPEND};
use crate::replay_tests::{
    base_config, message, on_config, switched_off, Account, BedConfig, TestBed, Trace,
    BAKED_POST_RULES, BAKED_USER_RULES, HEADS, SWITCHED_OFF_HEAD,
};

const SUSPEND: PageAction = PageAction::Suspend {};
const LIVENESS: PageAction = PageAction::LivenessCheck {};
const SHR7: PageAction = PageAction::SpamHighRecall {
    lifetime: Lifetime::Days7,
};
const DAY_MS: i64 = 86_400_000;

const BB1: (&str, &str) = HEADS[0];
const LABEL: &str = "anchor_campaign_suspend";

fn now_ms() -> i64 {
    now_millis()
}

fn evidence(level: ConfidenceLevel) -> ConfidenceEvidence {
    let (lb, reads, humans) = match level {
        ConfidenceLevel::High => (0.951, 59, 0),
        ConfidenceLevel::Mid => (0.88, 50, 2),
        ConfidenceLevel::Low => (0.72, 12, 0),
    };
    ConfidenceEvidence {
        precision_lower_bound: lb,
        reads,
        humans,
        read_at_ms: now_ms() - DAY_MS,
        link: "documents/read.md".into(),
    }
}

fn proto_evidence(e: &ConfidenceEvidence) -> proto::ConfidenceEvidence {
    proto::ConfidenceEvidence {
        precision_lower_bound: e.precision_lower_bound,
        reads: e.reads,
        humans: e.humans,
        read_at_ms: e.read_at_ms,
        link: e.link.clone(),
    }
}

fn with_level(
    mut s: ScoreResult,
    level: proto::Confidence,
    ev: Option<ConfidenceEvidence>,
) -> ScoreResult {
    s.confidence = level as i32;
    s.confidence_evidence = ev.as_ref().map(proto_evidence);
    s
}

fn bb1(account: Account) -> ScoreResult {
    message(EntityType::User, BB1.0, BB1.1, &[LABEL], account)
}

fn high(account: Account) -> ScoreResult {
    with_level(
        bb1(account),
        proto::Confidence::High,
        Some(evidence(ConfidenceLevel::High)),
    )
}

fn defaults(head: &str, entity: ControlEntity) -> ActionControlRecord {
    ActionControlRecord {
        entity_type: entity,
        ..record(head, None, None, ControlMode::ConfidenceDefaults, vec![])
    }
}

fn level_record(
    head: &str,
    classifier: Option<&str>,
    level: ConfidenceLevel,
    mode: ControlMode,
    actions: Vec<PageAction>,
) -> ActionControlRecord {
    ActionControlRecord {
        confidence: Some(level),
        evidence: Some(Box::new(evidence(level))),
        ..record(head, classifier, None, mode, actions)
    }
}

async fn bed_with(records: Vec<ActionControlRecord>, config: serde_json::Value) -> TestBed {
    TestBed::new(BedConfig {
        user_rules: BAKED_USER_RULES,
        post_rules: BAKED_POST_RULES,
        config,
        off_switches: vec![switched_off()],
        control_records: records,
    })
    .await
}

async fn bed(records: Vec<ActionControlRecord>) -> TestBed {
    bed_with(records, on_config(false, None)).await
}

fn info(t: &Trace) -> &BTreeMap<String, String> {
    &t.outcomes.last().expect("one outcome").info
}

fn without_expiry(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(o) => o
            .iter()
            .map(|(k, v)| {
                let v = if k == "expiresAtMsec" {
                    serde_json::Value::Null
                } else {
                    without_expiry(v)
                };
                (k.clone(), v)
            })
            .collect(),
        serde_json::Value::Array(a) => a.iter().map(without_expiry).collect(),
        other => other.clone(),
    }
}

fn effect(t: &Trace) -> (String, Vec<(String, serde_json::Value)>, i64, Vec<String>) {
    (
        t.status.clone(),
        t.ais
            .iter()
            .map(|(k, v)| (k.clone(), without_expiry(v)))
            .collect(),
        t.tickets,
        t.dedup_calls.clone(),
    )
}

fn assert_records_only(t: &Trace, forced: &str) {
    let want = if info(t).contains_key("action_specs") {
        "dry_run"
    } else {
        action_control::STATUS_NO_ACTIONS
    };
    assert_eq!(t.status, want, "{t:?}");
    assert!(t.outcomes.last().unwrap().dry_run);
    assert_eq!(
        info(t)["matched_rule_id"],
        action_control::PAGE_RULE_ID,
        "never the rules"
    );
    assert!(t.ais.is_empty(), "no AIS call");
    assert_eq!(t.tickets, 0, "no daily-cap ticket");
    assert!(
        t.dedup_calls.iter().all(|c| c.starts_with("peek")),
        "a peek, never a claim or a write: {:?}",
        t.dedup_calls
    );
    let i = info(t);
    assert_eq!(i["action_control_mode"], "recording");
    assert_eq!(i["dry_run_source"], "page");
    assert_eq!(i["action_control_forced_recording"], forced);
    assert!(!i.contains_key("confidence_acted_level"));
}

#[tokio::test]
async fn without_a_level_or_a_record_the_outcome_is_unchanged() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![]).await;
    let plain = b.run_fresh(&bb1(Account::Clean)).await;
    let o = plain.outcomes.last().unwrap();
    assert_eq!(o.confidence, 0);
    assert!(o.confidence_evidence.is_none());
    assert!(!o.info.keys().any(|k| k.starts_with("confidence")));
    let json = serde_json::to_value(o).unwrap();
    for key in [
        "confidence",
        "confidence_evidence",
        "raw_score",
        "score",
        "calibration_id",
    ] {
        assert!(json.get(key).is_none(), "{key} stays off the JSON: {json}");
    }

    let mut leveled = high(Account::Clean);
    leveled.raw_score = 4.2;
    leveled.score = 0.985;
    leveled.calibration_id = "platt@3".into();
    let t = b.run_fresh(&leveled).await;
    assert_eq!(effect(&t), effect(&plain));
    let o = t.outcomes.last().unwrap();
    assert_eq!(o.confidence, proto::Confidence::High as i32);
    assert_eq!((o.raw_score, o.score), (4.2, 0.985));
    assert_eq!(o.calibration_id, "platt@3");
    assert_eq!(
        o.confidence_evidence.as_ref().map(|e| e.reads),
        Some(59),
        "evidence passes through"
    );
    assert_eq!(o.info["confidence_level"], "high");
    assert_eq!(o.info["confidence_source"], "producer");
    assert_eq!(o.info["confidence_path"], "routing_label");
    assert_eq!(o.info["action_control_mode"], "rules");
}

#[tokio::test]
async fn the_default_table_acts_like_a_live_record_with_the_same_actions_for_every_account() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let by_level = bed(vec![defaults(BB1.0, ControlEntity::User)]).await;
    for (level, proto_level, actions) in [
        (
            ConfidenceLevel::High,
            proto::Confidence::High,
            vec![SUSPEND],
        ),
        (
            ConfidenceLevel::Mid,
            proto::Confidence::Mid,
            vec![LIVENESS, SHR7],
        ),
        (ConfidenceLevel::Low, proto::Confidence::Low, vec![LIVENESS]),
    ] {
        let by_record = bed(vec![record(BB1.0, None, None, ControlMode::Live, actions)]).await;
        for account in Account::ALL {
            let msg = with_level(bb1(account), proto_level, Some(evidence(level)));
            let want = by_record.run_fresh(&bb1(account)).await;
            let got = by_level.run_fresh(&msg).await;
            assert_eq!(effect(&got), effect(&want), "{level:?} {account:?}");
            let (wi, gi) = (info(&want), info(&got));
            assert_eq!(
                gi["matched_rule_id"], wi["matched_rule_id"],
                "{level:?} {account:?}"
            );
            assert_eq!(gi.get("action_specs"), wi.get("action_specs"));
            assert_eq!(gi["confidence_path"], "defaults");
            assert_eq!(gi["action_control_lkey"], format!("user/h/{}", BB1.0));
            if got.status == "success" {
                assert_eq!(gi["confidence_acted_level"], level.as_str());
                assert_eq!(gi["confidence_evidence_supports"], level.as_str());
            }
        }
    }
    for (account, reason) in [
        (Account::Followers5k, "high_follower_count"),
        (Account::Followers200k, "very_high_follower_count"),
        (Account::HighCred, "pagerank_skipped"),
        (Account::NotFound, "user_not_found"),
        (Account::Suspended, ALREADY_SUSPENDED),
    ] {
        let t = by_level.run_fresh(&high(account)).await;
        assert_eq!(t.status, reason, "{account:?}");
        assert!(t.ais.is_empty() && t.tickets == 0, "{account:?}");
    }
}

#[tokio::test]
async fn the_test_user_gate_off_switch_and_cap_still_stop_a_level() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![
        defaults(BB1.0, ControlEntity::User),
        defaults(SWITCHED_OFF_HEAD, ControlEntity::User),
    ])
    .await;

    let mut test_user = high(Account::Clean);
    test_user.user_id = (1 << 53) | (1 << 11);
    assert!(test_user::is_test_user_id(test_user.user_id));
    let t = b.run_fresh(&test_user).await;
    assert!(t.ais.is_empty() && t.tickets == 0, "{t:?}");
    assert_ne!(t.status, "success");

    let off = with_level(
        message(
            EntityType::User,
            SWITCHED_OFF_HEAD,
            "clf_v1",
            &[LABEL],
            Account::Clean,
        ),
        proto::Confidence::High,
        Some(evidence(ConfidenceLevel::High)),
    );
    let t = b.run_fresh(&off).await;
    assert_eq!(t.status, off_switch::STATUS_SWITCHED_OFF);
    assert!(t.ais.is_empty());

    let capped = bed_with(
        vec![defaults(BB1.0, ControlEntity::User)],
        on_config(false, Some(0)),
    )
    .await;
    let t = capped.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, "max_enforcement_reached");
    assert!(t.ais.is_empty());

    let staging = bed_with(
        vec![defaults(BB1.0, ControlEntity::User)],
        on_config(true, None),
    )
    .await;
    let t = staging.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, "dry_run");
    assert!(t.ais.is_empty());
    assert_eq!(info(&t)["dry_run_source"], "global");
}

#[tokio::test]
async fn an_overturn_hold_stops_a_level_driven_suspend() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let mut config = on_config(false, None);
    config["overturn_hold_gate"] = serde_json::json!({"mode": "enforce"});
    let mut b = bed_with(vec![defaults(BB1.0, ControlEntity::User)], config).await;
    b.ctx.hold_gate = Arc::new(HoldGate::new(Some(FakeHoldStore::new(vec![Hold {
        user_id: Account::Clean.user_id(),
        hold_id: 7,
        head: Some(BB1.0.into()),
        expires_at: chrono::Utc::now() + chrono::Duration::days(30),
        case_group_id: Some(63),
        reason: "appeal_overturned".into(),
        action_kind: HOLD_KIND_SUSPEND.into(),
        label: None,
    }]))));
    let t = b.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, overturn_hold::STATUS_HOLD_OVERTURNED);
    assert!(t.ais.is_empty() && t.tickets == 0);
}

#[tokio::test]
async fn a_post_level_never_waives_the_author_ceiling() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let head = "SpamEmbeddingPtosDistilled";
    for records in [
        vec![defaults(head, ControlEntity::Post)],
        vec![level_record(
            head,
            None,
            ConfidenceLevel::High,
            ControlMode::Live,
            vec![PageAction::SuspendAuthor {}],
        )],
    ] {
        let b = bed(records).await;
        for (account, want) in [
            (Account::Followers200k, "very_high_follower_count"),
            (Account::Clean, "success"),
        ] {
            let mut s = with_level(
                message(
                    EntityType::Post,
                    head,
                    "clf",
                    &["anchor_campaign_post_label"],
                    account,
                ),
                proto::Confidence::High,
                Some(evidence(ConfidenceLevel::High)),
            );
            s.skip_author_credibility_prechecks = true;
            let t = b.run_fresh(&s).await;
            assert_eq!(t.status, want, "{account:?}");
            if want != "success" {
                assert!(t.ais.is_empty() && t.tickets == 0);
            }
        }
    }
}

#[tokio::test]
async fn missing_old_or_weak_evidence_only_softens() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![defaults(BB1.0, ControlEntity::User)]).await;
    let names = |t: &Trace| -> Vec<String> {
        t.ais
            .iter()
            .flat_map(|(_, r)| {
                let actions: Vec<&serde_json::Value> = match r.get("actions") {
                    Some(serde_json::Value::Array(a)) => a.iter().collect(),
                    _ => r.get("action").into_iter().collect(),
                };
                actions
                    .into_iter()
                    .filter_map(|a| a.as_object().and_then(|o| o.keys().next().cloned()))
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let liveness_only = bed(vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Live,
        vec![LIVENESS],
    )])
    .await
    .run_fresh(&bb1(Account::Clean))
    .await;
    let mid_set = bed(vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Live,
        vec![LIVENESS, SHR7],
    )])
    .await
    .run_fresh(&bb1(Account::Clean))
    .await;

    let t = b
        .run_fresh(&with_level(
            bb1(Account::Clean),
            proto::Confidence::High,
            Some(evidence(ConfidenceLevel::Mid)),
        ))
        .await;
    assert_eq!(names(&t), names(&mid_set));
    assert_eq!(info(&t)["confidence_acted_level"], "mid");
    assert_eq!(info(&t)["confidence_level"], "high");

    let t = b
        .run_fresh(&with_level(
            bb1(Account::Clean),
            proto::Confidence::High,
            Some(evidence(ConfidenceLevel::Low)),
        ))
        .await;
    assert_eq!(names(&t), names(&liveness_only));
    assert_eq!(info(&t)["confidence_acted_level"], "low");

    let mut old = evidence(ConfidenceLevel::High);
    old.read_at_ms = now_ms() - 31 * DAY_MS;
    let mut weak = evidence(ConfidenceLevel::High);
    weak.precision_lower_bound = 0.5;
    let mut typed = evidence(ConfidenceLevel::High);
    typed.reads = 5;
    for (ev, forced) in [
        (None, "evidence_missing"),
        (Some(old), "evidence_below_floor"),
        (Some(weak), "evidence_below_floor"),
        (Some(typed), "evidence_below_floor"),
    ] {
        for level in [
            proto::Confidence::High,
            proto::Confidence::Mid,
            proto::Confidence::Low,
        ] {
            let t = b
                .run_fresh(&with_level(bb1(Account::Clean), level, ev.clone()))
                .await;
            assert_records_only(&t, forced);
        }
    }
}

#[tokio::test]
async fn an_override_names_a_person_and_expires() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let written = now_ms() - DAY_MS;
    let with_override = |expires_at_ms: i64| ActionControlRecord {
        updated_at_ms: written,
        evidence_override: Some(Box::new(EvidenceOverride {
            by: "operator@example.com".into(),
            reason: "seeded from today's rules while the read runs".into(),
            expires_at_ms,
        })),
        ..defaults(BB1.0, ControlEntity::User)
    };
    let no_read = with_level(bb1(Account::Clean), proto::Confidence::High, None);

    let b = bed(vec![with_override(now_ms() + DAY_MS)]).await;
    let t = b.run_fresh(&no_read).await;
    assert_eq!(t.status, "success");
    assert_eq!(info(&t)["confidence_override_by"], "operator@example.com");
    assert_eq!(info(&t)["confidence_acted_level"], "high");

    let b = bed(vec![with_override(now_ms() - 1)]).await;
    assert_records_only(&b.run_fresh(&no_read).await, "evidence_missing");

    let no_person = ActionControlRecord {
        evidence_override: Some(Box::new(EvidenceOverride {
            by: " ".into(),
            reason: "seeded".into(),
            expires_at_ms: now_ms() + DAY_MS,
        })),
        ..with_override(now_ms() + DAY_MS)
    };
    for rec in [with_override(written + 8 * DAY_MS), no_person] {
        let b = bed(vec![rec]).await;
        assert_records_only(&b.run_fresh(&no_read).await, "unreadable");
    }
}

#[tokio::test]
async fn unspecified_never_acts_and_unknown_values_read_as_low() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![defaults(BB1.0, ControlEntity::User)]).await;
    assert_records_only(
        &b.run_fresh(&bb1(Account::Clean)).await,
        "confidence_missing",
    );

    let mut future = bb1(Account::Clean);
    future.confidence = 9;
    future.confidence_evidence = Some(proto_evidence(&evidence(ConfidenceLevel::High)));
    let t = b.run_fresh(&future).await;
    assert_eq!(info(&t)["confidence_level"], "low");
    assert_eq!(info(&t)["confidence_acted_level"], "low");
    assert_eq!(t.outcomes[0].confidence, proto::Confidence::Low as i32);
}

#[tokio::test]
async fn level_records_win_over_defaults_most_specific_first() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![
        defaults(BB1.0, ControlEntity::User),
        level_record(
            BB1.0,
            None,
            ConfidenceLevel::High,
            ControlMode::Live,
            vec![LIVENESS],
        ),
        level_record(
            BB1.0,
            Some(BB1.1),
            ConfidenceLevel::High,
            ControlMode::Recording,
            vec![SUSPEND],
        ),
    ])
    .await;
    let t = b.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, "dry_run", "classifier+level wins");
    assert_eq!(
        info(&t)["action_control_lkey"],
        format!("user/cl/{}/{}/high", BB1.0, BB1.1)
    );
    assert_eq!(info(&t)["confidence_path"], "level_record");

    let mut other_classifier = high(Account::Clean);
    other_classifier.model_version = "other_v1".into();
    let t = b.run_fresh(&other_classifier).await;
    assert_eq!(t.status, "success");
    assert_eq!(
        info(&t)["action_control_lkey"],
        format!("user/l/{}/high", BB1.0)
    );

    let t = b
        .run_fresh(&with_level(
            bb1(Account::Clean),
            proto::Confidence::Mid,
            Some(evidence(ConfidenceLevel::Mid)),
        ))
        .await;
    assert_eq!(t.status, "success");
    assert_eq!(info(&t)["confidence_path"], "defaults");

    let b = bed(vec![level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Live,
        vec![LIVENESS],
    )])
    .await;
    let plain = bed(vec![]).await.run_fresh(&bb1(Account::Clean)).await;
    let t = b
        .run_fresh(&with_level(
            bb1(Account::Clean),
            proto::Confidence::Mid,
            Some(evidence(ConfidenceLevel::Mid)),
        ))
        .await;
    assert_eq!(effect(&t), effect(&plain));
    assert_eq!(info(&t)["action_control_mode"], "rules");
}

#[tokio::test]
async fn a_recording_level_takes_no_dedup_claim_and_no_ticket() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Recording,
        vec![SUSPEND],
    )])
    .await;
    let t = b.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, "dry_run");
    assert_eq!(t.dedup_calls, vec!["peek(class=3)=None"]);
    assert_eq!(t.tickets, 0);
    assert!(t.ais.is_empty());
    assert_eq!(
        info(&t)["action_specs"],
        r#"[{"kind":"suspend","permanent":false,"policy":"PlatformManipulation"}]"#
    );
}

#[tokio::test]
async fn every_failure_state_ends_in_recording_only() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let off = bed_with(
        vec![defaults(BB1.0, ControlEntity::User)],
        base_config(false, None),
    )
    .await;
    assert_records_only(&off.run_fresh(&high(Account::Clean)).await, "kill_switch");

    let (cse_head, cse_model) = HEADS[1];
    let cse = bed(vec![defaults(cse_head, ControlEntity::User)]).await;
    let s = with_level(
        message(
            EntityType::User,
            cse_head,
            cse_model,
            &[LABEL],
            Account::Clean,
        ),
        proto::Confidence::High,
        Some(evidence(ConfidenceLevel::High)),
    );
    assert_records_only(&cse.run_fresh(&s).await, "cse_lock");

    let stale = bed(vec![defaults(BB1.0, ControlEntity::User)]).await;
    stale
        .ctx
        .action_control
        .age_for_test(std::time::Duration::from_secs(3_600));
    assert_records_only(
        &stale.run_fresh(&high(Account::Clean)).await,
        "stale_snapshot",
    );

    let mut bad = level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Live,
        vec![SUSPEND],
    );
    bad.band = Some(LABEL.into());
    let broken = bed(vec![bad]).await;
    assert_records_only(&broken.run_fresh(&high(Account::Clean)).await, "unreadable");

    let mut config = on_config(false, None);
    config["action_control"]["confidence"] = serde_json::json!({"label_map": {LABEL: "very_high"}});
    let plain = bed(vec![]).await.run_fresh(&bb1(Account::Clean)).await;
    let ignored = bed_with(vec![], config).await;
    let t = ignored.run_fresh(&bb1(Account::Clean)).await;
    assert_eq!(effect(&t), effect(&plain));
    assert!(!info(&t).contains_key("confidence_level"));
}

#[tokio::test]
async fn the_label_map_ships_empty_and_translates_when_set() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let records = vec![level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Live,
        vec![SUSPEND],
    )];
    let shipped = bed(records.clone()).await;
    let t = shipped.run_fresh(&bb1(Account::Clean)).await;
    assert_eq!(info(&t)["action_control_mode"], "rules");
    assert!(!info(&t).contains_key("confidence_level"));

    let mut config = on_config(false, None);
    config["action_control"]["confidence"] =
        serde_json::json!({"label_map": {"*_suspend": "high"}});
    let mapped = bed_with(records, config).await;
    let t = mapped.run_fresh(&bb1(Account::Clean)).await;
    assert_eq!(
        t.status, "success",
        "the level record's own read certifies it"
    );
    assert_eq!(info(&t)["confidence_source"], "label_map");
    assert_eq!(t.outcomes[0].confidence, proto::Confidence::High as i32);
}

#[tokio::test]
async fn a_malformed_confidence_config_records_every_level_and_leaves_the_rules_alone() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let rules_today = bed(vec![]).await.run_fresh(&bb1(Account::Clean)).await;
    for bad in [
        serde_json::json!({"label_map": {LABEL: "very_high"}}),
        serde_json::json!({"evidence_max_age_days": "30"}),
        serde_json::json!({"evidence_max_age_days": -1}),
        serde_json::json!({"floors": {"high": 0.5, "mid": 0.5, "low": 0.5}}),
        serde_json::json!({"label_map": {}, "max_age": 30}),
        serde_json::json!("on"),
    ] {
        let mut config = on_config(false, None);
        config["action_control"]["confidence"] = bad.clone();
        for records in [
            vec![defaults(BB1.0, ControlEntity::User)],
            vec![level_record(
                BB1.0,
                None,
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            )],
        ] {
            let b = bed_with(records, config.clone()).await;
            for level in [
                proto::Confidence::High,
                proto::Confidence::Mid,
                proto::Confidence::Low,
            ] {
                let s = with_level(
                    bb1(Account::Clean),
                    level,
                    Some(evidence(ConfidenceLevel::High)),
                );
                let t = b.run_fresh(&s).await;
                if info(&t)["action_control_mode"] == "rules" {
                    assert_ne!(level, proto::Confidence::High, "{bad}");
                    continue;
                }
                assert_records_only(&t, "confidence_config_unreadable");
            }
        }
        let b = bed_with(vec![defaults("OtherHead", ControlEntity::User)], config).await;
        let t = b.run_fresh(&high(Account::Clean)).await;
        assert_eq!(effect(&t), effect(&rules_today), "{bad}");
        assert_eq!(info(&t)["action_control_mode"], "rules");
    }
}

#[tokio::test]
async fn future_dated_records_and_reads_are_not_trusted() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let now = now_ms();

    let mut read_tomorrow = level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Live,
        vec![SUSPEND],
    );
    read_tomorrow.evidence.as_mut().unwrap().read_at_ms = now + DAY_MS;
    let b = bed(vec![read_tomorrow]).await;
    assert_records_only(&b.run_fresh(&high(Account::Clean)).await, "unreadable");

    for mut rec in [
        defaults(BB1.0, ControlEntity::User),
        record(BB1.0, None, None, ControlMode::Live, vec![SUSPEND]),
    ] {
        rec.updated_at_ms = now + DAY_MS;
        let b = bed(vec![rec]).await;
        assert_records_only(&b.run_fresh(&high(Account::Clean)).await, "unreadable");
    }

    let b = bed(vec![defaults(BB1.0, ControlEntity::User)]).await;
    let mut tomorrow = evidence(ConfidenceLevel::High);
    tomorrow.read_at_ms = now + DAY_MS;
    let s = with_level(bb1(Account::Clean), proto::Confidence::High, Some(tomorrow));
    assert_records_only(&b.run_fresh(&s).await, "evidence_below_floor");

    let mut skewed = evidence(ConfidenceLevel::High);
    skewed.read_at_ms = now + 60_000;
    let mut rec = defaults(BB1.0, ControlEntity::User);
    rec.updated_at_ms = now + 60_000;
    let b = bed(vec![rec]).await;
    let s = with_level(bb1(Account::Clean), proto::Confidence::High, Some(skewed));
    assert_eq!(b.run_fresh(&s).await.status, "success");
}

#[tokio::test]
async fn v5_round5_recording_level_with_no_actions_has_no_side_effect() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let b = bed(vec![level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Recording,
        vec![],
    )])
    .await;
    let t = b.run_fresh(&high(Account::Clean)).await;
    assert_eq!(t.status, action_control::STATUS_NO_ACTIONS, "{t:?}");
    assert!(t.ais.is_empty(), "no AIS call: {t:?}");
    assert_eq!(t.tickets, 0, "no daily-cap ticket: {t:?}");
    assert!(
        t.dedup_calls.iter().all(|c| c.starts_with("peek")),
        "peek only, never a claim: {:?}",
        t.dedup_calls
    );
    assert_eq!(info(&t)["action_control_mode"], "recording");
}

#[tokio::test]
async fn a_level_with_no_actions_records_no_actions_never_the_rules() {
    let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
    let post_head = "SpamEmbeddingPtosDistilled";
    let post_low = with_level(
        message(
            EntityType::Post,
            post_head,
            "clf",
            &["anchor_campaign_post_label"],
            Account::Clean,
        ),
        proto::Confidence::Low,
        Some(evidence(ConfidenceLevel::Low)),
    );
    let mut keyed_by_band_and_level = level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Live,
        vec![SUSPEND],
    );
    keyed_by_band_and_level.band = Some(LABEL.into());
    let recording_no_actions = level_record(
        BB1.0,
        None,
        ConfidenceLevel::High,
        ControlMode::Recording,
        vec![],
    );
    for (records, msg, forced) in [
        (vec![recording_no_actions], high(Account::Clean), None),
        (
            vec![defaults(BB1.0, ControlEntity::User)],
            bb1(Account::Clean),
            Some("confidence_missing"),
        ),
        (
            vec![defaults(post_head, ControlEntity::Post)],
            post_low,
            Some("no_default_actions"),
        ),
        (
            vec![keyed_by_band_and_level],
            high(Account::Clean),
            Some("unreadable"),
        ),
    ] {
        let t = bed(records).await.run_fresh(&msg).await;
        let i = info(&t);
        assert_eq!(
            t.status,
            action_control::STATUS_NO_ACTIONS,
            "{forced:?} {t:?}"
        );
        assert!(t.outcomes.last().unwrap().dry_run);
        assert_eq!(i["matched_rule_id"], action_control::PAGE_RULE_ID);
        assert!(!i.contains_key("action_specs"), "{forced:?}");
        assert_eq!(i["action_control_mode"], "recording");
        assert_eq!(
            i.get(action_control::FORCED_RECORDING_KEY)
                .map(String::as_str),
            forced
        );
        assert!(t.ais.is_empty() && t.tickets == 0, "{forced:?}");
        assert!(
            t.dedup_calls.iter().all(|c| c.starts_with("peek")),
            "{forced:?} {:?}",
            t.dedup_calls
        );
    }

    let t = bed(vec![record(
        BB1.0,
        None,
        None,
        ControlMode::Recording,
        vec![],
    )])
    .await
    .run_fresh(&bb1(Account::Clean))
    .await;
    assert_eq!(t.status, "dry_run");
    assert_ne!(info(&t)["matched_rule_id"], action_control::PAGE_RULE_ID);
}
