use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};
use xai_manhattan::{LkeySelector, NativeManhattanClient, Tenant};

pub use xai_abuse_enforcement_types::action_control::{
    ACTION_CONTROL_RECORD_VERSION, ActionControlHistoryEntry, ActionControlHistoryResponse,
    ActionControlListResponse, ActionControlPutRequest, ActionControlRecord, ActionControlSeen,
    ActionControlView, AllowedActions, ConfidenceEvidence, ConfidenceLevel, ConfidenceLevelView,
    ConfidenceView, ControlEntity, ControlMode, CseLists, EvidenceOverride, Lifetime, PageAction,
    PodActionControlView, SafetyChecksByEntity, UnreadableRecord,
};

mod levels;
pub use levels::level_lkey;

use crate::confidence::{ConfidenceConfig, LevelOutcome};
use crate::decision::ActionSpec;
use crate::facts::EntityType;
use crate::growthbook::DynamicConfig;
use crate::metrics;
use crate::off_switch::valid_name;

pub const MH_PKEY: &str = "action_control";
pub const SEEN_PKEY: &str = "action_control_seen";
pub const HISTORY_PKEY_PREFIX: &str = "action_control_history:";
pub const CONFIG_KEY: &str = "action_control";
pub const DRY_RUN_LABEL_PREFIX: &str = "dry_run__";
pub const FACTORY_LABEL: &str = "abuse_platform_requested_actions";

pub const REFRESH_INTERVAL: Duration = Duration::from_secs(5);
pub const STALE_AFTER: Duration = Duration::from_secs(60);
const SCAN_TIMEOUT: Duration = Duration::from_secs(5);
const SCAN_LIMIT: u32 = 10_000;
const FIRST_LOAD_BACKOFF: Duration = Duration::from_secs(1);
const FIRST_LOAD_LOG_EVERY: u64 = 12;
const ENSURE_LOADED_RETRY: Duration = Duration::from_secs(1);
const REFRESH_ERROR_LOG_EVERY: u64 = 60;
pub const RECORD_TTL_SECS: u64 = 10 * 365 * 86_400;
pub const HISTORY_TTL_SECS: u64 = 10 * 365 * 86_400;
pub const SEEN_TTL_SECS: u64 = 30 * 86_400;
pub const SEEN_WRITE_EVERY: Duration = Duration::from_secs(600);
const SEEN_QUEUE: usize = 256;
const SEEN_GUARD_MAX: usize = 10_000;
const LOCAL_WRITE_GRACE: Duration = Duration::from_secs(30);
const MAX_REASON_LEN: usize = 1_000;
const MAX_NAME_LEN: usize = 256;
pub const HISTORY_LIMIT_DEFAULT: usize = 50;
pub const HISTORY_LIMIT_MAX: usize = 500;
pub const CONFIRM_WORD: &str = "ENFORCE";
pub const PAGE_RULE_ID: &str = "page";
pub const STATUS_NO_ACTIONS: &str = "no_actions";
pub const FORCED_RECORDING_KEY: &str = "action_control_forced_recording";

pub const CSE_HEADS: &[&str] = &[
    "CseReportsEmbeddingV2UserAggGe4",
    "CseRtScoreHiV2DryRun",
    "NearDupEmbeddingCseMatch",
    "NearDupEmbeddingCseNcmecMatch",
];
pub const CSE_LABELS: &[&str] = &[
    "anchor_campaign_suspend_cse",
    "cse_reports_embedding_v2",
    "cse_rt_dryrun_cse",
];

pub fn lkey_for(
    entity: ControlEntity,
    head: &str,
    classifier: Option<&str>,
    band: Option<&str>,
) -> String {
    let e = entity.as_str();
    match (classifier, band) {
        (None, None) => format!("{e}/h/{head}"),
        (Some(c), None) => format!("{e}/c/{head}/{c}"),
        (None, Some(b)) => format!("{e}/b/{head}/{b}"),
        (Some(c), Some(b)) => format!("{e}/cb/{head}/{c}/{b}"),
    }
}

pub fn record_lkey(rec: &ActionControlRecord) -> String {
    match rec.confidence {
        Some(level) => level_lkey(rec.entity_type, &rec.head, rec.classifier.as_deref(), level),
        None => lkey_for(
            rec.entity_type,
            &rec.head,
            rec.classifier.as_deref(),
            rec.band.as_deref(),
        ),
    }
}

pub fn seen_lkey(seen: &ActionControlSeen) -> String {
    format!(
        "{}/{}/{}/{}",
        seen.entity_type.as_str(),
        seen.head,
        seen.classifier,
        seen.band
    )
}

pub fn strip_dry_run_prefix(label: &str) -> &str {
    label.strip_prefix(DRY_RUN_LABEL_PREFIX).unwrap_or(label)
}

pub fn has_dry_run_prefix(labels: &[String]) -> bool {
    labels.iter().any(|l| l.starts_with(DRY_RUN_LABEL_PREFIX))
}

pub fn entity_of(e: EntityType) -> ControlEntity {
    match e {
        EntityType::User => ControlEntity::User,
        EntityType::Post => ControlEntity::Post,
    }
}


pub fn page_specs(actions: &[PageAction]) -> Vec<ActionSpec> {
    let rank = |a: &PageAction| match a {
        PageAction::SpamHighRecall { .. } | PageAction::PostSpamHighRecall { .. } => 0,
        PageAction::LivenessCheck {} => 1,
        PageAction::Suspend {} | PageAction::SuspendAuthor {} => 2,
    };
    let mut sorted: Vec<PageAction> = actions.to_vec();
    sorted.sort_by_key(rank);
    sorted
        .into_iter()
        .map(|a| match a {
            PageAction::SpamHighRecall { lifetime } => ActionSpec::AddLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: lifetime.ttl_msec(),
            },
            PageAction::PostSpamHighRecall { lifetime } => ActionSpec::AddPostLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: lifetime.ttl_msec(),
            },
            PageAction::LivenessCheck {} => ActionSpec::SpamLivenessCheck,
            PageAction::Suspend {} | PageAction::SuspendAuthor {} => ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into(),
            },
        })
        .collect()
}

pub fn is_label_only(actions: &[PageAction]) -> bool {
    matches!(actions, [PageAction::SpamHighRecall { .. }])
}

fn ttl_days(ttl_msec: i64) -> Value {
    if ttl_msec % 86_400_000 == 0 {
        Value::from(ttl_msec / 86_400_000)
    } else {
        Value::from(ttl_msec as f64 / 86_400_000.0)
    }
}

pub fn action_specs_value(specs: &[ActionSpec]) -> Value {
    let mut out = Vec::new();
    for spec in specs {
        match spec {
            ActionSpec::SuspendUser { perm, policy } => out.push(serde_json::json!({
                "kind": "suspend", "permanent": perm, "policy": policy,
            })),
            ActionSpec::AddLabelsV2 { labels, ttl_msec }
            | ActionSpec::AddPostLabelsV2 { labels, ttl_msec } => {
                let kind = if matches!(spec, ActionSpec::AddLabelsV2 { .. }) {
                    "label"
                } else {
                    "post_label"
                };
                for label in labels {
                    let mut o = serde_json::json!({"kind": kind, "label": label});
                    match ttl_msec {
                        Some(t) => {
                            o["ttl_days"] = ttl_days(*t);
                            o["permanent"] = Value::Bool(false);
                        }
                        None => o["permanent"] = Value::Bool(true),
                    }
                    out.push(o);
                }
            }
            ActionSpec::SpamLivenessCheck => {
                out.push(serde_json::json!({"kind": "liveness_check"}))
            }
            ActionSpec::Arkose => out.push(serde_json::json!({"kind": "arkose"})),
            ActionSpec::Captcha => out.push(serde_json::json!({"kind": "captcha"})),
        }
    }
    Value::Array(out)
}

pub fn action_specs_json(specs: &[ActionSpec]) -> String {
    action_specs_value(specs).to_string()
}

pub fn has_cse_policy(specs: &[ActionSpec]) -> bool {
    specs
        .iter()
        .any(|s| matches!(s, ActionSpec::SuspendUser { policy, .. } if policy == "Cse"))
}

fn page_class(actions: &[PageAction]) -> u8 {
    page_specs(actions)
        .iter()
        .map(ActionSpec::dedup_action_class)
        .max()
        .unwrap_or(crate::dedup_cache::CLASS_NONE)
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlConfig {
                pub enabled: bool,
        pub ignore_store: bool,
    pub cse_heads: HashSet<String>,
    pub cse_labels: HashSet<String>,
                pub aliases: HashMap<String, String>,
                pub updated_by_keys: Option<HashSet<String>>,
        pub confidence: ConfidenceConfig,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ignore_store: false,
            cse_heads: CSE_HEADS.iter().map(|s| (*s).to_owned()).collect(),
            cse_labels: CSE_LABELS.iter().map(|s| (*s).to_owned()).collect(),
            aliases: HashMap::new(),
            updated_by_keys: None,
            confidence: ConfidenceConfig::default(),
        }
    }
}

impl ControlConfig {
                    pub fn from_config(config: Option<&Value>) -> Self {
        static LAST_BAD: Mutex<Option<String>> = Mutex::new(None);
        let mut out = Self::default();
        let Some(v) = config
            .and_then(|c| c.get(CONFIG_KEY))
            .filter(|v| !v.is_null())
        else {
            return out;
        };
        out.confidence = ConfidenceConfig::from_value(v.get(crate::confidence::CONFIG_KEY));
        let mut bad = Vec::new();
        match v.get("enabled") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(b)) => out.enabled = *b,
            Some(other) => {
                out.enabled = false;
                bad.push(format!("enabled={other}"));
            }
        }
        match v.get("ignore_store") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(b)) => out.ignore_store = *b,
            Some(other) => bad.push(format!("ignore_store={other}")),
        }
        for (key, set) in [
            ("cse_heads", &mut out.cse_heads),
            ("cse_labels", &mut out.cse_labels),
        ] {
            if let Some(arr) = v.get(key).and_then(Value::as_array) {
                set.extend(
                    arr.iter()
                        .filter_map(Value::as_str)
                        .filter(|s| !s.trim().is_empty())
                        .map(str::to_owned),
                );
            }
        }
        if let Some(map) = v.get("aliases").and_then(Value::as_object) {
            out.aliases.extend(map.iter().filter_map(|(new, old)| {
                let old = old.as_str()?;
                (valid_name(new) && valid_name(old)).then(|| (new.clone(), old.to_owned()))
            }));
        }
        match v.get("updated_by_keys") {
            None | Some(Value::Null) => {}
            Some(keys) => {
                let parsed: Option<HashSet<String>> = keys.as_array().and_then(|arr| {
                    arr.iter()
                        .map(|k| {
                            k.as_str()
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .map(str::to_owned)
                        })
                        .collect()
                });
                if parsed.is_none() {
                    bad.push(format!("updated_by_keys={keys}"));
                }
                out.updated_by_keys = Some(parsed.unwrap_or_default());
            }
        }
        if !v.is_object() {
            out.enabled = false;
            bad.push(v.to_string());
        }
        if !bad.is_empty() {
            let bad = bad.join(", ");
            let mut last = LAST_BAD.lock().unwrap_or_else(|p| p.into_inner());
            if last.as_deref() != Some(bad.as_str()) {
                error!(
                    value = %bad,
                    "GrowthBook {CONFIG_KEY} is malformed; page-live scopes run as recording only until it is fixed"
                );
                *last = Some(bad);
            }
        }
        out
    }

    pub fn may_name_updater(&self, key_id: &str) -> bool {
        self.updated_by_keys
            .as_ref()
            .is_none_or(|keys| keys.contains(key_id))
    }

    pub fn is_cse(&self, head: &str, bands: &[&str]) -> bool {
        self.cse_heads.contains(head) || bands.iter().any(|b| self.cse_labels.contains(*b))
    }

            pub fn scope_is_cse(&self, head: &str, bands: &[&str]) -> bool {
        self.is_cse(head, bands)
            || self
                .aliases
                .get(head)
                .is_some_and(|old| self.is_cse(old, bands))
            || self
                .aliases
                .iter()
                .any(|(new, old)| old == head && self.is_cse(new, bands))
    }

                pub fn seen_cse(&self, head: &str, bands: &[&str], rules_chose_cse: bool) -> Vec<bool> {
        let unexplained = rules_chose_cse && !bands.iter().any(|b| self.cse_labels.contains(*b));
        bands
            .iter()
            .map(|b| self.is_cse(head, &[b]) || unexplained)
            .collect()
    }
}


#[async_trait]
pub trait ControlStore: Send + Sync {
    async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>>;
    async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>>;
    async fn put(
        &self,
        pkey: &str,
        lkey: &str,
        value: Vec<u8>,
        ttl_secs: u64,
    ) -> anyhow::Result<()>;
}

pub struct ManhattanControlStore {
    client: Arc<NativeManhattanClient>,
    tenant: Tenant,
}

impl ManhattanControlStore {
    pub fn new(client: Arc<NativeManhattanClient>, tenant: Tenant) -> Self {
        Self { client, tenant }
    }
}

#[async_trait]
impl ControlStore for ManhattanControlStore {
    async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
        let items = self
            .client
            .scan(
                self.tenant.clone(),
                [pkey],
                LkeySelector::Range {
                    from: None,
                    to: None,
                },
                SCAN_LIMIT,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if items.len() >= SCAN_LIMIT as usize {
            warn!(
                pkey,
                limit = SCAN_LIMIT,
                "action-control scan hit its row limit; rows past it are not read"
            );
        }
        Ok(items
            .iter()
            .filter_map(|it| {
                let lkey =
                    String::from_utf8(it.lkey().into_byte_vecs().into_iter().next()?).ok()?;
                Some((lkey, it.value().as_bytes().to_vec()))
            })
            .collect())
    }

    async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self
            .client
            .get(self.tenant.clone(), [pkey], [lkey])
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .map(|it| it.value().into_bytes()))
    }

    async fn put(
        &self,
        pkey: &str,
        lkey: &str,
        value: Vec<u8>,
        ttl_secs: u64,
    ) -> anyhow::Result<()> {
        self.client
            .put_with_ttl(
                self.tenant.clone(),
                [pkey],
                [lkey],
                value,
                crate::manhattan::ttl_nanos_from_secs(ttl_secs),
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}


#[derive(Default)]
struct Snapshot {
        records: HashMap<String, ActionControlRecord>,
            unreadable: HashMap<String, (String, u64)>,
    loaded_at: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matched {
    pub lkey: String,
    pub record: ActionControlRecord,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub matched: Option<Matched>,
        pub mode: ControlMode,
        pub page_actions: Vec<PageAction>,
        pub forced: Option<&'static str>,
            pub dedup_class: Option<u8>,
        pub producer_dry: bool,
            pub confidence: Option<LevelOutcome>,
            pub by_level: bool,
}

impl Default for Resolved {
    fn default() -> Self {
        Self {
            matched: None,
            mode: ControlMode::Rules,
            page_actions: Vec::new(),
            forced: None,
            dedup_class: None,
            producer_dry: false,
            confidence: None,
            by_level: false,
        }
    }
}

impl Resolved {
        pub fn page_driven(&self) -> bool {
        self.mode != ControlMode::Rules && (self.by_level || !self.page_actions.is_empty())
    }

    pub fn recording(&self) -> bool {
        self.mode == ControlMode::Recording
    }

                pub fn record_only(&self) -> bool {
        self.recording() || self.producer_dry
    }

        pub fn dry_run_source(&self, global: bool) -> Option<&'static str> {
        if global {
            Some("global")
        } else if self.recording() {
            Some("page")
        } else if self.producer_dry {
            Some("producer")
        } else {
            None
        }
    }

            pub fn incoming_dedup_class(&self, rules: u8) -> u8 {
        self.dedup_class.unwrap_or(rules)
    }

        pub fn lock_cse(&mut self) {
        self.lock("cse_lock");
    }

            pub fn lock_rules_unknown(&mut self) {
        self.lock("rules_eval_error");
    }

    fn lock(&mut self, reason: &'static str) {
        if self.mode == ControlMode::Live {
            self.mode = ControlMode::Recording;
            self.forced = Some(reason);
        }
    }

            pub fn info_keys(&self) -> Vec<(String, String)> {
        let mut keys = vec![(
            "action_control_mode".to_owned(),
            self.mode.as_str().to_owned(),
        )];
        if let Some(m) = &self.matched {
            keys.extend([
                (
                    "action_control_scope".to_owned(),
                    m.record.scope().to_owned(),
                ),
                (
                    "action_control_updated_by".to_owned(),
                    m.record.updated_by.clone(),
                ),
                (
                    "action_control_updated_at_ms".to_owned(),
                    m.record.updated_at_ms.to_string(),
                ),
                (
                    "action_control_version".to_owned(),
                    m.record.version.to_string(),
                ),
                ("action_control_lkey".to_owned(), m.lkey.clone()),
            ]);
        }
        if let Some(f) = self.forced {
            keys.push((FORCED_RECORDING_KEY.to_owned(), f.to_owned()));
        }
        if let Some(c) = &self.confidence {
            let mut c = c.clone();
            if self.mode != ControlMode::Live {
                c.acted = None;
            }
            keys.extend(c.info_keys());
        }
        keys
    }

            pub fn stamp_confidence(&self, outcome: &mut xai_abuse_proto::enforcement::DecisionOutcome) {
        if let Some(c) = &self.confidence {
            outcome.confidence = crate::confidence::level_to_proto(c.claimed) as i32;
        }
    }
}

pub struct ResolveInput<'a> {
    pub head: &'a str,
    pub classifier: &'a str,
    pub labels: &'a [String],
    pub entity: EntityType,
    pub factory: bool,
    pub producer_dry: bool,
        pub confidence: Option<crate::confidence::Claim>,
}

enum Hit<'a> {
    Record(&'a str, &'a ActionControlRecord),
    Unreadable,
    Ambiguous,
}

impl Snapshot {
    fn at<'a>(&'a self, lkey: &str) -> Option<Hit<'a>> {
        if let Some((k, r)) = self.records.get_key_value(lkey) {
            return Some(Hit::Record(k, r));
        }
        self.unreadable.get(lkey).map(|_| Hit::Unreadable)
    }

                    fn levels(
        entity: ControlEntity,
        head: &str,
        alias: Option<&str>,
        classifier: &str,
        bands: &[&str],
    ) -> Vec<Vec<Vec<String>>> {
        let names: Vec<&str> = std::iter::once(head).chain(alias).collect();
        let slot = |c: Option<&str>, b: Option<&str>| -> Vec<String> {
            names.iter().map(|n| lkey_for(entity, n, c, b)).collect()
        };
        let c = (!classifier.is_empty() && valid_name(classifier)).then_some(classifier);
        let mut levels = Vec::with_capacity(4);
        if let Some(c) = c {
            levels.push(bands.iter().map(|b| slot(Some(c), Some(b))).collect());
            levels.push(vec![slot(Some(c), None)]);
        }
        levels.push(bands.iter().map(|b| slot(None, Some(b))).collect());
        levels.push(vec![slot(None, None)]);
        levels
    }

    fn find(&self, levels: &[Vec<Vec<String>>]) -> Option<Hit<'_>> {
        for level in levels {
            let hits: Vec<Hit<'_>> = level
                .iter()
                .filter_map(|slot| slot.iter().find_map(|k| self.at(k)))
                .collect();
            match hits.len() {
                0 => continue,
                1 => return hits.into_iter().next(),
                _ => return Some(Hit::Ambiguous),
            }
        }
        None
    }

                    fn plain_decider(
        &self,
        entity: ControlEntity,
        head: &str,
        alias: Option<&str>,
        classifier: &str,
        bands: &[&str],
    ) -> Option<Hit<'_>> {
        self.find(&Self::levels(entity, head, alias, classifier, bands))
    }

            fn of_rows(rows: &[(String, Vec<u8>)], except: &str) -> Snapshot {
        let mut snap = Snapshot::default();
        for (lkey, value) in rows.iter().filter(|(k, _)| k != except) {
            match parse_record(value) {
                Ok(rec) => {
                    snap.records.insert(record_lkey(&rec), rec);
                }
                Err(err) => {
                    snap.unreadable
                        .insert(lkey.clone(), (err, raw_version(value)));
                }
            }
        }
        snap.records.remove(except);
        snap
    }
}

#[allow(clippy::too_many_arguments)]
fn effective(
    record: &ActionControlRecord,
    entity: EntityType,
    factory: bool,
    cse: bool,
    cfg: &ControlConfig,
    stale: bool,
    safety_ok: bool,
) -> (ControlMode, Vec<PageAction>, Option<&'static str>) {
    let entity_ok = record.entity_type == entity_of(entity)
        && record
            .actions
            .iter()
            .all(|a| a.entity() == record.entity_type);
    match record.mode {
        ControlMode::Rules => (ControlMode::Rules, Vec::new(), None),
        ControlMode::ConfidenceDefaults => (
            ControlMode::Recording,
            Vec::new(),
            Some("confidence_missing"),
        ),
        ControlMode::Recording if record.actions.is_empty() => {
            (ControlMode::Recording, Vec::new(), None)
        }
        ControlMode::Recording if factory => {
            (ControlMode::Recording, Vec::new(), Some("factory_managed"))
        }
        ControlMode::Recording if !entity_ok => {
            (ControlMode::Recording, Vec::new(), Some("wrong_entity"))
        }
        ControlMode::Recording => (ControlMode::Recording, record.actions.clone(), None),
        ControlMode::Live if factory => {
            (ControlMode::Recording, Vec::new(), Some("factory_managed"))
        }
        ControlMode::Live if record.actions.is_empty() || !entity_ok => {
            (ControlMode::Recording, Vec::new(), Some("wrong_entity"))
        }
        ControlMode::Live => {
            let forced = if cse {
                Some("cse_lock")
            } else if !cfg.enabled {
                Some("kill_switch")
            } else if stale {
                Some("stale_snapshot")
            } else if !safety_ok {
                Some("safety_checks_changed")
            } else {
                None
            };
            let mode = if forced.is_some() {
                ControlMode::Recording
            } else {
                ControlMode::Live
            };
            (mode, record.actions.clone(), forced)
        }
    }
}


#[derive(Debug, PartialEq)]
pub enum SetError {
        Invalid(&'static str, String),
            Conflict(Option<ActionControlRecord>, Option<u64>),
        LostRace(Option<ActionControlRecord>),
        Ignored,
        Store(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Validated {
        pub cse: bool,
}

#[derive(Debug)]
pub struct SetOutcome {
    pub view: ActionControlView,
    pub before: Option<ActionControlRecord>,
    pub harsher_because: Vec<String>,
    pub history_written: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeenOutcome {
    Queued,
    Recent,
    Dropped,
    Invalid,
}

pub struct ActionControl {
    store: Arc<dyn ControlStore>,
    snapshot: RwLock<Arc<Snapshot>>,
    local_writes: Mutex<HashMap<String, (ActionControlRecord, Instant)>>,
    created_at: Instant,
    scan_timeout: Duration,
    pub env: String,
    pub pod: String,
    seen_guard: Mutex<HashMap<String, Instant>>,
    seen_tx: mpsc::Sender<ActionControlSeen>,
    seen_rx: Mutex<Option<mpsc::Receiver<ActionControlSeen>>>,
        #[cfg(test)]
    age_offset: Mutex<Duration>,
}

fn new_nonce() -> u64 {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    nanos
        ^ ((std::process::id() as u64) << 32)
        ^ SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn label_days(a: &PageAction) -> Option<i64> {
    match a {
        PageAction::SpamHighRecall { lifetime } | PageAction::PostSpamHighRecall { lifetime } => {
            Some(lifetime.ttl_msec().map_or(i64::MAX, |t| t / 86_400_000))
        }
        _ => None,
    }
}

pub fn harsher_because(
    before: Option<&ActionControlRecord>,
    after: &ActionControlRecord,
    rules_specs: Option<&[Value]>,
) -> Vec<String> {
    let before_mode = before.map_or(ControlMode::Rules, |b| b.mode);
    let before_actions: &[PageAction] = before.map_or(&[], |b| &b.actions);
    let mut why = Vec::new();
    match (before_mode, after.mode) {
        (ControlMode::Live, ControlMode::Live) => {}
        (_, ControlMode::Live) => why.push("turns the scope live".to_owned()),
        (ControlMode::Recording, ControlMode::Rules) => {
            why.push("hands the scope back to the GrowthBook rules, which act".to_owned())
        }
        (ControlMode::Live, ControlMode::Rules) => {
            let page = action_specs_value(&page_specs(before_actions));
            let page = page.as_array().map(Vec::as_slice).unwrap_or_default();
            match rules_specs {
                Some(rules) if !rules.is_empty() && rules.iter().all(|r| covered_by(r, page)) => {}
                _ => why.push(
                    "hands the scope back to the GrowthBook rules, whose actions may be harsher"
                        .to_owned(),
                ),
            }
        }
        _ => {}
    }
    if after.mode == ControlMode::Live && before_mode == ControlMode::Live {
        for a in &after.actions {
            let same_kind = before_actions.iter().find(|b| b.kind() == a.kind());
            match (a, same_kind) {
                (PageAction::Suspend {} | PageAction::SuspendAuthor {}, None) => {
                    why.push("adds suspend".into())
                }
                (PageAction::LivenessCheck {}, None) => why.push("adds liveness check".into()),
                (_, None) => why.push("adds a label".into()),
                (_, Some(b)) => {
                    if let (Some(new), Some(old)) = (label_days(a), label_days(b))
                        && new > old
                    {
                        why.push("makes the label last longer".into());
                    }
                }
            }
        }
    }
    why
}

fn covered_by(rule: &Value, page: &[Value]) -> bool {
    let field = |o: &Value, k: &str| o.get(k).cloned().unwrap_or(Value::Null);
    let permanent = |o: &Value| o.get("permanent").and_then(Value::as_bool) == Some(true);
    let days = |o: &Value| {
        if permanent(o) {
            f64::INFINITY
        } else {
            o.get("ttl_days")
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY)
        }
    };
    page.iter().any(|p| {
        if field(rule, "kind") != field(p, "kind") {
            return false;
        }
        match rule.get("kind").and_then(Value::as_str) {
            Some("suspend") => {
                field(rule, "policy") == field(p, "policy") && (!permanent(rule) || permanent(p))
            }
            Some("label" | "post_label") => {
                field(rule, "label") == field(p, "label") && days(rule) <= days(p)
            }
            Some("liveness_check") => true,
            _ => false,
        }
    })
}

impl ActionControl {
    pub fn new(store: Arc<dyn ControlStore>, env: String, pod: String) -> Self {
        let (seen_tx, seen_rx) = mpsc::channel(SEEN_QUEUE);
        Self {
            store,
            snapshot: RwLock::new(Arc::new(Snapshot::default())),
            local_writes: Mutex::new(HashMap::new()),
            created_at: Instant::now(),
            scan_timeout: SCAN_TIMEOUT,
            env,
            pod,
            seen_guard: Mutex::new(HashMap::new()),
            seen_tx,
            seen_rx: Mutex::new(Some(seen_rx)),
            #[cfg(test)]
            age_offset: Mutex::new(Duration::ZERO),
        }
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn age_for_test(&self, by: Duration) {
        *self.age_offset.lock().unwrap() = by;
    }

    fn snapshot_age(&self, snap: &Snapshot) -> Option<Duration> {
        let age = snap.loaded_at?.elapsed();
        #[cfg(test)]
        let age = age + *self.age_offset.lock().unwrap();
        Some(age)
    }

        pub fn has_records(&self) -> bool {
        let snap = self.snapshot();
        !snap.records.is_empty() || !snap.unreadable.is_empty()
    }

    pub fn is_loaded(&self) -> bool {
        self.snapshot().loaded_at.is_some()
    }

            pub fn ready(&self, cfg: &ControlConfig) -> bool {
        cfg.ignore_store || self.is_loaded()
    }

        pub fn snapshot_age_ms(&self) -> i64 {
        self.snapshot_age(&self.snapshot())
            .map_or(-1, |a| a.as_millis() as i64)
    }

        pub fn is_stale(&self) -> bool {
        self.snapshot_age(&self.snapshot())
            .is_none_or(|a| a > STALE_AFTER)
    }

            pub fn resolve(
        &self,
        input: &ResolveInput<'_>,
        cfg: &ControlConfig,
        safety_ok: impl FnOnce(EntityType) -> bool,
    ) -> Resolved {
        let mut out = Resolved {
            producer_dry: input.producer_dry,
            confidence: input.confidence.as_ref().map(LevelOutcome::routing_label),
            ..Default::default()
        };
        if cfg.ignore_store || input.head.is_empty() {
            return out;
        }
        let snap = self.snapshot();
        if snap.records.is_empty() && snap.unreadable.is_empty() {
            return out;
        }
        let mut safety_ok = Some(safety_ok);
        let mut safety_seen: Option<bool> = None;
        let mut safety = |e: EntityType| {
            *safety_seen.get_or_insert_with(|| safety_ok.take().is_none_or(|f| f(e)))
        };
        let mut bands: Vec<&str> = input
            .labels
            .iter()
            .map(|l| strip_dry_run_prefix(l))
            .filter(|l| valid_name(l))
            .collect();
        bands.sort_unstable();
        bands.dedup();
        let entity = entity_of(input.entity);
        let stale = self.snapshot_age(&snap).is_none_or(|a| a > STALE_AFTER);
        let cse = cfg.scope_is_cse(input.head, &bands);
        let by_level = input.confidence.as_ref().and_then(|claim| {
            levels::resolve_level(
                &snap,
                &levels::LevelContext {
                    entity: input.entity,
                    head: input.head,
                    alias: cfg.aliases.get(input.head).map(String::as_str),
                    classifier: input.classifier,
                    factory: input.factory,
                    cse,
                    stale,
                },
                claim,
                cfg,
                &mut safety,
            )
        });
        let hit = match by_level {
            Some(r) => {
                out = Resolved {
                    producer_dry: input.producer_dry,
                    by_level: true,
                    ..r
                };
                None
            }
            None => snap.plain_decider(
                entity,
                input.head,
                cfg.aliases.get(input.head).map(String::as_str),
                input.classifier,
                &bands,
            ),
        };
        match hit {
            None => {}
            Some(Hit::Ambiguous) => {
                out.mode = ControlMode::Recording;
                out.forced = Some("band_ambiguous");
            }
            Some(Hit::Unreadable) => {
                out.mode = ControlMode::Recording;
                out.forced = Some("unreadable");
            }
            Some(Hit::Record(lkey, record)) => {
                let (mode, actions, forced) = effective(
                    record,
                    input.entity,
                    input.factory,
                    cse,
                    cfg,
                    stale,
                    record.mode != ControlMode::Live || safety(input.entity),
                );
                out.mode = mode;
                out.page_actions = actions;
                out.forced = forced;
                out.by_level = record.mode == ControlMode::ConfidenceDefaults;
                out.matched = Some(Matched {
                    lkey: lkey.to_owned(),
                    record: record.clone(),
                });
            }
        }
        out.dedup_class = out.page_driven().then(|| page_class(&out.page_actions));
        if let Some(m) = &out.matched {
            metrics::ACTION_CONTROL_DECISIONS_TOTAL
                .with_label_values(&[
                    m.record.head.as_str(),
                    out.mode.as_str(),
                    out.forced.unwrap_or(""),
                ])
                .inc();
        }
        out
    }


            pub async fn refresh(&self) -> anyhow::Result<usize> {
        let scanned = match tokio::time::timeout(self.scan_timeout, self.store.scan(MH_PKEY)).await
        {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!(
                "action-control scan timed out after {} ms",
                self.scan_timeout.as_millis()
            )),
        };
        let scanned = scanned.and_then(|rows| {
            if rows.len() >= SCAN_LIMIT as usize {
                Err(anyhow::anyhow!(
                    "action-control scan returned {} rows, its limit; the list may be incomplete",
                    rows.len()
                ))
            } else {
                Ok(rows)
            }
        });
        let rows = match scanned {
            Ok(rows) => rows,
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                metrics::ACTION_CONTROL_REFRESH_TOTAL
                    .with_label_values(&["error"])
                    .inc();
                self.publish_metrics();
                return Err(e);
            }
        };
        let n = rows.len();
        let prev = self.snapshot();
        let mut records = HashMap::with_capacity(n);
        let mut unreadable = HashMap::new();
        for (lkey, value) in rows {
            match parse_record(&value) {
                Ok(rec) => {
                    records.insert(record_lkey(&rec), rec);
                }
                Err(err) => {
                    metrics::ACTION_CONTROL_UNREADABLE_TOTAL.inc();
                    warn!(lkey = %lkey, error = %err, "action-control record does not parse; keeping the previous copy if any");
                    match prev.records.get(&lkey) {
                        Some(old) => {
                            records.insert(lkey, old.clone());
                        }
                        None => {
                            unreadable.insert(lkey, (err, raw_version(&value)));
                        }
                    }
                }
            }
        }
        {
            let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
            local.retain(|_, (_, at)| at.elapsed() < LOCAL_WRITE_GRACE);
            for (key, (rec, _)) in local.iter() {
                if records.get(key).is_none_or(|s| s.version < rec.version) {
                    unreadable.remove(key);
                    records.insert(key.clone(), rec.clone());
                }
            }
            self.publish(records, unreadable, true);
        }
        metrics::ACTION_CONTROL_REFRESH_TOTAL
            .with_label_values(&["ok"])
            .inc();
        Ok(n)
    }

    fn publish(
        &self,
        records: HashMap<String, ActionControlRecord>,
        unreadable: HashMap<String, (String, u64)>,
        loaded_now: bool,
    ) {
        {
            let mut slot = self.snapshot.write().unwrap_or_else(|p| p.into_inner());
            let loaded_at = if loaded_now {
                Some(Instant::now())
            } else {
                slot.loaded_at
            };
            *slot = Arc::new(Snapshot {
                records,
                unreadable,
                loaded_at,
            });
        }
        self.publish_metrics();
    }

    fn publish_metrics(&self) {
        let snap = self.snapshot();
        metrics::ACTION_CONTROL_LOADED.set(i64::from(snap.loaded_at.is_some()));
        for mode in [
            ControlMode::Rules,
            ControlMode::Recording,
            ControlMode::Live,
        ] {
            metrics::ACTION_CONTROL_ENTRIES
                .with_label_values(&[mode.as_str()])
                .set(snap.records.values().filter(|r| r.mode == mode).count() as i64);
        }
        let age = snap.loaded_at.unwrap_or(self.created_at).elapsed();
        metrics::ACTION_CONTROL_SNAPSHOT_AGE_SECONDS.set(age.as_secs() as i64);
    }

                pub async fn ensure_loaded(&self, cfg: &ControlConfig) {
        if !self.ready(cfg) {
            self.read_until_loaded(ENSURE_LOADED_RETRY, u64::MAX).await;
        }
    }

    async fn read_until_loaded(&self, retry: Duration, max_attempts: u64) -> bool {
        let mut attempt = 0u64;
        loop {
            attempt += 1;
            match self.refresh().await {
                Ok(_) => return true,
                Err(e) => {
                    if attempt == 1 || attempt.is_multiple_of(FIRST_LOAD_LOG_EVERY) {
                        error!(
                            error = %e,
                            attempts = attempt,
                            "action control: no successful read yet; holding this message until one succeeds"
                        );
                    }
                    if attempt >= max_attempts {
                        return false;
                    }
                    tokio::time::sleep(retry).await;
                }
            }
        }
    }

                            pub fn spawn_refresh(self: &Arc<Self>, dynamic_config: DynamicConfig) -> JoinHandle<()> {
        self.spawn_refresh_with(
            move || dynamic_config.action_control_config().ignore_store,
            REFRESH_INTERVAL,
            FIRST_LOAD_BACKOFF,
        )
    }

    fn spawn_refresh_with(
        self: &Arc<Self>,
        ignore_store: impl Fn() -> bool + Send + 'static,
        interval: Duration,
        first_backoff: Duration,
    ) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            let mut failures: u64 = 0;
            let mut backoff = first_backoff;
            loop {
                if ignore_store() {
                    metrics::ACTION_CONTROL_REFRESH_TOTAL
                        .with_label_values(&["ignored"])
                        .inc();
                    this.publish_metrics();
                    tokio::time::sleep(interval).await;
                    continue;
                }
                let first = !this.is_loaded();
                match this.refresh().await {
                    Ok(n) => {
                        if first {
                            info!(
                                records = n,
                                attempts = failures + 1,
                                since_start_ms = this.created_at.elapsed().as_millis() as u64,
                                "action control: first read ok"
                            );
                        } else if failures > 0 {
                            info!(failures, "action control: read ok again");
                        }
                        failures = 0;
                    }
                    Err(e) if first => {
                        failures += 1;
                        if failures == 1 || failures.is_multiple_of(FIRST_LOAD_LOG_EVERY) {
                            error!(
                                error = %e,
                                failures,
                                "action control: no successful read yet; this pod takes no Kafka work until one succeeds"
                            );
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY) {
                            warn!(
                                error = %e,
                                failures,
                                age_ms = this.snapshot_age_ms(),
                                "action control: read failed; keeping the last good list (live scopes run as recording only after 60 s)"
                            );
                        }
                    }
                }
                let wait = if this.is_loaded() {
                    interval
                } else {
                    let wait = backoff;
                    backoff = (backoff * 2).min(interval);
                    wait
                };
                tokio::time::sleep(wait).await;
            }
        })
    }


            pub fn record_seen(&self, seen: ActionControlSeen) -> SeenOutcome {
        self.record_seen_at(seen, Instant::now())
    }

    fn record_seen_at(&self, seen: ActionControlSeen, now: Instant) -> SeenOutcome {
        let band_ok = seen.band == "-" || valid_name(&seen.band);
        let classifier_ok = seen.classifier == "-" || valid_name(&seen.classifier);
        if !valid_name(&seen.head) || !band_ok || !classifier_ok {
            metrics::ACTION_CONTROL_SEEN_TOTAL
                .with_label_values(&["invalid"])
                .inc();
            return SeenOutcome::Invalid;
        }
        let key = seen_lkey(&seen);
        {
            let mut guard = self.seen_guard.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(at) = guard.get(&key)
                && now.saturating_duration_since(*at) < SEEN_WRITE_EVERY
            {
                return SeenOutcome::Recent;
            }
            if guard.len() >= SEEN_GUARD_MAX && !guard.contains_key(&key) {
                guard.retain(|_, at| now.saturating_duration_since(*at) < SEEN_WRITE_EVERY);
                if guard.len() >= SEEN_GUARD_MAX {
                    metrics::ACTION_CONTROL_SEEN_TOTAL
                        .with_label_values(&["dropped"])
                        .inc();
                    return SeenOutcome::Dropped;
                }
            }
            guard.insert(key.clone(), now);
        }
        match self.seen_tx.try_send(seen) {
            Ok(()) => {
                metrics::ACTION_CONTROL_SEEN_TOTAL
                    .with_label_values(&["queued"])
                    .inc();
                SeenOutcome::Queued
            }
            Err(_) => {
                metrics::ACTION_CONTROL_SEEN_TOTAL
                    .with_label_values(&["dropped"])
                    .inc();
                self.forget_seen(&key, Some(now));
                SeenOutcome::Dropped
            }
        }
    }

    fn forget_seen(&self, key: &str, at: Option<Instant>) {
        let mut guard = self.seen_guard.lock().unwrap_or_else(|p| p.into_inner());
        if at.is_none_or(|at| guard.get(key) == Some(&at)) {
            guard.remove(key);
        }
    }

        pub fn spawn_seen_writer(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        let mut rx = self
            .seen_rx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()?;
        let this = self.clone();
        Some(tokio::spawn(async move {
            let mut failures: u64 = 0;
            while let Some(seen) = rx.recv().await {
                let key = seen_lkey(&seen);
                let value = serde_json::to_vec(&seen).unwrap_or_default();
                let result = match tokio::time::timeout(
                    this.scan_timeout,
                    this.store.put(SEEN_PKEY, &key, value, SEEN_TTL_SECS),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("seen write timed out")),
                };
                match result {
                    Ok(()) => {
                        failures = 0;
                        metrics::ACTION_CONTROL_SEEN_TOTAL
                            .with_label_values(&["written"])
                            .inc();
                    }
                    Err(e) => {
                        failures += 1;
                        metrics::MANHATTAN_ERRORS_TOTAL.inc();
                        metrics::ACTION_CONTROL_SEEN_TOTAL
                            .with_label_values(&["error"])
                            .inc();
                        this.forget_seen(&key, None);
                        if failures == 1 || failures.is_multiple_of(REFRESH_ERROR_LOG_EVERY) {
                            warn!(error = %e, failures, key, "action control: seen write failed; the next decision retries");
                        }
                    }
                }
            }
        }))
    }

    async fn scan_seen(&self) -> anyhow::Result<Vec<ActionControlSeen>> {
        let rows = match tokio::time::timeout(self.scan_timeout, self.store.scan(SEEN_PKEY)).await {
            Ok(r) => r?,
            Err(_) => anyhow::bail!("seen scan timed out"),
        };
        let mut seen: Vec<ActionControlSeen> = rows
            .iter()
            .filter_map(|(_, v)| serde_json::from_slice(v).ok())
            .collect();
        seen.sort_by(|a, b| {
            (&a.head, &a.classifier, &a.band, a.entity_type.as_str()).cmp(&(
                &b.head,
                &b.classifier,
                &b.band,
                b.entity_type.as_str(),
            ))
        });
        Ok(seen)
    }


    fn view(
        &self,
        lkey: &str,
        rec: &ActionControlRecord,
        cfg: &ControlConfig,
        stale: bool,
        safety: &SafetyChecksByEntity,
        factory_heads: &HashSet<(ControlEntity, String)>,
    ) -> ActionControlView {
        let entity = rec.entity_type;
        let safety_ok = match entity {
            ControlEntity::User => safety.user.ok,
            ControlEntity::Post => safety.post.ok,
        };
        let bands: Vec<&str> = rec.band.as_deref().into_iter().collect();
        let as_level = if rec.mode == ControlMode::ConfidenceDefaults {
            Some(ActionControlRecord {
                mode: ControlMode::Live,
                actions: PageAction::menu(rec.entity_type)[..1].to_vec(),
                ..rec.clone()
            })
        } else if rec.confidence.is_some() {
            Some(rec.clone())
        } else {
            None
        };
        let (mode, _, forced) = effective(
            as_level.as_ref().unwrap_or(rec),
            match entity {
                ControlEntity::User => EntityType::User,
                ControlEntity::Post => EntityType::Post,
            },
            factory_heads.contains(&(entity, rec.head.clone())),
            cfg.scope_is_cse(&rec.head, &bands),
            cfg,
            stale,
            safety_ok,
        );
        let mode = match (rec.mode, mode) {
            (ControlMode::ConfidenceDefaults, ControlMode::Live) => ControlMode::ConfidenceDefaults,
            (_, m) => m,
        };
        ActionControlView {
            record: rec.clone(),
            scope: rec.scope().to_owned(),
            lkey: lkey.to_owned(),
            effective_mode: mode,
            forced_recording_reason: forced.map(str::to_owned),
        }
    }

        pub async fn list(
        &self,
        cfg: &ControlConfig,
        safety: SafetyChecksByEntity,
    ) -> ActionControlListResponse {
        let (refreshed, seen) = tokio::join!(self.refresh(), self.scan_seen());
        let (source, error) = match refreshed {
            Ok(_) => ("store", None),
            Err(e) => ("pod_snapshot", Some(e.to_string())),
        };
        let (seen, seen_error) = match seen {
            Ok(s) => (s, None),
            Err(e) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                (Vec::new(), Some(e.to_string()))
            }
        };
        let factory_heads: HashSet<(ControlEntity, String)> = seen
            .iter()
            .filter(|s| s.factory)
            .map(|s| (s.entity_type, s.head.clone()))
            .collect();
        let snap = self.snapshot();
        let stale = self.is_stale();
        let mut records: Vec<ActionControlView> = snap
            .records
            .iter()
            .map(|(k, r)| self.view(k, r, cfg, stale, &safety, &factory_heads))
            .collect();
        records.sort_by(|a, b| a.lkey.cmp(&b.lkey));
        let mut unreadable: Vec<UnreadableRecord> = snap
            .unreadable
            .iter()
            .map(|(k, (e, v))| UnreadableRecord {
                lkey: k.clone(),
                error: e.clone(),
                version: *v,
            })
            .collect();
        unreadable.sort_by(|a, b| a.lkey.cmp(&b.lkey));
        let mut cse_heads: Vec<String> = cfg.cse_heads.iter().cloned().collect();
        cse_heads.sort();
        let mut cse_labels: Vec<String> = cfg.cse_labels.iter().cloned().collect();
        cse_labels.sort();
        ActionControlListResponse {
            feature_version: ACTION_CONTROL_RECORD_VERSION,
            env: self.env.clone(),
            pod: self.pod.clone(),
            enabled: cfg.enabled,
            ignore_store: cfg.ignore_store,
            source: source.to_owned(),
            error,
            snapshot_age_ms: self.snapshot_age_ms(),
            stale,
            stale_after_ms: STALE_AFTER.as_millis() as i64,
            records,
            unreadable,
            seen,
            seen_error,
            safety_checks: safety,
            allowed_actions: AllowedActions {
                user: PageAction::menu(ControlEntity::User),
                post: PageAction::menu(ControlEntity::Post),
            },
            cse: CseLists {
                heads: cse_heads,
                labels: cse_labels,
            },
            confidence: Some(levels::view(cfg)),
        }
    }

        pub fn pod_view(&self) -> PodActionControlView {
        let snap = self.snapshot();
        let mut page_managed: Vec<String> = snap
            .records
            .iter()
            .filter(|(_, r)| r.mode != ControlMode::Rules)
            .map(|(k, _)| k.clone())
            .chain(snap.unreadable.keys().cloned())
            .collect();
        page_managed.sort();
        PodActionControlView {
            loaded: snap.loaded_at.is_some(),
            snapshot_age_ms: self.snapshot_age_ms(),
            stale: self.is_stale(),
            records: snap.records.len(),
            page_managed,
        }
    }

            pub fn validate(
        req: &ActionControlPutRequest,
        cfg: &ControlConfig,
        seen: &[ActionControlSeen],
    ) -> Result<Validated, SetError> {
        let bad = |code: &'static str, msg: String| Err(SetError::Invalid(code, msg));
        if !valid_name(&req.head) {
            return bad(
                "invalid_head",
                format!("head must be 1-{MAX_NAME_LEN} characters of [A-Za-z0-9_.:@-]"),
            );
        }
        if req.classifier.as_deref().is_some_and(|c| !valid_name(c)) {
            return bad(
                "invalid_classifier",
                "classifier must be null or 1-256 characters of [A-Za-z0-9_.:@-]".into(),
            );
        }
        if req
            .band
            .as_deref()
            .is_some_and(|b| !valid_name(b) || b.starts_with(DRY_RUN_LABEL_PREFIX))
        {
            return bad(
                "invalid_band",
                "band must be null or a routing label (1-256 characters of [A-Za-z0-9_.:@-], no dry_run__ prefix)"
                    .into(),
            );
        }
        let reason = req.reason.trim();
        if reason.is_empty() {
            return bad(
                "reason_required",
                "say why in reason; it is kept in the history".into(),
            );
        }
        if reason.len() > MAX_REASON_LEN {
            return bad(
                "reason_too_long",
                format!("reason is longer than {MAX_REASON_LEN} bytes"),
            );
        }
        if req
            .updated_by
            .as_deref()
            .is_some_and(|u| u.len() > MAX_NAME_LEN)
        {
            return bad(
                "updated_by_too_long",
                format!("updated_by is longer than {MAX_NAME_LEN} bytes"),
            );
        }
        match req.mode {
            ControlMode::Rules if !req.actions.is_empty() => {
                return bad(
                    "rules_takes_no_actions",
                    "mode rules follows the GrowthBook rules; send actions: []".into(),
                );
            }
            ControlMode::Live if req.actions.is_empty() => {
                return bad(
                    "live_needs_actions",
                    "mode live needs at least one action".into(),
                );
            }
            _ => {}
        }
        let mut kinds = HashSet::new();
        for a in &req.actions {
            if !kinds.insert(a.kind()) {
                return bad(
                    "duplicate_action",
                    format!("{} appears more than once", a.kind()),
                );
            }
        }
        if let Some(a) = req
            .actions
            .iter()
            .find(|a| !PageAction::menu(a.entity()).contains(a))
        {
            return bad(
                "invalid_lifetime",
                format!(
                    "{} is not offered with that lifetime; a post label lasts 30 days or permanently",
                    a.kind()
                ),
            );
        }
        let entity = req.entity_type;
        if req.actions.iter().any(|a| a.entity() != entity) {
            return bad(
                "wrong_entity_actions",
                format!(
                    "a {} record takes only {} actions",
                    entity.as_str(),
                    entity.as_str()
                ),
            );
        }
        let head_rows: Vec<&ActionControlSeen> =
            seen.iter().filter(|s| s.head == req.head).collect();
        if !head_rows.is_empty() && head_rows.iter().all(|s| s.entity_type != entity) {
            return bad(
                "wrong_entity",
                format!(
                    "this head has only sent {} decisions; a {} record would never match",
                    head_rows[0].entity_type.as_str(),
                    entity.as_str()
                ),
            );
        }
        let scope_rows: Vec<&ActionControlSeen> = head_rows
            .into_iter()
            .filter(|s| s.entity_type == entity)
            .filter(|s| req.classifier.as_ref().is_none_or(|c| &s.classifier == c))
            .filter(|s| req.band.as_ref().is_none_or(|b| &s.band == b))
            .collect();
        let bands: Vec<&str> = req.band.as_deref().into_iter().collect();
        let cse = cfg.scope_is_cse(&req.head, &bands) || scope_rows.iter().any(|s| s.cse);
        let acts = matches!(
            req.mode,
            ControlMode::Live | ControlMode::ConfidenceDefaults
        );
        if acts && cse {
            return bad(
                "cse_recording_only",
                "CSE heads and labels are enforced by the GrowthBook rules and can't be made live from this page".into(),
            );
        }
        let factory = scope_rows.iter().any(|s| s.factory);
        if factory && (acts || (req.mode == ControlMode::Recording && !req.actions.is_empty())) {
            return bad(
                "factory_managed",
                "this head's actions come from its Model Factory policy; change them there (the off switch and recording without actions are allowed here)"
                    .into(),
            );
        }
        Ok(Validated { cse })
    }

    async fn get_record(
        &self,
        lkey: &str,
    ) -> Result<Option<(Option<ActionControlRecord>, u64)>, SetError> {
        let raw = match tokio::time::timeout(self.scan_timeout, self.store.get(MH_PKEY, lkey)).await
        {
            Ok(Ok(raw)) => raw,
            Ok(Err(e)) => {
                metrics::MANHATTAN_ERRORS_TOTAL.inc();
                return Err(SetError::Store(e.to_string()));
            }
            Err(_) => return Err(SetError::Store("store read timed out".into())),
        };
        Ok(raw.map(|bytes| (parse_record(&bytes).ok(), raw_version(&bytes))))
    }

        #[allow(clippy::too_many_arguments)]
    pub async fn set(
        &self,
        req: ActionControlPutRequest,
        who: &str,
        key_id: &str,
        now_ms: i64,
        cfg: &ControlConfig,
        safety: &SafetyChecksByEntity,
    ) -> Result<SetOutcome, SetError> {
        if cfg.ignore_store {
            return Err(SetError::Ignored);
        }
        let seen = self
            .scan_seen()
            .await
            .map_err(|e| SetError::Store(format!("the seen list could not be read: {e}")))?;
        let checked = Self::validate(&req, cfg, &seen)?;
        levels::validate_confidence(&req, now_ms)?;
        let lkey = match req.confidence {
            Some(level) => level_lkey(req.entity_type, &req.head, req.classifier.as_deref(), level),
            None => lkey_for(
                req.entity_type,
                &req.head,
                req.classifier.as_deref(),
                req.band.as_deref(),
            ),
        };
        let stored = self.get_record(&lkey).await?;
        let stored_version = stored.as_ref().map(|(_, v)| *v);
        let before = stored.and_then(|(r, _)| r);
        if stored_version != req.expected_version {
            return Err(SetError::Conflict(before, stored_version));
        }
        let mut rec = ActionControlRecord {
            v: ACTION_CONTROL_RECORD_VERSION,
            entity_type: req.entity_type,
            head: req.head.clone(),
            classifier: req.classifier.clone(),
            band: req.band.clone(),
            mode: req.mode,
            actions: req.actions.clone(),
            reason: req.reason.trim().to_owned(),
            updated_by: who.to_owned(),
            key_id: key_id.to_owned(),
            updated_at_ms: now_ms,
            version: stored_version.unwrap_or(0) + 1,
            write_nonce: new_nonce(),
            confidence: req.confidence,
            evidence: req.evidence.clone().map(Box::new),
            evidence_override: req.evidence_override.clone().map(Box::new),
        };
        rec.v = rec.schema_version();
        let recording = || ActionControlRecord {
            mode: ControlMode::Recording,
            actions: Vec::new(),
            ..rec.clone()
        };
        let priors: Vec<Option<ActionControlRecord>> = match (&before, stored_version) {
            (Some(b), _) => vec![Some(b.clone())],
            (None, Some(_)) => vec![Some(recording())],
            (None, None) => {
                let opted_in = |r: ActionControlRecord| match (r.mode, req.confidence) {
                    (ControlMode::ConfidenceDefaults, Some(level)) => ActionControlRecord {
                        mode: ControlMode::Live,
                        actions: PageAction::confidence_defaults(r.entity_type, level),
                        ..r
                    },
                    (ControlMode::ConfidenceDefaults, None) => ActionControlRecord {
                        mode: ControlMode::Recording,
                        actions: Vec::new(),
                        ..r
                    },
                    _ => r,
                };
                let rows = self.scan_rows().await?;
                let snap = Snapshot::of_rows(&rows, &lkey);
                let alias = cfg.aliases.get(&req.head).map(String::as_str);
                let names: Vec<&str> = std::iter::once(req.head.as_str()).chain(alias).collect();
                let exact = match (req.confidence, req.band.as_deref()) {
                    (Some(level), _) => match snap.level_decider(
                        req.entity_type,
                        &names,
                        req.classifier.as_deref(),
                        level,
                    ) {
                        levels::LevelDecider::Record { record, .. } => {
                            Some(Some(opted_in(record.clone())))
                        }
                        levels::LevelDecider::Unreadable(_) => Some(Some(recording())),
                        levels::LevelDecider::Nothing => None,
                    },
                    (None, Some(band)) => Some(
                        match snap.plain_decider(
                            req.entity_type,
                            &req.head,
                            alias,
                            req.classifier.as_deref().unwrap_or(""),
                            &[band],
                        ) {
                            Some(Hit::Record(_, r)) => Some(opted_in(r.clone())),
                            Some(_) => Some(recording()),
                            None => None,
                        },
                    ),
                    (None, None) => None,
                };
                match exact {
                    Some(p) => vec![p],
                    None => {
                        let shadowed = Self::shadowed(&lkey, &req, cfg, &rows);
                        let head_covers = shadowed.iter().flatten().any(|r| {
                            r.classifier.is_none() && r.band.is_none() && r.confidence.is_none()
                        });
                        let mut all: Vec<Option<ActionControlRecord>> = shadowed
                            .into_iter()
                            .map(|r| Some(opted_in(r.unwrap_or_else(recording))))
                            .collect();
                        if !head_covers {
                            all.push(None);
                        }
                        all
                    }
                }
            }
        };
        let unmasks = priors
            .iter()
            .any(|p| p.as_ref().is_some_and(|p| p.mode != ControlMode::Rules));
        if checked.cse && req.mode == ControlMode::Rules && unmasks {
            return Err(SetError::Invalid(
                "cse_recording_only",
                "CSE heads and labels can't be moved back to the GrowthBook rules from this page, because that would turn enforcement on"
                    .into(),
            ));
        }
        let scope_rows: Vec<&ActionControlSeen> = seen
            .iter()
            .filter(|s| s.head == req.head && s.entity_type == req.entity_type)
            .filter(|s| req.classifier.as_ref().is_none_or(|c| &s.classifier == c))
            .filter(|s| req.band.as_ref().is_none_or(|b| &s.band == b))
            .collect();
        let rules_specs: Option<Vec<Value>> = (!scope_rows.is_empty()).then(|| {
            scope_rows
                .iter()
                .flat_map(|s| s.action_specs.as_array().cloned().unwrap_or_default())
                .collect()
        });
        let mut harsher: Vec<String> = Vec::new();
        for prior in &priors {
            let why = harsher_because(
                prior.as_ref().map(levels::for_compare).as_ref(),
                &levels::for_compare(&rec),
                rules_specs.as_deref(),
            )
            .into_iter()
            .chain(levels::defaults_harsher(prior.as_ref(), &rec))
            .chain(levels::evidence_harsher(
                prior.as_ref(),
                &rec,
                now_ms,
                cfg.confidence.evidence_max_age_ms,
            ));
            for w in why {
                if !harsher.contains(&w) {
                    harsher.push(w);
                }
            }
        }
        if !harsher.is_empty() && req.confirm.as_deref() != Some(CONFIRM_WORD) {
            return Err(SetError::Invalid(
                "confirm_required",
                format!(
                    "this change enforces more ({}); send confirm: \"{CONFIRM_WORD}\"",
                    harsher.join("; ")
                ),
            ));
        }
        let value = serde_json::to_vec(&rec).map_err(|e| SetError::Store(e.to_string()))?;
        if let Err(e) = self.store.put(MH_PKEY, &lkey, value, RECORD_TTL_SECS).await {
            metrics::MANHATTAN_ERRORS_TOTAL.inc();
            return Err(SetError::Store(e.to_string()));
        }
        let entry = ActionControlHistoryEntry {
            at_ms: now_ms,
            version: rec.version,
            lkey: lkey.clone(),
            scope: rec.scope().to_owned(),
            updated_by: rec.updated_by.clone(),
            key_id: key_id.to_owned(),
            reason: rec.reason.clone(),
            env: self.env.clone(),
            before: before.clone(),
            after: rec.clone(),
        };
        let history_written = match serde_json::to_vec(&entry) {
            Ok(v) => self
                .store
                .put(
                    &format!("{HISTORY_PKEY_PREFIX}{}", rec.head),
                    &format!("{now_ms:013}-{:010}-{lkey}", rec.version),
                    v,
                    HISTORY_TTL_SECS,
                )
                .await
                .map_err(|e| warn!(error = %e, lkey, "action control: history write failed"))
                .is_ok(),
            Err(_) => false,
        };
        let readback = match self.get_record(&lkey).await {
            Ok(Some((Some(r), _))) => r,
            Ok(_) | Err(_) => {
                return Err(SetError::Store(
                    "the write landed but reading it back failed; pods pick up whatever the store holds within 5 s, so reload to see it"
                        .into(),
                ));
            }
        };
        if readback.write_nonce != rec.write_nonce {
            return Err(SetError::LostRace(Some(readback)));
        }
        self.apply_local(lkey.clone(), rec.clone());
        info!(
            lkey,
            mode = rec.mode.as_str(),
            version = rec.version,
            updated_by = %rec.updated_by,
            key_id,
            "action control written"
        );
        let factory_heads: HashSet<(ControlEntity, String)> = seen
            .iter()
            .filter(|s| s.factory)
            .map(|s| (s.entity_type, s.head.clone()))
            .collect();
        let view = self.view(&lkey, &rec, cfg, self.is_stale(), safety, &factory_heads);
        Ok(SetOutcome {
            view,
            before,
            harsher_because: harsher,
            history_written,
        })
    }

                        fn shadowed(
        own_lkey: &str,
        req: &ActionControlPutRequest,
        cfg: &ControlConfig,
        rows: &[(String, Vec<u8>)],
    ) -> Vec<Option<ActionControlRecord>> {
        let e = req.entity_type;
        let names: Vec<&str> = std::iter::once(req.head.as_str())
            .chain(cfg.aliases.get(&req.head).map(String::as_str))
            .collect();
        let (c, b) = (req.classifier.as_deref(), req.band.as_deref());
        let mut exact: Vec<String> = Vec::new();
        let mut band_prefixes: Vec<String> = Vec::new();
        for n in &names {
            exact.push(lkey_for(e, n, c, b));
            exact.push(lkey_for(e, n, None, None));
            match (c, b) {
                (Some(c), Some(b)) => {
                    exact.push(lkey_for(e, n, Some(c), None));
                    exact.push(lkey_for(e, n, None, Some(b)));
                }
                (Some(_), None) => band_prefixes.push(format!("{}/b/{n}/", e.as_str())),
                _ => {}
            }
            if req.confidence.is_some() {
                band_prefixes.push(format!("{}/b/{n}/", e.as_str()));
                match c {
                    Some(c) => band_prefixes.push(format!("{}/cb/{n}/{c}/", e.as_str())),
                    None => {
                        band_prefixes.push(format!("{}/c/{n}/", e.as_str()));
                        band_prefixes.push(format!("{}/cb/{n}/", e.as_str()));
                    }
                }
            }
        }
        rows.iter()
            .filter(|(k, _)| k != own_lkey)
            .filter(|(k, _)| exact.contains(k) || band_prefixes.iter().any(|p| k.starts_with(p)))
            .map(|(_, v)| parse_record(v).ok())
            .collect()
    }

        async fn scan_rows(&self) -> Result<Vec<(String, Vec<u8>)>, SetError> {
        match tokio::time::timeout(self.scan_timeout, self.store.scan(MH_PKEY)).await {
            Ok(Ok(rows)) if rows.len() < SCAN_LIMIT as usize => Ok(rows),
            Ok(Ok(_)) => Err(SetError::Store(
                "the record list is at its row limit".into(),
            )),
            Ok(Err(e)) => Err(SetError::Store(e.to_string())),
            Err(_) => Err(SetError::Store("record scan timed out".into())),
        }
    }

    fn apply_local(&self, lkey: String, rec: ActionControlRecord) {
        let mut local = self.local_writes.lock().unwrap_or_else(|p| p.into_inner());
        local.insert(lkey.clone(), (rec.clone(), Instant::now()));
        let snap = self.snapshot();
        let mut records = snap.records.clone();
        let mut unreadable = snap.unreadable.clone();
        unreadable.remove(&lkey);
        records.insert(lkey, rec);
        self.publish(records, unreadable, false);
    }

        pub async fn history(&self, head: &str, limit: usize) -> ActionControlHistoryResponse {
        let limit = limit.clamp(1, HISTORY_LIMIT_MAX);
        let pkey = format!("{HISTORY_PKEY_PREFIX}{head}");
        match tokio::time::timeout(self.scan_timeout, self.store.scan(&pkey)).await {
            Ok(Ok(rows)) => {
                let mut entries: Vec<ActionControlHistoryEntry> = rows
                    .iter()
                    .filter_map(|(_, v)| serde_json::from_slice(v).ok())
                    .collect();
                entries.sort_by(|a, b| {
                    (b.at_ms, b.version, &b.lkey).cmp(&(a.at_ms, a.version, &a.lkey))
                });
                entries.truncate(limit);
                ActionControlHistoryResponse {
                    head: head.to_owned(),
                    entries,
                    error: None,
                }
            }
            Ok(Err(e)) => ActionControlHistoryResponse {
                head: head.to_owned(),
                entries: Vec::new(),
                error: Some(e.to_string()),
            },
            Err(_) => ActionControlHistoryResponse {
                head: head.to_owned(),
                entries: Vec::new(),
                error: Some("history read timed out".into()),
            },
        }
    }
}

fn raw_version(bytes: &[u8]) -> u64 {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|v| v.get("version").and_then(Value::as_u64))
        .unwrap_or(0)
}

fn parse_record(bytes: &[u8]) -> Result<ActionControlRecord, String> {
    let rec = ActionControlRecord::parse_strict(bytes)?;
    if rec.v > ACTION_CONTROL_RECORD_VERSION {
        return Err(format!("record version {} is newer than this pod", rec.v));
    }
    if !valid_name(&rec.head)
        || rec.classifier.as_deref().is_some_and(|c| !valid_name(c))
        || rec.band.as_deref().is_some_and(|b| !valid_name(b))
    {
        return Err("invalid head, classifier or band".into());
    }
    if let Some(problem) = levels::record_problem(&rec, crate::now_millis()) {
        return Err(problem.into());
    }
    Ok(rec)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        #[derive(Default)]
    pub(crate) struct MemControlStore {
        pub rows: Mutex<HashMap<(String, String), Vec<u8>>>,
        pub fail_scan: AtomicBool,
        pub fail_put: AtomicBool,
        pub fail_get: AtomicBool,
        pub scans: AtomicUsize,
                        pub clobber_with: Mutex<Option<Vec<u8>>>,
    }

    impl MemControlStore {
        pub(crate) fn put_record(&self, rec: &ActionControlRecord) {
            self.rows.lock().unwrap().insert(
                (MH_PKEY.into(), record_lkey(rec)),
                serde_json::to_vec(rec).unwrap(),
            );
        }

        pub(crate) fn put_raw(&self, lkey: &str, value: &[u8]) {
            self.rows
                .lock()
                .unwrap()
                .insert((MH_PKEY.into(), lkey.into()), value.to_vec());
        }

        pub(crate) fn pkey_rows(&self, pkey: &str) -> Vec<(String, Vec<u8>)> {
            let mut rows: Vec<(String, Vec<u8>)> = self
                .rows
                .lock()
                .unwrap()
                .iter()
                .filter(|((p, _), _)| p == pkey)
                .map(|((_, l), v)| (l.clone(), v.clone()))
                .collect();
            rows.sort();
            rows
        }
    }

    #[async_trait]
    impl ControlStore for MemControlStore {
        async fn scan(&self, pkey: &str) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
            self.scans.fetch_add(1, Ordering::Relaxed);
            if self.fail_scan.load(Ordering::Relaxed) {
                anyhow::bail!("scan failed (test)");
            }
            Ok(self.pkey_rows(pkey))
        }

        async fn get(&self, pkey: &str, lkey: &str) -> anyhow::Result<Option<Vec<u8>>> {
            if self.fail_get.load(Ordering::Relaxed) {
                anyhow::bail!("get failed (test)");
            }
            Ok(self
                .rows
                .lock()
                .unwrap()
                .get(&(pkey.to_owned(), lkey.to_owned()))
                .cloned())
        }

        async fn put(
            &self,
            pkey: &str,
            lkey: &str,
            value: Vec<u8>,
            _ttl: u64,
        ) -> anyhow::Result<()> {
            if self.fail_put.load(Ordering::Relaxed) {
                anyhow::bail!("put failed (test)");
            }
            let value = match (pkey == MH_PKEY, self.clobber_with.lock().unwrap().clone()) {
                (true, Some(other)) => other,
                _ => value,
            };
            self.rows
                .lock()
                .unwrap()
                .insert((pkey.to_owned(), lkey.to_owned()), value);
            Ok(())
        }
    }

        pub(crate) fn record(
        head: &str,
        classifier: Option<&str>,
        band: Option<&str>,
        mode: ControlMode,
        actions: Vec<PageAction>,
    ) -> ActionControlRecord {
        ActionControlRecord {
            v: ACTION_CONTROL_RECORD_VERSION,
            entity_type: actions.first().map_or(ControlEntity::User, |a| a.entity()),
            head: head.into(),
            classifier: classifier.map(Into::into),
            band: band.map(Into::into),
            mode,
            actions,
            reason: format!("reason for {head}"),
            updated_by: "operator@example.com".into(),
            key_id: "pv2".into(),
            updated_at_ms: 1_000,
            version: 1,
            write_nonce: 7,
            confidence: None,
            evidence: None,
            evidence_override: None,
        }
    }

    pub(crate) async fn loaded(
        records: &[ActionControlRecord],
    ) -> (Arc<ActionControl>, Arc<MemControlStore>) {
        let store = Arc::new(MemControlStore::default());
        for r in records {
            store.put_record(r);
        }
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            "pod-0".into(),
        ));
        ac.refresh().await.unwrap();
        (ac, store)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use std::sync::atomic::Ordering;

    const SUSPEND: PageAction = PageAction::Suspend {};
    const LIVENESS: PageAction = PageAction::LivenessCheck {};
    const SHR7: PageAction = PageAction::SpamHighRecall {
        lifetime: Lifetime::Days7,
    };
    const SHR30: PageAction = PageAction::SpamHighRecall {
        lifetime: Lifetime::Days30,
    };

        fn on() -> ControlConfig {
        ControlConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn input<'a>(head: &'a str, classifier: &'a str, labels: &'a [String]) -> ResolveInput<'a> {
        ResolveInput {
            head,
            classifier,
            labels,
            entity: EntityType::User,
            factory: false,
            producer_dry: false,
            confidence: None,
        }
    }

    fn baked_safety() -> SafetyChecksByEntity {
        let rules = crate::rules::RulesCache::new();
        SafetyChecksByEntity {
            user: crate::safety_checks::status(
                EntityType::User,
                &rules.resolve(EntityType::User, None),
            ),
            post: crate::safety_checks::status(
                EntityType::Post,
                &rules.resolve(EntityType::Post, None),
            ),
        }
    }

    fn labels(ls: &[&str]) -> Vec<String> {
        ls.iter().map(|l| (*l).to_owned()).collect()
    }

    fn resolve(ac: &ActionControl, head: &str, classifier: &str, ls: &[&str]) -> Resolved {
        let ls = labels(ls);
        ac.resolve(&input(head, classifier, &ls), &on(), |_| true)
    }

    fn live(
        head: &str,
        c: Option<&str>,
        b: Option<&str>,
        a: Vec<PageAction>,
    ) -> ActionControlRecord {
        record(head, c, b, ControlMode::Live, a)
    }

    #[tokio::test]
    async fn most_specific_record_wins() {
        let (ac, _) = loaded(&[
            live("H", None, None, vec![SUSPEND]),
            live("H", None, Some("band_a"), vec![LIVENESS]),
            live("H", Some("c1"), None, vec![SHR7]),
            live("H", Some("c1"), Some("band_a"), vec![SHR30]),
        ])
        .await;
        assert_eq!(
            resolve(&ac, "H", "c1", &["band_a"]).page_actions,
            vec![SHR30]
        );
        assert_eq!(
            resolve(&ac, "H", "c1", &["band_b"]).page_actions,
            vec![SHR7]
        );
        assert_eq!(
            resolve(&ac, "H", "c2", &["band_a"]).page_actions,
            vec![LIVENESS]
        );
        assert_eq!(resolve(&ac, "H", "c2", &[]).page_actions, vec![SUSPEND]);
        let r = resolve(&ac, "H", "c1", &["band_a"]);
        assert_eq!(r.matched.unwrap().lkey, "user/cb/H/c1/band_a");
        assert_eq!(r.mode, ControlMode::Live);
    }

    #[tokio::test]
    async fn no_record_follows_the_rules() {
        let (ac, _) = loaded(&[live("Other", None, None, vec![SUSPEND])]).await;
        let r = resolve(&ac, "H", "c", &["x"]);
        assert_eq!(r, Resolved::default());
        assert!(!r.record_only());
        assert_eq!(r.incoming_dedup_class(2), 2);
        assert_eq!(resolve(&ac, "", "c", &[]).matched, None);
    }

    #[tokio::test]
    async fn band_matches_through_the_dry_run_prefix() {
        let (ac, _) = loaded(&[live("H", None, Some("band_a"), vec![LIVENESS])]).await;
        let r = resolve(&ac, "H", "c", &["dry_run__band_a"]);
        assert_eq!(r.page_actions, vec![LIVENESS]);
        for ls in [&["band_a", "band_a"][..], &["band_a", "dry_run__band_a"]] {
            let r = resolve(&ac, "H", "c", ls);
            assert_eq!(
                (r.mode, r.page_actions),
                (ControlMode::Live, vec![LIVENESS]),
                "{ls:?}"
            );
        }
    }

    #[tokio::test]
    async fn two_band_records_at_one_level_record_only() {
        let (ac, _) = loaded(&[
            live("H", None, Some("a"), vec![SUSPEND]),
            record("H", None, Some("b"), ControlMode::Rules, vec![]),
            live("H", None, None, vec![SUSPEND]),
        ])
        .await;
        let r = resolve(&ac, "H", "c", &["a", "b"]);
        assert_eq!(r.mode, ControlMode::Recording);
        assert_eq!(r.forced, Some("band_ambiguous"));
        assert!(r.page_actions.is_empty(), "records what the rules would do");
    }

    #[tokio::test]
    async fn unreadable_record_keeps_previous_copy_else_records_only() {
        let (ac, store) = loaded(&[live("H", None, None, vec![SUSPEND])]).await;
        store.put_raw("user/h/H", b"{not json");
        store.put_raw(
            "user/h/New",
            br#"{"v":1,"entity_type":"user","head":"New","mode":"live","version":1,"future":1}"#,
        );
        store.put_raw(
            "user/h/Newer",
            br#"{"v":9,"entity_type":"user","head":"Newer","mode":"rules","version":1}"#,
        );
        ac.refresh().await.unwrap();
        assert_eq!(
            resolve(&ac, "H", "c", &[]).mode,
            ControlMode::Live,
            "previous copy kept"
        );
        for head in ["New", "Newer"] {
            let r = resolve(&ac, head, "c", &[]);
            assert_eq!(r.mode, ControlMode::Recording, "{head}");
            assert_eq!(r.forced, Some("unreadable"));
        }
        assert_eq!(
            ac.pod_view().page_managed,
            vec!["user/h/H", "user/h/New", "user/h/Newer"]
        );
    }

    #[tokio::test]
    async fn kill_switch_records_live_scopes_and_keeps_recording() {
        let (ac, _) = loaded(&[
            live("L", None, None, vec![SUSPEND]),
            record("R", None, None, ControlMode::Recording, vec![]),
            record("G", None, None, ControlMode::Rules, vec![]),
        ])
        .await;
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        let ls = labels(&[]);
        let l = ac.resolve(&input("L", "c", &ls), &killed, |_| true);
        assert_eq!(
            (l.mode, l.forced),
            (ControlMode::Recording, Some("kill_switch"))
        );
        assert_eq!(l.page_actions, vec![SUSPEND]);
        let r = ac.resolve(&input("R", "c", &ls), &killed, |_| true);
        assert_eq!(r.mode, ControlMode::Recording);
        let g = ac.resolve(&input("G", "c", &ls), &killed, |_| true);
        assert_eq!(g.mode, ControlMode::Rules);
    }

    #[tokio::test]
    async fn stale_copy_records_live_scopes() {
        let (ac, store) = loaded(&[
            live("L", None, None, vec![SUSPEND]),
            record("R", None, None, ControlMode::Recording, vec![]),
        ])
        .await;
        store.fail_scan.store(true, Ordering::Relaxed);
        assert!(ac.refresh().await.is_err());
        assert_eq!(
            resolve(&ac, "L", "c", &[]).mode,
            ControlMode::Live,
            "fresh enough"
        );
        ac.age_for_test(STALE_AFTER + Duration::from_secs(1));
        let l = resolve(&ac, "L", "c", &[]);
        assert_eq!(
            (l.mode, l.forced),
            (ControlMode::Recording, Some("stale_snapshot"))
        );
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
        assert!(ac.is_stale());
    }

    #[tokio::test]
    async fn changed_safety_checks_record_live_scopes() {
        let (ac, _) = loaded(&[live("L", None, None, vec![SUSPEND])]).await;
        let ls = labels(&[]);
        let r = ac.resolve(&input("L", "c", &ls), &on(), |_| false);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("safety_checks_changed"))
        );
    }

    #[tokio::test]
    async fn cse_is_locked_by_exact_identity_and_panda_is_not() {
        let (ac, _) = loaded(&[
            live("NearDupEmbeddingCseMatch", None, None, vec![SUSPEND]),
            live(
                "PandaReportsEmbeddingV10RoughUserAggGe4",
                None,
                None,
                vec![SUSPEND],
            ),
            live(
                "AnchorHead",
                None,
                Some("anchor_campaign_suspend_cse"),
                vec![SUSPEND],
            ),
            live("ExtraCse", None, None, vec![SUSPEND]),
        ])
        .await;
        let cse = resolve(
            &ac,
            "NearDupEmbeddingCseMatch",
            "cse_cluster_neardup_v1",
            &[],
        );
        assert_eq!(
            (cse.mode, cse.forced),
            (ControlMode::Recording, Some("cse_lock"))
        );
        let panda = resolve(
            &ac,
            "PandaReportsEmbeddingV10RoughUserAggGe4",
            "cse_reports_embedding_v10_rough_post_1024_std",
            &["panda_reports_embedding_v10_rough_spam"],
        );
        assert_eq!(
            panda.mode,
            ControlMode::Live,
            "Panda spam heads are not CSE"
        );
        let label = resolve(&ac, "AnchorHead", "c", &["anchor_campaign_suspend_cse"]);
        assert_eq!(label.forced, Some("cse_lock"));
        let mut cfg = on();
        let ls = labels(&[]);
        assert_eq!(
            ac.resolve(&input("ExtraCse", "c", &ls), &cfg, |_| true)
                .mode,
            ControlMode::Live
        );
        cfg.cse_heads.insert("ExtraCse".into());
        assert_eq!(
            ac.resolve(&input("ExtraCse", "c", &ls), &cfg, |_| true)
                .forced,
            Some("cse_lock")
        );
        let mut lockable = Resolved {
            mode: ControlMode::Live,
            page_actions: vec![SUSPEND],
            ..Default::default()
        };
        lockable.lock_cse();
        assert_eq!(
            (lockable.mode, lockable.forced),
            (ControlMode::Recording, Some("cse_lock"))
        );
    }

    #[tokio::test]
    async fn factory_messages_never_take_page_actions() {
        let (ac, _) = loaded(&[
            live("F", None, None, vec![SUSPEND]),
            record("FR", None, None, ControlMode::Recording, vec![LIVENESS]),
            record("FE", None, None, ControlMode::Recording, vec![]),
        ])
        .await;
        let ls = labels(&[]);
        for (head, forced) in [
            ("F", Some("factory_managed")),
            ("FR", Some("factory_managed")),
            ("FE", None),
        ] {
            let mut i = input(head, "c", &ls);
            i.factory = true;
            let r = ac.resolve(&i, &on(), |_| true);
            assert_eq!(r.mode, ControlMode::Recording, "{head}");
            assert!(r.page_actions.is_empty(), "{head}: the factory chooses");
            assert_eq!(r.forced, forced, "{head}");
        }
    }

    #[tokio::test]
    async fn a_shared_head_name_keeps_user_and_post_records_apart() {
        let head = "SharedSpamBot";
        let post_input = |ls: &'static [String]| ResolveInput {
            entity: EntityType::Post,
            ..input(head, "c", ls)
        };
        let (ac, _) = loaded(&[live(head, None, None, vec![SUSPEND])]).await;
        let user = resolve(&ac, head, "c", &[]);
        assert_eq!(
            (user.mode, user.page_actions),
            (ControlMode::Live, vec![SUSPEND])
        );
        let post = ac.resolve(&post_input(&[]), &on(), |_| true);
        assert_eq!(
            post,
            Resolved::default(),
            "a user record never touches post decisions"
        );
        let author = PageAction::SuspendAuthor {};
        let (ac, _) = loaded(&[
            live(head, None, None, vec![SUSPEND]),
            live(head, None, None, vec![author]),
        ])
        .await;
        assert_eq!(resolve(&ac, head, "c", &[]).page_actions, vec![SUSPEND]);
        let post = ac.resolve(&post_input(&[]), &on(), |_| true);
        assert_eq!(post.page_actions, vec![author]);
        assert_eq!(post.matched.unwrap().lkey, "post/h/SharedSpamBot");
        let mut odd = live(head, None, None, vec![author]);
        odd.entity_type = ControlEntity::User;
        let (ac, _) = loaded(&[odd]).await;
        let r = resolve(&ac, head, "c", &[]);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("wrong_entity"))
        );
        assert!(r.page_actions.is_empty());
    }

    #[tokio::test]
    async fn ignore_store_ignores_every_record() {
        let (ac, _) = loaded(&[record("R", None, None, ControlMode::Recording, vec![])]).await;
        let cfg = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        let ls = labels(&[]);
        assert_eq!(
            ac.resolve(&input("R", "c", &ls), &cfg, |_| true).mode,
            ControlMode::Rules
        );
    }

    #[tokio::test]
    async fn renamed_head_uses_the_old_name_through_an_alias() {
        let (ac, _) = loaded(&[record(
            "OldName",
            None,
            None,
            ControlMode::Recording,
            vec![],
        )])
        .await;
        let ls = labels(&[]);
        let mut cfg = on();
        assert_eq!(
            ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true).mode,
            ControlMode::Rules
        );
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let r = ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true);
        assert_eq!(r.mode, ControlMode::Recording);
        assert_eq!(r.matched.unwrap().lkey, "user/h/OldName");
    }

    #[tokio::test]
    async fn alias_records_keep_their_specificity() {
        let (ac, _) = loaded(&[
            live("NewName", None, None, vec![SUSPEND]),
            record("OldName", None, Some("b1"), ControlMode::Recording, vec![]),
            live("NewName", None, Some("b2"), vec![LIVENESS]),
            record("OldName", None, Some("b2"), ControlMode::Recording, vec![]),
            live("OldName", None, Some("b3"), vec![LIVENESS]),
        ])
        .await;
        let mut cfg = on();
        cfg.aliases.insert("NewName".into(), "OldName".into());
        let run = |ls: &[&str]| {
            let ls = labels(ls);
            ac.resolve(&input("NewName", "c", &ls), &cfg, |_| true)
        };
        let r = run(&["b1"]);
        assert_eq!(
            (r.mode, r.matched.unwrap().lkey),
            (ControlMode::Recording, "user/b/OldName/b1".to_owned()),
            "the old name's band record beats the new name's head record"
        );
        let r = run(&["b2"]);
        assert_eq!(
            r.matched.unwrap().lkey,
            "user/b/NewName/b2",
            "same scope: the new name wins"
        );
        let r = run(&["b2", "b3"]);
        assert_eq!(
            (r.mode, r.forced),
            (ControlMode::Recording, Some("band_ambiguous")),
            "bands split across the two names"
        );
    }

    #[tokio::test]
    async fn the_view_follows_a_rename_onto_a_cse_head() {
        let (ac, _) = loaded(&[live("OldName", None, None, vec![SUSPEND])]).await;
        let mut cfg = on();
        cfg.aliases
            .insert("NearDupEmbeddingCseMatch".into(), "OldName".into());
        let ls = labels(&[]);
        let leftover = ac.resolve(&input("OldName", "c", &ls), &cfg, |_| true);
        assert_eq!(
            (leftover.mode, leftover.forced),
            (ControlMode::Recording, Some("cse_lock")),
            "the old name's leftover traffic is locked too"
        );
        let list = ac.list(&cfg, baked_safety()).await;
        assert_eq!(list.records[0].effective_mode, ControlMode::Recording);
        assert_eq!(
            list.records[0].forced_recording_reason.as_deref(),
            Some("cse_lock")
        );
    }

    #[tokio::test]
    async fn put_refuses_when_the_seen_list_cant_be_read() {
        let (ac, store) = loaded(&[]).await;
        store.fail_scan.store(true, Ordering::Relaxed);
        assert!(matches!(
            set(&ac, put("H", ControlMode::Live, vec![SUSPEND])).await,
            Err(SetError::Store(_))
        ));
        assert!(store.pkey_rows(MH_PKEY).is_empty());
    }

    #[tokio::test]
    async fn page_driven_decisions_claim_their_own_strongest_action() {
        let (ac, _) = loaded(&[
            live("A", None, None, vec![LIVENESS, SHR7]),
            record("R", None, None, ControlMode::Recording, vec![SHR7]),
            record("G", None, None, ControlMode::Rules, vec![]),
            record("E", None, None, ControlMode::Recording, vec![]),
        ])
        .await;
        let a = resolve(&ac, "A", "c", &[]);
        assert_eq!((a.dedup_class, a.incoming_dedup_class(3)), (Some(2), 2));
        assert_eq!(resolve(&ac, "R", "c", &[]).incoming_dedup_class(3), 1);
        for head in ["G", "E", "None"] {
            let r = resolve(&ac, head, "c", &[]);
            assert_eq!(r.dedup_class, None, "{head}");
            assert_eq!(
                r.incoming_dedup_class(3),
                3,
                "{head}: same class as the rules"
            );
        }
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        let ls = labels(&[]);
        let r = ac.resolve(&input("A", "c", &ls), &killed, |_| true);
        assert!(r.record_only(), "kill switch: peeks, never claims");
        assert_eq!(r.incoming_dedup_class(3), 2);
    }

    #[test]
    fn recording_sources() {
        let page = Resolved {
            mode: ControlMode::Recording,
            ..Default::default()
        };
        assert!(page.record_only());
        assert_eq!(page.dry_run_source(false), Some("page"));
        assert_eq!(page.dry_run_source(true), Some("global"));
        let producer = Resolved {
            producer_dry: true,
            ..Default::default()
        };
        assert_eq!(producer.dry_run_source(false), Some("producer"));
        assert_eq!(Resolved::default().dry_run_source(false), None);
    }

    #[test]
    fn page_specs_are_ordered_label_liveness_suspend() {
        let specs = page_specs(&[SUSPEND, LIVENESS, SHR7]);
        assert_eq!(
            specs,
            vec![
                ActionSpec::AddLabelsV2 {
                    labels: vec!["SpamHighRecall".into()],
                    ttl_msec: Some(604_800_000)
                },
                ActionSpec::SpamLivenessCheck,
                ActionSpec::SuspendUser {
                    perm: false,
                    policy: "PlatformManipulation".into()
                },
            ]
        );
        assert_eq!(
            page_specs(&[PageAction::PostSpamHighRecall {
                lifetime: Lifetime::Permanent
            }]),
            vec![ActionSpec::AddPostLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: None
            }]
        );
        assert!(is_label_only(&[SHR30]));
        assert!(!is_label_only(&[SHR30, LIVENESS]));
    }

    #[test]
    fn action_specs_have_the_contract_shape() {
        let v = action_specs_value(&[
            ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into(),
            },
            ActionSpec::AddLabelsV2 {
                labels: vec!["SpamHighRecall".into(), "Other".into()],
                ttl_msec: Some(7 * 86_400_000),
            },
            ActionSpec::AddPostLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: None,
            },
            ActionSpec::AddLabelsV2 {
                labels: vec!["X".into()],
                ttl_msec: Some(43_200_000),
            },
            ActionSpec::SpamLivenessCheck,
            ActionSpec::Arkose,
            ActionSpec::Captcha,
        ]);
        assert_eq!(
            v,
            serde_json::json!([
                {"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"},
                {"kind": "label", "label": "SpamHighRecall", "ttl_days": 7, "permanent": false},
                {"kind": "label", "label": "Other", "ttl_days": 7, "permanent": false},
                {"kind": "post_label", "label": "SpamHighRecall", "permanent": true},
                {"kind": "label", "label": "X", "ttl_days": 0.5, "permanent": false},
                {"kind": "liveness_check"},
                {"kind": "arkose"},
                {"kind": "captcha"},
            ])
        );
    }

    #[test]
    fn config_parsing_fails_toward_less_enforcement() {
        use serde_json::json;
        let c = |v: serde_json::Value| ControlConfig::from_config(Some(&v));
        assert!(
            !ControlConfig::from_config(None).enabled,
            "no config = not on"
        );
        assert!(!c(json!({})).enabled, "no action_control object = not on");
        assert!(!c(json!({CONFIG_KEY: null})).enabled);
        assert!(
            !c(json!({CONFIG_KEY: {}})).enabled,
            "enabled absent = not on"
        );
        assert!(!c(json!({CONFIG_KEY: {"enabled": null}})).enabled);
        assert!(c(json!({CONFIG_KEY: {"enabled": true}})).enabled);
        assert!(!c(json!({CONFIG_KEY: {"enabled": false}})).enabled);
        assert!(
            !c(json!({CONFIG_KEY: {"enabled": "no"}})).enabled,
            "malformed = killed"
        );
        assert!(!c(json!({CONFIG_KEY: false})).enabled, "malformed = killed");
        assert!(!c(json!({CONFIG_KEY: {"ignore_store": "yes"}})).ignore_store);
        assert!(c(json!({CONFIG_KEY: {"ignore_store": true}})).ignore_store);
        let extra = c(
            json!({CONFIG_KEY: {"cse_heads": ["X", ""], "cse_labels": ["y"],
            "aliases": {"New": "Old", "bad/name": "Old"}}}),
        );
        assert!(
            extra.cse_heads.contains("X") && extra.cse_heads.contains("NearDupEmbeddingCseMatch")
        );
        assert!(extra.cse_labels.contains("y"));
        assert_eq!(extra.aliases.len(), 1);
    }

    #[tokio::test]
    async fn a_scan_cut_at_its_limit_is_a_failed_read() {
        let (ac, store) = loaded(&[record("R", None, None, ControlMode::Recording, vec![])]).await;
        for i in 0..SCAN_LIMIT {
            store.put_record(&record(
                &format!("Filler{i}"),
                None,
                None,
                ControlMode::Rules,
                vec![],
            ));
        }
        assert!(ac.refresh().await.is_err());
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
        assert_eq!(ac.pod_view().records, 1, "last good list kept");
    }

    #[test]
    fn seen_cse_is_per_band() {
        let cfg = on();
        assert_eq!(
            cfg.seen_cse("H", &["spam_band", "anchor_campaign_suspend_cse"], true),
            vec![false, true],
            "a CSE label explains the rules' Cse action"
        );
        assert_eq!(cfg.seen_cse("H", &["a", "b"], true), vec![true, true]);
        assert_eq!(cfg.seen_cse("H", &["a", "b"], false), vec![false, false]);
        assert_eq!(
            cfg.seen_cse("NearDupEmbeddingCseMatch", &["a", "-"], false),
            vec![true, true]
        );
    }

    #[tokio::test]
    async fn failed_reads_keep_the_last_good_list() {
        let (ac, store) = loaded(&[record("R", None, None, ControlMode::Recording, vec![])]).await;
        store.fail_scan.store(true, Ordering::Relaxed);
        for _ in 0..3 {
            assert!(ac.refresh().await.is_err());
        }
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
    }

    #[tokio::test]
    async fn a_pod_booted_under_ignore_store_reads_before_deciding() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("R", None, None, ControlMode::Recording, vec![]));
        let ac = ActionControl::new(store.clone(), "prod".into(), String::new());
        let ignoring = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        ac.ensure_loaded(&ignoring).await;
        assert!(!ac.is_loaded(), "the break-glass flag reads nothing");
        ac.ensure_loaded(&on()).await;
        assert!(ac.is_loaded());
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
    }

    #[tokio::test]
    async fn an_unreadable_record_shows_its_version_and_can_be_replaced() {
        let (ac, store) = loaded(&[]).await;
        store.put_raw(
            "user/h/H",
            br#"{"v":1,"entity_type":"user","head":"H","mode":"live","version":7,"x":1}"#,
        );
        ac.refresh().await.unwrap();
        let list = ac.list(&on(), baked_safety()).await;
        assert_eq!(list.unreadable[0].version, 7);
        match set(&ac, put("H", ControlMode::Recording, vec![])).await {
            Err(SetError::Conflict(None, Some(7))) => {}
            other => panic!("{other:?}"),
        }
        let mut replace = put("H", ControlMode::Recording, vec![]);
        replace.expected_version = Some(7);
        assert_eq!(set(&ac, replace).await.unwrap().view.record.version, 8);
    }

    async fn wait_for_scans(store: &MemControlStore, n: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while store.scans.load(Ordering::Relaxed) < n {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("reads keep retrying");
    }

    #[tokio::test]
    async fn the_background_read_retries_until_the_first_success() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("R", None, None, ControlMode::Recording, vec![]));
        store.fail_scan.store(true, Ordering::Relaxed);
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            String::new(),
        ));
        let task = ac.spawn_refresh_with(
            || false,
            Duration::from_millis(40),
            Duration::from_millis(2),
        );
        wait_for_scans(&store, 4).await;
        assert!(!ac.is_loaded());
        assert!(!ac.ready(&on()), "a failed first read is not an empty list");
        assert!(ac.is_stale(), "never loaded counts as stale");

        store.fail_scan.store(false, Ordering::Relaxed);
        tokio::time::timeout(Duration::from_secs(5), async {
            while !ac.ready(&on()) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("ready once a read succeeds");
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
        task.abort();
    }

    #[tokio::test]
    async fn ignore_store_reads_nothing_and_does_not_wait() {
        let store = Arc::new(MemControlStore::default());
        store.fail_scan.store(true, Ordering::Relaxed);
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            String::new(),
        ));
        let ignoring = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        let task =
            ac.spawn_refresh_with(|| true, Duration::from_millis(10), Duration::from_millis(2));
        assert!(ac.ready(&ignoring));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(store.scans.load(Ordering::Relaxed), 0);
        task.abort();
    }

    #[tokio::test]
    async fn a_message_read_has_no_time_out() {
        let store = Arc::new(MemControlStore::default());
        store.put_record(&record("R", None, None, ControlMode::Recording, vec![]));
        store.fail_scan.store(true, Ordering::Relaxed);
        let ac = Arc::new(ActionControl::new(
            store.clone(),
            "prod".into(),
            String::new(),
        ));
        assert!(!ac.read_until_loaded(Duration::from_millis(1), 3).await);
        let task = {
            let ac = ac.clone();
            tokio::spawn(async move {
                ac.read_until_loaded(Duration::from_millis(5), u64::MAX)
                    .await
            })
        };
        wait_for_scans(&store, 6).await;
        assert!(!task.is_finished(), "no time-out: still waiting");
        store.fail_scan.store(false, Ordering::Relaxed);
        assert!(task.await.unwrap());
        assert_eq!(resolve(&ac, "R", "c", &[]).mode, ControlMode::Recording);
    }


    fn put(head: &str, mode: ControlMode, actions: Vec<PageAction>) -> ActionControlPutRequest {
        ActionControlPutRequest {
            head: head.into(),
            classifier: None,
            band: None,
            mode,
            actions,
            reason: "because".into(),
            expected_version: None,
            confirm: Some(CONFIRM_WORD.into()),
            entity_type: ControlEntity::User,
            updated_by: None,
            confidence: None,
            evidence: None,
            evidence_override: None,
        }
    }

    async fn set(ac: &ActionControl, req: ActionControlPutRequest) -> Result<SetOutcome, SetError> {
        ac.set(
            req,
            "operator@example.com",
            "pv2",
            1_791_000_000_000,
            &on(),
            &baked_safety(),
        )
        .await
    }

    fn code(r: Result<SetOutcome, SetError>) -> &'static str {
        match r {
            Err(SetError::Invalid(c, _)) => c,
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_validation_codes() {
        let (ac, store) = loaded(&[]).await;
        let seen =
            |head: &str, factory: bool, cse: bool, entity: ControlEntity| ActionControlSeen {
                head: head.into(),
                classifier: "c".into(),
                band: "-".into(),
                entity_type: entity,
                factory,
                cse,
                last_seen_ms: 1,
                rule_id: String::new(),
                action_specs: serde_json::json!([]),
            };
        for s in [
            seen("Factory", true, false, ControlEntity::User),
            seen("SeenCse", false, true, ControlEntity::User),
            seen("PostHead", false, false, ControlEntity::Post),
        ] {
            store.rows.lock().unwrap().insert(
                (SEEN_PKEY.into(), seen_lkey(&s)),
                serde_json::to_vec(&s).unwrap(),
            );
        }
        let cases: Vec<(ActionControlPutRequest, &str)> = vec![
            (put("a/b", ControlMode::Rules, vec![]), "invalid_head"),
            (
                ActionControlPutRequest {
                    classifier: Some("x y".into()),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "invalid_classifier",
            ),
            (
                ActionControlPutRequest {
                    band: Some("dry_run__b".into()),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "invalid_band",
            ),
            (
                ActionControlPutRequest {
                    reason: " ".into(),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "reason_required",
            ),
            (
                ActionControlPutRequest {
                    reason: "r".repeat(1001),
                    ..put("H", ControlMode::Rules, vec![])
                },
                "reason_too_long",
            ),
            (
                put("H", ControlMode::Rules, vec![SUSPEND]),
                "rules_takes_no_actions",
            ),
            (put("H", ControlMode::Live, vec![]), "live_needs_actions"),
            (
                put("H", ControlMode::Live, vec![SHR7, SHR30]),
                "duplicate_action",
            ),
            (
                put(
                    "H",
                    ControlMode::Live,
                    vec![SUSPEND, PageAction::SuspendAuthor {}],
                ),
                "wrong_entity_actions",
            ),
            (
                ActionControlPutRequest {
                    entity_type: ControlEntity::Post,
                    ..put("PostHead", ControlMode::Live, vec![SUSPEND])
                },
                "wrong_entity_actions",
            ),
            (
                put("PostHead", ControlMode::Recording, vec![]),
                "wrong_entity",
            ),
            (
                ActionControlPutRequest {
                    entity_type: ControlEntity::Post,
                    ..put(
                        "PostHead",
                        ControlMode::Live,
                        vec![PageAction::PostSpamHighRecall {
                            lifetime: Lifetime::Days7,
                        }],
                    )
                },
                "invalid_lifetime",
            ),
            (
                put(
                    "NearDupEmbeddingCseNcmecMatch",
                    ControlMode::Live,
                    vec![SUSPEND],
                ),
                "cse_recording_only",
            ),
            (
                ActionControlPutRequest {
                    band: Some("cse_reports_embedding_v2".into()),
                    ..put("H", ControlMode::Live, vec![SUSPEND])
                },
                "cse_recording_only",
            ),
            (
                put("SeenCse", ControlMode::Live, vec![SUSPEND]),
                "cse_recording_only",
            ),
            (
                put("Factory", ControlMode::Live, vec![SUSPEND]),
                "factory_managed",
            ),
            (
                put("Factory", ControlMode::Recording, vec![SUSPEND]),
                "factory_managed",
            ),
            (
                ActionControlPutRequest {
                    confirm: None,
                    ..put("H", ControlMode::Live, vec![SUSPEND])
                },
                "confirm_required",
            ),
        ];
        for (req, want) in cases {
            assert_eq!(code(set(&ac, req.clone()).await), want, "{req:?}");
        }
        assert!(store.pkey_rows(MH_PKEY).is_empty(), "nothing written");
        assert!(
            set(&ac, put("Factory", ControlMode::Recording, vec![]))
                .await
                .is_ok()
        );
        let cse = put("NearDupEmbeddingCseMatch", ControlMode::Recording, vec![]);
        assert!(set(&ac, cse).await.is_ok());
        let p = set(
            &ac,
            ActionControlPutRequest {
                entity_type: ControlEntity::Post,
                ..put(
                    "PostHead",
                    ControlMode::Live,
                    vec![PageAction::SuspendAuthor {}],
                )
            },
        )
        .await;
        assert!(p.is_ok(), "{p:?}");
    }

    #[tokio::test]
    async fn recording_back_to_the_rules_is_harsher_and_refused_for_cse() {
        let (ac, store) = loaded(&[]).await;
        let back = |head: &str, confirm: Option<&str>| ActionControlPutRequest {
            expected_version: Some(1),
            confirm: confirm.map(Into::into),
            ..put(head, ControlMode::Rules, vec![])
        };
        for head in ["H", "NearDupEmbeddingCseMatch"] {
            set(&ac, put(head, ControlMode::Recording, vec![]))
                .await
                .unwrap();
        }
        assert_eq!(code(set(&ac, back("H", None)).await), "confirm_required");
        let out = set(&ac, back("H", Some(CONFIRM_WORD))).await.unwrap();
        assert_eq!(
            out.harsher_because,
            vec!["hands the scope back to the GrowthBook rules, which act"]
        );
        for confirm in [None, Some(CONFIRM_WORD)] {
            assert_eq!(
                code(set(&ac, back("NearDupEmbeddingCseMatch", confirm)).await),
                "cse_recording_only",
                "{confirm:?}"
            );
        }
        let cse_label = |mode, version| ActionControlPutRequest {
            band: Some("anchor_campaign_suspend_cse".into()),
            expected_version: version,
            ..put("AnchorHead", mode, vec![])
        };
        set(&ac, cse_label(ControlMode::Recording, None))
            .await
            .unwrap();
        assert_eq!(
            code(set(&ac, cse_label(ControlMode::Rules, Some(1))).await),
            "cse_recording_only"
        );
        store.put_raw(
            "user/h/CseRtScoreHiV2DryRun",
            br#"{"v":1,"entity_type":"user","head":"CseRtScoreHiV2DryRun","mode":"recording","version":4,"x":1}"#,
        );
        let mut r = back("CseRtScoreHiV2DryRun", Some(CONFIRM_WORD));
        r.expected_version = Some(4);
        assert_eq!(code(set(&ac, r).await), "cse_recording_only");
        assert_eq!(
            resolve(&ac, "NearDupEmbeddingCseMatch", "c", &[]).mode,
            ControlMode::Recording,
            "still recording only"
        );
    }

    #[tokio::test]
    async fn put_writes_versions_history_and_applies_at_once() {
        let (ac, store) = loaded(&[]).await;
        let out = set(&ac, put("H", ControlMode::Recording, vec![]))
            .await
            .unwrap();
        assert_eq!(out.view.record.version, 1);
        assert_eq!(out.view.record.updated_by, "operator@example.com");
        assert_eq!(out.view.record.key_id, "pv2");
        assert!(out.history_written);
        assert_eq!(
            resolve(&ac, "H", "c", &[]).mode,
            ControlMode::Recording,
            "applied locally"
        );
        match set(&ac, put("H", ControlMode::Rules, vec![])).await {
            Err(SetError::Conflict(Some(cur), Some(1))) => assert_eq!(cur.version, 1),
            other => panic!("{other:?}"),
        }
        let mut next = put("H", ControlMode::Live, vec![SUSPEND]);
        next.expected_version = Some(1);
        let out = set(&ac, next).await.unwrap();
        assert_eq!(out.view.record.version, 2);
        assert_eq!(out.before.unwrap().mode, ControlMode::Recording);
        assert_eq!(out.harsher_because, vec!["turns the scope live"]);
        let h = ac.history("H", 50).await;
        assert_eq!(h.entries.len(), 2);
        assert_eq!(h.entries[0].version, 2, "newest first");
        assert_eq!(h.entries[0].before.as_ref().unwrap().version, 1);
        assert_eq!(h.entries[1].before, None);
        assert_eq!(ac.history("H", 1).await.entries.len(), 1);
        assert_eq!(
            store.pkey_rows(MH_PKEY).len(),
            1,
            "one record per scope, never deleted"
        );
    }

    #[tokio::test]
    async fn put_reports_a_lost_race() {
        let (ac, store) = loaded(&[]).await;
        let mut other = record("H", None, None, ControlMode::Rules, vec![]);
        other.write_nonce = 99;
        *store.clobber_with.lock().unwrap() = Some(serde_json::to_vec(&other).unwrap());
        match set(&ac, put("H", ControlMode::Recording, vec![])).await {
            Err(SetError::LostRace(Some(cur))) => assert_eq!(cur.write_nonce, 99),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            ac.history("H", 10).await.entries.len(),
            1,
            "history keeps the write that was live"
        );
    }

    #[tokio::test]
    async fn a_rules_record_that_shadows_recording_is_harsher_and_refused_for_cse() {
        let (ac, store) = loaded(&[
            record("H", None, None, ControlMode::Recording, vec![]),
            record(
                "NearDupEmbeddingCseMatch",
                None,
                None,
                ControlMode::Recording,
                vec![],
            ),
            record("OldName", None, None, ControlMode::Recording, vec![]),
            record("H2", None, Some("bx"), ControlMode::Recording, vec![]),
            record(
                "CseRtScoreHiV2DryRun",
                None,
                Some("bx"),
                ControlMode::Recording,
                vec![],
            ),
        ])
        .await;
        let clf_rules = |head: &str| ActionControlPutRequest {
            classifier: Some("clf".into()),
            confirm: None,
            ..put(head, ControlMode::Rules, vec![])
        };
        assert_eq!(code(set(&ac, clf_rules("H2")).await), "confirm_required");
        assert_eq!(
            code(set(&ac, clf_rules("CseRtScoreHiV2DryRun")).await),
            "cse_recording_only"
        );
        store.put_record(&record(
            "NearDupEmbeddingCseNcmecMatch",
            None,
            None,
            ControlMode::Recording,
            vec![],
        ));
        let band_rules = |head: &str, confirm: Option<&str>| ActionControlPutRequest {
            band: Some("some_band".into()),
            confirm: confirm.map(Into::into),
            ..put(head, ControlMode::Rules, vec![])
        };
        assert_eq!(
            code(set(&ac, band_rules("H", None)).await),
            "confirm_required"
        );
        for head in ["NearDupEmbeddingCseMatch", "NearDupEmbeddingCseNcmecMatch"] {
            for confirm in [None, Some(CONFIRM_WORD)] {
                assert_eq!(
                    code(set(&ac, band_rules(head, confirm)).await),
                    "cse_recording_only",
                    "{head}"
                );
            }
        }
        let mut cfg = on();
        cfg.aliases.insert("NewName".into(), "OldName".into());
        cfg.aliases
            .insert("NewCse".into(), "NearDupEmbeddingCseMatch".into());
        let renamed = |head: &str, confirm: Option<&str>| ActionControlPutRequest {
            confirm: confirm.map(Into::into),
            ..put(head, ControlMode::Rules, vec![])
        };
        let safety = baked_safety();
        let set_cfg = |req| ac.set(req, "w", "k", 1, &cfg, &safety);
        assert_eq!(
            code(set_cfg(renamed("NewName", None)).await),
            "confirm_required"
        );
        assert_eq!(
            code(set_cfg(renamed("NewCse", Some(CONFIRM_WORD))).await),
            "cse_recording_only"
        );
        cfg.aliases
            .insert("NearDupEmbeddingCseNcmecMatch".into(), "OldName".into());
        let set_cfg = |req| ac.set(req, "w", "k", 1, &cfg, &safety);
        let mut old = renamed("OldName", Some(CONFIRM_WORD));
        old.expected_version = Some(1);
        assert_eq!(code(set_cfg(old).await), "cse_recording_only");
        cfg.aliases.remove("NearDupEmbeddingCseNcmecMatch");
        let set_cfg = |req| ac.set(req, "w", "k", 1, &cfg, &safety);
        assert!(
            set_cfg(renamed("NewName", Some(CONFIRM_WORD)))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn put_answer_shows_this_pods_effective_mode() {
        let (ac, _) = loaded(&[]).await;
        let killed = ControlConfig {
            enabled: false,
            ..Default::default()
        };
        let out = ac
            .set(
                put("H", ControlMode::Live, vec![SUSPEND]),
                "w",
                "k",
                1,
                &killed,
                &baked_safety(),
            )
            .await
            .unwrap();
        assert_eq!(out.view.effective_mode, ControlMode::Recording);
        assert_eq!(
            out.view.forced_recording_reason.as_deref(),
            Some("kill_switch")
        );
    }

    #[tokio::test]
    async fn an_unreadable_read_back_is_not_applied_locally() {
        let (ac, store) = loaded(&[]).await;
        *store.clobber_with.lock().unwrap() = Some(b"{not json".to_vec());
        assert!(matches!(
            set(&ac, put("H", ControlMode::Recording, vec![])).await,
            Err(SetError::Store(_))
        ));
        assert_eq!(resolve(&ac, "H", "c", &[]).mode, ControlMode::Rules);
    }

    #[tokio::test]
    async fn failed_store_write_changes_nothing() {
        let (ac, store) = loaded(&[]).await;
        store.fail_put.store(true, Ordering::Relaxed);
        assert!(matches!(
            set(&ac, put("H", ControlMode::Recording, vec![])).await,
            Err(SetError::Store(_))
        ));
        assert_eq!(resolve(&ac, "H", "c", &[]).mode, ControlMode::Rules);
        store.fail_put.store(false, Ordering::Relaxed);
        store.fail_get.store(true, Ordering::Relaxed);
        assert!(matches!(
            set(&ac, put("H", ControlMode::Recording, vec![])).await,
            Err(SetError::Store(_))
        ));
        let ignore = ControlConfig {
            ignore_store: true,
            ..Default::default()
        };
        assert_eq!(
            ac.set(
                put("H", ControlMode::Rules, vec![]),
                "w",
                "k",
                1,
                &ignore,
                &baked_safety()
            )
            .await
            .unwrap_err(),
            SetError::Ignored
        );
    }

    #[test]
    fn harsher_changes() {
        let r = |mode, actions: Vec<PageAction>| record("H", None, None, mode, actions);
        let h =
            |b: Option<&ActionControlRecord>, a: &ActionControlRecord| harsher_because(b, a, None);
        assert_eq!(
            h(None, &r(ControlMode::Recording, vec![])),
            Vec::<String>::new()
        );
        assert!(!h(None, &r(ControlMode::Live, vec![LIVENESS])).is_empty());
        let rec = r(ControlMode::Recording, vec![]);
        assert!(
            !h(Some(&rec), &r(ControlMode::Rules, vec![])).is_empty(),
            "recording → rules acts"
        );
        let l = r(ControlMode::Live, vec![SHR7]);
        assert!(
            h(Some(&l), &r(ControlMode::Live, vec![SHR7, SUSPEND]))
                .contains(&"adds suspend".to_owned())
        );
        assert!(
            h(Some(&l), &r(ControlMode::Live, vec![SHR7, LIVENESS]))
                .contains(&"adds liveness check".to_owned())
        );
        assert!(
            h(Some(&l), &r(ControlMode::Live, vec![SHR30]))
                .contains(&"makes the label last longer".to_owned())
        );
        let shorter = h(
            Some(&r(ControlMode::Live, vec![SHR30])),
            &r(ControlMode::Live, vec![SHR7]),
        );
        assert!(shorter.is_empty(), "{shorter:?}");
        assert!(
            h(Some(&l), &r(ControlMode::Recording, vec![SHR7])).is_empty(),
            "live → recording is softer"
        );
        let back = |page: Vec<PageAction>, rules: serde_json::Value| {
            harsher_because(
                Some(&r(ControlMode::Live, page)),
                &r(ControlMode::Rules, vec![]),
                Some(rules.as_array().unwrap()),
            )
        };
        use serde_json::json;
        let temp_suspend =
            json!({"kind": "suspend", "permanent": false, "policy": "PlatformManipulation"});
        let label7 =
            json!({"kind": "label", "label": "SpamHighRecall", "ttl_days": 7, "permanent": false});
        assert!(back(vec![SHR7, SUSPEND], json!([label7, temp_suspend])).is_empty());
        assert!(
            back(vec![SHR30], json!([label7])).is_empty(),
            "rules' label is shorter"
        );
        assert!(
            !back(vec![SHR30], json!([])).is_empty(),
            "seen rows with no actions: unknown"
        );
        for harsher in [
            json!([{"kind": "suspend", "permanent": true, "policy": "PlatformManipulation"}]),
            json!([{"kind": "suspend", "permanent": false, "policy": "Cse"}]),
            json!([{"kind": "label", "label": "SpamHighRecall", "permanent": true}]),
            json!([{"kind": "label", "label": "Other", "ttl_days": 7, "permanent": false}]),
            json!([{"kind": "arkose"}]),
        ] {
            assert!(
                !back(vec![SHR30, SUSPEND, LIVENESS], harsher.clone()).is_empty(),
                "{harsher}"
            );
        }
        assert!(
            !harsher_because(Some(&l), &r(ControlMode::Rules, vec![]), None).is_empty(),
            "rules' actions unknown"
        );
    }

    #[tokio::test]
    async fn seen_rows_are_written_at_most_once_per_window() {
        let (ac, store) = loaded(&[]).await;
        let row = |band: &str| ActionControlSeen {
            head: "H".into(),
            classifier: "c@1".into(),
            band: band.into(),
            entity_type: ControlEntity::User,
            factory: false,
            cse: false,
            last_seen_ms: 5,
            rule_id: "act_x".into(),
            action_specs: serde_json::json!([{"kind": "suspend"}]),
        };
        let t0 = Instant::now();
        assert_eq!(ac.record_seen_at(row("a"), t0), SeenOutcome::Queued);
        assert_eq!(
            ac.record_seen_at(row("a"), t0 + Duration::from_secs(1)),
            SeenOutcome::Recent
        );
        assert_eq!(ac.record_seen_at(row("-"), t0), SeenOutcome::Queued);
        assert_eq!(ac.record_seen_at(row("a/b"), t0), SeenOutcome::Invalid);
        ac.spawn_seen_writer().unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while store.pkey_rows(SEEN_PKEY).len() < 2 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let list = ac.list(&on(), baked_safety()).await;
        assert_eq!(list.seen.len(), 2);
        assert_eq!(list.seen[0].band, "-");
        assert_eq!(list.source, "store");
    }


    const PUT_NOW: i64 = 1_791_000_000_000;

    fn level_put(
        head: &str,
        level: ConfidenceLevel,
        mode: ControlMode,
        actions: Vec<PageAction>,
    ) -> ActionControlPutRequest {
        ActionControlPutRequest {
            confidence: Some(level),
            ..put(head, mode, actions)
        }
    }

    fn read(lb: f64, reads: u32, humans: u32) -> ConfidenceEvidence {
        ConfidenceEvidence {
            precision_lower_bound: lb,
            reads,
            humans,
            read_at_ms: PUT_NOW - 86_400_000,
            link: "documents/read.md".into(),
        }
    }

    fn person(expires_at_ms: i64) -> EvidenceOverride {
        EvidenceOverride {
            by: "operator@example.com".into(),
            reason: "seeded while the read runs".into(),
            expires_at_ms,
        }
    }

    #[tokio::test]
    async fn confidence_put_validation_codes() {
        let (ac, _) = loaded(&[]).await;
        let cases: Vec<(ActionControlPutRequest, &str)> = vec![
            (
                ActionControlPutRequest {
                    band: Some("b".into()),
                    ..level_put("H", ConfidenceLevel::High, ControlMode::Recording, vec![])
                },
                "invalid_scope",
            ),
            (
                ActionControlPutRequest {
                    band: Some("b".into()),
                    ..put("H", ControlMode::ConfidenceDefaults, vec![])
                },
                "invalid_scope",
            ),
            (
                put("H", ControlMode::ConfidenceDefaults, vec![SUSPEND]),
                "defaults_take_no_actions",
            ),
            (
                ActionControlPutRequest {
                    evidence: Some(read(0.96, 59, 0)),
                    ..put("H", ControlMode::Live, vec![SUSPEND])
                },
                "invalid_evidence",
            ),
            (
                ActionControlPutRequest {
                    evidence: Some(read(0.96, 5, 6)),
                    ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
                },
                "invalid_evidence",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(person(PUT_NOW + 86_400_000)),
                    ..level_put("H", ConfidenceLevel::High, ControlMode::Recording, vec![])
                },
                "invalid_override",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(person(PUT_NOW + 8 * 86_400_000)),
                    ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
                },
                "invalid_override",
            ),
            (
                ActionControlPutRequest {
                    evidence_override: Some(EvidenceOverride {
                        by: " ".into(),
                        ..person(PUT_NOW + 86_400_000)
                    }),
                    ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
                },
                "invalid_override",
            ),
            (
                put(CSE_HEADS[0], ControlMode::ConfidenceDefaults, vec![]),
                "cse_recording_only",
            ),
            (
                level_put(
                    CSE_HEADS[0],
                    ConfidenceLevel::High,
                    ControlMode::Live,
                    vec![SUSPEND],
                ),
                "cse_recording_only",
            ),
        ];
        for (req, want) in cases {
            let label = format!("{req:?}");
            assert_eq!(code(set(&ac, req).await), want, "{label}");
        }
    }

    #[tokio::test]
    async fn level_records_are_written_at_their_own_key_and_version() {
        let (ac, _) = loaded(&[]).await;
        let out = set(
            &ac,
            ActionControlPutRequest {
                evidence: Some(read(0.96, 59, 0)),
                ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
            },
        )
        .await
        .unwrap();
        assert_eq!(out.view.scope, "confidence");
        assert_eq!(out.view.lkey, "user/l/H/high");
        assert_eq!(out.view.record.v, ACTION_CONTROL_RECORD_VERSION);
        let band = set(
            &ac,
            ActionControlPutRequest {
                band: Some("b".into()),
                ..put("H", ControlMode::Recording, vec![])
            },
        )
        .await
        .unwrap();
        assert_eq!(
            band.view.record.v,
            xai_abuse_enforcement_types::action_control::BAND_RECORD_VERSION,
            "a band record stays readable by older images"
        );
        let c = set(
            &ac,
            ActionControlPutRequest {
                classifier: Some("clf".into()),
                ..level_put("H", ConfidenceLevel::Mid, ControlMode::Recording, vec![])
            },
        )
        .await
        .unwrap();
        assert_eq!(c.view.lkey, "user/cl/H/clf/mid");
        assert_eq!(c.view.scope, "classifier+confidence");
    }

    #[tokio::test]
    async fn opting_into_the_defaults_and_out_to_the_rules_need_the_confirmation() {
        let (ac, _) = loaded(&[]).await;
        let unconfirmed = |req: ActionControlPutRequest| ActionControlPutRequest {
            confirm: None,
            ..req
        };
        assert_eq!(
            code(
                set(
                    &ac,
                    unconfirmed(put("H", ControlMode::ConfidenceDefaults, vec![]))
                )
                .await
            ),
            "confirm_required"
        );
        let on = set(&ac, put("H", ControlMode::ConfidenceDefaults, vec![]))
            .await
            .unwrap();
        assert!(
            on.harsher_because
                .iter()
                .any(|w| w.contains("default table")),
            "{:?}",
            on.harsher_because
        );
        let back = |mode| ActionControlPutRequest {
            expected_version: Some(1),
            ..unconfirmed(put("H", mode, vec![]))
        };
        assert_eq!(
            code(set(&ac, back(ControlMode::Rules)).await),
            "confirm_required"
        );
        let softer = set(&ac, back(ControlMode::Recording)).await.unwrap();
        assert!(
            softer.harsher_because.is_empty(),
            "{:?}",
            softer.harsher_because
        );
    }

    #[tokio::test]
    async fn stronger_evidence_or_a_new_override_needs_the_confirmation() {
        let (ac, _) = loaded(&[]).await;
        let mid_read = ActionControlPutRequest {
            evidence: Some(read(0.88, 50, 2)),
            ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
        };
        set(&ac, mid_read).await.unwrap();
        let high_read = ActionControlPutRequest {
            evidence: Some(read(0.96, 59, 0)),
            expected_version: Some(1),
            confirm: None,
            ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
        };
        assert_eq!(code(set(&ac, high_read.clone()).await), "confirm_required");
        let ok = set(
            &ac,
            ActionControlPutRequest {
                confirm: Some(CONFIRM_WORD.into()),
                ..high_read
            },
        )
        .await
        .unwrap();
        assert!(
            ok.harsher_because.iter().any(|w| w.contains("at high")),
            "{:?}",
            ok.harsher_because
        );
        let with_override = ActionControlPutRequest {
            evidence: Some(read(0.96, 59, 0)),
            evidence_override: Some(person(PUT_NOW + 86_400_000)),
            expected_version: Some(2),
            confirm: None,
            ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
        };
        assert_eq!(code(set(&ac, with_override).await), "confirm_required");
    }

    #[tokio::test]
    async fn a_first_classifier_level_write_is_judged_against_the_head_level_record() {
        let head_level = |mode, actions| ActionControlRecord {
            confidence: Some(ConfidenceLevel::High),
            ..record("H", None, None, mode, actions)
        };
        let opt_in = record("H", None, None, ControlMode::ConfidenceDefaults, vec![]);
        let classifier_live_suspend = ActionControlPutRequest {
            classifier: Some("clf".into()),
            confirm: None,
            ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
        };
        for (stored, harsher) in [
            (head_level(ControlMode::Recording, vec![]), true),
            (head_level(ControlMode::Live, vec![LIVENESS]), true),
            (head_level(ControlMode::Live, vec![SUSPEND]), false),
        ] {
            let label = format!("{:?} {:?}", stored.mode, stored.actions);
            let (ac, _) = loaded(&[opt_in.clone(), stored]).await;
            let got = set(&ac, classifier_live_suspend.clone()).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{label}");
            } else {
                assert!(got.is_ok(), "{label}: {got:?}");
            }
        }
    }

    #[test]
    fn stored_time_boundaries_are_exact() {
        let now = PUT_NOW;
        let skew = crate::confidence::CLOCK_SKEW_MS;
        let day = 86_400_000;
        let mut r = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        r.updated_at_ms = now + skew;
        assert_eq!(
            levels::record_problem(&r, now),
            None,
            "written exactly the skew ahead"
        );
        r.updated_at_ms += 1;
        assert!(levels::record_problem(&r, now).is_some());

        let mut lv = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        lv.confidence = Some(ConfidenceLevel::High);
        lv.evidence = Some(Box::new(ConfidenceEvidence {
            read_at_ms: now + skew,
            ..read(0.96, 59, 0)
        }));
        assert_eq!(
            levels::record_problem(&lv, now),
            None,
            "read exactly the skew ahead"
        );
        lv.evidence.as_mut().unwrap().read_at_ms += 1;
        assert!(levels::record_problem(&lv, now).is_some());

        let with_override = |updated_at_ms: i64, expires_at_ms: i64| {
            let mut r = record("H", None, None, ControlMode::ConfidenceDefaults, vec![]);
            r.updated_at_ms = updated_at_ms;
            r.evidence_override = Some(Box::new(EvidenceOverride {
                expires_at_ms,
                ..person(0)
            }));
            levels::record_problem(&r, now)
        };
        assert_eq!(with_override(now, now + 7 * day), None, "exactly 7 days");
        assert!(with_override(now, now + 7 * day + 1).is_some());
        assert!(with_override(now, now).is_some(), "expires at its write");
        assert_eq!(with_override(now, now + 1), None);
        assert!(
            with_override(now + skew, now + 60_000).is_some(),
            "expires before its write"
        );
    }

    #[test]
    fn the_get_view_shows_an_unreadable_confidence_config() {
        let mut cfg = on();
        assert_eq!(levels::view(&cfg).config_error, None);
        cfg.confidence = crate::confidence::ConfidenceConfig::from_value(Some(
            &serde_json::json!({"label_map": {"x": "very_high"}}),
        ));
        let view = levels::view(&cfg);
        assert!(view.config_error.is_some_and(|e| e.contains("label_map")));
        assert_eq!(view.evidence_max_age_ms, 0);
        assert!(view.label_map.is_empty());
    }

    #[tokio::test]
    async fn a_stored_record_breaking_the_level_rules_is_unreadable() {
        let mut both = record("H", None, Some("b"), ControlMode::Live, vec![SUSPEND]);
        both.confidence = Some(ConfidenceLevel::High);
        assert!(parse_record(&serde_json::to_vec(&both).unwrap()).is_err());
        let mut defaults_with_actions = record(
            "H",
            None,
            None,
            ControlMode::ConfidenceDefaults,
            vec![SUSPEND],
        );
        defaults_with_actions.v = ACTION_CONTROL_RECORD_VERSION;
        assert!(parse_record(&serde_json::to_vec(&defaults_with_actions).unwrap()).is_err());
        let mut stray_evidence = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        stray_evidence.evidence = Some(Box::new(read(0.96, 59, 0)));
        assert!(parse_record(&serde_json::to_vec(&stray_evidence).unwrap()).is_err());
        let mut newer = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        newer.v = ACTION_CONTROL_RECORD_VERSION + 1;
        assert!(parse_record(&serde_json::to_vec(&newer).unwrap()).is_err());

        let now = crate::now_millis();
        let mut written_later = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        written_later.updated_at_ms = now + 3_600_000;
        assert!(parse_record(&serde_json::to_vec(&written_later).unwrap()).is_err());
        written_later.updated_at_ms = now + 60_000;
        assert!(
            parse_record(&serde_json::to_vec(&written_later).unwrap()).is_ok(),
            "within the clock skew"
        );
        let mut read_later = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        read_later.confidence = Some(ConfidenceLevel::High);
        read_later.evidence = Some(Box::new(ConfidenceEvidence {
            read_at_ms: now + 3_600_000,
            ..read(0.96, 59, 0)
        }));
        assert!(parse_record(&serde_json::to_vec(&read_later).unwrap()).is_err());
        let day = 86_400_000;
        let with_override = |by: &str, updated_at_ms: i64, expires_at_ms: i64| {
            let mut r = record("H", None, None, ControlMode::ConfidenceDefaults, vec![]);
            r.updated_at_ms = updated_at_ms;
            r.evidence_override = Some(Box::new(EvidenceOverride {
                expires_at_ms,
                ..person(0)
            }));
            r.evidence_override.as_mut().unwrap().by = by.into();
            serde_json::to_vec(&r).unwrap()
        };
        assert!(parse_record(&with_override("a", now, now + day)).is_ok());
        assert!(
            parse_record(&with_override("a", now - day, now - 1)).is_ok(),
            "expired is inactive, not unreadable"
        );
        assert!(
            parse_record(&with_override("a", now, now - 1)).is_err(),
            "expires before its write"
        );
        assert!(parse_record(&with_override("", now, now + day)).is_err());
        assert!(parse_record(&with_override("a", now, now + 8 * day)).is_err());
    }

    #[tokio::test]
    async fn v5_round3_every_harsher_precedence_path_needs_confirmation() {
        let no_confirm = |req: ActionControlPutRequest| ActionControlPutRequest {
            confirm: None,
            ..req
        };
        let level_record =
            |head: &str,
             classifier: Option<&str>,
             level: ConfidenceLevel,
             mode: ControlMode,
             actions: Vec<PageAction>| ActionControlRecord {
                confidence: Some(level),
                ..record(head, classifier, None, mode, actions)
            };

        let (ac, _) = loaded(&[level_record(
            "H",
            None,
            ConfidenceLevel::High,
            ControlMode::Recording,
            vec![],
        )])
        .await;
        let req = ActionControlPutRequest {
            classifier: Some("clf".into()),
            ..no_confirm(level_put(
                "H",
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            ))
        };
        assert_eq!(code(set(&ac, req).await), "confirm_required");

        let (ac, _) = loaded(&[record(
            "H",
            None,
            None,
            ControlMode::ConfidenceDefaults,
            vec![],
        )])
        .await;
        let req = no_confirm(level_put(
            "H",
            ConfidenceLevel::Low,
            ControlMode::Live,
            vec![SUSPEND],
        ));
        assert_eq!(code(set(&ac, req).await), "confirm_required");

        let (ac, _) = loaded(&[record(
            "H",
            Some("clf"),
            Some("band"),
            ControlMode::Recording,
            vec![],
        )])
        .await;
        let req = ActionControlPutRequest {
            classifier: Some("clf".into()),
            ..no_confirm(level_put(
                "H",
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            ))
        };
        assert_eq!(code(set(&ac, req).await), "confirm_required");

        let (ac, _) = loaded(&[level_record(
            "OldH",
            None,
            ConfidenceLevel::High,
            ControlMode::Recording,
            vec![],
        )])
        .await;
        let mut cfg = on();
        cfg.aliases.insert("NewH".into(), "OldH".into());
        let req = ActionControlPutRequest {
            classifier: Some("clf".into()),
            ..no_confirm(level_put(
                "NewH",
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            ))
        };
        let got = ac
            .set(
                req,
                "operator@example.com",
                "pv2",
                PUT_NOW,
                &cfg,
                &baked_safety(),
            )
            .await;
        assert_eq!(code(got), "confirm_required");

        let (ac, _) = loaded(&[
            level_record(
                "ZNew",
                None,
                ConfidenceLevel::High,
                ControlMode::Recording,
                vec![],
            ),
            level_record(
                "AOld",
                None,
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            ),
        ])
        .await;
        let mut cfg = on();
        cfg.aliases.insert("ZNew".into(), "AOld".into());
        let req = ActionControlPutRequest {
            classifier: Some("clf".into()),
            ..no_confirm(level_put(
                "ZNew",
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            ))
        };
        let got = ac
            .set(
                req,
                "operator@example.com",
                "pv2",
                PUT_NOW,
                &cfg,
                &baked_safety(),
            )
            .await;
        assert_eq!(code(got), "confirm_required");

        for (new, old, requested) in [
            ("AliasHead", CSE_HEADS[0], "AliasHead"),
            (CSE_HEADS[0], "AliasHead", "AliasHead"),
        ] {
            let (ac, _) = loaded(&[]).await;
            let mut cfg = on();
            cfg.aliases.insert(new.into(), old.into());
            let req = no_confirm(level_put(
                requested,
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            ));
            let got = ac
                .set(
                    req,
                    "operator@example.com",
                    "pv2",
                    PUT_NOW,
                    &cfg,
                    &baked_safety(),
                )
                .await;
            assert_eq!(code(got), "cse_recording_only", "{new} -> {old}");
        }
    }

    #[tokio::test]
    async fn v5_round3_cse_aliases_are_refused_in_both_directions() {
        for (new, old, requested) in [
            ("AliasHead", CSE_HEADS[0], "AliasHead"),
            (CSE_HEADS[0], "AliasHead", "AliasHead"),
        ] {
            let (ac, _) = loaded(&[]).await;
            let mut cfg = on();
            cfg.aliases.insert(new.into(), old.into());
            let req = ActionControlPutRequest {
                confirm: None,
                ..level_put(
                    requested,
                    ConfidenceLevel::High,
                    ControlMode::Live,
                    vec![SUSPEND],
                )
            };
            let got = ac
                .set(
                    req,
                    "operator@example.com",
                    "pv2",
                    PUT_NOW,
                    &cfg,
                    &baked_safety(),
                )
                .await;
            assert_eq!(code(got), "cse_recording_only", "{new} -> {old}");
        }
    }

    fn level_rec(head: &str, mode: ControlMode, actions: Vec<PageAction>) -> ActionControlRecord {
        ActionControlRecord {
            confidence: Some(ConfidenceLevel::High),
            ..record(head, None, None, mode, actions)
        }
    }

        fn decided_lkey(ac: &ActionControl, cfg: &ControlConfig, head: &str) -> Option<String> {
        let ls: Vec<String> = Vec::new();
        let input = ResolveInput {
            confidence: Some(crate::confidence::Claim {
                level: ConfidenceLevel::High,
                source: crate::confidence::Source::Producer,
                evidence: Some(read(0.96, 59, 0)),
                now_ms: crate::now_millis(),
            }),
            ..input(head, "clf", &ls)
        };
        ac.resolve(&input, cfg, |_| true).matched.map(|m| m.lkey)
    }

    async fn put_with(
        ac: &ActionControl,
        cfg: &ControlConfig,
        req: ActionControlPutRequest,
    ) -> Result<SetOutcome, SetError> {
        ac.set(
            req,
            "operator@example.com",
            "pv2",
            PUT_NOW,
            cfg,
            &baked_safety(),
        )
        .await
    }

    #[tokio::test]
    async fn renamed_heads_are_compared_as_the_decision_path_resolves_them() {
        let classifier_suspend = |head: &str| ActionControlPutRequest {
            classifier: Some("clf".into()),
            confirm: None,
            ..level_put(
                head,
                ConfidenceLevel::High,
                ControlMode::Live,
                vec![SUSPEND],
            )
        };
        for (new, old, new_mode, old_mode, harsher) in [
            (
                "ZNew",
                "AOld",
                ControlMode::Recording,
                ControlMode::Live,
                true,
            ),
            (
                "ANew",
                "ZOld",
                ControlMode::Recording,
                ControlMode::Live,
                true,
            ),
            (
                "ZNew",
                "AOld",
                ControlMode::Live,
                ControlMode::Recording,
                false,
            ),
            (
                "ANew",
                "ZOld",
                ControlMode::Live,
                ControlMode::Recording,
                false,
            ),
        ] {
            let actions = |m| {
                if m == ControlMode::Live {
                    vec![SUSPEND]
                } else {
                    vec![]
                }
            };
            let (ac, _) = loaded(&[
                level_rec(new, new_mode, actions(new_mode)),
                level_rec(old, old_mode, actions(old_mode)),
            ])
            .await;
            let mut cfg = on();
            cfg.aliases.insert(new.into(), old.into());
            let label = format!("{new} {new_mode:?} / {old} {old_mode:?}");
            assert_eq!(
                decided_lkey(&ac, &cfg, new).as_deref(),
                Some(format!("user/l/{new}/high").as_str()),
                "{label}"
            );
            let got = put_with(&ac, &cfg, classifier_suspend(new)).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{label}");
            } else {
                assert!(got.is_ok(), "{label}: {got:?}");
            }
        }
    }

    #[tokio::test]
    async fn a_first_head_level_write_counts_the_classifier_records_it_takes_over() {
        let head_suspends = record("H", None, None, ControlMode::Live, vec![SUSPEND]);
        let head_level_suspend = ActionControlPutRequest {
            confirm: None,
            ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
        };
        for (taken_over, harsher) in [
            (
                record("H", Some("clf"), None, ControlMode::Recording, vec![]),
                true,
            ),
            (
                record("H", Some("clf"), Some("b1"), ControlMode::Recording, vec![]),
                true,
            ),
            (
                record("H", Some("clf"), None, ControlMode::Live, vec![SUSPEND]),
                false,
            ),
        ] {
            let label = taken_over.scope().to_owned() + &format!(" {:?}", taken_over.mode);
            let (ac, _) = loaded(&[head_suspends.clone(), taken_over]).await;
            let got = set(&ac, head_level_suspend.clone()).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{label}");
            } else {
                assert!(got.is_ok(), "{label}: {got:?}");
            }
        }
    }

    #[tokio::test]
    async fn a_three_name_rename_chain_is_compared_one_hop_like_the_decision_path() {
        for (a_mode, b_mode, harsher) in [
            (ControlMode::Live, ControlMode::Recording, true),
            (ControlMode::Recording, ControlMode::Live, false),
        ] {
            let actions = |m| {
                if m == ControlMode::Live {
                    vec![SUSPEND]
                } else {
                    vec![]
                }
            };
            let (ac, _) = loaded(&[
                level_rec("A", a_mode, actions(a_mode)),
                level_rec("B", b_mode, actions(b_mode)),
            ])
            .await;
            let mut cfg = on();
            cfg.aliases.insert("C".into(), "B".into());
            cfg.aliases.insert("B".into(), "A".into());
            assert_eq!(
                decided_lkey(&ac, &cfg, "C").as_deref(),
                Some("user/l/B/high")
            );
            let req = ActionControlPutRequest {
                classifier: Some("clf".into()),
                confirm: None,
                ..level_put("C", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
            };
            let got = put_with(&ac, &cfg, req).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "A {a_mode:?} B {b_mode:?}");
            } else {
                assert!(got.is_ok(), "A {a_mode:?} B {b_mode:?}: {got:?}");
            }
        }
    }

    #[tokio::test]
    async fn band_scopes_of_a_renamed_head_are_compared_like_the_decision_path() {
        for (new_mode, old_mode, harsher) in [
            (ControlMode::Recording, ControlMode::Live, true),
            (ControlMode::Live, ControlMode::Recording, false),
        ] {
            let band = |head: &str, mode| {
                let actions = if mode == ControlMode::Live {
                    vec![SUSPEND]
                } else {
                    vec![]
                };
                record(head, None, Some("b1"), mode, actions)
            };
            let (ac, _) = loaded(&[band("ZNew", new_mode), band("AOld", old_mode)]).await;
            let mut cfg = on();
            cfg.aliases.insert("ZNew".into(), "AOld".into());
            let ls = vec!["b1".to_owned()];
            let decided = ac
                .resolve(&input("ZNew", "clf", &ls), &cfg, |_| true)
                .matched
                .map(|m| m.lkey);
            assert_eq!(decided.as_deref(), Some("user/b/ZNew/b1"));
            let req = ActionControlPutRequest {
                classifier: Some("clf".into()),
                band: Some("b1".into()),
                confirm: None,
                ..put("ZNew", ControlMode::Live, vec![SUSPEND])
            };
            let got = put_with(&ac, &cfg, req).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{new_mode:?}");
            } else {
                assert!(got.is_ok(), "{new_mode:?}: {got:?}");
            }
        }
    }

    #[tokio::test]
    async fn v5_round5_one_suspending_classifier_cannot_hide_harsher_head_level_write() {
        let head_liveness = record("H", None, None, ControlMode::Live, vec![LIVENESS]);
        let narrow =
            |band: Option<&str>| record("H", Some("clf-a"), band, ControlMode::Live, vec![SUSPEND]);
        for taken_over in [narrow(None), narrow(Some("b1"))] {
            let scope = taken_over.scope().to_owned();
            let (ac, _) = loaded(&[head_liveness.clone(), taken_over]).await;
            let req = ActionControlPutRequest {
                confirm: None,
                ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
            };
            assert_eq!(
                code(set(&ac, req).await),
                "confirm_required",
                "{scope} suspend must not hide that every other classifier goes from liveness to suspend"
            );
        }
    }

    #[tokio::test]
    async fn v5_round5_one_suspending_classifier_band_cannot_hide_harsher_head_level_write() {
        let (ac, _) = loaded(&[
            record("H", None, None, ControlMode::Live, vec![LIVENESS]),
            record(
                "H",
                Some("clf-a"),
                Some("b1"),
                ControlMode::Live,
                vec![SUSPEND],
            ),
        ])
        .await;
        let req = ActionControlPutRequest {
            confirm: None,
            ..level_put("H", ConfidenceLevel::High, ControlMode::Live, vec![SUSPEND])
        };
        assert_eq!(
            code(set(&ac, req).await),
            "confirm_required",
            "one classifier+band suspend must not hide that every other classifier goes from liveness to suspend"
        );
    }

    #[tokio::test]
    async fn a_broad_write_needs_confirmation_if_harsher_than_any_record_it_takes_over() {
        let head_level =
            |actions: Vec<PageAction>, confirm: Option<&str>| ActionControlPutRequest {
                confirm: confirm.map(Into::into),
                ..level_put("H", ConfidenceLevel::High, ControlMode::Live, actions)
            };
        let (ac, _) = loaded(&[
            record("H", None, None, ControlMode::Live, vec![LIVENESS]),
            record("H", Some("clf-a"), None, ControlMode::Live, vec![SUSPEND]),
            record("H", Some("clf-b"), None, ControlMode::Recording, vec![]),
        ])
        .await;
        assert_eq!(
            code(set(&ac, head_level(vec![SUSPEND], None)).await),
            "confirm_required"
        );
        let out = set(&ac, head_level(vec![SUSPEND], Some(CONFIRM_WORD)))
            .await
            .unwrap();
        assert!(
            out.harsher_because.iter().any(|w| w == "adds suspend"),
            "{:?}",
            out.harsher_because
        );
        assert!(
            out.harsher_because
                .iter()
                .any(|w| w == "turns the scope live"),
            "{:?}",
            out.harsher_because
        );

        for (band_mode, harsher) in [(ControlMode::Recording, true), (ControlMode::Live, false)] {
            let band_actions = if band_mode == ControlMode::Live {
                vec![LIVENESS]
            } else {
                vec![]
            };
            let (ac, _) = loaded(&[
                record("H", None, None, ControlMode::Live, vec![LIVENESS]),
                record("H", Some("clf-a"), Some("b1"), band_mode, band_actions),
            ])
            .await;
            let got = set(&ac, head_level(vec![LIVENESS], None)).await;
            if harsher {
                assert_eq!(code(got), "confirm_required", "{band_mode:?}");
            } else {
                assert!(got.is_ok(), "{band_mode:?}: {got:?}");
            }
        }

        let (ac, _) = loaded(&[
            record("H", Some("clf-a"), None, ControlMode::Live, vec![SUSPEND]),
            record("H", Some("clf-b"), None, ControlMode::Recording, vec![]),
        ])
        .await;
        let head = ActionControlPutRequest {
            confirm: None,
            ..put("H", ControlMode::Live, vec![SUSPEND])
        };
        assert_eq!(code(set(&ac, head.clone()).await), "confirm_required");
        let (ac, _) = loaded(&[
            record("OldH", None, None, ControlMode::Live, vec![SUSPEND]),
            record("H", Some("clf-b"), None, ControlMode::Recording, vec![]),
        ])
        .await;
        let mut cfg = on();
        cfg.aliases.insert("H".into(), "OldH".into());
        let got = put_with(&ac, &cfg, head).await;
        assert!(got.is_ok(), "{got:?}");
    }
}
