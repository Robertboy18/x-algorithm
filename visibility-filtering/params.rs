use arc_swap::ArcSwap;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use xai_feature_switches::{FeatureSwitches, RecipientBuilder, Value};

pub const NSFW_GATING_COUNTRIES_KEY: &str = "rust_vf_nsfw_gating_countries";

const SCALA_NSFW_GATING_FILE: &str = "country_specific_nsfw_content_gating.yml";
const SCALA_NSFW_GATING_COUNTRIES_KEY: &str = "country_specific_nsfw_content_gating_countries";

const DRIFT_COUNTER: &str = "nsfw_gating_countries_drift";

pub fn default_nsfw_gating_countries() -> Vec<String> {
    [
        "ar", "au", "br", "ca", "de", "es", "fr", "gb", "id", "it", "kr", "mx", "nl", "ph", "pt",
        "th",
    ]
    .map(str::to_string)
    .to_vec()
}

pub(crate) struct NsfwGatingCountries {
    countries: ArcSwap<Vec<String>>,
}

impl NsfwGatingCountries {
    pub fn starting_at_default() -> Self {
        Self {
            countries: ArcSwap::from_pointee(default_nsfw_gating_countries()),
        }
    }

    pub fn contains(&self, country_code: &str) -> bool {
        self.countries.load().iter().any(|c| c == country_code)
    }

    #[cfg(test)]
    pub fn refresh_from(&self, feature_switches: &FeatureSwitches) {
        let (_, resolved) = resolve_with_origin(feature_switches);
        self.countries.store(Arc::new(resolved));
    }

    pub fn refresh_and_check_drift(&self, feature_switches: &FeatureSwitches, fs_path: &str) {
        let (origin, resolved) = resolve_with_origin(feature_switches);
        self.countries.store(Arc::new(resolved.clone()));
        check_drift(origin, &resolved, fs_path);
    }

    pub fn spawn_refresh(
        self: &Arc<Self>,
        feature_switches: Arc<FeatureSwitches>,
        fs_path: String,
    ) {
        let cache = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                cache.refresh_and_check_drift(&feature_switches, &fs_path);
            }
        });
    }
}

fn lowercased_codes(values: &[Value]) -> Option<Vec<String>> {
    values
        .iter()
        .map(|v| v.as_str().map(str::to_ascii_lowercase))
        .collect()
}

fn resolve_with_origin(feature_switches: &FeatureSwitches) -> (&'static str, Vec<String>) {
    let configured = feature_switches
        .match_recipient(&RecipientBuilder::new().build())
        .get_array_no_impression(NSFW_GATING_COUNTRIES_KEY)
        .and_then(|values| lowercased_codes(values));
    match configured {
        Some(list) => ("config", list),
        None => ("default", default_nsfw_gating_countries()),
    }
}

fn check_drift(origin: &str, resolved: &[String], fs_path: &str) {
    let default = default_nsfw_gating_countries();
    let matches_default = set_eq(resolved, &default);
    let Some(scala_list) = scala_nsfw_gating_countries(fs_path) else {
        tracing::debug!(
            fs_path,
            scala_file = SCALA_NSFW_GATING_FILE,
            "scala gating file absent or unparsable; skipping drift check"
        );
        return;
    };
    let matches_scala = set_eq(resolved, &scala_list);
    if matches_scala {
        tracing::info!(
            key = NSFW_GATING_COUNTRIES_KEY,
            origin,
            resolved = ?resolved,
            scala_list = ?scala_list,
            matches_scala,
            matches_default,
            "nsfw gating countries: drift check"
        );
    } else {
        tracing::warn!(
            key = NSFW_GATING_COUNTRIES_KEY,
            origin,
            resolved = ?resolved,
            scala_list = ?scala_list,
            matches_scala,
            matches_default,
            "nsfw gating countries: drift check"
        );
        if let Some(sr) = xai_stats_receiver::global_stats_receiver() {
            sr.incr(DRIFT_COUNTER, &[], 1);
        }
    }
}

fn scala_nsfw_gating_countries(fs_path: &str) -> Option<Vec<String>> {
    let path = Path::new(fs_path).parent()?.join(SCALA_NSFW_GATING_FILE);
    let features = xai_feature_switches::load_yaml_file(&path).ok()?;
    features
        .iter()
        .find_map(|f| f.parameters.get(SCALA_NSFW_GATING_COUNTRIES_KEY))
        .and_then(|param| param.default_value.as_array())
        .and_then(|values| lowercased_codes(values))
}

fn set_eq(a: &[String], b: &[String]) -> bool {
    a.iter().collect::<BTreeSet<_>>() == b.iter().collect::<BTreeSet<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(yaml: &str) -> FeatureSwitches {
        FeatureSwitches::load_string(yaml).unwrap()
    }

    #[test]
    fn refresh_reads_key_and_fails_open() {
        let cache = NsfwGatingCountries::starting_at_default();
        assert!(cache.contains("de"));
        assert!(!cache.contains("xx"));

        cache.refresh_from(&engine(
            r#"
rust_vf:
  parameters:
    rust_vf_nsfw_gating_countries:
      type: array
      default:
      - "XX"
"#,
        ));
        assert!(cache.contains("xx"));
        assert!(!cache.contains("de"));

        cache.refresh_from(&engine("other:\n  parameters: {}\n"));
        assert!(cache.contains("de"));
        assert!(!cache.contains("xx"));
    }

    #[test]
    fn malformed_value_falls_back_whole_not_partial() {
        let cache = NsfwGatingCountries::starting_at_default();
        cache.refresh_from(&engine(
            r#"
rust_vf:
  parameters:
    rust_vf_nsfw_gating_countries:
      type: array
      default:
      - "xx"
      - 7
"#,
        ));
        assert!(!cache.contains("xx"));
        assert!(cache.contains("de"));
    }

    #[test]
    fn scala_list_parses_from_sibling_file_and_tolerates_absence() {
        let dir = tempfile::tempdir().unwrap();
        let fs_path = dir.path().join("rust_vf.yml");
        assert_eq!(scala_nsfw_gating_countries(fs_path.to_str().unwrap()), None);

        std::fs::write(
            dir.path().join(SCALA_NSFW_GATING_FILE),
            r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_countries:
      type: array
      default:
      - "de"
      - "FR"
"#,
        )
        .unwrap();
        assert_eq!(
            scala_nsfw_gating_countries(fs_path.to_str().unwrap()),
            Some(vec!["de".to_string(), "fr".to_string()])
        );
    }
}
