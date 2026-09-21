pub const ENV_DUAL_CALL_HARNESS_ENABLED: &str = "VF_DUAL_CALL_HARNESS_ENABLED";
pub const ENV_FALLBACK_CACHE_ENABLED: &str = "VF_FALLBACK_CACHE_ENABLED";
pub const ENV_CACHE_WARM_ENABLED: &str = "VF_CACHE_WARM_ENABLED";
pub const ENV_APP_ENV: &str = "APP_ENV";
pub const ENV_FS_PATH: &str = "VF_FS_PATH";
pub const ENV_GIZMODUCK_CLIENT_ID: &str = "VF_GIZMODUCK_CLIENT_ID";
pub const ENV_TWEMCACHE_CLIENT_NAME: &str = "VF_TWEMCACHE_CLIENT_NAME";

const DEFAULT_FS_PATH: &str = "/usr/local/config/features/visibility/main/rust_vf.yml";

pub fn fs_path() -> String {
    std::env::var(ENV_FS_PATH).unwrap_or_else(|_| DEFAULT_FS_PATH.to_string())
}

pub(crate) fn gizmoduck_client_id() -> String {
    resolve_gizmoduck_client_id(
        std::env::var(ENV_GIZMODUCK_CLIENT_ID).ok().as_deref(),
        std::env::var(ENV_APP_ENV).ok().as_deref(),
    )
}

pub(crate) fn twemcache_client_name() -> String {
    resolve_twemcache_client_name(std::env::var(ENV_TWEMCACHE_CLIENT_NAME).ok().as_deref())
}

pub fn resolve_gizmoduck_client_id(configured: Option<&str>, app_env: Option<&str>) -> String {
    match configured {
        Some(id) => id.to_string(),
        None => format!("visibility-filtering-service.{}", app_env.unwrap_or("prod")),
    }
}

pub fn resolve_twemcache_client_name(configured: Option<&str>) -> String {
    configured
        .unwrap_or("visibility-filtering-service")
        .to_string()
}

pub(crate) fn dual_call_harness_enabled() -> bool {
    parse_env_flag(std::env::var(ENV_DUAL_CALL_HARNESS_ENABLED).ok().as_deref())
}

pub(crate) fn fallback_cache_enabled() -> bool {
    parse_env_flag(std::env::var(ENV_FALLBACK_CACHE_ENABLED).ok().as_deref())
}

pub(crate) fn author_id_fallback_enabled() -> bool {
    std::env::var("VF_AUTHOR_ID_FALLBACK_ENABLED")
        .ok()
        .is_none_or(|value| parse_env_flag(Some(&value)))
}

pub(crate) fn author_id_fallback_capacity() -> usize {
    std::env::var("VF_AUTHOR_ID_FALLBACK_CAPACITY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|capacity| *capacity > 0)
        .unwrap_or(1_000_000)
}

pub(crate) fn cache_warm_enabled() -> bool {
    parse_env_flag(std::env::var(ENV_CACHE_WARM_ENABLED).ok().as_deref())
}

fn parse_env_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_env_flag, resolve_gizmoduck_client_id, resolve_twemcache_client_name};

    #[test]
    fn client_ids_default_to_historical_values_and_overrides_win() {
        assert_eq!(
            resolve_gizmoduck_client_id(None, Some("prod")),
            "visibility-filtering-service.prod"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(None, Some("staging")),
            "visibility-filtering-service.staging"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(None, None),
            "visibility-filtering-service.prod"
        );
        assert_eq!(
            resolve_twemcache_client_name(None),
            "visibility-filtering-service"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(Some("xai-vf-service.staging"), Some("staging")),
            "xai-vf-service.staging"
        );
    }

    #[test]
    fn parses_enabled_environment_values() {
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert!(parse_env_flag(Some(value)), "{value}");
        }
    }

    #[test]
    fn missing_or_disabled_environment_values_are_off() {
        assert!(!parse_env_flag(None));
        for value in ["", "0", "false", "FALSE", "no", "off", "other"] {
            assert!(!parse_env_flag(Some(value)), "{value}");
        }
    }
}
