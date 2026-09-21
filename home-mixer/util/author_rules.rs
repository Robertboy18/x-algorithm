use moka::sync::SegmentedCache;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use xai_feature_switches::{
    FeatureSwitches, FromFeatureSwitch, FromValue, Param, RecipientBuilder,
};

const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const CACHE_SIZE: u64 = 500_000;
const NUM_SHARDS: usize = 64;

fn impress_cache_key(author_id: u64, experiment: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    author_id.hash(&mut hasher);
    experiment.hash(&mut hasher);
    hasher.finish()
}

pub struct AuthorRulesEvaluator {
    feature_switches: Arc<FeatureSwitches>,
    impressed: SegmentedCache<u64, ()>,
}

impl AuthorRulesEvaluator {
    pub fn new(feature_switches: Arc<FeatureSwitches>) -> Self {
        Self {
            feature_switches,
            impressed: SegmentedCache::builder(NUM_SHARDS)
                .max_capacity(CACHE_SIZE)
                .time_to_live(CACHE_TTL)
                .build(),
        }
    }

    pub fn get<T: FromFeatureSwitch + FromValue>(&self, author_id: u64, p: impl Param<T>) -> T {
        let recipient = RecipientBuilder::new().user_id(author_id).build();
        let result = self.feature_switches.eval_author_rules(&recipient);

        if let Some(experiment) = result.experiment_for_key(p.key()) {
            let first_touch = AtomicBool::new(false);
            self.impressed
                .get_with(impress_cache_key(author_id, experiment), || {
                    first_touch.store(true, Ordering::Relaxed);
                });
            if first_touch.load(Ordering::Relaxed) {
                result.log_impression_for_key(p.key());
            }
        }

        result.param(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::param;
    use xai_feature_switches::{
        BucketMembership, ExperimentBucket, FeatureSwitches, MockExperimentBucketsChooser,
        SpyingBucketImpressor,
    };

    param!(TestBoost, f64, "rust_home_mixer_boost", 1.0);
    param!(TestCap, f64, "rust_home_mixer_cap", 10.0);

    fn build_fs(
        yaml: &str,
        buckets: BucketMembership,
        impressor: Arc<SpyingBucketImpressor>,
    ) -> Arc<FeatureSwitches> {
        Arc::new(
            FeatureSwitches::with_options(
                xai_feature_switches::load_yaml_string(yaml).unwrap(),
                Arc::new(MockExperimentBucketsChooser::with_buckets(buckets)),
                impressor,
                None,
                false,
            )
            .unwrap(),
        )
    }

    const YAML: &str = r#"
rust_home_mixer:
  parameters:
    rust_home_mixer_boost:
      type: double
      default: 1.0
  rules:
    - query: "[my_exp author_bucket_membership treatment]"
      values:
        rust_home_mixer_boost: 1.5
"#;

    const MULTI_EXP_YAML: &str = r#"
rust_home_mixer:
  parameters:
    rust_home_mixer_boost:
      type: double
      default: 1.0
    rust_home_mixer_cap:
      type: double
      default: 10.0
  rules:
    - query: "[exp_alpha author_bucket_membership treatment]"
      values:
        rust_home_mixer_boost: 1.5
    - query: "[exp_beta author_bucket_membership control]"
      values:
        rust_home_mixer_cap: 20.0
"#;

    #[test]
    fn impression_once_per_author_experiment() {
        let mut buckets = BucketMembership::new();
        buckets.add(ExperimentBucket::new("my_exp", "treatment").with_version(1));
        let impressor = Arc::new(SpyingBucketImpressor::new());
        let evaluator = AuthorRulesEvaluator::new(build_fs(YAML, buckets, impressor.clone()));

        assert_eq!(evaluator.get(42, TestBoost), 1.5);
        assert_eq!(evaluator.get(42, TestBoost), 1.5);
        assert_eq!(impressor.impression_count("my_exp"), 1);
    }

    #[test]
    fn unbucketed_author_gets_fs_default() {
        let impressor = Arc::new(SpyingBucketImpressor::new());
        let evaluator =
            AuthorRulesEvaluator::new(build_fs(YAML, BucketMembership::new(), impressor.clone()));

        assert_eq!(evaluator.get(42, TestBoost), 1.0);
        assert_eq!(impressor.impression_count("my_exp"), 0);
    }

    #[test]
    fn impression_per_experiment_for_same_author() {
        let mut buckets = BucketMembership::new();
        buckets.add(ExperimentBucket::new("exp_alpha", "treatment").with_version(1));
        buckets.add(ExperimentBucket::new("exp_beta", "control").with_version(1));
        let impressor = Arc::new(SpyingBucketImpressor::new());
        let evaluator =
            AuthorRulesEvaluator::new(build_fs(MULTI_EXP_YAML, buckets, impressor.clone()));

        assert_eq!(evaluator.get(42, TestBoost), 1.5);
        assert_eq!(evaluator.get(42, TestCap), 20.0);
        assert_eq!(impressor.impression_count("exp_alpha"), 1);
        assert_eq!(impressor.impression_count("exp_beta"), 1);

        assert_eq!(evaluator.get(42, TestBoost), 1.5);
        assert_eq!(evaluator.get(42, TestCap), 20.0);
        assert_eq!(impressor.impression_count("exp_alpha"), 1);
        assert_eq!(impressor.impression_count("exp_beta"), 1);
    }
}
