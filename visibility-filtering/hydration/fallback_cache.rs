use std::sync::atomic::{AtomicU64, Ordering};

use quick_cache::sync::Cache;
use quick_cache::OptionsBuilder;

use crate::hydration::batch::{Hydrated, RawHydrationBatch};
use crate::hydration::metrics::{record_fallback_cache_entries, record_fallback_cache_keys};

const CACHE_SHARDS: usize = 64;
const OCCUPANCY_SAMPLE_INTERVAL: u64 = 1024;

pub(crate) struct FallbackCache<V> {
    facet: &'static str,
    entries: Cache<u64, V>,
    resolved_batches: AtomicU64,
}

impl<V: Clone> FallbackCache<V> {
    #[expect(
        clippy::expect_used,
        reason = "both options `build` requires are set, so it cannot fail"
    )]
    pub(crate) fn new(facet: &'static str, capacity: usize) -> Self {
        let options = OptionsBuilder::new()
            .shards(CACHE_SHARDS)
            .estimated_items_capacity(capacity)
            .weight_capacity(u64::try_from(capacity).unwrap_or(u64::MAX))
            .build()
            .expect("capacity options are set");
        let entries = Cache::with_options(
            options,
            Default::default(),
            Default::default(),
            Default::default(),
        );
        entries.reserve(capacity);
        Self {
            facet,
            entries,
            resolved_batches: AtomicU64::new(0),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_capacity(facet: &'static str) -> Self {
        Self::new(facet, 8)
    }

    pub(crate) fn resolve_hydration_batch(
        &self,
        batch: RawHydrationBatch<V>,
    ) -> RawHydrationBatch<V> {
        let mut fresh = 0;
        let mut stale = 0;
        let mut not_found = 0;
        let mut partial = 0;
        let mut unavailable = 0;
        let resolved = batch
            .into_hydrated()
            .into_iter()
            .map(|(key, hydrated)| {
                let hydrated = match hydrated {
                    Hydrated::Found(value) => {
                        self.entries.insert(key, value.clone());
                        fresh += 1;
                        Hydrated::Found(value)
                    }
                    Hydrated::NotFound => {
                        self.entries.remove(&key);
                        not_found += 1;
                        Hydrated::NotFound
                    }
                    Hydrated::Partial(value) => {
                        partial += 1;
                        Hydrated::Partial(value)
                    }
                    Hydrated::Failed(error) => match self.entries.get(&key) {
                        Some(value) => {
                            stale += 1;
                            Hydrated::Found(value)
                        }
                        None => {
                            unavailable += 1;
                            Hydrated::Failed(error)
                        }
                    },
                };
                (key, hydrated)
            })
            .collect();

        record_fallback_cache_keys(self.facet, fresh, stale, not_found, partial, unavailable);
        if self
            .resolved_batches
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(OCCUPANCY_SAMPLE_INTERVAL)
        {
            record_fallback_cache_entries(self.facet, self.entries.len());
        }
        RawHydrationBatch::from_hydrated(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::{HydrationBatch, HydrationError};

    fn cache() -> FallbackCache<String> {
        FallbackCache::with_test_capacity("test")
    }

    fn batch(
        entries: impl IntoIterator<Item = (u64, Hydrated<String>)>,
    ) -> HydrationBatch<u64, String> {
        HydrationBatch::from_hydrated(entries.into_iter().collect())
    }

    fn failed() -> Hydrated<String> {
        Hydrated::Failed(HydrationError::Timeout)
    }

    #[test]
    fn recovers_only_resident_failed_keys() {
        let cache = cache();
        cache.resolve_hydration_batch(batch([(1, Hydrated::Found("cached".to_string()))]));

        let resolved = cache.resolve_hydration_batch(batch([
            (1, failed()),
            (2, failed()),
            (3, Hydrated::Found("fresh".to_string())),
        ]));

        assert_eq!(resolved.get(&1), Some(&"cached".to_string()));
        assert!(matches!(resolved.hydrated(&2), Some(Hydrated::Failed(_))));
        assert_eq!(resolved.get(&3), Some(&"fresh".to_string()));
    }

    #[test]
    fn a_later_value_replaces_the_cached_one() {
        let cache = cache();
        cache.resolve_hydration_batch(batch([(1, Hydrated::Found("old".to_string()))]));
        cache.resolve_hydration_batch(batch([(1, Hydrated::Found("new".to_string()))]));

        let failed = cache.resolve_hydration_batch(batch([(1, failed())]));

        assert_eq!(failed.get(&1), Some(&"new".to_string()));
    }

    #[test]
    fn a_partial_answer_keeps_the_complete_entry_for_a_later_failure() {
        let cache = cache();
        cache.resolve_hydration_batch(batch([(1, Hydrated::Found("complete".to_string()))]));

        let partial =
            cache.resolve_hydration_batch(batch([(1, Hydrated::Partial("partial".to_string()))]));
        let later = cache.resolve_hydration_batch(batch([(1, failed())]));

        assert_eq!(
            partial.hydrated(&1),
            Some(&Hydrated::Partial("partial".to_string()))
        );
        assert_eq!(
            later.hydrated(&1),
            Some(&Hydrated::Found("complete".to_string()))
        );
    }

    #[test]
    fn a_partial_answer_creates_no_entry() {
        let cache = cache();
        cache.resolve_hydration_batch(batch([(1, Hydrated::Partial("partial".to_string()))]));

        let later = cache.resolve_hydration_batch(batch([(1, failed())]));

        assert_eq!(later.hydrated(&1), Some(&failed()));
    }

    #[test]
    fn authoritative_not_found_invalidates_stale_value() {
        let cache = cache();
        cache.resolve_hydration_batch(batch([(1, Hydrated::Found("cached".to_string()))]));

        cache.resolve_hydration_batch(batch([(1, Hydrated::NotFound)]));
        let failed = cache.resolve_hydration_batch(batch([(1, failed())]));

        assert!(matches!(failed.hydrated(&1), Some(Hydrated::Failed(_))));
    }
}
