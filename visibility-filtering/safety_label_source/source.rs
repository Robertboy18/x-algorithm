use quanta::Clock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::info;
use xai_visibility_filtering_proto as vf_pb;

use super::expiring_cache::{ExpiringCache, Lookup};
use super::lookup::{LookupError, RemoteSource};
use super::metrics::{self, BatchStage, CacheResult, CacheTier};

const CACHE_CAPACITY: usize = 1_000_000;

const SNOWFLAKE_EPOCH_MS: u64 = 1288834974657;
const SNOWFLAKE_TIMESTAMP_SHIFT: u32 = 22;
const YOUNG_TWEET_AGE: Duration = Duration::from_secs(5 * 60);
const SHORT_TTL: Duration = Duration::from_secs(30);
const LONG_TTL: Duration = Duration::from_secs(60);

fn tweet_age(tweet_id: u64, now: SystemTime) -> Option<Duration> {
    let created_ms = (tweet_id >> SNOWFLAKE_TIMESTAMP_SHIFT).checked_add(SNOWFLAKE_EPOCH_MS)?;
    let created_at = UNIX_EPOCH.checked_add(Duration::from_millis(created_ms))?;
    now.duration_since(created_at).ok()
}

fn ttl_for_tweet(tweet_id: u64, now: SystemTime) -> Option<Duration> {
    let age = tweet_age(tweet_id, now)?;
    if age < YOUNG_TWEET_AGE {
        Some(SHORT_TTL.min(YOUNG_TWEET_AGE - age))
    } else {
        Some(LONG_TTL)
    }
}

pub struct SafetyLabelSource {
    cache: ExpiringCache<u64, Arc<vf_pb::SafetyLabelMap>>,
    remote: Arc<RemoteSource>,
}

impl SafetyLabelSource {
    pub(crate) fn new(remote: Arc<RemoteSource>) -> Self {
        Self::with_clock(remote, Clock::new())
    }

    fn with_clock(remote: Arc<RemoteSource>, clock: Clock) -> Self {
        info!(
            cache_capacity = CACHE_CAPACITY,
            young_ttl_secs = SHORT_TTL.as_secs(),
            old_ttl_secs = LONG_TTL.as_secs(),
            age_threshold_secs = YOUNG_TWEET_AGE.as_secs(),
            "Local safety-label cache initialized (single pool, per-entry TTL)"
        );
        Self {
            cache: ExpiringCache::with_clock(CACHE_CAPACITY, clock),
            remote,
        }
    }

    pub async fn get(
        &self,
        ids: &[u64],
    ) -> HashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>> {
        let total = ids.len();
        let mut results: HashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>> =
            HashMap::with_capacity(total);

        let (local_misses, expired) = self.get_local(ids, &mut results);
        let batch_size = local_misses.len();

        let remote_results = self.remote.get(&local_misses).await;
        self.backfill_local(remote_results, &mut results);

        self.emit_stats(total - batch_size, batch_size, expired);

        results
    }

    fn get_local(
        &self,
        ids: &[u64],
        results: &mut HashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
    ) -> (Vec<u64>, usize) {
        let mut misses = Vec::with_capacity(ids.len());
        let mut expired = 0;
        for &tweet_id in ids {
            match self.cache.get(&tweet_id) {
                Lookup::Found(labels) => {
                    results.insert(tweet_id, Ok(labels));
                }
                Lookup::Expired => {
                    expired += 1;
                    misses.push(tweet_id);
                }
                Lookup::NotFound => misses.push(tweet_id),
            }
        }
        (misses, expired)
    }

    fn backfill_local(
        &self,
        remote_results: HashMap<u64, Result<vf_pb::SafetyLabelMap, LookupError>>,
        results: &mut HashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
    ) {
        let wall_now = SystemTime::now();
        for (id, result) in remote_results {
            let result = result.map(Arc::new);
            if let Ok(label_map) = &result
                && let Some(ttl) = ttl_for_tweet(id, wall_now)
            {
                self.cache.insert(id, Arc::clone(label_map), ttl);
            }
            results.insert(id, result);
        }
    }

    fn emit_stats(&self, hits: usize, batch_size: usize, expired: usize) {
        metrics::record_cache_keys(CacheTier::Local, CacheResult::Hit, hits);
        metrics::record_cache_keys(CacheTier::Local, CacheResult::Miss, batch_size - expired);
        metrics::record_cache_keys(CacheTier::Local, CacheResult::Expired, expired);
        metrics::record_batch_size(BatchStage::LocalMiss, batch_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use tonic::async_trait;
    use xai_cache::{KVCacheError, Key, Value};
    use xai_x_thrift::tweet_safety_label::SafetyLabelType;

    use crate::safety_label_source::codec::{LkeyBytes, MvalBytes, RawSafetyLabel};
    use crate::safety_label_source::lookup::RemoteSource;
    use crate::safety_label_source::manhattan::ManhattanSource;
    use crate::safety_label_source::mh_client::{FetchResult, ManhattanLabelFetcher};
    use crate::safety_label_source::twemcache::{CacheRead, TwemcacheSource};

    struct FakeTwemcache {
        results: HashMap<Key, std::result::Result<Option<Value>, KVCacheError>>,
        fetched_keys: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl CacheRead for FakeTwemcache {
        async fn multi_get(
            &self,
            keys: &[Key],
        ) -> HashMap<Key, std::result::Result<Option<Value>, KVCacheError>> {
            self.fetched_keys.fetch_add(keys.len(), Ordering::SeqCst);
            self.results.clone()
        }
    }

    struct FakeLabelFetcher {
        items: HashMap<i64, Vec<RawSafetyLabel>>,
    }

    #[async_trait]
    impl ManhattanLabelFetcher for FakeLabelFetcher {
        async fn fetch_labels(
            &self,
            tweet_ids: &[i64],
        ) -> Result<Vec<FetchResult>, xai_manhattan::ManhattanError> {
            Ok(tweet_ids
                .iter()
                .map(|id| Ok(self.items.get(id).cloned().unwrap_or_default()))
                .collect())
        }
    }

    fn make_source(
        cache_results: HashMap<Key, std::result::Result<Option<Value>, KVCacheError>>,
        mh_items: HashMap<i64, Vec<RawSafetyLabel>>,
    ) -> SafetyLabelSource {
        make_source_with_clock(cache_results, mh_items, Clock::new())
    }

    fn make_source_with_clock(
        cache_results: HashMap<Key, std::result::Result<Option<Value>, KVCacheError>>,
        mh_items: HashMap<i64, Vec<RawSafetyLabel>>,
        clock: Clock,
    ) -> SafetyLabelSource {
        make_counting_source_with_clock(cache_results, mh_items, clock).0
    }

    fn make_counting_source(
        cache_results: HashMap<Key, std::result::Result<Option<Value>, KVCacheError>>,
        mh_items: HashMap<i64, Vec<RawSafetyLabel>>,
    ) -> (SafetyLabelSource, Arc<AtomicUsize>) {
        make_counting_source_with_clock(cache_results, mh_items, Clock::new())
    }

    fn make_counting_source_with_clock(
        cache_results: HashMap<Key, std::result::Result<Option<Value>, KVCacheError>>,
        mh_items: HashMap<i64, Vec<RawSafetyLabel>>,
        clock: Clock,
    ) -> (SafetyLabelSource, Arc<AtomicUsize>) {
        let remote_keys = Arc::new(AtomicUsize::new(0));
        let twemcache = Arc::new(TwemcacheSource::with_cache(Arc::new(FakeTwemcache {
            results: cache_results,
            fetched_keys: remote_keys.clone(),
        })));
        let manhattan = Arc::new(ManhattanSource::new(Arc::new(FakeLabelFetcher {
            items: mh_items,
        })));
        let remote = Arc::new(RemoteSource::new(twemcache, manhattan));
        (SafetyLabelSource::with_clock(remote, clock), remote_keys)
    }

    fn cache_key(tweet_id: u64) -> Key {
        Key::new(format!("slm_{tweet_id}").into_bytes()).unwrap()
    }

    fn cached_one_label() -> Vec<u8> {
        vec![
            0x0b, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x0c, 0x00, 0x03, 0x0c, 0x5f, 0xff, 0x0d,
            0x69, 0x14, 0x0c, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x01, 0xe7, 0x7c, 0x00, 0x00,
        ]
    }

    fn mval_with_score(score: f64) -> Vec<u8> {
        let mut buf = vec![0x0C, 0x00, 0x04];
        buf.extend_from_slice(&[0x04, 0x00, 0x01]);
        buf.extend_from_slice(&score.to_bits().to_be_bytes());
        buf.push(0x00);
        buf.push(0x00);
        buf
    }

    fn raw_label(lt: SafetyLabelType, mval: Vec<u8>) -> RawSafetyLabel {
        RawSafetyLabel {
            lkey: LkeyBytes(xai_safety_label_store::types::encode_lkey(lt)),
            mval: MvalBytes(mval),
        }
    }

    fn cached_value_not_found() -> Vec<u8> {
        crate::safety_label_source::cached_value::tests::cached_value_not_found()
    }

    fn tweet_id_created_at(created_at: SystemTime) -> u64 {
        let since_epoch = created_at.duration_since(UNIX_EPOCH).unwrap();
        let created_ms = since_epoch
            .as_secs()
            .checked_mul(1000)
            .and_then(|secs_ms| secs_ms.checked_add(u64::from(since_epoch.subsec_millis())))
            .unwrap();
        created_ms.checked_sub(SNOWFLAKE_EPOCH_MS).unwrap() << SNOWFLAKE_TIMESTAMP_SHIFT
    }

    fn future_tweet_id() -> u64 {
        tweet_id_created_at(SystemTime::now() + Duration::from_secs(60))
    }

    #[tokio::test]
    async fn get_backfills_remote_successes_into_l1() {
        let (source, remote_keys) = make_counting_source(
            HashMap::from([(cache_key(42), Ok(Some(cached_one_label())))]),
            HashMap::new(),
        );

        let results1 = source.get(&[42]).await;
        let first = Arc::clone(results1.get(&42).unwrap().as_ref().unwrap());
        assert_eq!(remote_keys.load(Ordering::SeqCst), 1);
        assert!(source.cache.expiry_of(&42).is_some());

        let results2 = source.get(&[42]).await;
        assert!(Arc::ptr_eq(
            results2.get(&42).unwrap().as_ref().unwrap(),
            &first
        ));
        assert_eq!(remote_keys.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn get_does_not_backfill_future_tweet_id_into_l1() {
        let tweet_id = future_tweet_id();
        let source = make_source(
            HashMap::from([(cache_key(tweet_id), Ok(Some(cached_one_label())))]),
            HashMap::new(),
        );

        let results = source.get(&[tweet_id]).await;

        assert!(results.get(&tweet_id).unwrap().is_ok());
        assert!(source.cache.expiry_of(&tweet_id).is_none());
    }

    #[tokio::test]
    async fn get_merges_remote_results() {
        let source = make_source(
            HashMap::from([
                (cache_key(10), Ok(Some(cached_one_label()))),
                (cache_key(20), Err(KVCacheError::Io("shard timeout".into()))),
                (cache_key(30), Ok(None)),
            ]),
            HashMap::from([(
                20,
                vec![raw_label(SafetyLabelType::SPAM, mval_with_score(0.7))],
            )]),
        );

        let results = source.get(&[10, 20, 30]).await;

        assert_eq!(results.len(), 3);
        assert!(results.get(&10).unwrap().is_ok());
        assert!(results.get(&20).unwrap().is_ok());
        assert!(results.get(&30).unwrap().is_ok());
    }

    #[tokio::test]
    async fn get_backfills_negative_cache_into_l1() {
        let (source, remote_keys) = make_counting_source(
            HashMap::from([(cache_key(42), Ok(Some(cached_value_not_found())))]),
            HashMap::new(),
        );

        let results1 = source.get(&[42]).await;
        assert!(
            results1
                .get(&42)
                .unwrap()
                .as_ref()
                .unwrap()
                .labels
                .is_empty()
        );
        assert!(source.cache.expiry_of(&42).is_some());

        let results2 = source.get(&[42]).await;
        assert!(
            results2
                .get(&42)
                .unwrap()
                .as_ref()
                .unwrap()
                .labels
                .is_empty()
        );
        assert_eq!(remote_keys.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn get_empty_input_returns_empty() {
        let source = make_source(HashMap::new(), HashMap::new());

        let results = source.get(&[]).await;

        assert!(results.is_empty());
    }

    #[test]
    fn young_tweet_gets_short_ttl() {
        let tweet_id = tweet_id_created_at(SystemTime::now() - Duration::from_secs(60));
        assert_eq!(ttl_for_tweet(tweet_id, SystemTime::now()), Some(SHORT_TTL));
    }

    #[test]
    fn old_tweet_gets_long_ttl() {
        let tweet_id = tweet_id_created_at(SystemTime::now() - Duration::from_secs(600));
        assert_eq!(ttl_for_tweet(tweet_id, SystemTime::now()), Some(LONG_TTL));
    }

    #[test]
    fn future_tweet_is_not_cached() {
        let tweet_id = future_tweet_id();
        assert_eq!(ttl_for_tweet(tweet_id, SystemTime::now()), None);
    }

    #[test]
    fn near_boundary_tweet_ttl_clamped_to_window() {
        let tweet_id =
            tweet_id_created_at(SystemTime::now() - (YOUNG_TWEET_AGE - Duration::from_secs(10)));
        let ttl = ttl_for_tweet(tweet_id, SystemTime::now()).expect("young tweet has a ttl");
        assert!(ttl <= Duration::from_secs(10));
        assert!(ttl < SHORT_TTL);
    }

    #[tokio::test]
    async fn expired_entry_is_refetched_and_restamped() {
        let (clock, mock) = Clock::mock();
        let tweet_id = tweet_id_created_at(SystemTime::now() - Duration::from_secs(60));
        let source = make_source_with_clock(
            HashMap::from([(cache_key(tweet_id), Ok(Some(cached_one_label())))]),
            HashMap::new(),
            clock,
        );

        source.get(&[tweet_id]).await;
        let first_expiry = source.cache.expiry_of(&tweet_id).expect("backfilled");

        mock.increment(SHORT_TTL + Duration::from_secs(1));
        source.get(&[tweet_id]).await;
        let restamped = source.cache.expiry_of(&tweet_id).expect("re-backfilled");

        assert!(restamped > first_expiry);
    }
}
