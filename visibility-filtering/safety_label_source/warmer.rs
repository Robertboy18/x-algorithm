use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tonic::async_trait;
use xai_strato::{encode, StratoGrpc};

use super::metrics::{self, WarmKeyResult};

const WARM_COLUMN_PATH: &str = "visibility/baseTweetSafetyLabelMap";
const WARM_CHANNEL_CAPACITY: usize = 1024;
const WARM_LINGER: Duration = Duration::from_millis(500);
const WARM_FETCH_MAX_KEYS: usize = 50;

pub(crate) trait Warmer: Send + Sync {
    fn warm(&self, miss_ids: Vec<u64>);
}

#[async_trait]
pub(crate) trait WarmFetcher: Send + Sync {
    async fn fetch(&self, ids: &[u64]) -> usize;
}

pub(crate) struct StratoWarmFetcher {
    grpc: StratoGrpc,
}

impl StratoWarmFetcher {
    pub(crate) fn new(grpc: StratoGrpc) -> Self {
        Self { grpc }
    }
}

#[async_trait]
impl WarmFetcher for StratoWarmFetcher {
    async fn fetch(&self, ids: &[u64]) -> usize {
        let calls = ids
            .iter()
            .map(|id| {
                (
                    WARM_COLUMN_PATH.to_string(),
                    "fetch".to_string(),
                    vec![encode(&(*id as i64, ()))],
                )
            })
            .collect();
        self.grpc
            .batch_call(calls, None)
            .await
            .iter()
            .filter(|result| result.is_err())
            .count()
    }
}

pub(crate) struct CacheWarmer {
    tx: mpsc::Sender<Vec<u64>>,
}

impl CacheWarmer {
    pub(crate) fn spawn(fetcher: Arc<dyn WarmFetcher>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(WARM_CHANNEL_CAPACITY);
        tokio::spawn(drain(rx, fetcher));
        Arc::new(Self { tx })
    }

    #[cfg(test)]
    pub(crate) fn without_drain_task(capacity: usize) -> (Self, mpsc::Receiver<Vec<u64>>) {
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx }, rx)
    }
}

impl Warmer for CacheWarmer {
    fn warm(&self, miss_ids: Vec<u64>) {
        metrics::record_cache_warm_keys(WarmKeyResult::EligibleMiss, miss_ids.len());
        if miss_ids.is_empty() {
            return;
        }
        let count = miss_ids.len();
        match self.tx.try_send(miss_ids) {
            Ok(()) => metrics::record_cache_warm_keys(WarmKeyResult::Enqueued, count),
            Err(_) => metrics::record_cache_warm_keys(WarmKeyResult::DroppedChannelFull, count),
        }
    }
}

async fn drain(mut rx: mpsc::Receiver<Vec<u64>>, fetcher: Arc<dyn WarmFetcher>) {
    while let Some(mut ids) = rx.recv().await {
        let deadline = tokio::time::Instant::now() + WARM_LINGER;
        while ids.len() < WARM_FETCH_MAX_KEYS {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(mut more)) => ids.append(&mut more),
                _ => break,
            }
        }
        for chunk in ids.chunks(WARM_FETCH_MAX_KEYS) {
            let failed = fetcher.fetch(chunk).await;
            metrics::record_cache_warm_keys(WarmKeyResult::FetchIssued, chunk.len());
            metrics::record_cache_warm_keys(WarmKeyResult::FetchFailed, failed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct FakeFetcher {
        batches: Mutex<Vec<Vec<u64>>>,
        failed_per_batch: usize,
    }

    impl FakeFetcher {
        fn new(failed_per_batch: usize) -> Arc<Self> {
            Arc::new(Self {
                batches: Mutex::new(Vec::new()),
                failed_per_batch,
            })
        }

        fn batches(&self) -> Vec<Vec<u64>> {
            self.batches.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl WarmFetcher for FakeFetcher {
        async fn fetch(&self, ids: &[u64]) -> usize {
            self.batches.lock().unwrap().push(ids.to_vec());
            self.failed_per_batch
        }
    }

    #[tokio::test(start_paused = true)]
    async fn drain_lingers_then_flushes_in_chunks() {
        let fetcher = FakeFetcher::new(0);
        let warmer = CacheWarmer::spawn(fetcher.clone());

        warmer.warm((0..30).collect());
        warmer.warm((30..60).collect());
        tokio::time::sleep(WARM_LINGER * 2).await;
        warmer.warm(vec![100]);
        tokio::time::sleep(WARM_LINGER * 2).await;

        let batches = fetcher.batches();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], (0..50).collect::<Vec<u64>>());
        assert_eq!(batches[1], (50..60).collect::<Vec<u64>>());
        assert_eq!(batches[2], vec![100]);
    }

    #[tokio::test(start_paused = true)]
    async fn accumulation_is_capped_per_flush() {
        let fetcher = FakeFetcher::new(0);
        let warmer = CacheWarmer::spawn(fetcher.clone());

        for start in (0..150).step_by(30) {
            warmer.warm((start..start + 30).collect());
        }
        tokio::time::sleep(WARM_LINGER * 2).await;

        let batches = fetcher.batches();
        assert!(batches
            .iter()
            .all(|batch| batch.len() <= WARM_FETCH_MAX_KEYS));
        assert_eq!(batches.concat(), (0..150).collect::<Vec<u64>>());
    }

    #[tokio::test(start_paused = true)]
    async fn fetch_failure_does_not_stop_the_drain() {
        let fetcher = FakeFetcher::new(1);
        let warmer = CacheWarmer::spawn(fetcher.clone());

        warmer.warm(vec![1]);
        tokio::time::sleep(WARM_LINGER * 2).await;
        warmer.warm(vec![2]);
        tokio::time::sleep(WARM_LINGER * 2).await;

        assert_eq!(fetcher.batches(), vec![vec![1], vec![2]]);
    }

    #[tokio::test]
    async fn full_channel_drops_without_blocking() {
        let (warmer, mut rx) = CacheWarmer::without_drain_task(1);

        warmer.warm(vec![1]);
        warmer.warm(vec![2]);

        assert_eq!(rx.try_recv(), Ok(vec![1]));
        assert!(rx.try_recv().is_err(), "the second publish was dropped");
    }
}
