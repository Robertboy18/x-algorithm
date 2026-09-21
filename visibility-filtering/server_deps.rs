use crate::clients::socialgraph_client::ProdSocialgraphClient;
use crate::evaluate_tweets::EvaluateTweetsEndpoint;
use crate::filter::{EvaluationStatus, FilterRequest, FilterResponse, FilterTweets};
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use crate::hydration::tes_composite::ProdTweetForVisibilitySource;
use crate::hydration::HydrationPipeline;
use crate::models::{RawCandidate, TweetId};
use crate::reference_compare::ReferenceCompareHarness;
use crate::rules::SafetyLevel;
use crate::safety_label_source::lookup::RemoteSource;
use crate::safety_label_source::manhattan::ManhattanSource;
use crate::safety_label_source::twemcache::TwemcacheSource;
use crate::safety_label_source::warmer::{CacheWarmer, StratoWarmFetcher, Warmer};
use crate::safety_label_source::{ManhattanLabelFetcher, MhLabelClient, SafetyLabelSource};
use crate::server::VFServer;
use anyhow::Context;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use xai_cache::discovery::WilyDiscovery;
use xai_cache::{
    CacheClient, ClientBuilder, ClientConfig, HashAlgorithm, Key, Protocol, TlsConfig,
};
use xai_core_entities::gizmoduck_client::ProdGizmoduckClient;
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants, TESRpcConstants};
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CLIENT_ID, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_core_entities::tweet_entity_service_client::ProdTESClient;
use xai_strato::StratoGrpc;
use xai_visibility_filtering::vf_client::{StratoVfClient, VfClient};
use xai_x_rpc::balanced_channel::LbPolicy;
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::retry::RetryConfig;
use xai_x_rpc::timed_buffer::DEFAULT_BUFFER_MAX_WAIT;
use xai_x_rpc::total_timeout::DEFAULT_TOTAL_TIMEOUT;
use xai_xds_client::StartFrom;

const CACHE_PATH: &str = "/s/cache/safety_label_store:twemcaches";
const READINESS_PROBE_PORT: u16 = 8081;

const CLIENT_INIT_RETRY_BUDGET: Duration = Duration::from_secs(240);
const CLIENT_INIT_MAX_BACKOFF: Duration = Duration::from_secs(15);
const CLIENT_INIT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

async fn init_client_with_retry<T, E, Fut>(
    client: &str,
    deadline: tokio::time::Instant,
    mut build: impl FnMut() -> Fut,
) -> Result<T, String>
where
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut backoff = Duration::from_secs(1);
    let mut attempt = 1u32;
    loop {
        if tokio::time::Instant::now() >= deadline {
            error!(
                client,
                attempt, "Client init skipped; init deadline exhausted"
            );
            return Err("init deadline exhausted before attempt".to_string());
        }
        let error = match tokio::time::timeout(CLIENT_INIT_ATTEMPT_TIMEOUT, build()).await {
            Ok(Ok(t)) => {
                if attempt > 1 {
                    info!(client, attempt, "Client init succeeded after retry");
                }
                return Ok(t);
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ),
        };
        if tokio::time::Instant::now() + backoff >= deadline {
            error!(
                client,
                attempt,
                error = %error,
                "Client init failed; retry budget exhausted"
            );
            return Err(error);
        }
        warn!(
            client,
            attempt,
            backoff_secs = backoff.as_secs(),
            error = %error,
            "Client init failed; retrying"
        );
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(CLIENT_INIT_MAX_BACKOFF);
        attempt += 1;
    }
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub async fn build_prod_server(
    datacenter: &str,
    feature_switches: Arc<xai_feature_switches::FeatureSwitches>,
) -> VFServer {
    info!("Initializing prod clients for datacenter={}", datacenter);

    let init_deadline = tokio::time::Instant::now() + CLIENT_INIT_RETRY_BUDGET;

    let deterministic_aperture = std::env::var("APP_ENV").as_deref() == Ok("prod");
    let fallback_cache_enabled = crate::config::fallback_cache_enabled();
    let fallback_cache = fallback_cache_enabled
        .then(crate::hydration::gizmoduck_hydrator::GizmoduckAuthorHydrator::fallback_cache);
    let author_id_fallback_enabled = crate::config::author_id_fallback_enabled();
    let author_id_fallback_capacity = crate::config::author_id_fallback_capacity();
    let author_id_fallback_cache = author_id_fallback_enabled.then(|| {
        crate::hydration::tes_hydrator::TesHydrator::author_id_fallback_cache(
            author_id_fallback_capacity,
        )
    });

    let tes_client = Arc::new(
        init_client_with_retry("tes", init_deadline, || async move {
            let strato = build_xds_strato(
                XdsStratoParams {
                    name: "tes-xds",
                    xds_listener: "tweet-entity-service.prod.tweet-entity-service:fed-grpc",
                    tls_domain: format!(
                        "tweet-entity-service.tweet-entity-service.prod.{datacenter}.s2s.twttr.net"
                    ),
                    lb_policy: LbPolicy::penalized_peak_ewma(),
                    client_id: S2S_CLIENT_ID.clone(),
                    retry_config: Some(RetryConfig::for_idempotent()),
                    max_batch_size: TESRpcConstants::max_batch_size(),
                    enable_serve_within: true,
                },
                deterministic_aperture,
            )
            .await?;
            anyhow::Ok(ProdTESClient {
                grpc_client: Arc::new(strato),
            })
        })
        .await
        .expect("Failed to initialize TES client"),
    );

    let gizmoduck_client_id = crate::config::gizmoduck_client_id();
    let gizmoduck_client: Arc<
        dyn xai_core_entities::gizmoduck_client::GizmoduckClient + Send + Sync,
    > = Arc::new(
        init_client_with_retry("gizmoduck", init_deadline, || {
            let client_id = gizmoduck_client_id.clone();
            async move {
                let strato = build_xds_strato(
                    XdsStratoParams {
                        name: "gizmoduck-xds",
                        xds_listener: "gizmoduck.prod.gizmoduck:fed-grpc",
                        tls_domain: format!("gizmoduck.gizmoduck.prod.{datacenter}.s2s.twttr.net"),
                        lb_policy: LbPolicy::least_request(),
                        client_id,
                        retry_config: None,
                        max_batch_size: GizmoduckRpcConstants::max_batch_size(),
                        enable_serve_within: false,
                    },
                    deterministic_aperture,
                )
                .await?;
                anyhow::Ok(ProdGizmoduckClient {
                    grpc_client: Arc::new(strato),
                })
            }
        })
        .await
        .expect("Failed to initialize Gizmoduck client"),
    );
    info!(
        deterministic_aperture,
        "TES and Gizmoduck clients ready over xDS"
    );

    let sg_client: Arc<dyn crate::clients::socialgraph_client::SocialgraphClient + Send + Sync> =
        Arc::new(
            init_client_with_retry("socialgraph", init_deadline, || {
                ProdSocialgraphClient::new(
                    datacenter,
                    &S2S_CHAIN_PATH,
                    &S2S_CRT_PATH,
                    &S2S_KEY_PATH,
                )
            })
            .await
            .expect("Failed to initialize SocialGraph client"),
        );

    let mh_label_client: Arc<dyn ManhattanLabelFetcher> = Arc::new(
        init_client_with_retry("manhattan", init_deadline, || {
            let s2s = xai_manhattan::s2s::S2sConfig {
                client_cert_path: S2S_CRT_PATH.clone(),
                client_key_path: S2S_KEY_PATH.clone(),
                ca_cert_path: S2S_CHAIN_PATH.clone(),
            };
            MhLabelClient::new(datacenter, s2s, deterministic_aperture)
        })
        .await
        .expect("Failed to initialize MhLabelClient"),
    );

    let twemcache_client_name = crate::config::twemcache_client_name();
    let twemcache = Arc::new(
        init_client_with_retry("twemcache", init_deadline, || {
            let name = twemcache_client_name.clone();
            let zone = datacenter.to_string();
            async move {
                let discovery = Arc::new(WilyDiscovery::new(CACHE_PATH, name, zone).await?);
                let config = ClientConfig::builder()
                    .request_timeout(Duration::from_millis(20))
                    .connect_timeout(Duration::from_secs(5))
                    .hash_algorithm(HashAlgorithm::FNV1)
                    .connections_per_endpoint(2)
                    .depth_cap(100)
                    .failure_accrual_enabled(false)
                    .build();
                ClientBuilder::new(Protocol::Memcached, discovery, config)
                    .with_tls(TlsConfig {
                        ca_cert_path: S2S_CHAIN_PATH.clone(),
                        client_cert_path: S2S_CRT_PATH.clone(),
                        client_key_path: S2S_KEY_PATH.clone(),
                    })
                    .build()
                    .await
            }
        })
        .await
        .expect("Failed to create twemcache client"),
    );
    info!("Cache client connected to {CACHE_PATH}");

    let reference_compare = build_reference_compare_harness(datacenter, init_deadline).await;

    warm_cache(&twemcache).await;
    warm_manhattan(mh_label_client.as_ref()).await;

    let cache_warmer = build_cache_warmer(datacenter, init_deadline, deterministic_aperture).await;

    let twemcache_source = Arc::new(TwemcacheSource::new(twemcache));
    let manhattan_source = Arc::new(ManhattanSource::new(mh_label_client));
    let mut remote = RemoteSource::new(twemcache_source, manhattan_source);
    if let Some(warmer) = cache_warmer {
        remote = remote.with_warmer(warmer);
    }
    let remote = Arc::new(remote);
    let safety_label_source = Arc::new(SafetyLabelSource::new(remote));

    let tweet_source = Arc::new(ProdTweetForVisibilitySource {
        grpc_client: tes_client.grpc_client.clone(),
    });
    let hydration_pipeline = HydrationPipeline::new(
        tes_client,
        tweet_source,
        gizmoduck_client,
        sg_client,
        safety_label_source.clone(),
        fallback_cache,
        author_id_fallback_cache,
    );
    let gating_countries = Arc::new(crate::params::NsfwGatingCountries::starting_at_default());
    let fs_path = crate::config::fs_path();
    gating_countries.refresh_and_check_drift(&feature_switches, &fs_path);
    gating_countries.spawn_refresh(feature_switches, fs_path);
    let rule_engine = crate::rules::RuleEngine::with_nsfw_gating_countries(gating_countries);
    let (home_rule_count, recommendations_rule_count) = rule_engine.rule_counts();
    let filter_tweets = Arc::new(FilterTweets::new(hydration_pipeline, rule_engine));

    warm_filter_tweets(&filter_tweets).await;

    info!(
        hydrator_count = 5,
        fallback_cache_enabled,
        author_id_fallback_enabled,
        author_id_fallback_capacity,
        home_rule_count,
        recommendations_rule_count,
        "VFServer initialized with prod clients"
    );

    VFServer::from_endpoints(
        EvaluateTweetsEndpoint::new(filter_tweets.clone()),
        FilterTweetsEndpoint::new(filter_tweets, reference_compare),
        GetSafetyLabelsEndpoint::new(safety_label_source),
    )
}

async fn build_reference_compare_harness(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
) -> Option<Arc<ReferenceCompareHarness>> {
    #[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
    let should_build = crate::reference_compare::should_build_harness(
        crate::config::dual_call_harness_enabled(),
        std::env::var("APP_ENV").ok().as_deref(),
    )
    .unwrap_or_else(|misconfiguration| panic!("{misconfiguration}"));
    if !should_build {
        return None;
    }

    let client_id = format!(
        "visibility-filtering-service.{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "staging".to_string())
    );
    #[expect(
        clippy::expect_used,
        reason = "startup fail-fast: init failure is fatal"
    )]
    let strato: Arc<dyn VfClient + Send + Sync> = Arc::new(
        init_client_with_retry("strato_vf", init_deadline, || {
            let client_id = client_id.clone();
            async move {
                StratoVfClient::new(
                    S2S_CHAIN_PATH.clone(),
                    S2S_CRT_PATH.clone(),
                    S2S_KEY_PATH.clone(),
                    client_id,
                    datacenter.to_string(),
                )
                .await
                .map_err(|e| e.to_string())
            }
        })
        .await
        .expect("Failed to initialize Strato VF client (reference comparator)"),
    );
    Some(Arc::new(ReferenceCompareHarness::new(strato, datacenter)))
}

const CACHE_WARM_REQUEST_TIMEOUT_MS: u64 = 500;

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
async fn build_cache_warmer(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
    deterministic_aperture: bool,
) -> Option<Arc<dyn Warmer>> {
    if !crate::config::cache_warm_enabled() {
        return None;
    }

    let client_id = format!(
        "visibility-filtering-service.{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "prod".to_string())
    );
    let grpc = init_client_with_retry("strato_cache_warm", init_deadline, || {
        let config = xai_strato::StratoGrpcConfig {
            ca_cert_path: S2S_CHAIN_PATH.clone(),
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            aperture_size: Some(STRATO_APERTURE_SIZE),
            deterministic_aperture,
            connect_timeout_ms: 400,
            request_timeout_ms: CACHE_WARM_REQUEST_TIMEOUT_MS,
            client_id: Some(client_id.clone()),
            service_url: format!("stratostore.stratoserver.prod.{datacenter}.s2s.twttr.net"),
            zone: datacenter.to_string(),
            ..Default::default()
        };
        async move { StratoGrpc::new(config).await.map_err(|e| e.to_string()) }
    })
    .await
    .expect("Failed to initialize Strato cache-warm client");
    info!("L2 cache warmer enabled");
    Some(CacheWarmer::spawn(Arc::new(StratoWarmFetcher::new(grpc))))
}

const STRATO_REQUEST_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const STRATO_CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
const STRATO_APERTURE_SIZE: usize = 12;
const XDS_EAGER_RESOLUTION_TIMEOUT: Duration = Duration::from_secs(5);

struct XdsStratoParams {
    name: &'static str,
    xds_listener: &'static str,
    tls_domain: String,
    lb_policy: LbPolicy,
    client_id: String,
    retry_config: Option<RetryConfig>,
    max_batch_size: usize,
    enable_serve_within: bool,
}

async fn build_xds_strato(
    params: XdsStratoParams,
    deterministic_aperture: bool,
) -> anyhow::Result<StratoGrpc> {
    let mut builder = ChannelBuilder::new(params.name)
        .tls(
            TlsMode::mtls_from_env()
                .context("S2S cert env vars required for mTLS")?
                .with_domain_override(params.tls_domain),
        )
        .request_timeout(STRATO_REQUEST_TIMEOUT)
        .connect_timeout(STRATO_CONNECT_TIMEOUT)
        .xds(StartFrom::Lds(params.xds_listener.to_string()))
        .await
        .with_context(|| format!("failed to initialize xDS for {}", params.xds_listener))?
        .eager_resolution(XDS_EAGER_RESOLUTION_TIMEOUT)
        .aperture(STRATO_APERTURE_SIZE)
        .readiness_probe(READINESS_PROBE_PORT, "/ready")
        .buffer_max_wait(DEFAULT_BUFFER_MAX_WAIT)
        .total_timeout(DEFAULT_TOTAL_TIMEOUT);
    if deterministic_aperture {
        builder = builder.deterministic();
    }

    let channel = builder
        .build_load_balanced(params.lb_policy)
        .await
        .with_context(|| {
            format!(
                "failed to build xDS LoadBalancedChannel for {}",
                params.name
            )
        })?;

    let mut strato = StratoGrpc::from_load_balanced_channel(
        channel,
        None,
        Some(params.client_id),
        params.retry_config,
        params.max_batch_size,
    );
    if params.enable_serve_within {
        strato = strato.with_op_context(xai_strato::strato_proto::OpContext {
            serve_within: Some(xai_strato::strato_proto::ServeWithin {
                duration_microseconds: STRATO_REQUEST_TIMEOUT.as_micros() as i64,
                round_trip_allowance_microseconds: 10_000,
            }),
            ..Default::default()
        });
    }
    Ok(strato)
}

async fn warm_cache(twemcache: &CacheClient) {
    let start = Instant::now();
    let key = match Key::new(b"slm_warmup".to_vec()) {
        Ok(key) => key,
        Err(e) => {
            warn!(error = %e, "Cache warmup key construction failed (non-fatal)");
            return;
        }
    };
    let latency_ms = || start.elapsed().as_millis() as u64;
    match twemcache.multi_get(std::slice::from_ref(&key)).await {
        Ok(map) => match map.get(&key) {
            Some(Ok(_)) => info!(latency_ms = latency_ms(), "Cache warmup succeeded"),
            Some(Err(e)) => warn!(
                latency_ms = latency_ms(),
                error = %e,
                "Cache warmup failed (non-fatal)"
            ),
            None => warn!(
                latency_ms = latency_ms(),
                "Cache warmup returned no response (non-fatal)"
            ),
        },
        Err(e) => warn!(
            latency_ms = latency_ms(),
            error = %e,
            "Cache warmup failed (non-fatal)"
        ),
    }
}

const WARM_FILTER_TWEETS_TWEET_ID: u64 = 20;
const WARM_FILTER_TWEETS_CONSECUTIVE: u32 = 3;
const WARM_FILTER_TWEETS_DEADLINE: Duration = Duration::from_secs(20);
const WARM_FILTER_TWEETS_SLEEP: Duration = Duration::from_millis(500);

fn warm_filter_tweets_request() -> FilterRequest {
    FilterRequest {
        viewer_id: None,
        country_code: None,
        safety_level: SafetyLevel::TimelineHomeRecommendations,
        candidates: vec![RawCandidate {
            tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
            request_author_id: None,
        }],
    }
}

fn filter_tweets_response_is_warm(response: &FilterResponse) -> bool {
    response
        .outcomes
        .iter()
        .all(|outcome| outcome.status != EvaluationStatus::UnresolvedAuthor)
}

async fn warm_filter_tweets(filter_tweets: &FilterTweets) {
    let start = Instant::now();
    let mut attempts = 0u32;
    let mut consecutive_warm = 0u32;
    while start.elapsed() < WARM_FILTER_TWEETS_DEADLINE {
        attempts += 1;
        let response = filter_tweets.run(warm_filter_tweets_request()).await;
        if filter_tweets_response_is_warm(&response) {
            consecutive_warm += 1;
            if consecutive_warm >= WARM_FILTER_TWEETS_CONSECUTIVE {
                info!(
                    attempts,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "filter_tweets warmup succeeded"
                );
                return;
            }
        } else {
            consecutive_warm = 0;
        }
        if start.elapsed() + WARM_FILTER_TWEETS_SLEEP < WARM_FILTER_TWEETS_DEADLINE {
            tokio::time::sleep(WARM_FILTER_TWEETS_SLEEP).await;
        } else {
            break;
        }
    }
    warn!(
        attempts,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "filter_tweets warmup deadline expired (non-fatal)"
    );
}

async fn warm_manhattan(manhattan: &dyn ManhattanLabelFetcher) {
    let start = Instant::now();
    match manhattan.fetch_labels(&[0]).await {
        Ok(_) => info!(
            latency_ms = start.elapsed().as_millis() as u64,
            "Manhattan warmup succeeded"
        ),
        Err(e) => warn!(
            latency_ms = start.elapsed().as_millis() as u64,
            error = %e,
            "Manhattan warmup failed (non-fatal)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterOutcome;
    use crate::models::VfAction;
    use crate::rules::Verdict;
    use std::cell::Cell;
    use xai_visibility_filtering::models::FilteredReason;

    fn deadline_in(budget: Duration) -> tokio::time::Instant {
        tokio::time::Instant::now() + budget
    }

    #[tokio::test]
    async fn reference_compare_harness_not_built_without_flag() {
        let harness =
            build_reference_compare_harness("atla", deadline_in(Duration::from_secs(1))).await;
        assert!(harness.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_recovers_from_transient_failures() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();

        let result = init_client_with_retry("test", deadline_in(Duration::from_secs(240)), || {
            attempts.set(attempts.get() + 1);
            let n = attempts.get();
            async move {
                if n < 3 {
                    Err("discovery timed out")
                } else {
                    Ok(n)
                }
            }
        })
        .await;

        assert_eq!(result, Ok(3));
        assert_eq!(attempts.get(), 3);
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_gives_up_at_deadline() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();
        let budget = Duration::from_secs(240);

        let result: Result<(), String> =
            init_client_with_retry("test", deadline_in(budget), || {
                attempts.set(attempts.get() + 1);
                async { Err("discovery timed out") }
            })
            .await;

        assert_eq!(result, Err("discovery timed out".to_string()));
        assert!(attempts.get() > 1);
        assert!(start.elapsed() < budget);
        assert!(start.elapsed() >= budget - CLIENT_INIT_MAX_BACKOFF);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_skips_attempt_when_deadline_already_passed() {
        let deadline = tokio::time::Instant::now();
        tokio::time::sleep(Duration::from_secs(1)).await;
        let attempts = Cell::new(0u32);

        let result: Result<(), String> = init_client_with_retry("test", deadline, || {
            attempts.set(attempts.get() + 1);
            async { Err("unreachable") }
        })
        .await;

        assert_eq!(
            result,
            Err("init deadline exhausted before attempt".to_string())
        );
        assert_eq!(attempts.get(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_times_out_hanging_attempt() {
        let start = tokio::time::Instant::now();

        let result: Result<(), String> = init_client_with_retry(
            "test",
            deadline_in(CLIENT_INIT_ATTEMPT_TIMEOUT / 2),
            || async {
                std::future::pending::<()>().await;
                Err("unreachable")
            },
        )
        .await;

        assert_eq!(
            result,
            Err(format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ))
        );
        assert_eq!(start.elapsed(), CLIENT_INIT_ATTEMPT_TIMEOUT);
    }

    fn response(verdicts: Vec<Verdict>) -> FilterResponse {
        let outcomes: Vec<FilterOutcome> = verdicts
            .into_iter()
            .map(|verdict| FilterOutcome {
                tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
                status: if verdict.decided_by == Verdict::unresolved_author().decided_by {
                    EvaluationStatus::UnresolvedAuthor
                } else {
                    EvaluationStatus::Evaluated
                },
                verdict,
                safety_labels: None,
            })
            .collect();
        FilterResponse { outcomes }
    }

    #[test]
    fn allow_verdict_is_warm() {
        let response = response(vec![Verdict {
            action: VfAction::Allow,
            decided_by: None,
        }]);
        assert!(filter_tweets_response_is_warm(&response));
    }

    #[test]
    fn drop_by_real_rule_is_warm() {
        let response = response(vec![Verdict {
            action: VfAction::Drop(FilteredReason::AuthorIsSuspended),
            decided_by: Some("DropSuspendedAuthorRule"),
        }]);
        assert!(filter_tweets_response_is_warm(&response));
    }

    #[test]
    fn unresolved_author_verdict_is_not_warm() {
        let response = response(vec![Verdict::unresolved_author()]);
        assert!(!filter_tweets_response_is_warm(&response));
    }
}
