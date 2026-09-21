// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::cmp;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::slice;
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::{BoxFuture, join_all};
use lazy_static::lazy_static;
use rand::prelude::*;
use tonic::transport::Channel;

use crate::adler32::adler32_combine;
use crate::emb_table::{get_channels_list_entries, send_entries};
use crate::grpc_util::TRANSFER_FAILED_SENTINEL;

pub struct TensorBuf<'a> {
    pub key: String,
    pub buf: &'a mut [u8],
}

#[derive(Debug, Clone)]
pub struct DownloadMeta {
    pub prefix: String,
    pub checksums_json: String,
    pub created_timestamp: f64,
}

#[derive(Debug)]
pub enum CopyPortError {
    NoNewer {
        new_prefix: String,
        prev_prefix: String,
    },
    Grpc(String),
    SizeMismatch {
        key: String,
        listed: usize,
        requested: usize,
    },
    NotFound {
        key: String,
    },
    TransferFailed(String),
    Timeout(String),
    Other(String),
}

impl std::fmt::Display for CopyPortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNewer {
                new_prefix,
                prev_prefix,
            } => write!(
                f,
                "new prefix {new_prefix} is no greater than old prefix {prev_prefix}"
            ),
            Self::Grpc(s) => write!(f, "gRPC error: {s}"),
            Self::SizeMismatch {
                key,
                listed,
                requested,
            } => write!(
                f,
                "size mismatch on {key}: listed {listed}, requested {requested}"
            ),
            Self::NotFound { key } => write!(f, "cannot find {key}"),
            Self::TransferFailed(s) => write!(f, "transfer failed: {s}"),
            Self::Timeout(s) => write!(f, "timeout: {s}"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for CopyPortError {}

const MAX_PACING_SLEEP: Duration = Duration::from_secs(60);

pub async fn resolve_copy_urls(urls: &str) -> Result<String, CopyPortError> {
    let mut out: Vec<String> = Vec::new();
    for part in urls.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (scheme, host, port) = parse_host_port(part)?;
        let lookup = format!("{host}:{port}");
        let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&lookup)
            .await
            .map_err(|e| {
                CopyPortError::Other(format!("copy_port DNS lookup failed for {lookup}: {e}"))
            })?
            .collect();
        if addrs.is_empty() {
            return Err(CopyPortError::Other(format!(
                "copy_port DNS returned no addresses for {lookup}"
            )));
        }
        let v4: Vec<_> = addrs.iter().filter(|a| a.ip().is_ipv4()).copied().collect();
        let use_addrs = if v4.is_empty() { &addrs[..] } else { &v4[..] };
        for sa in use_addrs {
            out.push(format!("{scheme}://{}:{port}", sa.ip()));
        }
    }
    if out.is_empty() {
        return Err(CopyPortError::Other(
            "copy_port urls empty after resolve".into(),
        ));
    }
    let mut seen = HashSet::new();
    out.retain(|u| seen.insert(u.clone()));
    log::info!(
        "copy_port: resolved {} endpoint(s) from input urls",
        out.len()
    );
    Ok(out.join(","))
}

fn parse_host_port(s: &str) -> Result<(&'static str, String, u16), CopyPortError> {
    let s = s.trim();
    let (scheme, s) = if let Some(rest) = s.strip_prefix("https://") {
        ("https", rest)
    } else {
        ("http", s.trim_start_matches("http://"))
    };
    let (host, port_str) = s.rsplit_once(':').ok_or_else(|| {
        CopyPortError::Other(format!(
            "bad copy_url '{s}' (want host:port or http(s)://host:port)"
        ))
    })?;
    let port: u16 = port_str
        .parse()
        .map_err(|_| CopyPortError::Other(format!("bad copy_url port in '{s}'")))?;
    if host.is_empty() {
        return Err(CopyPortError::Other(format!("bad copy_url host in '{s}'")));
    }
    Ok((scheme, host.to_string(), port))
}

fn checkpoint_prefix_from_path(path: &str) -> Result<String, CopyPortError> {
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() < 2 {
        return Err(CopyPortError::Other(format!(
            "bad path for prefix extraction (need elapsed_samples_*/run_id): {path}"
        )));
    }
    let n = parts.len();
    Ok(format!("{}/{}/", parts[n - 2], parts[n - 1]))
}

fn parse_elapsed_from_prefix(prefix: &str) -> Option<u64> {
    let first = prefix.split('/').next()?;
    let n = first.strip_prefix("elapsed_samples_")?;
    n.parse().ok()
}

fn newest_full_prefix(entries: &[Vec<(String, usize)>]) -> String {
    full_prefixes_newest_first(entries)
        .into_iter()
        .next()
        .unwrap_or_default()
}

fn full_prefixes_newest_first(entries: &[Vec<(String, usize)>]) -> Vec<String> {
    let n_active = entries.iter().filter(|e| !e.is_empty()).count();
    if n_active == 0 {
        return Vec::new();
    }
    let mut counts = BTreeMap::<&str, usize>::new();
    for entry_list in entries {
        if entry_list.is_empty() {
            continue;
        }
        let mut prev_prefix = "";
        for (name, _) in entry_list {
            if let Some(i) = name.find('/')
                && let Some(j) = name[i + 1..].find('/')
            {
                let prefix = &name[..i + j + 1];
                if prev_prefix != prefix {
                    prev_prefix = prefix;
                    *counts.entry(prefix).or_insert(0) += 1;
                }
            }
        }
    }
    counts
        .into_iter()
        .rev()
        .filter(|(_, count)| *count == n_active)
        .map(|(name, _)| name.to_string())
        .collect()
}

fn is_newer_prefix(prefix: &str, elapsed_samples: u64) -> bool {
    let prev = format!("elapsed_samples_{elapsed_samples:018}/");
    !prefix.is_empty() && prefix.len() >= prev.len() && prefix[..prev.len()] > prev[..]
}

fn set_or_check(slot: &mut usize, value: usize, what: &str) -> Result<(), CopyPortError> {
    if value == 0 {
        return Err(CopyPortError::Other(format!("unexpected zero {what}")));
    }
    if *slot == 0 {
        *slot = value;
    } else if *slot != value {
        return Err(CopyPortError::Other(format!(
            "{what} mismatch: expected {slot}, got {value}"
        )));
    }
    Ok(())
}

async fn connect_and_list(
    urls: &str,
    list_prefix: String,
) -> Result<(Vec<Channel>, Vec<Vec<(String, usize)>>), CopyPortError> {
    let urls = resolve_copy_urls(urls).await?;
    let (channels, entries) = get_channels_list_entries(urls, list_prefix)
        .await
        .map_err(|s| CopyPortError::Grpc(s.message().to_string()))?;
    let (channels, entries): (Vec<_>, Vec<_>) = channels
        .into_iter()
        .zip(entries)
        .filter(|(_, e)| !e.is_empty())
        .unzip();
    if channels.is_empty() {
        return Err(CopyPortError::Grpc(
            "no copy_port channels connected".into(),
        ));
    }
    log::info!(
        "copy_port: {} channel(s) with non-empty listings",
        channels.len()
    );
    Ok((channels, entries))
}

type TransferFuture = BoxFuture<'static, (usize, u32)>;
type DenseDownloadPlan = (Vec<TransferFuture>, Vec<usize>, Vec<u8>);

async fn join_transfers(
    futures: impl IntoIterator<Item = TransferFuture>,
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    join_all(futures.into_iter().map(tokio::task::spawn))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| CopyPortError::Other(format!("copy_port download task join: {e}")))
}

async fn run_downloads(
    futures: Vec<TransferFuture>,
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    if futures.is_empty() {
        return Ok(Vec::new());
    }
    match rate_limit_bytes_per_sec {
        Some(limit) if limit > 0 => {
            let max_c = max_concurrent_downloads
                .unwrap_or(futures.len().max(1))
                .max(1);
            join_rate_limited(futures, limit, max_c).await
        }
        _ => join_transfers(futures).await,
    }
}

async fn join_rate_limited(
    futures: Vec<TransferFuture>,
    rate_limit_bytes_per_sec: u64,
    max_concurrent: usize,
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    let max_concurrent = max_concurrent.max(1);
    let start = Instant::now();
    let mut total_bytes: u64 = 0;
    let mut results = Vec::with_capacity(futures.len());
    let mut iter = futures.into_iter();
    loop {
        let batch: Vec<_> = iter.by_ref().take(max_concurrent).collect();
        if batch.is_empty() {
            break;
        }
        let batch_size = batch.len();
        let batch_results = join_transfers(batch).await?;
        let failed = batch_results
            .iter()
            .filter(|r| r.0 == TRANSFER_FAILED_SENTINEL)
            .count();
        if failed > 0 {
            log::error!(
                "rate_limiter: {failed}/{batch_size} transfers in batch failed; \
                 skipping remaining downloads so the caller can fail and retry"
            );
            results.extend(batch_results);
            return Ok(results);
        }
        let mut batch_bytes: u64 = 0;
        for result in batch_results {
            batch_bytes = batch_bytes.saturating_add(result.0 as u64);
            results.push(result);
        }
        total_bytes = total_bytes.saturating_add(batch_bytes);
        let elapsed = start.elapsed();
        let expected =
            Duration::try_from_secs_f64(total_bytes as f64 / rate_limit_bytes_per_sec as f64)
                .unwrap_or(Duration::MAX);
        let sleep = expected.saturating_sub(elapsed);
        let (sleep, clamped) = if sleep > MAX_PACING_SLEEP {
            (MAX_PACING_SLEEP, true)
        } else {
            (sleep, false)
        };
        if clamped {
            log::warn!(
                "rate_limiter: pacing sleep for {total_bytes} bytes exceeds \
                 {MAX_PACING_SLEEP:?}; capping"
            );
        }
        if !sleep.is_zero() {
            log::info!(
                "rate_limiter: batch of {batch_size} downloaded {:.2} GiB \
                 ({:.2} GiB total) in {:.2}s, sleeping {:.2}s under {:.2} GiB/s",
                batch_bytes as f64 / (1u64 << 30) as f64,
                total_bytes as f64 / (1u64 << 30) as f64,
                elapsed.as_secs_f64(),
                sleep.as_secs_f64(),
                rate_limit_bytes_per_sec as f64 / (1u64 << 30) as f64,
            );
            tokio::time::sleep(sleep).await;
        }
    }
    Ok(results)
}

fn check_transfer_results(
    results: &[(usize, u32)],
    expected_sizes: &[usize],
) -> Result<(), CopyPortError> {
    for ((count, _), size) in results.iter().zip(expected_sizes.iter()) {
        if *count == TRANSFER_FAILED_SENTINEL {
            return Err(CopyPortError::TransferFailed(
                "gRPC/RDMA transfer failed (check Rust logs)".into(),
            ));
        }
        if *count != *size {
            return Err(CopyPortError::TransferFailed(format!(
                "size mismatch: wanted {size}, got {count}"
            )));
        }
    }
    Ok(())
}

fn sfence_after_download() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

pub async fn probe_newer_checkpoint(
    elapsed_samples: u64,
    urls: &str,
) -> Result<Option<(String, u64)>, CopyPortError> {
    let (_channels, entries) = connect_and_list(urls, String::new()).await?;
    let prefix = newest_full_prefix(&entries);
    if !is_newer_prefix(&prefix, elapsed_samples) {
        return Ok(None);
    }
    let elapsed = parse_elapsed_from_prefix(&prefix).unwrap_or(elapsed_samples);
    Ok(Some((prefix, elapsed)))
}

fn choose_prefix(
    target_prefix: Option<&str>,
    entries: &[Vec<(String, usize)>],
) -> Result<String, CopyPortError> {
    let Some(target) = target_prefix.map(|t| t.trim_end_matches('/')) else {
        return Ok(newest_full_prefix(entries));
    };
    let dir = format!("{target}/");
    for (i, inner) in entries.iter().enumerate() {
        if !inner.iter().any(|(name, _)| name.starts_with(&dir)) {
            return Err(CopyPortError::Other(format!(
                "target prefix {target} not present on channel {i}/{}",
                entries.len()
            )));
        }
    }
    Ok(target.to_string())
}

pub async fn download_dense_weight(
    elapsed_samples: u64,
    urls: &str,
    tensors: &mut [TensorBuf<'_>],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    target_prefix: Option<&str>,
) -> Result<DownloadMeta, CopyPortError> {
    let (channels, entries) = connect_and_list(urls, String::new()).await?;
    let prefix = choose_prefix(target_prefix, &entries)?;
    if target_prefix.is_none() && !is_newer_prefix(&prefix, elapsed_samples) {
        return Err(CopyPortError::NoNewer {
            new_prefix: prefix,
            prev_prefix: format!("elapsed_samples_{elapsed_samples:018}/"),
        });
    }

    let channel_idx = rand::rng().random_range(0..channels.len());
    let index = index_dense_listing(&prefix, &entries[channel_idx]);
    let (futures, sizes, checksums_buf) =
        build_dense_downloads(&channels[channel_idx], &index, tensors)?;

    let results =
        run_downloads(futures, rate_limit_bytes_per_sec, max_concurrent_downloads).await?;
    sfence_after_download();
    check_transfer_results(&results, &sizes)?;

    let checksums_json = String::from_utf8(checksums_buf).unwrap_or_default();
    let created_timestamp = serde_json::from_str::<serde_json::Value>(&checksums_json)
        .ok()
        .and_then(|v| v.get("created_timestamp")?.as_f64())
        .unwrap_or(0.0);

    Ok(DownloadMeta {
        prefix,
        checksums_json,
        created_timestamp,
    })
}

pub async fn download_dense_and_embeddings(
    elapsed_samples: u64,
    urls: &str,
    tensors: &mut [TensorBuf<'_>],
    emb: &mut [u8],
    pe: Option<&mut [u8]>,
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<(DownloadMeta, u32, Option<u32>), CopyPortError> {
    let t_all = Instant::now();
    let (channels, entries) = connect_and_list(urls, String::new()).await?;
    let prefix = newest_full_prefix(&entries);
    if !is_newer_prefix(&prefix, elapsed_samples) {
        return Err(CopyPortError::NoNewer {
            new_prefix: prefix,
            prev_prefix: format!("elapsed_samples_{elapsed_samples:018}/"),
        });
    }

    let channel_idx = rand::rng().random_range(0..channels.len());
    let index = index_dense_listing(&prefix, &entries[channel_idx]);
    let (futures, sizes, checksums_buf) =
        build_dense_downloads(&channels[channel_idx], &index, tensors)?;
    let results =
        run_downloads(futures, rate_limit_bytes_per_sec, max_concurrent_downloads).await?;
    sfence_after_download();
    check_transfer_results(&results, &sizes)?;

    let checksums_json = String::from_utf8(checksums_buf).unwrap_or_default();
    let created_timestamp = serde_json::from_str::<serde_json::Value>(&checksums_json)
        .ok()
        .and_then(|v| v.get("created_timestamp")?.as_f64())
        .unwrap_or(0.0);
    let bytes: u64 = tensors.iter().map(|t| t.buf.len() as u64).sum();
    let secs = t_all.elapsed().as_secs_f64();
    let gbs = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    log::info!(
        "copy_port: dense weights loaded bytes={bytes} in {:.2}s ({:.2} GB/s)",
        secs,
        gbs
    );

    let prefix_slash = if prefix.ends_with('/') {
        prefix.clone()
    } else {
        format!("{prefix}/")
    };
    let stripped: Vec<Vec<(String, usize)>> = entries
        .iter()
        .map(|list| {
            list.iter()
                .filter_map(|(n, sz)| {
                    n.strip_prefix(&prefix_slash)
                        .or_else(|| n.strip_prefix(&prefix).and_then(|s| s.strip_prefix('/')))
                        .map(|rest| (rest.to_string(), *sz))
                })
                .collect()
        })
        .collect();

    let t_emb = Instant::now();
    let emb_ck = download_sharded_with_channels(
        &channels,
        &stripped,
        &prefix_slash,
        "emb_table",
        emb,
        rate_limit_bytes_per_sec,
        max_concurrent_downloads,
    )
    .await?;
    let secs = t_emb.elapsed().as_secs_f64();
    let gbs = if secs > 0.0 {
        emb.len() as f64 / secs / 1e9
    } else {
        0.0
    };
    log::info!(
        "copy_port: emb_table loaded bytes={} in {:.2}s ({:.2} GB/s)",
        emb.len(),
        secs,
        gbs
    );

    let pe_ck = if let Some(pe_buf) = pe.filter(|b| !b.is_empty()) {
        log::info!(
            "copy_port: downloading post_embeddings prefix={prefix_slash} bytes={}",
            pe_buf.len()
        );
        Some(
            download_sharded_with_channels(
                &channels,
                &stripped,
                &prefix_slash,
                "post_embeddings.embeddings",
                pe_buf,
                rate_limit_bytes_per_sec,
                max_concurrent_downloads,
            )
            .await?,
        )
    } else {
        None
    };

    Ok((
        DownloadMeta {
            prefix,
            checksums_json,
            created_timestamp,
        },
        emb_ck,
        pe_ck,
    ))
}

async fn download_sharded_with_channels(
    channels: &[Channel],
    entries: &[Vec<(String, usize)>],
    prefix: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<u32, CopyPortError> {
    let layout = ShardedLayout::from_listing(name, entries)?;
    layout.check_buffer_size(name, buf.len())?;
    let (futures, expected, schedule) =
        spawn_sharded_downloads(&layout, prefix, name, channels, buf).await?;
    let concurrent = max_concurrent_downloads.or(Some((channels.len() / 2).max(1)));
    let results = run_downloads(futures, rate_limit_bytes_per_sec, concurrent).await?;
    sfence_after_download();
    let results = results_in_piece_order(results, &schedule)?;
    combine_transfer_checksums(&results, &expected)
}

pub const BUNDLE_MANIFEST_NAME: &str = "export/MANIFEST.json";

pub async fn find_bundle_checkpoint(
    elapsed_samples: u64,
    urls: &str,
) -> Result<Option<(String, u64)>, CopyPortError> {
    let (_channels, entries) = connect_and_list(urls, String::new()).await?;
    select_bundle_checkpoint(&entries, elapsed_samples)
}

fn select_bundle_checkpoint(
    entries: &[Vec<(String, usize)>],
    elapsed_samples: u64,
) -> Result<Option<(String, u64)>, CopyPortError> {
    let mut newest_unbundled: Option<String> = None;
    for prefix in full_prefixes_newest_first(entries) {
        if !is_newer_prefix(&prefix, elapsed_samples) {
            break;
        }
        let manifest_name = format!("{prefix}/{BUNDLE_MANIFEST_NAME}");
        let on_all_channels = entries.iter().filter(|l| !l.is_empty()).all(|list| {
            list.iter()
                .any(|(name, size)| name == &manifest_name && *size > 0)
        });
        if on_all_channels {
            if let Some(skipped) = &newest_unbundled {
                log::warn!(
                    "copy_port: newest checkpoint {skipped} has no complete \
                     {BUNDLE_MANIFEST_NAME}; using older bundled checkpoint {prefix}"
                );
            }
            let elapsed = parse_elapsed_from_prefix(&prefix).unwrap_or(elapsed_samples);
            return Ok(Some((prefix, elapsed)));
        }
        newest_unbundled.get_or_insert(manifest_name);
    }
    match newest_unbundled {
        Some(key) => Err(CopyPortError::NotFound { key }),
        None => Ok(None),
    }
}

pub async fn download_named_files(
    prefix: &str,
    urls: &str,
    names: &[String],
) -> Result<Vec<Vec<u8>>, CopyPortError> {
    let prefix = format!("{}/", prefix.trim_end_matches('/'));
    let (channels, entries) = connect_and_list(urls, prefix.clone()).await?;
    let complete: Vec<usize> = (0..channels.len())
        .filter(|&i| {
            let listing: HashMap<&str, usize> =
                entries[i].iter().map(|(n, s)| (n.as_str(), *s)).collect();
            names
                .iter()
                .all(|n| listing.contains_key(n.strip_prefix(&prefix).unwrap_or(n)))
        })
        .collect();
    let Some(&channel_idx) = complete.as_slice().choose(&mut rand::rng()) else {
        return Err(CopyPortError::NotFound {
            key: format!(
                "{prefix}{{{}}} (no channel lists all files)",
                names.join(",")
            ),
        });
    };
    let listing: HashMap<&str, usize> = entries[channel_idx]
        .iter()
        .map(|(name, size)| (name.as_str(), *size))
        .collect();

    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(names.len());
    let mut sizes: Vec<usize> = Vec::with_capacity(names.len());
    let mut futures: Vec<TransferFuture> = Vec::with_capacity(names.len());
    for name in names {
        let relative = name.strip_prefix(&prefix).unwrap_or(name);
        let Some(&size) = listing.get(relative) else {
            return Err(CopyPortError::NotFound {
                key: format!("{prefix}{relative}"),
            });
        };
        let full_name = format!("{prefix}{relative}");
        let mut buf = vec![0u8; size];
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut buf[..]) };
        buffers.push(buf);
        sizes.push(size);
        futures.push(Box::pin(send_entries(
            channels[channel_idx].clone(),
            vec![full_name.into_bytes()],
            vec![0],
            vec![size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        )));
    }

    let results = run_downloads(futures, None, None).await?;
    sfence_after_download();
    check_transfer_results(&results, &sizes)?;
    Ok(buffers)
}

pub async fn download_embedding_table(
    path: &str,
    urls: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<u32, CopyPortError> {
    download_embedding_table_with_conns(
        path,
        urls,
        name,
        buf,
        rate_limit_bytes_per_sec,
        max_concurrent_downloads,
        crate::emb_table::peer_conns_per_source(),
    )
    .await
}

async fn download_embedding_table_with_conns(
    path: &str,
    urls: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    peer_conns_per_source: usize,
) -> Result<u32, CopyPortError> {
    let prefix = checkpoint_prefix_from_path(path)?;
    let resolved = resolve_copy_urls(urls).await?;
    let (mut channels, entries) = connect_and_list(&resolved, prefix.clone()).await?;
    let layout = ShardedLayout::from_listing(name, &entries)?;
    layout.check_buffer_size(name, buf.len())?;
    if matches!(layout.ownership, ShardOwnership::Replicated { .. }) {
        crate::emb_table::expand_replicated_channels(
            &resolved,
            &prefix,
            &mut channels,
            peer_conns_per_source,
        )
        .await;
    }

    let (futures, expected, schedule) =
        spawn_sharded_downloads(&layout, &prefix, name, &channels, buf).await?;
    let concurrent = max_concurrent_downloads.or(Some((channels.len() / 2).max(1)));
    let results = run_downloads(futures, rate_limit_bytes_per_sec, concurrent).await?;
    sfence_after_download();
    let results = results_in_piece_order(results, &schedule)?;
    combine_transfer_checksums(&results, &expected)
}

struct DenseListing<'a> {
    names_sizes: HashMap<&'a str, (&'a str, usize)>,
    checksums_name: String,
    checksums_size: usize,
}

fn index_dense_listing<'a>(prefix: &str, entries: &'a [(String, usize)]) -> DenseListing<'a> {
    let checksums_name = format!("{prefix}/checksums.0.json");
    let mut names_sizes = HashMap::new();
    let mut checksums_size = 0usize;
    for (name, size) in entries {
        let i = prefix.len() + 1;
        if !(name.starts_with(prefix) && name.len() > i) {
            continue;
        }
        if let Some(j) = name[i..].rfind("/c") {
            names_sizes.insert(&name[i..i + j], (name.as_str(), *size));
        } else if name.as_str() == checksums_name {
            checksums_size = *size;
        }
    }
    DenseListing {
        names_sizes,
        checksums_name,
        checksums_size,
    }
}

fn build_dense_downloads(
    channel: &Channel,
    index: &DenseListing<'_>,
    tensors: &mut [TensorBuf<'_>],
) -> Result<DenseDownloadPlan, CopyPortError> {
    let mut sizes = Vec::with_capacity(tensors.len() + 1);
    let mut futures = Vec::with_capacity(tensors.len() + 1);

    for t in tensors.iter_mut() {
        let Some(&(full_name, size)) = index.names_sizes.get(t.key.as_str()) else {
            return Err(CopyPortError::NotFound { key: t.key.clone() });
        };
        if t.buf.len() != size {
            return Err(CopyPortError::SizeMismatch {
                key: t.key.clone(),
                listed: size,
                requested: t.buf.len(),
            });
        }
        sizes.push(size);
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut t.buf[..]) };
        let fut: TransferFuture = Box::pin(send_entries(
            channel.clone(),
            vec![full_name.as_bytes().to_vec()],
            vec![0],
            vec![size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        ));
        futures.push(fut);
    }

    let mut checksums = vec![0u8; index.checksums_size];
    if index.checksums_size != 0 {
        sizes.push(index.checksums_size);
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut checksums[..]) };
        let fut: TransferFuture = Box::pin(send_entries(
            channel.clone(),
            vec![index.checksums_name.as_bytes().to_vec()],
            vec![0],
            vec![index.checksums_size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        ));
        futures.push(fut);
    }

    Ok((futures, sizes, checksums))
}

lazy_static! {
    pub(crate) static ref PEER_SEND_MAX_BYTES: usize = {
        std::env::var("COPY_PORT_PEER_SEND_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(16 << 30)
    };
}

pub(crate) fn replicated_send_ranges(
    total: usize,
    n_channels: usize,
    max_pieces_per_send: usize,
) -> Vec<(usize, usize, usize)> {
    if total == 0 || n_channels == 0 {
        return Vec::new();
    }
    let cap = cmp::max(1, cmp::min(max_pieces_per_send, total.div_ceil(n_channels)));
    let mut ranges = Vec::new();
    let mut lo = 0;
    while lo < total {
        let hi = cmp::min(total, lo.saturating_add(cap));
        ranges.push((ranges.len() % n_channels, lo, hi));
        lo = hi;
    }
    ranges
}

pub(crate) fn peer_send_max_pieces(piece_bytes: usize) -> usize {
    cmp::max(1, *PEER_SEND_MAX_BYTES / cmp::max(1, piece_bytes))
}

pub(crate) enum ShardOwnership {
    Sharded {
        owner: HashMap<String, usize>,
        pieces_per_rank: usize,
    },
    Replicated {
        total_pieces: usize,
    },
}

pub(crate) fn classify_shard_ownership(
    name: &str,
    entries: &[Vec<(String, usize)>],
) -> Result<(usize, ShardOwnership), CopyPortError> {
    let name_prefix = format!("{name}/c/");
    let n_channels = entries.len();
    let mut piece_bytes = 0usize;
    let mut listings: HashMap<String, (usize, usize)> = HashMap::new();
    let mut per_channel = vec![0usize; n_channels];

    for (channel_idx, inner) in entries.iter().enumerate() {
        for (key, size) in inner {
            if !key.starts_with(&name_prefix) {
                continue;
            }
            set_or_check(&mut piece_bytes, *size, "shard piece size")?;
            listings.entry(key.clone()).or_insert((channel_idx, 0)).1 += 1;
            per_channel[channel_idx] += 1;
        }
    }
    if piece_bytes == 0 || listings.is_empty() {
        return Err(CopyPortError::NotFound {
            key: name.to_string(),
        });
    }

    let total = listings.len();
    let mut pieces_per_rank = 0usize;
    let mut zero_ok = true;
    for (channel_idx, &count) in per_channel.iter().enumerate() {
        if count == 0 {
            log::warn!("copy_port: channel {channel_idx} listed 0 {name} pieces; skipping");
            continue;
        }
        if pieces_per_rank == 0 {
            pieces_per_rank = count;
        } else if count != pieces_per_rank {
            zero_ok = false;
        }
    }
    let sharded = zero_ok && pieces_per_rank != 0 && listings.values().all(|&(_, c)| c == 1);
    if sharded {
        return Ok((
            piece_bytes,
            ShardOwnership::Sharded {
                owner: listings.into_iter().map(|(k, (c, _))| (k, c)).collect(),
                pieces_per_rank,
            },
        ));
    }
    let replicated =
        listings.values().all(|&(_, c)| c == n_channels) && per_channel.iter().all(|&c| c == total);
    if replicated {
        return Ok((
            piece_bytes,
            ShardOwnership::Replicated {
                total_pieces: total,
            },
        ));
    }
    Err(CopyPortError::Other(format!(
        "inconsistent shard ownership for {name}: {total} pieces across {n_channels} channels \
         are neither exclusively sharded nor fully replicated"
    )))
}

pub(crate) fn shuffle_sharded_schedule(n: usize, name: &str) -> Vec<usize> {
    let mut schedule: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return schedule;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Ok(id) = std::env::var("POD_NAME").or_else(|_| std::env::var("HOSTNAME")) {
        id.hash(&mut hasher);
    }
    name.hash(&mut hasher);
    let mut rng = rand::rngs::StdRng::seed_from_u64(hasher.finish());
    schedule.shuffle(&mut rng);
    schedule
}

pub(crate) fn restore_piece_order<T>(shuffled: Vec<T>, schedule: &[usize]) -> Option<Vec<T>> {
    if shuffled.len() != schedule.len() {
        return None;
    }
    let mut out: Vec<Option<T>> = (0..schedule.len()).map(|_| None).collect();
    for (item, &orig) in shuffled.into_iter().zip(schedule) {
        if orig >= out.len() || out[orig].is_some() {
            return None;
        }
        out[orig] = Some(item);
    }
    out.into_iter().collect()
}

fn results_in_piece_order(
    results: Vec<(usize, u32)>,
    schedule: &[usize],
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    if results
        .iter()
        .any(|(sent, _)| *sent == TRANSFER_FAILED_SENTINEL)
    {
        return Err(CopyPortError::TransferFailed(
            "gRPC/RDMA transfer failed (check Rust logs)".into(),
        ));
    }
    restore_piece_order(results, schedule).ok_or_else(|| {
        CopyPortError::TransferFailed("shuffled download result count/order mismatch".into())
    })
}

pub(crate) fn combine_transfer_checksums(
    results: &[(usize, u32)],
    expected: &[usize],
) -> Result<u32, CopyPortError> {
    let mut checksum = 1u32;
    for (&(sent, sum), &want) in results.iter().zip(expected) {
        if sent == TRANSFER_FAILED_SENTINEL {
            return Err(CopyPortError::TransferFailed(
                "gRPC/RDMA transfer failed".into(),
            ));
        }
        if sent != want {
            return Err(CopyPortError::TransferFailed(format!(
                "bad sent size: wanted {want}, got {sent}"
            )));
        }
        adler32_combine(&mut checksum, sum, sent);
    }
    Ok(checksum)
}

struct ShardedLayout {
    piece_bytes: usize,
    ownership: ShardOwnership,
}

impl ShardedLayout {
    fn from_listing(name: &str, entries: &[Vec<(String, usize)>]) -> Result<Self, CopyPortError> {
        let (piece_bytes, ownership) = classify_shard_ownership(name, entries)?;
        Ok(Self {
            piece_bytes,
            ownership,
        })
    }

    fn total_pieces(&self) -> usize {
        match &self.ownership {
            ShardOwnership::Sharded { owner, .. } => owner.len(),
            ShardOwnership::Replicated { total_pieces } => *total_pieces,
        }
    }

    fn listed_bytes(&self) -> usize {
        self.piece_bytes * self.total_pieces()
    }

    fn check_buffer_size(&self, name: &str, buf_len: usize) -> Result<(), CopyPortError> {
        let listed = self.listed_bytes();
        if listed != buf_len {
            match &self.ownership {
                ShardOwnership::Sharded {
                    owner,
                    pieces_per_rank,
                } => {
                    let n_ranks = if *pieces_per_rank == 0 {
                        0
                    } else {
                        owner.len() / pieces_per_rank
                    };
                    log::error!(
                        "copy_port: {name} size mismatch listed={listed} buf={buf_len} \
                         shard_owners={n_ranks} pieces_per_rank={pieces_per_rank}"
                    );
                }
                ShardOwnership::Replicated { total_pieces } => {
                    log::error!(
                        "copy_port: {name} size mismatch listed={listed} buf={buf_len} \
                         replicated_pieces={total_pieces}"
                    );
                }
            }
            return Err(CopyPortError::SizeMismatch {
                key: name.to_string(),
                listed,
                requested: buf_len,
            });
        }
        Ok(())
    }
}

async fn spawn_sharded_downloads(
    layout: &ShardedLayout,
    prefix: &str,
    name: &str,
    channels: &[Channel],
    buf: &mut [u8],
) -> Result<(Vec<TransferFuture>, Vec<usize>, Vec<usize>), CopyPortError> {
    let name_prefix = format!("{name}/c/");
    let piece = layout.piece_bytes;

    let ranges: Vec<(usize, usize, usize)> = match &layout.ownership {
        ShardOwnership::Sharded {
            owner,
            pieces_per_rank,
        } => {
            let step = *pieces_per_rank;
            if step == 0 {
                return Err(CopyPortError::Other("pieces_per_rank is 0".into()));
            }
            let n_ranks = owner.len() / step;
            if n_ranks == 0 {
                return Err(CopyPortError::NotFound {
                    key: name.to_string(),
                });
            }
            let mut ranges = Vec::with_capacity(n_ranks);
            for i in 0..n_ranks {
                let shard_idx = i * step;
                let key = format!("{name_prefix}{shard_idx}/0");
                let Some(&idx) = owner.get(&key) else {
                    return Err(CopyPortError::NotFound { key });
                };
                ranges.push((idx, shard_idx, shard_idx + step));
            }
            ranges
        }
        ShardOwnership::Replicated { total_pieces } => {
            replicated_send_ranges(*total_pieces, channels.len(), peer_send_max_pieces(piece))
        }
    };

    let schedule = match &layout.ownership {
        ShardOwnership::Sharded { .. } => shuffle_sharded_schedule(ranges.len(), name),
        ShardOwnership::Replicated { .. } => (0..ranges.len()).collect(),
    };
    if matches!(layout.ownership, ShardOwnership::Sharded { .. }) && ranges.len() > 1 {
        log::info!(
            "copy_port: shard schedule shuffled name={name} n={} first={:?}",
            schedule.len(),
            &schedule[..schedule.len().min(8)]
        );
    }

    #[cfg(target_os = "linux")]
    let (contexts, devicez, mrx) = {
        use crate::emb_table::register_tensor_for_download;
        use crate::ibverbs_util::make_ib_contexts;

        let rdma = matches!(layout.ownership, ShardOwnership::Sharded { .. });
        let contexts = Arc::new(if rdma {
            make_ib_contexts().map_err(|e| CopyPortError::Other(format!("RDMA context: {e}")))?
        } else {
            Vec::new()
        });
        let devicez = Arc::new(
            (0..ranges.len())
                .map(|i| {
                    if contexts.is_empty() {
                        Vec::new()
                    } else {
                        vec![i * contexts.len() / ranges.len()]
                    }
                })
                .collect::<Vec<_>>(),
        );
        let mrx = if contexts.is_empty() {
            (0..ranges.len()).map(|_| Arc::new(Vec::new())).collect()
        } else {
            let step = ranges.first().map(|(_, a, b)| b - a).unwrap_or(0);
            let b: &'static mut [u8] = unsafe { mem::transmute(&mut buf[..]) };
            register_tensor_for_download(
                piece,
                ranges.len(),
                step,
                contexts.clone(),
                devicez.clone(),
                b,
            )
            .await
            .map_err(|s| CopyPortError::Other(s.message().to_string()))?
        };
        (contexts, devicez, mrx)
    };

    let expected: Vec<usize> = ranges.iter().map(|&(_, a, b)| (b - a) * piece).collect();
    let mut futures = Vec::with_capacity(ranges.len());
    for &i in &schedule {
        let (idx, a, b) = ranges[i];
        let n = b - a;
        let slice = &mut buf[a * piece..b * piece];
        let slice: &'static mut [u8] =
            unsafe { mem::transmute(slice::from_raw_parts_mut(slice.as_mut_ptr(), slice.len())) };
        let fut: TransferFuture = Box::pin(send_entries(
            channels[idx].clone(),
            (a..b)
                .map(|j| format!("{prefix}{name_prefix}{j}/0").into_bytes())
                .collect(),
            vec![0; n],
            vec![piece; n],
            slice,
            format!("rank={i}"),
            #[cfg(target_os = "linux")]
            (devicez[i].clone(), contexts.clone(), mrx[i].clone()),
        ));
        futures.push(fut);
    }
    Ok((futures, expected, schedule))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_from_trailing_slash_path() {
        let p =
            checkpoint_prefix_from_path("/ckpt/run/elapsed_samples_000000000012345678/abc123def/")
                .unwrap();
        assert_eq!(p, "elapsed_samples_000000000012345678/abc123def/");
    }

    #[test]
    fn prefix_from_no_trailing_slash() {
        let p =
            checkpoint_prefix_from_path("/ckpt/run/elapsed_samples_000000000012345678/abc123def")
                .unwrap();
        assert_eq!(p, "elapsed_samples_000000000012345678/abc123def/");
    }

    #[test]
    fn prefix_rejects_short_path() {
        assert!(checkpoint_prefix_from_path("only_one").is_err());
    }

    #[test]
    fn set_or_check_first_then_match() {
        let mut slot = 0usize;
        set_or_check(&mut slot, 64, "piece").unwrap();
        assert_eq!(slot, 64);
        set_or_check(&mut slot, 64, "piece").unwrap();
        assert!(set_or_check(&mut slot, 32, "piece").is_err());
        assert!(set_or_check(&mut slot, 0, "piece").is_err());
    }

    fn ckpt_listing(elapsed: u64, bundled: bool) -> Vec<(String, usize)> {
        let prefix = format!("elapsed_samples_{elapsed:018}/run");
        let mut l = vec![(format!("{prefix}/dense/w"), 8)];
        if bundled {
            l.push((format!("{prefix}/{BUNDLE_MANIFEST_NAME}"), 128));
        }
        l
    }

    fn ckpt_prefix(elapsed: u64) -> String {
        format!("elapsed_samples_{elapsed:018}/run")
    }

    #[test]
    fn full_prefixes_are_newest_first_and_require_every_active_channel() {
        let mut a = ckpt_listing(10, true);
        a.extend(ckpt_listing(20, true));
        a.extend(ckpt_listing(30, true));
        let mut b = ckpt_listing(10, true);
        b.extend(ckpt_listing(20, true));
        let entries = vec![a, b, Vec::new()];
        assert_eq!(
            full_prefixes_newest_first(&entries),
            vec![ckpt_prefix(20), ckpt_prefix(10)]
        );
        assert_eq!(newest_full_prefix(&entries), ckpt_prefix(20));
    }

    #[test]
    fn bundle_discovery_picks_newest_bundled_checkpoint() {
        let mut l = ckpt_listing(10, true);
        l.extend(ckpt_listing(20, true));
        let entries = vec![l.clone(), l];
        let (prefix, elapsed) = select_bundle_checkpoint(&entries, 0).unwrap().unwrap();
        assert_eq!(prefix, ckpt_prefix(20));
        assert_eq!(elapsed, 20);
    }

    #[test]
    fn bundle_discovery_falls_back_past_unbundled_newest() {
        let mut a = ckpt_listing(10, true);
        a.extend(ckpt_listing(20, true));
        a.extend(ckpt_listing(30, true));
        let mut b = ckpt_listing(10, true);
        b.extend(ckpt_listing(20, true));
        b.extend(ckpt_listing(30, false));
        let entries = vec![a, b];
        let (prefix, elapsed) = select_bundle_checkpoint(&entries, 0).unwrap().unwrap();
        assert_eq!(prefix, ckpt_prefix(20));
        assert_eq!(elapsed, 20);
    }

    #[test]
    fn bundle_discovery_never_returns_older_than_current() {
        let mut l = ckpt_listing(10, true);
        l.extend(ckpt_listing(20, true));
        l.extend(ckpt_listing(30, false));
        let entries = vec![l.clone(), l];
        match select_bundle_checkpoint(&entries, 20) {
            Err(CopyPortError::NotFound { key }) => {
                assert_eq!(key, format!("{}/{BUNDLE_MANIFEST_NAME}", ckpt_prefix(30)));
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn bundle_discovery_none_when_nothing_newer() {
        let mut l = ckpt_listing(10, true);
        l.extend(ckpt_listing(20, true));
        let entries = vec![l.clone(), l];
        assert!(select_bundle_checkpoint(&entries, 20).unwrap().is_none());
        assert!(select_bundle_checkpoint(&entries, 25).unwrap().is_none());
    }

    #[test]
    fn bundle_discovery_rejects_zero_size_manifest() {
        let prefix = ckpt_prefix(10);
        let l = vec![
            (format!("{prefix}/dense/w"), 8),
            (format!("{prefix}/{BUNDLE_MANIFEST_NAME}"), 0),
        ];
        let entries = vec![l.clone(), l];
        assert!(matches!(
            select_bundle_checkpoint(&entries, 0),
            Err(CopyPortError::NotFound { .. })
        ));
    }

    fn listing(pieces: &[usize], size: usize) -> Vec<(String, usize)> {
        pieces
            .iter()
            .map(|j| (format!("emb/c/{j}/0"), size))
            .collect()
    }

    #[test]
    fn classify_sharded_owners() {
        let entries = vec![listing(&[0, 1], 64), listing(&[2, 3], 64)];
        let (piece, ownership) = classify_shard_ownership("emb", &entries).unwrap();
        assert_eq!(piece, 64);
        let ShardOwnership::Sharded {
            owner,
            pieces_per_rank,
        } = ownership
        else {
            panic!("expected sharded");
        };
        assert_eq!(pieces_per_rank, 2);
        assert_eq!(owner["emb/c/2/0"], 1);
        assert_eq!(owner["emb/c/0/0"], 0);
    }

    #[test]
    fn classify_replicated_peers() {
        let all = [0, 1, 2, 3, 4];
        let entries = vec![listing(&all, 64), listing(&all, 64), listing(&all, 64)];
        let (piece, ownership) = classify_shard_ownership("emb", &entries).unwrap();
        assert_eq!(piece, 64);
        assert!(matches!(
            ownership,
            ShardOwnership::Replicated { total_pieces: 5 }
        ));
    }

    #[test]
    fn classify_replicated_requires_prefiltering_empty_channels() {
        let all = [0, 1, 2, 3, 4];
        let with_stale = vec![listing(&all, 64), vec![], listing(&all, 64)];
        assert!(classify_shard_ownership("emb", &with_stale).is_err());
        let degraded = vec![listing(&all, 64), vec![]];
        let (_, ownership) = classify_shard_ownership("emb", &degraded).unwrap();
        assert!(matches!(ownership, ShardOwnership::Sharded { .. }));
        let filtered = vec![listing(&all, 64), listing(&all, 64)];
        let (_, ownership) = classify_shard_ownership("emb", &filtered).unwrap();
        assert!(matches!(
            ownership,
            ShardOwnership::Replicated { total_pieces: 5 }
        ));
    }

    #[test]
    fn classify_single_channel_stays_sharded() {
        let entries = vec![listing(&[0, 1, 2], 64)];
        let (_, ownership) = classify_shard_ownership("emb", &entries).unwrap();
        assert!(matches!(
            ownership,
            ShardOwnership::Sharded {
                pieces_per_rank: 3,
                ..
            }
        ));
    }

    #[test]
    fn classify_rejects_inconsistent_ownership() {
        let entries = vec![listing(&[0, 1], 64), listing(&[1, 2], 64)];
        assert!(classify_shard_ownership("emb", &entries).is_err());
        let entries = vec![listing(&[0], 64), listing(&[1], 32)];
        assert!(classify_shard_ownership("emb", &entries).is_err());
        assert!(classify_shard_ownership("emb", &[vec![], vec![]]).is_err());
    }

    fn replicated_block_range(total: usize, n: usize, i: usize) -> (usize, usize) {
        let step = total.div_ceil(n);
        (
            std::cmp::min(total, i * step),
            std::cmp::min(total, (i + 1) * step),
        )
    }

    #[test]
    fn replicated_block_ranges_cover_all_pieces() {
        assert_eq!(replicated_block_range(5, 2, 0), (0, 3));
        assert_eq!(replicated_block_range(5, 2, 1), (3, 5));
        assert_eq!(replicated_block_range(2, 4, 1), (1, 2));
        assert_eq!(replicated_block_range(2, 4, 2), (2, 2));
        for (total, n) in [(1, 1), (7, 3), (256, 4), (3, 8)] {
            let mut covered = 0;
            for i in 0..n {
                let (a, b) = replicated_block_range(total, n, i);
                assert_eq!(a, covered.min(total));
                covered = b.max(covered);
            }
            assert_eq!(covered, total);
        }
    }

    #[test]
    fn combine_checksums_validates_sizes() {
        assert!(combine_transfer_checksums(&[(64, 1)], &[64]).is_ok());
        assert!(combine_transfer_checksums(&[(63, 1)], &[64]).is_err());
        assert!(combine_transfer_checksums(&[(TRANSFER_FAILED_SENTINEL, 1)], &[64]).is_err());
    }

    #[test]
    fn shuffle_schedule_is_stable_permutation() {
        let a = shuffle_sharded_schedule(32, "emb_table");
        let b = shuffle_sharded_schedule(32, "emb_table");
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        let c = shuffle_sharded_schedule(32, "post_embeddings");
        assert_ne!(a, c);
        assert_ne!(a, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn restore_piece_order_inverts_schedule() {
        let schedule = vec![3, 0, 2, 1];
        let shuffled = vec!['d', 'a', 'c', 'b'];
        assert_eq!(
            restore_piece_order(shuffled, &schedule).unwrap(),
            vec!['a', 'b', 'c', 'd']
        );
        assert!(restore_piece_order(vec![1, 2], &[0, 1, 2]).is_none());
        assert!(restore_piece_order(vec![1, 2, 3], &[0, 0, 1]).is_none());
    }

    #[test]
    fn combine_after_restore_matches_piece_order() {
        let piece_order = vec![(10, 11u32), (10, 22), (10, 33)];
        let expected = vec![10usize, 10, 10];
        let direct = combine_transfer_checksums(&piece_order, &expected).unwrap();
        let schedule = vec![2, 0, 1];
        let shuffled = vec![piece_order[2], piece_order[0], piece_order[1]];
        let restored = restore_piece_order(shuffled.clone(), &schedule).unwrap();
        assert_eq!(
            combine_transfer_checksums(&restored, &expected).unwrap(),
            direct
        );
        assert_ne!(
            combine_transfer_checksums(&shuffled, &expected).unwrap(),
            direct
        );
    }

    #[test]
    fn replicated_send_ranges_chunk_within_blocks() {
        assert_eq!(
            replicated_send_ranges(10, 1, 4),
            vec![(0, 0, 4), (0, 4, 8), (0, 8, 10)]
        );
        assert_eq!(
            replicated_send_ranges(5, 2, 2),
            vec![(0, 0, 2), (1, 2, 4), (0, 4, 5)]
        );
        assert_eq!(replicated_send_ranges(2, 4, 8), vec![(0, 0, 1), (1, 1, 2)]);
        assert_eq!(
            replicated_send_ranges(3, 1, 0),
            vec![(0, 0, 1), (0, 1, 2), (0, 2, 3)]
        );
    }

    #[test]
    fn replicated_send_ranges_round_robin_spans_all_channels() {
        for (total, n, cap) in [(30, 2, 5), (30, 8, 2), (256, 4, 8), (17, 3, 1)] {
            let ranges = replicated_send_ranges(total, n, cap);
            assert!(ranges.len() >= n, "every channel must get work");
            for w in ranges.windows(n) {
                let mut seen: Vec<usize> = w.iter().map(|&(c, _, _)| c).collect();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), n, "window must span all {n} channels");
            }
        }
        let ranges = replicated_send_ranges(30, 3, usize::MAX);
        let channels: Vec<usize> = ranges.iter().map(|&(c, _, _)| c).collect();
        assert_eq!(channels, vec![0, 1, 2]);
    }

    #[test]
    fn replicated_send_ranges_golden_uncapped_matches_block_range() {
        for (total, n) in [(1, 1), (7, 3), (256, 4), (3, 8), (230, 2)] {
            let uncapped = replicated_send_ranges(total, n, usize::MAX);
            let blocks: Vec<_> = (0..n)
                .map(|i| {
                    let (a, b) = replicated_block_range(total, n, i);
                    (i, a, b)
                })
                .filter(|(_, a, b)| a < b)
                .collect();
            assert_eq!(uncapped, blocks);
        }
    }

    #[test]
    fn replicated_send_ranges_cover_all_pieces_in_order() {
        for (total, n, cap) in [(1, 1, 1), (7, 3, 2), (256, 4, 8), (3, 8, 1), (230, 2, 7)] {
            let ranges = replicated_send_ranges(total, n, cap);
            let mut covered = 0;
            for &(_, a, b) in &ranges {
                assert_eq!(a, covered, "ranges must be ascending and gap-free");
                assert!(b > a && b - a <= cap.max(1));
                covered = b;
            }
            assert_eq!(covered, total);
        }
    }

    #[test]
    fn peer_send_max_pieces_bounds() {
        assert_eq!(peer_send_max_pieces(1_920_004_096), 8);
        assert_eq!(peer_send_max_pieces(usize::MAX), 1);
        assert_eq!(peer_send_max_pieces(0), *PEER_SEND_MAX_BYTES);
    }

    #[test]
    fn choose_prefix_respects_target() {
        let entries = vec![
            vec![("elapsed_samples_1/run/x".to_string(), 1)],
            vec![("elapsed_samples_1/run/x".to_string(), 1)],
        ];
        assert_eq!(
            choose_prefix(Some("elapsed_samples_1/run"), &entries).unwrap(),
            "elapsed_samples_1/run"
        );
        assert_eq!(
            choose_prefix(Some("elapsed_samples_1/run/"), &entries).unwrap(),
            "elapsed_samples_1/run"
        );
        let partial = vec![entries[0].clone(), vec![]];
        assert!(choose_prefix(Some("elapsed_samples_1/run"), &partial).is_err());
    }

    fn tracking_downloads(
        n: usize,
        bytes: usize,
        hold: Duration,
        in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        starts: std::sync::Arc<std::sync::Mutex<Vec<Option<Instant>>>>,
    ) -> Vec<TransferFuture> {
        (0..n)
            .map(|i| {
                let in_flight = in_flight.clone();
                let peak = peak.clone();
                let starts = starts.clone();
                Box::pin(async move {
                    starts.lock().unwrap()[i] = Some(Instant::now());
                    let cur = in_flight.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    peak.fetch_max(cur, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(hold).await;
                    in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    (bytes, i as u32)
                }) as TransferFuture
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rate_limit_caps_in_flight() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = std::sync::Arc::new(std::sync::Mutex::new(vec![None; 6]));
        let futures = tracking_downloads(
            6,
            1,
            Duration::from_millis(80),
            in_flight,
            peak.clone(),
            starts,
        );
        let results = join_rate_limited(futures, 1 << 40, 2).await.unwrap();
        assert_eq!(
            results.iter().map(|r| r.1).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rate_limit_paces_between_batches() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = std::sync::Arc::new(std::sync::Mutex::new(vec![None; 4]));
        let bytes = 200 * 1024;
        let rate = 400 * 1024;
        let t0 = Instant::now();
        let results = run_downloads(
            tracking_downloads(
                4,
                bytes,
                Duration::from_millis(20),
                in_flight,
                peak.clone(),
                starts.clone(),
            ),
            Some(rate),
            Some(2),
        )
        .await
        .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(results.len(), 4);
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 2);

        let starts = starts.lock().unwrap();
        let s: Vec<Instant> = starts.iter().map(|t| t.expect("started")).collect();
        let first_batch_start = s[0].min(s[1]);
        let second_batch_start = s[2].min(s[3]);
        let between = second_batch_start.saturating_duration_since(first_batch_start);
        assert!(
            between >= Duration::from_millis(700),
            "second batch started {between:?} after the first"
        );

        let total_bytes = (4 * bytes) as f64;
        let min_elapsed = Duration::from_secs_f64(total_bytes / rate as f64 * 0.85);
        assert!(
            elapsed >= min_elapsed,
            "elapsed {elapsed:?} is below the {min_elapsed:?} floor for {total_bytes} bytes at {rate} B/s"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rate_limit_skips_remaining_on_sentinel() {
        let started = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let futures: Vec<TransferFuture> = (0..4)
            .map(|i| {
                let started = started.clone();
                Box::pin(async move {
                    started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i == 1 {
                        (TRANSFER_FAILED_SENTINEL, 0)
                    } else {
                        (1_000_000, i as u32)
                    }
                }) as TransferFuture
            })
            .collect();
        let t0 = Instant::now();
        let results = join_rate_limited(futures, 1, 2).await.unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "must not pace a failed batch"
        );
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.0 == TRANSFER_FAILED_SENTINEL));
        assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unlimited_path_spawns_all() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = std::sync::Arc::new(std::sync::Mutex::new(vec![None; 4]));
        let results = run_downloads(
            tracking_downloads(
                4,
                1,
                Duration::from_millis(80),
                in_flight,
                peak.clone(),
                starts,
            ),
            Some(0),
            Some(1),
        )
        .await
        .unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(
            peak.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "rate_limit=0 must ignore max_concurrent and spawn every future"
        );
    }
}

#[cfg(all(test, target_os = "linux", feature = "rdma-tests"))]
mod p2p_e2e_tests {
    use super::*;
    use crate::copy::{CopyService, ManifestSpec, Sender};
    use simd_adler32::adler32;

    const PREFIX: &str = "elapsed_samples_000000000000000042/testrun/";
    const PIECE: usize = 1000;
    const PIECES: usize = 5;

    async fn spawn_peer(backing: &std::path::Path) -> String {
        let sender = Arc::new(Sender::new(12).with_peer_serve(4, None));
        sender
            .publish_manifest(
                (0..PIECES)
                    .map(|j| ManifestSpec {
                        name: format!("{PREFIX}emb/c/{j}/0"),
                        file: backing.to_str().unwrap().to_string(),
                        offset: j * PIECE,
                        size: PIECE,
                    })
                    .collect(),
                vec![],
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        format!("127.0.0.1:{}", addr.port())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replicated_download_from_two_peers() {
        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 249) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let peer_a = spawn_peer(&path).await;
        let peer_b = spawn_peer(&path).await;
        let urls = format!("{peer_a},{peer_b}");

        let mut buf = vec![0u8; PIECE * PIECES];
        let checksum = download_embedding_table(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(buf, data);
        assert_eq!(checksum, adler32(&&data[..]));
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replicated_download_multi_connection() {
        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 251) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_mc_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let peer_a = spawn_peer(&path).await;
        let peer_b = spawn_peer(&path).await;
        let urls = format!("{peer_a},{peer_b}");

        let mut buf = vec![0u8; PIECE * PIECES];
        let checksum = download_embedding_table_with_conns(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
            2,
        )
        .await
        .unwrap();

        assert_eq!(buf, data);
        assert_eq!(checksum, adler32(&&data[..]));
        let _ = std::fs::remove_file(&path);
    }

    async fn spawn_stale_peer(backing: &std::path::Path) -> String {
        let sender = Arc::new(Sender::new(12).with_peer_serve(4, None));
        sender
            .publish_manifest(
                vec![ManifestSpec {
                    name: "elapsed_samples_000000000000000041/oldrun/emb/c/0/0".to_string(),
                    file: backing.to_str().unwrap().to_string(),
                    offset: 0,
                    size: PIECE,
                }],
                vec![],
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        format!("127.0.0.1:{}", addr.port())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replicated_multi_connection_excludes_stale_source() {
        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 253) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_stale_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let peer_a = spawn_peer(&path).await;
        let peer_b = spawn_peer(&path).await;
        let stale = spawn_stale_peer(&path).await;
        let urls = format!("{peer_a},{stale},{peer_b}");

        let mut buf = vec![0u8; PIECE * PIECES];
        let checksum = download_embedding_table_with_conns(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
            2,
        )
        .await
        .unwrap();

        assert_eq!(buf, data);
        assert_eq!(checksum, adler32(&&data[..]));
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial_test::serial(copy_tls_env)]
    async fn replicated_download_over_tls() {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};

        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(Vec::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let server_key = KeyPair::generate().unwrap();
        let server_cert = CertificateParams::new(vec!["copy.test".to_string()])
            .unwrap()
            .signed_by(&server_key, &ca_cert, &ca_key)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, pem: &str| {
            let p = dir.path().join(name);
            std::fs::write(&p, pem).unwrap();
            p.to_str().unwrap().to_string()
        };
        let (ca, cert, key) = (
            write("ca.crt", &ca_cert.pem()),
            write("tls.crt", &server_cert.pem()),
            write("tls.key", &server_key.serialize_pem()),
        );

        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 249) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_tls_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let sender = Arc::new(Sender::new(12).with_peer_serve(4, None));
        sender
            .publish_manifest(
                (0..PIECES)
                    .map(|j| ManifestSpec {
                        name: format!("{PREFIX}emb/c/{j}/0"),
                        file: path.to_str().unwrap().to_string(),
                        offset: j * PIECE,
                        size: PIECE,
                    })
                    .collect(),
                vec![],
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tls = crate::tls::ServerTlsOptions::from_paths(Some(cert), Some(key), None)
            .unwrap()
            .unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .tls_config(tls.load().unwrap())
                .unwrap()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        unsafe { std::env::set_var(crate::tls::ENV_TLS_CA, &ca) };
        unsafe { std::env::set_var(crate::tls::ENV_TLS_SERVER_NAME, "copy.test") };
        let urls = format!("https://127.0.0.1:{}", addr.port());
        let mut buf = vec![0u8; PIECE * PIECES];
        let result = download_embedding_table(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
        )
        .await;
        unsafe { std::env::remove_var(crate::tls::ENV_TLS_CA) };
        unsafe { std::env::remove_var(crate::tls::ENV_TLS_SERVER_NAME) };

        assert_eq!(buf, data);
        assert_eq!(result.unwrap(), adler32(&&data[..]));
        let _ = std::fs::remove_file(&path);
    }
}
