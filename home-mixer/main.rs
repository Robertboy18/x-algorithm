use std::time::Duration;

use clap::Parser;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_dark_traffic::RejectDarkTrafficLayer;
use xai_home_mixer::dark_traffic_setup;
use xai_home_mixer::params;
use xai_home_mixer::{HomeMixerConfig, HomeMixerServer, PhoenixXdsConfig, VmRankerXdsConfig};
use xai_home_mixer_proto as pb;
use xai_x_rpc::grpc_client::TlsMode;
use xai_x_rpc::wily_lookup_service::ShardCoordinate;
use xai_x_service_builder::XServiceBuilder;

#[derive(Parser, Debug)]
#[command(about = "HomeMixer gRPC Server")]
struct Args {
    #[arg(long, default_value_t = 8080u16)]
    grpc_port: u16,
    #[arg(long, default_value_t = 9090u16)]
    metrics_port: u16,
    #[arg(long, default_value_t = -1)]
    shard_coordinate: i16,
    #[arg(long, default_value_t = 500)]
    shard_total_size: u16,
    #[arg(long, default_value = "atla")]
    datacenter: String,
    #[arg(long, default_value = "")]
    otel_endpoint: String,

    #[arg(long, default_value = "penalized_peak_ewma")]
    phoenix_xds_lb_policy: String,
    #[arg(long, default_value = "tonic")]
    phoenix_xds_retrieval_lb_policy: String,
    #[arg(long, default_value_t = 500)]
    phoenix_xds_lb_default_rtt_ms: u64,
    #[arg(long, default_value_t = 4194304)]
    phoenix_xds_h2_stream_window_bytes: u32,
    #[arg(long, default_value_t = 16777216)]
    phoenix_xds_h2_connection_window_bytes: u32,
    #[arg(long, default_value_t = 4194304)]
    phoenix_xds_socket_buffer_bytes: u32,
    #[arg(long, default_value_t = 12)]
    phoenix_xds_aperture_size: u32,
    #[arg(long, default_value = "readiness")]
    phoenix_xds_health_probe: String,
    #[arg(long, default_value_t = 1500)]
    phoenix_xds_grpc_health_timeout_ms: u64,
    #[arg(long, default_value_t = 3000)]
    phoenix_xds_grpc_health_interval_ms: u64,
    #[arg(long, default_value_t = 9091)]
    phoenix_xds_readiness_port: u16,
    #[arg(long, default_value = "/readyz")]
    phoenix_xds_readiness_path: String,
    #[arg(long, default_value = "atla")]
    phoenix_xds_discovery_authority: String,

    #[arg(long, default_value = "tonic")]
    vm_ranker_xds_lb_policy: String,
    #[arg(long, default_value_t = 500)]
    vm_ranker_xds_lb_default_rtt_ms: u64,
    #[arg(long, default_value_t = 4194304)]
    vm_ranker_xds_h2_stream_window_bytes: u32,
    #[arg(long, default_value_t = 16777216)]
    vm_ranker_xds_h2_connection_window_bytes: u32,
    #[arg(long, default_value_t = 4194304)]
    vm_ranker_xds_socket_buffer_bytes: u32,
    #[arg(long, default_value_t = 12)]
    vm_ranker_xds_aperture_size: u32,
    #[arg(long, default_value = "readiness")]
    vm_ranker_xds_health_probe: String,
    #[arg(long, default_value_t = 1500)]
    vm_ranker_xds_grpc_health_timeout_ms: u64,
    #[arg(long, default_value_t = 3000)]
    vm_ranker_xds_grpc_health_interval_ms: u64,
    #[arg(long, default_value_t = 8080)]
    vm_ranker_xds_readiness_port: u16,
    #[arg(long, default_value = "/healthz")]
    vm_ranker_xds_readiness_path: String,
    #[arg(long, default_value = "atla")]
    vm_ranker_xds_discovery_authority: String,
}

fn parse_shard(args: &Args) -> Option<ShardCoordinate> {
    if args.shard_coordinate >= 0 {
        Some(ShardCoordinate {
            ordinal: args.shard_coordinate as u16,
            total_size: args.shard_total_size,
        })
    } else {
        None
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let shard_coordinate = parse_shard(&args);

    xai_stringcenter::init_from_file(params::STRINGCENTER_BUNDLE_PATH);

    quality_factor::init(70.0, 30.0);

    let phoenix_xds = PhoenixXdsConfig {
        lb_policy: args.phoenix_xds_lb_policy,
        secondary_lb_policy: args.phoenix_xds_retrieval_lb_policy,
        lb_default_rtt_ms: args.phoenix_xds_lb_default_rtt_ms,
        h2_stream_window_bytes: args.phoenix_xds_h2_stream_window_bytes,
        h2_connection_window_bytes: args.phoenix_xds_h2_connection_window_bytes,
        socket_buffer_bytes: args.phoenix_xds_socket_buffer_bytes,
        aperture_size: args.phoenix_xds_aperture_size,
        health_probe: args.phoenix_xds_health_probe,
        grpc_health_timeout_ms: args.phoenix_xds_grpc_health_timeout_ms,
        grpc_health_interval_ms: args.phoenix_xds_grpc_health_interval_ms,
        readiness_port: args.phoenix_xds_readiness_port,
        readiness_path: args.phoenix_xds_readiness_path,
        discovery_authority: args.phoenix_xds_discovery_authority,
    };

    let vm_ranker_xds = VmRankerXdsConfig {
        lb_policy: args.vm_ranker_xds_lb_policy,
        secondary_lb_policy: "tonic".to_string(),
        lb_default_rtt_ms: args.vm_ranker_xds_lb_default_rtt_ms,
        h2_stream_window_bytes: args.vm_ranker_xds_h2_stream_window_bytes,
        h2_connection_window_bytes: args.vm_ranker_xds_h2_connection_window_bytes,
        socket_buffer_bytes: args.vm_ranker_xds_socket_buffer_bytes,
        aperture_size: args.vm_ranker_xds_aperture_size,
        health_probe: args.vm_ranker_xds_health_probe,
        grpc_health_timeout_ms: args.vm_ranker_xds_grpc_health_timeout_ms,
        grpc_health_interval_ms: args.vm_ranker_xds_grpc_health_interval_ms,
        readiness_port: args.vm_ranker_xds_readiness_port,
        readiness_path: args.vm_ranker_xds_readiness_path,
        discovery_authority: args.vm_ranker_xds_discovery_authority,
    };

    XServiceBuilder::new("home-mixer")
        .grpc_port(args.grpc_port)
        .metrics_port(args.metrics_port)
        .datacenter(args.datacenter)
        .otel_endpoint(args.otel_endpoint)
        .with_featureswitches_experiment_logging(params::FS_PATH)
        .with_decider(params::decider_path(), None)
        .with_tls(TlsMode::server_mtls_from_env()?)
        .with_max_connection_age(Duration::from_secs(300))
        .with_reflection(pb::FILE_DESCRIPTOR_SET)
        .with_layer(dark_traffic_setup::resolve_layer())
        .with_layer(RejectDarkTrafficLayer::from_env())
        .http_routes(xai_profiling::profiling_router())
        .run::<HomeMixerServer>(HomeMixerConfig {
            shard_coordinate,
            phoenix_xds,
            vm_ranker_xds,
        })
        .await
}
