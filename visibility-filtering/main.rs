#![deny(
    clippy::dbg_macro,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used
)]

use clap::Parser;
use xai_dark_traffic::RejectDarkTrafficLayer;
use xai_grpc_compression::GrpcZstdLayer;
use xai_visibility_filtering_proto as vf_pb;
use xai_visibility_filtering_service::dark_traffic_setup;
use xai_visibility_filtering_service::server::VFServer;
use xai_x_rpc::grpc_client::TlsMode;
use xai_x_service_builder::XServiceBuilder;

#[derive(Parser, Debug)]
#[command(about = "Visibility Filtering gRPC Server")]
struct Args {
    #[arg(long, default_value_t = 50051u16)]
    grpc_port: u16,
    #[arg(long, default_value_t = 9090u16)]
    metrics_port: u16,
    #[arg(long, default_value = "atla")]
    datacenter: String,
    #[arg(long, default_value = "")]
    otel_endpoint: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    XServiceBuilder::new("visibility-filtering-service")
        .with_featureswitches(xai_visibility_filtering_service::config::fs_path())
        .grpc_port(args.grpc_port)
        .metrics_port(args.metrics_port)
        .datacenter(args.datacenter)
        .otel_endpoint(args.otel_endpoint)
        .with_tls(TlsMode::server_mtls_from_env()?)
        .with_reflection(vf_pb::FILE_DESCRIPTOR_SET)
        .with_layer(GrpcZstdLayer)
        .with_layer(dark_traffic_setup::resolve_layer())
        .with_layer(RejectDarkTrafficLayer::from_env())
        .http_routes(xai_profiling::profiling_router())
        .run::<VFServer>(())
        .await
}
