use crate::evaluate_tweets::EvaluateTweetsEndpoint;
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use std::sync::Arc;
use tonic::codec::CompressionEncoding;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub struct VFServer {
    evaluate_tweets: EvaluateTweetsEndpoint,
    filter_tweets: FilterTweetsEndpoint,
    get_safety_labels: GetSafetyLabelsEndpoint,
}

#[tonic::async_trait]
impl xai_x_service_builder::XService for VFServer {
    type Config = ();

    async fn build(ctx: xai_x_service_builder::ServiceContext<()>) -> Self {
        VFServer::new(&ctx.datacenter, ctx.feature_switches).await
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        routes.add_service(
            vf_pb::VisibilityFilteringServiceServer::from_arc(self)
                .accept_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Gzip),
        );
    }
}

impl VFServer {
    pub(crate) async fn new(
        datacenter: &str,
        feature_switches: Arc<xai_feature_switches::FeatureSwitches>,
    ) -> Self {
        crate::server_deps::build_prod_server(datacenter, feature_switches).await
    }

    pub(crate) fn from_endpoints(
        evaluate_tweets: EvaluateTweetsEndpoint,
        filter_tweets: FilterTweetsEndpoint,
        get_safety_labels: GetSafetyLabelsEndpoint,
    ) -> Self {
        Self {
            evaluate_tweets,
            filter_tweets,
            get_safety_labels,
        }
    }
}

#[tonic::async_trait]
impl vf_pb::VisibilityFilteringService for VFServer {
    async fn evaluate_tweets(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
        self.evaluate_tweets.handle(request).await
    }

    async fn filter_tweets(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        self.filter_tweets.handle(request).await
    }

    async fn get_safety_labels(
        &self,
        request: Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        self.get_safety_labels.handle(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::PureCoreData;
    use xai_core_entities::gizmoduck_client::MockGizmoduckClient;
    use xai_core_entities::tweet_entity_service_client::MockTESClient;
    use xai_visibility_filtering::evaluated::EvaluationResult;
    use xai_visibility_filtering::vf_client::XaiVfClient;
    use xai_x_thrift::action::{self, Action};

    #[tokio::test]
    async fn evaluate_tweets_loopback() {
        let tes = MockTESClient {
            core_data: [(
                1,
                Some(PureCoreData {
                    author_id: 100,
                    ..Default::default()
                }),
            )]
            .into(),
            ..Default::default()
        };
        let filter_tweets = Arc::new(crate::filter::test_support::filter_tweets_with_clients(
            Arc::new(tes),
            Arc::new(MockGizmoduckClient::default()),
        ));
        let server = VFServer::from_endpoints(
            EvaluateTweetsEndpoint::new(filter_tweets.clone()),
            FilterTweetsEndpoint::new(filter_tweets, None),
            GetSafetyLabelsEndpoint::new(crate::filter::test_support::safety_labels()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(
                    vf_pb::VisibilityFilteringServiceServer::new(server)
                        .accept_compressed(CompressionEncoding::Zstd),
                )
                .serve_with_incoming(futures::stream::unfold(listener, |listener| async {
                    Some((listener.accept().await.map(|(socket, _)| socket), listener))
                }))
                .await
                .unwrap();
        });
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let client = XaiVfClient::from_channel(channel);
        let tweet = |tweet_id, outer_tweet_id: Option<u64>| vf_pb::TweetData {
            tweet_id,
            quote_context: outer_tweet_id.map(|outer_tweet_id| vf_pb::QuoteContext {
                outer_tweet_id,
                outer_author_id: None,
            }),
        };
        let home = client
            .evaluate_tweets(vf_pb::EvaluateTweetsRequest {
                safety_level: 8,
                tweets: vec![tweet(1, None), tweet(1, Some(2)), tweet(2, None)],
                ..Default::default()
            })
            .await;
        handle.abort();
        let _ = handle.await;

        assert_eq!(
            home.unwrap(),
            vec![
                EvaluationResult::Evaluated(Box::new(Action::Allow(action::Allow::new()))),
                EvaluationResult::NotEvaluated,
                EvaluationResult::Failed,
            ]
        );
    }
}
