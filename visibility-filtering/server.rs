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
        VFServer::new(&ctx.datacenter).await
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
    pub(crate) async fn new(datacenter: &str) -> Self {
        crate::server_deps::build_prod_server(datacenter).await
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
    use crate::filter::FilterTweets;
    use crate::hydration::sources::InMemorySources;
    use crate::limited_actions_copy::LimitedActionsCopy;
    use crate::params::ClientSwitches;
    use crate::rules::RuleEngine;
    use crate::safety_label_source::lookup::{ManhattanLookup, RemoteSource, TwemcacheLookup};
    use crate::safety_label_source::types::{ManhattanOutcome, TwemcacheOutcome};
    use crate::safety_label_source::SafetyLabelSource;
    use std::collections::HashMap;
    use xai_visibility_filtering::evaluated::EvaluationResult;
    use xai_visibility_filtering::vf_client::XaiVfClient;
    use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;
    use xai_x_service_builder::XService;
    use xai_x_thrift::tweet_service::{TweetFieldsResultFound, TweetFieldsResultState};

    struct NoLabels;

    #[tonic::async_trait]
    impl TwemcacheLookup for NoLabels {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, TwemcacheOutcome> {
            ids.iter().map(|&id| (id, TwemcacheOutcome::Miss)).collect()
        }
    }

    #[tonic::async_trait]
    impl ManhattanLookup for NoLabels {
        async fn get(&self, ids: &[u64]) -> HashMap<u64, ManhattanOutcome> {
            ids.iter()
                .map(|&id| (id, ManhattanOutcome::Resolved(Default::default())))
                .collect()
        }
    }

    fn server(sources: InMemorySources) -> VFServer {
        let filter_tweets = Arc::new(FilterTweets::new(
            Arc::new(sources),
            RuleEngine::for_tests(),
        ));
        let labels = Arc::new(NoLabels);
        VFServer::from_endpoints(
            EvaluateTweetsEndpoint::new(
                filter_tweets.clone(),
                ClientSwitches::for_tests(),
                LimitedActionsCopy::from_json("[]"),
            ),
            FilterTweetsEndpoint::new(filter_tweets, None),
            GetSafetyLabelsEndpoint::new(Arc::new(SafetyLabelSource::new(Arc::new(
                RemoteSource::new(labels.clone(), labels),
            )))),
        )
    }

    async fn serve(server: VFServer) -> (tonic::transport::Channel, tokio::task::JoinHandle<()>) {
        let mut routes = tonic::service::RoutesBuilder::default();
        Arc::new(server).register(&mut routes);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_routes(routes.routes())
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
        (channel, handle)
    }

    #[tokio::test]
    async fn evaluate_tweets_loopback() {
        let (channel, handle) = serve(server(InMemorySources::default().tweet(1, 100))).await;
        let client = XaiVfClient::from_channel(channel);
        let tweet = |tweet_id, outer_tweet_id: Option<u64>| vf_pb::TweetData {
            tweet_id,
            quote_context: outer_tweet_id.map(|outer_tweet_id| vf_pb::QuoteContext {
                outer_tweet_id,
                outer_author_id: None,
            }),
        };
        let home = client
            .evaluate_tweets(
                vf_pb::EvaluateTweetsRequest {
                    safety_level: 8,
                    tweets: vec![tweet(1, None), tweet(1, Some(2)), tweet(2, None)],
                    ..Default::default()
                },
                &Default::default(),
            )
            .await;
        handle.abort();
        let _ = handle.await;

        assert_eq!(
            home.unwrap(),
            vec![
                EvaluationResult::Evaluated(Box::new(TweetFieldsResultState::Found(
                    TweetFieldsResultFound::new(None)
                ))),
                EvaluationResult::NotEvaluated,
                EvaluationResult::Failed,
            ]
        );
    }

    #[tokio::test]
    async fn filter_tweets_loopback() {
        let (channel, handle) = serve(server(InMemorySources::default())).await;
        let mut client = VisibilityFilteringServiceClient::new(channel)
            .send_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Gzip);
        let tweet = |tweet_id, author_id| vf_pb::TweetInput {
            tweet_id,
            author_id,
        };
        let response = client
            .filter_tweets(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: vec![tweet(2, Some(20)), tweet(1, None), tweet(2, Some(20))],
                viewer_id: None,
                country_code: None,
            })
            .await;
        handle.abort();
        let _ = handle.await;

        let results = response.unwrap().into_inner().results;
        let wire = |result: &vf_pb::TweetVisibilityResult| {
            (
                result.tweet_id,
                result.action.and_then(|a| a.kind),
                result.filtered_reason.clone().and_then(|r| r.reason),
            )
        };
        let allow = vf_pb::action::Kind::Allow(true);
        let drop = vf_pb::action::Kind::Drop(vf_pb::DropReason {});
        let unspecified = vf_pb::filtered_reason::Reason::UnspecifiedReason(true);
        assert_eq!(
            results.iter().map(wire).collect::<Vec<_>>(),
            vec![
                (2, Some(allow), None),
                (1, Some(drop), Some(unspecified)),
                (2, Some(allow), None),
            ]
        );
    }
}
