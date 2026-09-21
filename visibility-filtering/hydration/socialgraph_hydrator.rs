use crate::clients::socialgraph_client::SocialgraphClient;
use crate::hydration::batch::{HydrationBatch, TweetHydrationBatch};
use crate::hydration::metrics::{record_batch_size, timed_values};
use crate::hydration::{keyed_by_author, tweets_per_author};
use crate::models::{TweetCandidateInput, Viewer, ViewerAuthorRelationship};
use crate::rules::SafetyLevel;
use std::sync::Arc;
use std::time::Duration;

const CLIENT_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const CLIENT: &str = "socialgraph";

pub struct SocialgraphHydrator {
    pub sg_client: Arc<dyn SocialgraphClient + Send + Sync>,
}

impl SocialgraphHydrator {
    pub(crate) async fn hydrate(
        &self,
        candidates: &[TweetCandidateInput],
        viewer: Viewer,
        safety_level: SafetyLevel,
    ) -> TweetHydrationBatch<ViewerAuthorRelationship> {
        let Some(vid) = viewer.user_id() else {
            return HydrationBatch::from_values(
                candidates.iter().map(|c| c.tweet_id),
                candidates
                    .iter()
                    .map(|c| (c.tweet_id, ViewerAuthorRelationship::default()))
                    .collect(),
            );
        };

        let candidate_count_by_key = tweets_per_author(candidates);
        let author_ids: Vec<u64> = candidate_count_by_key.keys().map(|a| a.get()).collect();

        record_batch_size(CLIENT, candidates.len());
        let relationships = timed_values(
            CLIENT,
            "batch_check_relationships",
            safety_level,
            &candidate_count_by_key,
            CLIENT_TIMEOUT,
            async {
                let response = self
                    .sg_client
                    .batch_check_relationships(vid, &author_ids)
                    .await;
                keyed_by_author(&candidate_count_by_key, response)
            },
        )
        .await;

        relationships.project(candidates.iter().map(|c| (c.tweet_id, c.author_id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::FakeSocialgraphClient;
    use crate::hydration::batch::Hydrated;
    use crate::models::{resolve_candidate, RawCandidate, TweetId};
    use std::collections::HashMap;
    use xai_core_entities::entities::PureCoreData;

    fn candidate(tweet_id: u64, author_id: u64) -> TweetCandidateInput {
        resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(tweet_id),
                request_author_id: Some(author_id),
            },
            &HashMap::<TweetId, PureCoreData>::new(),
            &HashMap::new(),
        )
        .unwrap()
    }

    struct EmptyResponseSocialgraphClient;

    #[tonic::async_trait]
    impl SocialgraphClient for EmptyResponseSocialgraphClient {
        async fn batch_check_relationships(
            &self,
            _viewer_id: u64,
            _author_ids: &[u64],
        ) -> HashMap<u64, ViewerAuthorRelationship> {
            HashMap::new()
        }

        async fn batch_check_super_follows(
            &self,
            _viewer_id: u64,
            _author_ids: &[u64],
        ) -> Option<HashMap<u64, bool>> {
            Some(HashMap::new())
        }
    }

    #[tokio::test]
    async fn rpc_empty_response_marks_every_key_failed() {
        let hydrator = SocialgraphHydrator {
            sg_client: Arc::new(EmptyResponseSocialgraphClient),
        };
        let candidates = vec![candidate(1, 10), candidate(2, 20)];

        let relationships = hydrator
            .hydrate(&candidates, Viewer::LoggedIn(99), SafetyLevel::TimelineHome)
            .await;

        for tweet_id in [TweetId(1), TweetId(2)] {
            assert!(matches!(
                relationships.hydrated(&tweet_id),
                Some(Hydrated::Failed(_))
            ));
        }
    }

    #[tokio::test]
    async fn logged_out_viewer_is_found_default_even_for_duplicate_tweets() {
        let hydrator = SocialgraphHydrator {
            sg_client: Arc::new(FakeSocialgraphClient),
        };
        let candidates = vec![candidate(1, 10), candidate(1, 10), candidate(2, 20)];

        let relationships = hydrator
            .hydrate(&candidates, Viewer::LoggedOut, SafetyLevel::TimelineHome)
            .await;

        for tweet_id in [TweetId(1), TweetId(2)] {
            assert!(matches!(
                relationships.hydrated(&tweet_id),
                Some(Hydrated::Found(_))
            ));
        }
    }
}
