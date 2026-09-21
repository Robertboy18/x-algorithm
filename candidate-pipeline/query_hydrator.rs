use std::any::{type_name_of_val, Any};
use tonic::async_trait;

use crate::candidate_pipeline::{PipelineQuery, PipelineStage};
use crate::util;
use crate::SPAN_LEVEL;
use tracing::error;

#[async_trait]
pub trait QueryHydrator<Q>: Any + Send + Sync
where
    Q: PipelineQuery,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "query_hydrator", fields(name = self.name()))]
    async fn run(&self, query: &Q, _stage: PipelineStage) -> Result<Q, String> {
        match self.hydrate(query).await {
            Ok(hydrated) => Ok(hydrated),
            Err(err) => {
                error!("{} Failed: {}", self.name(), err);
                Err(err)
            }
        }
    }

    async fn hydrate(&self, query: &Q) -> Result<Q, String>;

    fn update(&self, query: &mut Q, hydrated: Q);

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
