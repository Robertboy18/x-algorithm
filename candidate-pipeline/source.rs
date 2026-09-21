use std::any::{type_name_of_val, Any};
use tonic::async_trait;

use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::record_source_fetched;
use crate::util;
use crate::SPAN_LEVEL;
use tracing::error;

#[async_trait]
pub trait Source<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(size=Bucket500To1000)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "source", fields(name = self.name()))]
    async fn run(&self, query: &Q, _stage: PipelineStage) -> Result<Vec<C>, String> {
        match self.source(query).await {
            Ok(candidates) => {
                record_source_fetched(self.name(), candidates.len());
                Ok(candidates)
            }
            Err(err) => {
                error!("{} Failed: {}", self.name(), err);
                Err(err)
            }
        }
    }

    async fn source(&self, query: &Q) -> Result<Vec<C>, String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
