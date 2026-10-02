use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::{FrameContext, FrameProcessorResult, VydError};
use std::any::Any;

/// Represents a type that can be dumped to parquet for offline analysis
pub trait Dumpable: Send + Sync + 'static {
    /// Flat row type suitable for Parquet export
    type Row: for<'a> serde::Deserialize<'a> + serde::Serialize + Send + Sync;

    /// Identifier fragment used for output filenames
    const TABLE_NAME: &'static str;

    /// Schema version (bump if layout of Row changes)
    const VERSION: u32;

    /// Convert into a row, always including run_id and frame_index
    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row;
}

pub trait FrameSource: Send + Sync {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext>;
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }
    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait FrameProcessor: Send + Sync {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult;
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }
    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait FrameSink: Send + Sync {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext);
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }

    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait StageFactory: Send + Sync {
    fn create(&self, params: Option<&toml::Value>) -> Result<Box<dyn Any>, VydError>;
    fn stage_type(&self) -> &'static str;
}
