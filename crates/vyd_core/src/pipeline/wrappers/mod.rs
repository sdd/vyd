use crate::pipeline::PipelineContext;
use crate::statistics::{PipelineStatistics, ProcessingType, StatsContext, StatsScope};
use crate::traits::{FrameProcessor, FrameSink, FrameSource};
use crate::{FrameContext, FrameProcessorResult, VydError};
use std::sync::Arc;
use std::time::Instant;

/// Common storage for a configured pipeline stage, tagged with its registered name.
pub struct Stage<T: ?Sized> {
    pub inner: Box<T>,
    pub stage_type: &'static str,
}

impl<T: ?Sized> Stage<T> {
    pub fn new(inner: Box<T>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }
}

pub type FrameSourceWrapper = Stage<dyn FrameSource>;
pub type FrameProcessorWrapper = Stage<dyn FrameProcessor>;
pub type FrameSinkWrapper = Stage<dyn FrameSink>;

impl Stage<dyn FrameSource> {
    pub fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.next_frame(ctx)
    }

    pub fn try_next_frame(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Result<Option<FrameContext>, VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner
            .try_next_frame(ctx)
            .map_err(|error| stage_error(self.stage_type, "next_frame", error))
    }

    pub fn get_processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.pipeline_ctx_init(ctx)
    }
    pub(crate) fn pipeline_finished(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.pipeline_finished(ctx)
    }
}

impl Stage<dyn FrameProcessor> {
    pub fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let _scope = stage_scope(ctx, self.stage_type);
        let start_time = Instant::now();
        let result = self.inner.process(frame_ctx, ctx);
        record_duration(ctx, self.stage_type, start_time);
        result
    }

    pub fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        let start_time = Instant::now();
        let result = self.inner.try_process(frame_ctx, ctx);
        record_duration(ctx, self.stage_type, start_time);
        result.map_err(|error| stage_error(self.stage_type, "process", error))
    }

    pub fn get_processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.pipeline_ctx_init(ctx)
    }

    pub(crate) fn pipeline_finished(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.pipeline_finished(ctx)
    }
}

impl FrameSink for Stage<dyn FrameSink> {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        let _scope = stage_scope(ctx, self.stage_type);
        let start_time = Instant::now();
        self.inner.consume(frame_ctx, ctx);
        record_duration(ctx, self.stage_type, start_time);
    }

    fn try_consume(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        let start_time = Instant::now();
        let result = self.inner.try_consume(frame_ctx, ctx);
        record_duration(ctx, self.stage_type, start_time);
        result.map_err(|error| stage_error(self.stage_type, "consume", error))
    }

    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.pipeline_ctx_init(ctx)
    }

    fn pipeline_finished(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        let _scope = stage_scope(ctx, self.stage_type);
        self.inner.pipeline_finished(ctx)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }
}

fn stage_scope(ctx: &PipelineContext, stage: &'static str) -> Option<StatsScope> {
    ctx.try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics")
        .ok()
        .map(|stats| StatsContext::new(stats.clone()).for_stage(stage).enter())
}

fn stage_error(stage: &'static str, operation: &'static str, error: VydError) -> VydError {
    VydError::StageFailed {
        stage,
        operation,
        source: Box::new(error),
    }
}

fn record_duration(ctx: &PipelineContext, stage_type: &'static str, start_time: Instant) {
    if let Ok(stats) = ctx.try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
        stats.record_stage_timing(stage_type, start_time.elapsed().as_micros() as u64);
    }
}
