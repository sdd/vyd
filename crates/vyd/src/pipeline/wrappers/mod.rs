use crate::pipeline::PipelineContext;
use crate::statistics::PipelineStatistics;
use crate::statistics::{ProcessingType, StatsContext};
use crate::traits::{FrameProcessor, FrameSink, FrameSource};
use crate::{FrameContext, FrameProcessorResult, VydError};
use std::sync::Arc;
use std::time::Instant;

pub struct FrameSourceWrapper {
    pub inner: Box<dyn FrameSource>,
    pub stage_type: &'static str,
}

impl FrameSourceWrapper {
    pub fn new(inner: Box<dyn FrameSource>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        // Set current stage context for Frame operations
        StatsContext::set_current_stage(self.stage_type.to_string());

        let result = self.inner.next_frame(ctx);

        StatsContext::clear_current_stage();
        result
    }

    pub fn get_processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_ctx_init(ctx)
    }
}

pub struct FrameProcessorWrapper {
    pub inner: Box<dyn FrameProcessor>,
    pub stage_type: &'static str,
}

impl FrameProcessorWrapper {
    pub fn new(inner: Box<dyn FrameProcessor>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let start_time = Instant::now();

        // Set current stage context for Frame operations
        StatsContext::set_current_stage(self.stage_type.to_string());

        let result = self.inner.process(frame_ctx, ctx);
        let duration_us = start_time.elapsed().as_micros() as u64;

        // Record timing if statistics are available
        if let Ok(stats) = ctx.try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
            stats.record_stage_timing(self.stage_type, duration_us);
        }

        StatsContext::clear_current_stage();
        result
    }

    pub fn get_processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_ctx_init(ctx)
    }

    pub(crate) fn pipeline_finished(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_finished(ctx)
    }
}

pub struct FrameSinkWrapper {
    pub inner: Box<dyn FrameSink>,
    pub stage_type: &'static str,
}

impl FrameSinkWrapper {
    pub fn new(inner: Box<dyn FrameSink>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }
}

impl FrameSink for FrameSinkWrapper {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        let start_time = Instant::now();

        StatsContext::set_current_stage(self.stage_type.to_string());

        self.inner.consume(frame_ctx, ctx);
        let duration_us = start_time.elapsed().as_micros() as u64;

        // Record timing if statistics are available
        if let Ok(stats) = ctx.try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
            stats.record_stage_timing(self.stage_type, duration_us);
        }

        StatsContext::clear_current_stage();
    }

    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_ctx_init(ctx)
    }

    fn pipeline_finished(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_finished(ctx)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }
}
