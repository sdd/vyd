use crate::caps::{FramePortSpecs, ProcessorPortSpecs};
use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::{FrameContext, FrameProcessorResult, VydError};
use std::any::Any;

pub trait FrameSource: Send + Sync {
    /// Formats this source can emit. The default matches the legacy Frame API.
    fn output_port(&self) -> FramePortSpecs {
        FramePortSpecs::default()
    }
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext>;
    /// Fallible runtime entry point. `Ok(None)` means normal end of stream.
    /// During migration, the default delegates to the existing source method.
    fn try_next_frame(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Result<Option<FrameContext>, VydError> {
        Ok(self.next_frame(ctx))
    }
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }
    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }
    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait FrameProcessor: Send + Sync {
    /// Input/output pairs this processor can handle. Format-changing pairs
    /// require a corresponding adapter; declaring them never converts pixels.
    fn ports(&self) -> ProcessorPortSpecs {
        ProcessorPortSpecs::default()
    }
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult;
    /// Fallible runtime entry point. Skip remains a successful processing result.
    fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, VydError> {
        Ok(self.process(frame_ctx, ctx))
    }
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
    /// Formats this sink accepts. The default matches the legacy Frame API.
    fn input_port(&self) -> FramePortSpecs {
        FramePortSpecs::default()
    }
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext);
    /// Fallible runtime entry point; defaults to the existing sink method.
    fn try_consume(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<(), VydError> {
        self.consume(frame_ctx, ctx);
        Ok(())
    }
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
