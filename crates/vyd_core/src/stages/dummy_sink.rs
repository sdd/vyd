use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::traits::FrameSink;
use crate::{FrameContext, VydError, register_vyd_frame_sink};
use std::sync::atomic::{AtomicUsize, Ordering};
use toml::Value;

pub struct DummySink;

impl DummySink {
    pub fn new(_config: Option<&Value>) -> Result<Self, VydError> {
        Ok(Self)
    }
}

impl FrameSink for DummySink {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        ctx.entry("frames_sunk".to_string())
            .or_insert_with(|| Box::new(AtomicUsize::new(0)));

        Ok(())
    }

    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        tracing::trace!(
            "Sink received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        if let Ok(counter) = ctx.try_get_as::<AtomicUsize>("frames_sunk") {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn name(&self) -> &str {
        "dummy_sink"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

register_vyd_frame_sink!(DummySink, "vyd");
