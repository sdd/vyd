use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::traits::FrameProcessor;
use crate::{FrameContext, FrameProcessorResult, VydError, register_vyd_frame_processor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use toml::Value;

pub struct DummyProcessor {
    processing_delay_ms: Option<u64>,
}

impl DummyProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, VydError> {
        let processing_delay_ms = config
            .and_then(|c| c.get("processing_delay_ms"))
            .and_then(|d| d.as_integer())
            .map(|delay| {
                u64::try_from(delay).map_err(|_| {
                    VydError::PluginInvalidConfigError(
                        "processing_delay_ms must be non-negative".into(),
                    )
                })
            })
            .transpose()?;

        Ok(Self {
            processing_delay_ms,
        })
    }
}

impl FrameProcessor for DummyProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), VydError> {
        ctx.entry("frames_processed".to_string())
            .or_insert_with(|| Box::new(AtomicUsize::new(0)));

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        tracing::trace!(
            "Processor received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        // Apply processing delay if configured
        if let Some(delay_ms) = self.processing_delay_ms {
            tracing::trace!("Applying processing delay of {} ms", delay_ms);
            std::thread::sleep(Duration::from_millis(delay_ms));
        }

        if let Ok(counter) = ctx.try_get_as::<AtomicUsize>("frames_processed") {
            counter.fetch_add(1, Ordering::SeqCst);
        }

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "dummy_processor"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

register_vyd_frame_processor!(DummyProcessor);
