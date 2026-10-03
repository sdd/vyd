use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::traits::FrameProcessor;
use crate::{
    DumpManager, Dumpable, FrameContext, FrameProcessorResult, VydError,
    register_vyd_frame_processor,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use toml::Value;

#[derive(Debug, Deserialize, Serialize)]
pub struct ExampleDataRow {
    pub run_id: u64,
    pub frame_index: usize,
    pub value: String,
}

/// Example data to dump
pub struct ExampleData {
    pub value: String,
}

impl Dumpable for ExampleData {
    type Row = ExampleDataRow;

    const TABLE_NAME: &'static str = "example_data";
    const VERSION: u32 = 1;

    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row {
        ExampleDataRow {
            run_id,
            frame_index,
            value: self.value.clone(),
        }
    }
}

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

        // Create and dump example data
        let example_data = ExampleData {
            value: format!("Frame {} from {}", frame_ctx.frame_index, self.name()),
        };

        // Check if we have a dump manager in the pipeline context
        if let Ok(dump_manager) = ctx.try_get_as_mut::<DumpManager>("dump_manager") {
            let result = dump_manager.dumper::<ExampleData>().and_then(|dumper| {
                dumper
                    .lock()
                    .map_err(|e| anyhow::anyhow!("dumper lock poisoned: {e}"))?
                    .dump(&example_data, frame_ctx.frame_index)
            });
            if let Err(err) = result {
                tracing::error!("Failed to dump example data: {:?}", err);
            }
        };

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
