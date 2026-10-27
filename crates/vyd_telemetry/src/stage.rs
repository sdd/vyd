use crate::vortex_file::{VortexReader, VortexWriter};
use crate::{StageCodec, record};
use anyhow::{Context, anyhow};
use std::sync::{Arc, Mutex};
use vyd_core::caps::ProcessorPortSpecs;
use vyd_core::pipeline::PipelineContext;
use vyd_core::statistics::ProcessingType;
use vyd_core::{FrameContext, FrameProcessor, FrameProcessorResult, VydError};

pub struct DumpingProcessor {
    inner: Box<dyn FrameProcessor>,
    codec: Arc<dyn StageCodec>,
    writer: Arc<Mutex<VortexWriter>>,
}

impl DumpingProcessor {
    pub fn new(
        inner: Box<dyn FrameProcessor>,
        codec: Arc<dyn StageCodec>,
        writer: Arc<Mutex<VortexWriter>>,
    ) -> Self {
        Self {
            inner,
            codec,
            writer,
        }
    }
}

impl FrameProcessor for DumpingProcessor {
    fn ports(&self) -> ProcessorPortSpecs {
        self.inner.ports()
    }

    fn process(
        &mut self,
        frame: &mut FrameContext,
        context: &mut PipelineContext,
    ) -> FrameProcessorResult {
        self.try_process(frame, context)
            .unwrap_or_else(|error| panic!("stage dump failed: {error}"))
    }

    fn try_process(
        &mut self,
        frame: &mut FrameContext,
        context: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, VydError> {
        let result = self.inner.try_process(frame, context)?;
        let batch = record::capture(self.codec.as_ref(), frame, result)?;
        self.writer
            .lock()
            .map_err(|_| anyhow!("Vortex writer lock poisoned"))?
            .write(batch)
            .context("write Vortex frame")?;
        Ok(result)
    }

    fn pipeline_ctx_init(&mut self, context: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_ctx_init(context)
    }

    fn pipeline_finished(&mut self, context: &mut PipelineContext) -> Result<(), VydError> {
        self.inner.pipeline_finished(context)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn processing_type(&self) -> ProcessingType {
        self.inner.processing_type()
    }
}

pub struct ReplayProcessor {
    codec: Arc<dyn StageCodec>,
    reader: VortexReader,
    allow_prefix_end: bool,
}

impl ReplayProcessor {
    pub fn new(codec: Arc<dyn StageCodec>, reader: VortexReader, allow_prefix_end: bool) -> Self {
        Self {
            codec,
            reader,
            allow_prefix_end,
        }
    }
}

impl FrameProcessor for ReplayProcessor {
    fn ports(&self) -> ProcessorPortSpecs {
        self.codec.ports()
    }

    fn process(
        &mut self,
        frame: &mut FrameContext,
        context: &mut PipelineContext,
    ) -> FrameProcessorResult {
        self.try_process(frame, context)
            .unwrap_or_else(|error| panic!("stage replay failed: {error}"))
    }

    fn try_process(
        &mut self,
        frame: &mut FrameContext,
        context: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, VydError> {
        let row = self.reader.next_row().context("read Vortex replay frame")?;
        let Some(row) = row else {
            if self.allow_prefix_end {
                context
                    .try_get_as::<Arc<vyd_core::statistics::PipelineStatistics>>(
                        "pipeline_statistics",
                    )?
                    .cancellation_token()
                    .cancel();
                return Ok(FrameProcessorResult::Skip);
            }
            return Err(VydError::PersistenceError(anyhow!(
                "replay ends before frame {}",
                frame.frame_index
            )));
        };
        record::restore(self.codec.as_ref(), frame, &row)
    }

    fn pipeline_finished(&mut self, context: &mut PipelineContext) -> Result<(), VydError> {
        let cancelled = context
            .try_get_as::<Arc<vyd_core::statistics::PipelineStatistics>>("pipeline_statistics")
            .is_ok_and(|stats| stats.cancellation_token().is_cancelled());
        if !cancelled
            && self
                .reader
                .next_row()
                .context("check Vortex replay end")?
                .is_some()
        {
            return Err(VydError::PersistenceError(anyhow!(
                "replay has frames after source end"
            )));
        }
        Ok(self.reader.close().context("close Vortex replay reader")?)
    }

    fn name(&self) -> &str {
        "vortex_replay"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
