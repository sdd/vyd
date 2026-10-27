//! Shared serial stage/lifecycle operations for the migration runners.
use super::{Pipeline, PipelineContext, PipelineObserver};
use crate::caps::PipelinePortDeclarations;
use crate::frame::factory::FrameFactory;
use crate::statistics::{PipelineStatistics, StatsContext};
use crate::{FrameContext, FrameProcessorResult, FrameSink, VydError};
use std::sync::Arc;
use std::time::Instant;

/// Owns a run's stages and context. Runtime adapters serialize access to this
/// session while legacy stage methods still require `&mut PipelineContext`.
/// Call `finish` after every attempted `start`, including initialization errors.
/// This is a migration interface, not a concurrent graph scheduler.
pub struct PipelineSession {
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Arc<PipelineStatistics>,
    observers: Vec<Box<dyn PipelineObserver>>,
    observers_started: usize,
    source_started: bool,
    stages_started: usize,
    sink_started: bool,
    start_attempted: bool,
    ready: bool,
    started_at: Instant,
    frames: u64,
}

impl PipelineSession {
    pub fn new(
        context: PipelineContext,
        pipeline: Pipeline,
        stats: Arc<PipelineStatistics>,
    ) -> Self {
        Self {
            context,
            pipeline,
            stats,
            observers: Vec::new(),
            observers_started: 0,
            source_started: false,
            stages_started: 0,
            sink_started: false,
            start_attempted: false,
            ready: false,
            started_at: Instant::now(),
            frames: 0,
        }
    }

    pub fn processor_count(&self) -> usize {
        self.pipeline.stages.len()
    }

    pub fn port_declarations(&self) -> PipelinePortDeclarations {
        PipelinePortDeclarations {
            source: self.pipeline.source.inner.output_port(),
            processors: self
                .pipeline
                .stages
                .iter()
                .map(|stage| stage.inner.ports())
                .collect(),
            sink: self.pipeline.sink.inner.input_port(),
        }
    }

    /// Whether this run has the single GPU device required by a Wgpu port.
    /// This remains false in CPU-only builds.
    pub fn has_wgpu_context(&self) -> bool {
        #[cfg(feature = "gpu-wgpu")]
        {
            self.context.wgpu_context().is_ok()
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        {
            false
        }
    }

    pub fn frame_factory(&self) -> &FrameFactory {
        self.context.frame_factory()
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn gpu_transfer(
        &self,
    ) -> Result<&Arc<crate::gpu::transfer::GpuTransfer>, crate::gpu::GpuContextError> {
        self.context.gpu_transfer()
    }

    pub fn start(&mut self) -> Result<(), VydError> {
        if self.start_attempted {
            return Err(VydError::GeneralPluginError(
                "session already started".into(),
            ));
        }
        self.start_attempted = true;
        #[cfg(feature = "gpu-wgpu")]
        self.context.freeze_wgpu_context();
        #[cfg(feature = "gpu-wgpu")]
        self.context.install_gpu_transfer(self.stats.clone())?;
        let _scope = StatsContext::new(self.stats.clone()).enter();
        self.context.put("pipeline_statistics", self.stats.clone());
        self.context.services().publish(self.stats.clone())?;
        self.stats.register_stage(
            self.pipeline.source.stage_type,
            self.pipeline.source.get_processing_type(),
        );
        for stage in &self.pipeline.stages {
            self.stats
                .register_stage(stage.stage_type, stage.get_processing_type());
        }
        self.stats.register_stage(
            self.pipeline.sink.stage_type,
            self.pipeline.sink.processing_type(),
        );
        self.observers = std::mem::take(&mut self.context.observers);
        for observer in &mut self.observers {
            self.observers_started += 1;
            observer.pipeline_started(&mut self.context)?;
        }
        self.source_started = true;
        self.pipeline.source.pipeline_ctx_init(&mut self.context)?;
        for stage in &mut self.pipeline.stages {
            self.stages_started += 1;
            stage.pipeline_ctx_init(&mut self.context)?;
        }
        self.sink_started = true;
        self.pipeline.sink.pipeline_ctx_init(&mut self.context)?;
        self.ready = true;
        Ok(())
    }

    fn check_ready(&self) -> Result<(), VydError> {
        if self.ready {
            Ok(())
        } else {
            Err(VydError::GeneralPluginError(
                "session is not initialized".into(),
            ))
        }
    }

    pub fn next_frame(&mut self) -> Result<Option<FrameContext>, VydError> {
        self.check_ready()?;
        if self.stats.cancellation_token().is_cancelled() {
            return Ok(None);
        }
        let _scope = StatsContext::new(self.stats.clone()).enter();
        let frame = self.pipeline.source.try_next_frame(&mut self.context)?;
        if frame.is_some() {
            self.frames += 1;
        }
        Ok(frame)
    }

    pub fn process(
        &mut self,
        index: usize,
        frame: &mut FrameContext,
    ) -> Result<FrameProcessorResult, VydError> {
        self.check_ready()?;
        let _scope = StatsContext::new(self.stats.clone()).enter();
        let stage = self.pipeline.stages.get_mut(index).ok_or_else(|| {
            VydError::GeneralPluginError(format!("invalid processor index {index}"))
        })?;
        stage.try_process(frame, &mut self.context)
    }

    pub fn consume(&mut self, frame: &mut FrameContext) -> Result<(), VydError> {
        self.check_ready()?;
        let _scope = StatsContext::new(self.stats.clone()).enter();
        self.pipeline.sink.try_consume(frame, &mut self.context)
    }

    /// Complete an emitted or skipped frame. Failed/cancelled frames do not
    /// invoke frame observers. `started` excludes source acquisition time.
    pub fn complete_frame(
        &mut self,
        frame: &FrameContext,
        started: Instant,
    ) -> Result<(), VydError> {
        self.check_ready()?;
        let _scope = StatsContext::new(self.stats.clone()).enter();
        let duration_us = started.elapsed().as_micros() as u64;
        self.stats.record_frame(duration_us);
        super::timing::log_frame_timing_summary(frame, self.frames, duration_us);
        if self.frames.is_multiple_of(100) {
            let elapsed = self.started_at.elapsed().as_secs_f64();
            tracing::info!(
                frames_processed = self.frames,
                elapsed_seconds = elapsed,
                fps = self.frames as f64 / elapsed,
                "Pipeline throughput"
            );
        }
        for observer in &mut self.observers {
            observer.frame_processed(frame, &mut self.context)?;
        }
        Ok(())
    }

    /// Finalize every attempted stage/observer, preserving the original failure.
    /// Runtime callers must first stop streaming callbacks before consuming this
    /// session; arbitrary application code is not forcibly cancelled here.
    pub fn finish(mut self, result: Result<(), VydError>) -> Result<PipelineContext, VydError> {
        let _scope = StatsContext::new(self.stats.clone()).enter();
        let mut failure = result.err();
        let mut finish = |result: Result<(), VydError>| {
            if let Err(error) = result {
                tracing::error!(?error, "Pipeline finalizer failed");
                if failure.is_none() {
                    failure = Some(error);
                }
            }
        };
        if self.source_started {
            tracing::debug!(stage = self.pipeline.source.stage_type, "Finalizing source");
            finish(self.pipeline.source.pipeline_finished(&mut self.context));
            tracing::debug!(stage = self.pipeline.source.stage_type, "Source finalized");
        }
        for stage in self.pipeline.stages.iter_mut().take(self.stages_started) {
            tracing::debug!(stage = stage.stage_type, "Finalizing processor");
            finish(stage.pipeline_finished(&mut self.context));
            tracing::debug!(stage = stage.stage_type, "Processor finalized");
        }
        if self.sink_started {
            tracing::debug!(stage = self.pipeline.sink.stage_type, "Finalizing sink");
            finish(self.pipeline.sink.pipeline_finished(&mut self.context));
        }
        for observer in self.observers.iter_mut().take(self.observers_started) {
            finish(observer.pipeline_finished(&mut self.context));
        }
        self.context.observers.splice(0..0, self.observers);
        tracing::info!(
            total_frames = self.frames,
            total_duration_seconds = self.started_at.elapsed().as_secs_f64(),
            "Pipeline finished"
        );
        if let Some(error) = failure {
            Err(error)
        } else {
            Ok(self.context)
        }
    }
}
