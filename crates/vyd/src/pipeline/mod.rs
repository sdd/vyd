use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use toml::Value;

pub(crate) mod timing;
pub(crate) mod wrappers;
pub use wrappers::Stage;

use crate::pipeline::timing::log_frame_timing_summary;
use crate::pipeline::wrappers::{FrameProcessorWrapper, FrameSinkWrapper, FrameSourceWrapper};
use crate::statistics::{PipelineStatistics, StatsContext};
use crate::traits::{FrameProcessor, FrameSink, FrameSource, StageFactory};
use crate::{FrameProcessorResult, VydError};

pub type PipelineContextValue = Box<dyn Any + Send + Sync>;

pub struct PipelineContext {
    metadata: HashMap<String, PipelineContextValue>,
    observers: Vec<Box<dyn PipelineObserver>>,
}

impl std::fmt::Debug for PipelineContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineContext")
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .field("observer_count", &self.observers.len())
            .finish()
    }
}

/// Lifecycle hooks for telemetry, logging, or other pipeline extensions.
/// Started hooks run before stage initialization. Finished hooks run after stage
/// finalizers, including when initialization or a frame hook returns an error.
pub trait PipelineObserver: Send + Sync {
    fn pipeline_started(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }

    /// Called for every sourced frame, including frames skipped by a processor.
    fn frame_processed(
        &mut self,
        _frame: &crate::FrameContext,
        _ctx: &mut PipelineContext,
    ) -> Result<(), VydError> {
        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), VydError> {
        Ok(())
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineContext {
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
            observers: Vec::new(),
        }
    }

    pub fn add_observer(&mut self, observer: impl PipelineObserver + 'static) {
        self.observers.push(Box::new(observer));
    }

    pub fn entry<K: Into<String>>(&mut self, key: K) -> Entry<'_, String, PipelineContextValue> {
        self.metadata.entry(key.into())
    }

    pub fn try_get_as<T: 'static>(&self, key: &str) -> Result<&T, VydError> {
        let Some(value) = self.metadata.get(key) else {
            tracing::trace!("key \"{}\" not present in frame context metadata", key);
            return Err(VydError::FrameMetadataNotFoundError);
        };

        let Some(downcasted) = value.downcast_ref::<T>() else {
            tracing::error!("Could not downcast metadata value to requested type");
            return Err(VydError::FrameMetadataTypeError);
        };

        Ok(downcasted)
    }

    pub fn try_get_as_mut<T: 'static>(&mut self, key: &str) -> Result<&mut T, VydError> {
        let Some(value) = self.metadata.get_mut(key) else {
            tracing::trace!("key \"{}\" not present in frame context metadata", key);
            return Err(VydError::FrameMetadataNotFoundError);
        };

        let Some(downcasted) = value.downcast_mut::<T>() else {
            tracing::error!("Could not downcast metadata value to requested type");
            return Err(VydError::FrameMetadataTypeError);
        };

        Ok(downcasted)
    }

    pub fn get_as<T: 'static>(&self, key: &str) -> &T {
        self.try_get_as(key).unwrap()
    }

    pub fn get_as_or_insert<T: Send + Sync + 'static, F>(&mut self, key: &str, default: F) -> &T
    where
        F: FnOnce() -> T,
    {
        match self.metadata.entry(key.to_string()) {
            Entry::Occupied(entry) => entry.into_mut().downcast_mut::<T>().expect(&format!(
                "Could not downcast metadata value to requested type '{}'",
                core::any::type_name::<T>()
            )),
            Entry::Vacant(entry) => {
                let value = default();
                entry.insert(Box::new(value)).downcast_mut::<T>().unwrap()
            }
        }
    }

    pub fn put<T: Any + Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.metadata.insert(key.to_string(), Box::new(value));
    }
}

#[derive(Deserialize, Debug)]
pub struct PipelineConfig {
    source: StageConfig,
    #[serde(default)]
    stages: Vec<StageConfig>,
    sink: StageConfig,
}

impl PipelineConfig {
    pub fn source_type(&self) -> &str {
        &self.source.stage_type
    }
}

#[derive(Deserialize, Debug)]
struct StageConfig {
    #[allow(unused)]
    name: Option<String>,
    stage_type: String,

    #[serde(flatten)]
    params: Option<Value>,
}

pub struct Pipeline {
    pub source: Stage<dyn FrameSource>,
    pub stages: Vec<Stage<dyn FrameProcessor>>,
    pub sink: Stage<dyn FrameSink>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("source", &self.source.stage_type)
            .field(
                "stages",
                &self.stages.iter().map(|s| s.stage_type).collect::<Vec<_>>(),
            )
            .field("sink", &self.sink.stage_type)
            .finish()
    }
}

inventory::collect!(&'static dyn StageFactory);

#[macro_export]
macro_rules! register_vyd_frame_source {
    ($source_type:ty) => {
        $crate::_register_vyd_stage_impl!(env!("CARGO_PKG_NAME"), $source_type, FrameSource);
    };
}

#[macro_export]
macro_rules! register_vyd_frame_processor {
    ($processor_type:ty) => {
        $crate::_register_vyd_stage_impl!(env!("CARGO_PKG_NAME"), $processor_type, FrameProcessor);
    };
}

#[macro_export]
macro_rules! register_vyd_frame_sink {
    ($sink_type:ty) => {
        $crate::_register_vyd_stage_impl!(env!("CARGO_PKG_NAME"), $sink_type, FrameSink);
    };
}

#[macro_export]
macro_rules! _register_vyd_stage_impl {
    ($crate_name:expr, $stage_type:ty, $trait_name:ident) => {
        $crate::paste::paste! {
            struct [<$stage_type Factory>];

            impl $crate::StageFactory for [<$stage_type Factory>] {
                fn create(&self, params: Option<&$crate::toml::Value>) -> Result<Box<dyn std::any::Any>, $crate::VydError> {
                    let instance = <$stage_type>::new(params)?;
                    Ok(Box::new(Box::new(instance) as Box<dyn $crate::$trait_name>) as Box<dyn std::any::Any>)
                }

                fn stage_type(&self) -> &'static str {
                    $crate::paste::paste! {
                        concat!($crate_name, "::", stringify!($stage_type))
                    }
                }
            }

            static [<$crate_name:snake:upper __ $stage_type:snake:upper _FACTORY>]: [<$stage_type Factory>] = [<$stage_type Factory>];

            $crate::inventory::submit! {
                &[<$crate_name:snake:upper __ $stage_type:snake:upper _FACTORY>] as &'static dyn $crate::StageFactory
            }
        }
    };
}

pub fn build_pipeline(config: &PipelineConfig) -> Result<Pipeline, VydError> {
    tracing::trace!(?config, "Building pipeline");

    // Create specific factory functions for each trait type
    fn find_source_factory(
        type_name: &str,
        params: Option<&Value>,
    ) -> Result<FrameSourceWrapper, VydError> {
        for factory in inventory::iter::<&dyn StageFactory>() {
            tracing::trace!("checking factory: {:?}", factory.stage_type());
            if factory.stage_type() == type_name {
                tracing::trace!(
                    "Found factory for source type: {}. Sending params {:?}",
                    type_name,
                    &params
                );
                let any_box = factory.create(params)?;

                // Try to downcast to Box<dyn FrameSource>
                let inner = *any_box.downcast::<Box<dyn FrameSource>>().map_err(|_| {
                    VydError::StageTypeMismatch {
                        kind: "source",
                        stage_type: type_name.into(),
                    }
                })?;

                return Ok(FrameSourceWrapper::new(inner, factory.stage_type()));
            }
        }
        Err(VydError::UnknownStage {
            kind: "source",
            stage_type: type_name.into(),
        })
    }

    fn find_processor_factory(
        type_name: &str,
        params: Option<&toml::Value>,
    ) -> Result<FrameProcessorWrapper, VydError> {
        for factory in inventory::iter::<&dyn StageFactory>() {
            if factory.stage_type() == type_name {
                let any_box = factory.create(params)?;

                let inner = *any_box.downcast::<Box<dyn FrameProcessor>>().map_err(|_| {
                    VydError::StageTypeMismatch {
                        kind: "processor",
                        stage_type: type_name.into(),
                    }
                })?;

                return Ok(FrameProcessorWrapper::new(inner, factory.stage_type()));
            }
        }
        Err(VydError::UnknownStage {
            kind: "processor",
            stage_type: type_name.into(),
        })
    }

    fn find_sink_factory(
        type_name: &str,
        params: Option<&toml::Value>,
    ) -> Result<FrameSinkWrapper, VydError> {
        for factory in inventory::iter::<&dyn StageFactory>() {
            if factory.stage_type() == type_name {
                let any_box = factory.create(params)?;

                let inner = *any_box.downcast::<Box<dyn FrameSink>>().map_err(|_| {
                    VydError::StageTypeMismatch {
                        kind: "sink",
                        stage_type: type_name.into(),
                    }
                })?;

                return Ok(FrameSinkWrapper::new(inner, factory.stage_type()));
            }
        }
        Err(VydError::UnknownStage {
            kind: "sink",
            stage_type: type_name.into(),
        })
    }

    tracing::trace!(params = ?config.source.params.as_ref(),
        stage_type = ?&config.source.stage_type,
        "trying to create source",
    );
    let source = find_source_factory(&config.source.stage_type, config.source.params.as_ref())?;
    let sink = find_sink_factory(&config.sink.stage_type, config.sink.params.as_ref())?;

    let mut stages: Vec<FrameProcessorWrapper> = Vec::new();
    for stage in &config.stages {
        stages.push(find_processor_factory(
            &stage.stage_type,
            stage.params.as_ref(),
        )?);
    }

    Ok(Pipeline {
        source,
        stages,
        sink,
    })
}

pub fn run_pipeline(
    mut pipeline_context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
) -> Result<PipelineContext, VydError> {
    let Pipeline {
        mut source,
        mut stages,
        mut sink,
    } = pipeline;
    let stats = stats.unwrap_or_else(|| Arc::new(PipelineStatistics::new()));
    pipeline_context.put("pipeline_statistics", stats.clone());
    StatsContext::set_global_stats(stats.clone());
    let _stats_scope = StatisticsScope;

    stats.register_stage(source.stage_type, source.get_processing_type());
    for stage in &stages {
        stats.register_stage(stage.stage_type, stage.get_processing_type());
    }
    stats.register_stage(sink.stage_type, sink.processing_type());

    let mut observers = std::mem::take(&mut pipeline_context.observers);
    let mut observers_started = 0;
    let mut source_started = false;
    let mut stages_started = 0;
    let mut sink_started = false;
    let pipeline_start = Instant::now();
    let mut frame_count = 0u64;

    let result = (|| {
        for observer in &mut observers {
            observers_started += 1;
            observer.pipeline_started(&mut pipeline_context)?;
        }
        source_started = true;
        source.pipeline_ctx_init(&mut pipeline_context)?;
        for stage in &mut stages {
            stages_started += 1;
            stage.pipeline_ctx_init(&mut pipeline_context)?;
        }
        sink_started = true;
        sink.pipeline_ctx_init(&mut pipeline_context)?;

        while stats.is_running() {
            let Some(mut frame) = source.next_frame(&mut pipeline_context) else {
                break;
            };
            let frame_start = Instant::now();
            frame_count += 1;
            let mut continue_processing = true;
            for stage in &mut stages {
                if matches!(
                    stage.process(&mut frame, &mut pipeline_context),
                    FrameProcessorResult::Skip
                ) {
                    continue_processing = false;
                    break;
                }
            }
            if continue_processing {
                sink.consume(&mut frame, &mut pipeline_context);
            }
            let duration_us = frame_start.elapsed().as_micros() as u64;
            stats.record_frame(duration_us);
            log_frame_timing_summary(&frame, frame_count, duration_us);
            for observer in &mut observers {
                observer.frame_processed(&frame, &mut pipeline_context)?;
            }
            if frame_count.is_multiple_of(100) {
                let elapsed = pipeline_start.elapsed().as_secs_f64();
                tracing::info!(
                    frames_processed = frame_count,
                    elapsed_seconds = elapsed,
                    fps = frame_count as f64 / elapsed,
                    "Pipeline throughput"
                );
            }
        }
        Ok(())
    })();

    // Attempt every finalizer even if a preceding hook or finalizer failed.
    let mut failure = result.err();
    let mut finish = |result: Result<(), VydError>| {
        if let Err(error) = result {
            tracing::error!(?error, "Pipeline finalizer failed");
            if failure.is_none() {
                failure = Some(error);
            }
        }
    };
    if source_started {
        finish(source.pipeline_finished(&mut pipeline_context));
    }
    for stage in stages.iter_mut().take(stages_started) {
        finish(stage.pipeline_finished(&mut pipeline_context));
    }
    if sink_started {
        finish(sink.pipeline_finished(&mut pipeline_context));
    }
    for observer in observers.iter_mut().take(observers_started) {
        finish(observer.pipeline_finished(&mut pipeline_context));
    }
    pipeline_context.observers.splice(0..0, observers);

    tracing::info!(
        total_frames = frame_count,
        total_duration_seconds = pipeline_start.elapsed().as_secs_f64(),
        "Pipeline finished"
    );
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(pipeline_context)
}

struct StatisticsScope;
impl Drop for StatisticsScope {
    fn drop(&mut self) {
        StatsContext::clear_global_stats();
    }
}

pub fn run_pipeline_with_config_file_path(
    path: &std::path::Path,
    stats: Arc<PipelineStatistics>,
) -> Result<PipelineContext, VydError> {
    let config_raw = std::fs::read_to_string(path)?;
    let config: PipelineConfig = toml::from_str(&config_raw)?;
    run_pipeline(
        PipelineContext::new(),
        build_pipeline(&config)?,
        Some(stats),
    )
}
