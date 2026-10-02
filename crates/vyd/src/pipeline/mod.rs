use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use toml::Value;

pub(crate) mod timing;
pub(crate) mod wrappers;

use crate::pipeline::timing::log_frame_timing_summary;
use crate::pipeline::wrappers::{FrameProcessorWrapper, FrameSinkWrapper, FrameSourceWrapper};
use crate::statistics::{PipelineStatistics, StatsContext};
use crate::traits::{FrameProcessor, FrameSink, FrameSource, StageFactory};
use crate::{FrameProcessorResult, VydError};

pub type PipelineContextValue = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
pub struct PipelineContext {
    metadata: HashMap<String, PipelineContextValue>,
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
        }
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

#[derive(Deserialize, Debug)]
struct StageConfig {
    #[allow(unused)]
    name: Option<String>,
    stage_type: String,

    #[serde(flatten)]
    params: Option<Value>,
}

pub struct Pipeline {
    pub source: FrameSourceWrapper,
    pub stages: Vec<FrameProcessorWrapper>,
    pub sink: FrameSinkWrapper,
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

    let stats = match stats {
        Some(stats) => stats,
        _ => Arc::new(PipelineStatistics::new()),
    };

    // Set up global statistics context
    pipeline_context.put("pipeline_statistics", stats.clone());
    StatsContext::set_global_stats(stats.clone());

    // Auto-register all stage processing types
    stats.register_stage("source", source.get_processing_type());
    for stage in &stages {
        stats.register_stage(stage.stage_type, stage.get_processing_type());
    }
    stats.register_stage("sink", sink.processing_type());

    // Add statistics to pipeline context if provided
    pipeline_context.put("pipeline_statistics", stats.clone());

    tracing::info!("Initializing pipeline context");
    let init_result = (|| {
        source.pipeline_ctx_init(&mut pipeline_context)?;
        for stage in stages.iter_mut() {
            stage.pipeline_ctx_init(&mut pipeline_context)?;
        }
        sink.pipeline_ctx_init(&mut pipeline_context)
    })();
    if let Err(error) = init_result {
        StatsContext::clear_global_stats();
        return Err(error);
    }
    tracing::info!(?pipeline_context, "pipeline context initialized");

    let pipeline_start = Instant::now();
    let mut frame_count = 0u64;

    while let Some(mut ctx) = source.next_frame(&mut pipeline_context) {
        // Check if we should stop (for Ctrl+C handling)
        if !stats.is_running() {
            tracing::info!("Received shutdown signal, stopping pipeline gracefully");
            break;
        }

        // Process any GStreamer timing data that came with this frame
        crate::pipeline::timing::process_gst_timing_data(&ctx, Some(stats.as_ref()));

        let frame_start = Instant::now();
        frame_count += 1;

        let mut continue_processing = true;
        for stage in stages.iter_mut() {
            if matches!(
                stage.process(&mut ctx, &mut pipeline_context),
                FrameProcessorResult::Skip
            ) {
                continue_processing = false;
                break;
            }
        }

        if continue_processing {
            sink.consume(&mut ctx, &mut pipeline_context);
        }

        let total_frame_duration = frame_start.elapsed();
        let total_frame_duration_us = total_frame_duration.as_micros() as u64;

        // Record statistics if available
        stats.record_frame(total_frame_duration_us);

        // Log comprehensive timing summary for this frame
        log_frame_timing_summary(&ctx, frame_count, total_frame_duration_us);

        // Log periodic throughput stats and flush dumpers
        if frame_count.is_multiple_of(100) {
            let elapsed = pipeline_start.elapsed();
            let fps = frame_count as f64 / elapsed.as_secs_f64();

            tracing::info!(
                frames_processed = frame_count,
                elapsed_seconds = elapsed.as_secs_f64(),
                fps,
                "Pipeline throughput"
            );

            // Periodically flush the dump manager to ensure data is written to disk
            if let Ok(dm) =
                pipeline_context.try_get_as_mut::<crate::dump_manager::DumpManager>("dump_manager")
            {
                // We need to get a mutable reference, but we only have immutable access through try_get_as
                // Instead of changing the API, we'll drop and re-access it with entry()
                if let Err(e) = dm.flush_all() {
                    tracing::warn!(error = ?e, "Failed to flush dump manager");
                }
            }
        }
    }

    let total_duration = pipeline_start.elapsed();
    let fps = frame_count as f64 / total_duration.as_secs_f64();

    tracing::info!(
        total_frames = frame_count,
        total_duration_seconds = total_duration.as_secs_f64(),
        average_fps = fps,
        "Pipeline completed"
    );

    for stage in stages.iter_mut() {
        if let Err(err) = stage.pipeline_finished(&mut pipeline_context) {
            tracing::error!("Error while running pipeline stage finalizer: {}", err);
        }
    }

    if let Err(err) = sink.pipeline_finished(&mut pipeline_context) {
        tracing::error!("Error while running pipeline sink finalizer: {}", err);
    }

    // Clean up global stats when pipeline completes
    StatsContext::clear_global_stats();

    if let Ok(dm) =
        pipeline_context.try_get_as_mut::<crate::dump_manager::DumpManager>("dump_manager")
    {
        if let Err(e) = dm.flush_all() {
            tracing::warn!(error = ?e, "Failed to flush dump manager");
        }
    }

    Ok(pipeline_context)
}

pub fn run_pipeline_with_config_file_path(
    path: &std::path::Path,
    stats: Arc<PipelineStatistics>,
    dump_dir: Option<PathBuf>,
) -> Result<PipelineContext, VydError> {
    let config_raw = std::fs::read_to_string(path)?;
    let config: PipelineConfig = toml::from_str(&config_raw)?;
    let pipeline = build_pipeline(&config)?;
    let mut pipeline_context = PipelineContext::new();

    // If dump_dir is provided, store the config content in the DumpManager
    if let Some(dump_dir) = dump_dir
        && let Ok(dm) = crate::dump_manager::DumpManager::new(&dump_dir)
    {
        // Store the config content
        let dm = dm.with_config(config_raw.clone());

        // Attempt to extract source information from config
        let dm = if let Some(source_name) = config
            .source
            .params
            .as_ref()
            .and_then(|v| v.get("file"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
        {
            dm.with_source(source_name)
        } else {
            dm
        };

        tracing::info!(dump_dir = ?dump_dir, "Initialized dump manager");

        if let Err(e) = dm.write_metadata() {
            tracing::warn!(error = ?e, "Failed to write dump metadata");
        }

        pipeline_context.put("dump_manager", dm);
    }

    run_pipeline(pipeline_context, pipeline, Some(stats))
}
