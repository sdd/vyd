use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use toml::Value;

mod services;
mod session;
pub use session::PipelineSession;
pub(crate) mod timing;
pub(crate) mod wrappers;
pub use services::PipelineServices;
pub use wrappers::Stage;

use crate::frame::factory::FrameFactory;
use crate::pipeline::wrappers::{FrameProcessorWrapper, FrameSinkWrapper, FrameSourceWrapper};
use crate::statistics::PipelineStatistics;
use crate::traits::{FrameProcessor, FrameSink, FrameSource, StageFactory};
use crate::{FrameProcessorResult, VydError};

pub type PipelineContextValue = Box<dyn Any + Send + Sync>;

pub struct PipelineContext {
    services: PipelineServices,
    frame_factory: FrameFactory,
    #[cfg(feature = "gpu-wgpu")]
    gpu: Option<Arc<crate::gpu::WgpuContext>>,
    #[cfg(feature = "gpu-wgpu")]
    gpu_frozen: bool,
    #[cfg(feature = "gpu-wgpu")]
    gpu_pool_limits: crate::gpu::pool::GpuPoolLimits,
    #[cfg(feature = "gpu-wgpu")]
    gpu_transfer: Option<Arc<crate::gpu::transfer::GpuTransfer>>,
    metadata: HashMap<String, PipelineContextValue>,
    observers: Vec<Box<dyn PipelineObserver>>,
}

impl std::fmt::Debug for PipelineContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineContext")
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .field("observer_count", &self.observers.len())
            .field("gpu_configured", &{
                #[cfg(feature = "gpu-wgpu")]
                {
                    self.gpu.is_some()
                }
                #[cfg(not(feature = "gpu-wgpu"))]
                {
                    false
                }
            })
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

    /// Called after each successfully handled frame, including processor skips.
    /// A source, processor or sink error terminates processing before this hook.
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
            services: PipelineServices::default(),
            frame_factory: FrameFactory::default(),
            #[cfg(feature = "gpu-wgpu")]
            gpu: None,
            #[cfg(feature = "gpu-wgpu")]
            gpu_frozen: false,
            #[cfg(feature = "gpu-wgpu")]
            gpu_pool_limits: crate::gpu::pool::GpuPoolLimits::new(128, 512 * 1024 * 1024)
                .expect("positive default GPU pool limits"),
            #[cfg(feature = "gpu-wgpu")]
            gpu_transfer: None,
            metadata: HashMap::new(),
            observers: Vec::new(),
        }
    }

    /// Clone this handle to share typed services/latest values with workers.
    /// The legacy string-keyed metadata and lifecycle observers remain local to
    /// this context during migration; they are not copied into the registry.
    pub fn services(&self) -> &PipelineServices {
        &self.services
    }

    /// Configure the per-pipeline CPU frame factory before stages start.
    pub fn set_frame_factory(&mut self, factory: FrameFactory) {
        self.frame_factory = factory;
    }

    pub fn frame_factory(&self) -> &FrameFactory {
        &self.frame_factory
    }

    /// Attach the pipeline's single GPU context before stages start. Reattaching
    /// the same context is harmless; a different device is rejected.
    #[cfg(feature = "gpu-wgpu")]
    pub fn set_wgpu_context(
        &mut self,
        gpu: Arc<crate::gpu::WgpuContext>,
    ) -> Result<(), crate::gpu::GpuContextError> {
        if self.gpu_frozen {
            return Err(crate::gpu::GpuContextError::ConfigurationLocked);
        }
        if let Some(existing) = &self.gpu {
            existing.ensure_same_device(&gpu)?;
        }
        self.gpu = Some(gpu);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn wgpu_context(
        &self,
    ) -> Result<&Arc<crate::gpu::WgpuContext>, crate::gpu::GpuContextError> {
        self.gpu
            .as_ref()
            .ok_or(crate::gpu::GpuContextError::NotConfigured)
    }

    /// Configure the bounded GPU resource pool before a pipeline starts.
    #[cfg(feature = "gpu-wgpu")]
    pub fn set_gpu_pool_limits(
        &mut self,
        limits: crate::gpu::pool::GpuPoolLimits,
    ) -> Result<(), crate::gpu::GpuContextError> {
        if self.gpu_frozen {
            return Err(crate::gpu::GpuContextError::ConfigurationLocked);
        }
        self.gpu_pool_limits = limits;
        Ok(())
    }

    /// The one transfer service shared by Rust stages and the GST adapter.
    /// It is installed before observer and stage initialization.
    #[cfg(feature = "gpu-wgpu")]
    pub fn gpu_transfer(
        &self,
    ) -> Result<&Arc<crate::gpu::transfer::GpuTransfer>, crate::gpu::GpuContextError> {
        self.gpu_transfer
            .as_ref()
            .ok_or(crate::gpu::GpuContextError::NotConfigured)
    }

    #[cfg(feature = "gpu-wgpu")]
    pub(crate) fn install_gpu_transfer(
        &mut self,
        stats: Arc<PipelineStatistics>,
    ) -> Result<(), VydError> {
        if let Some(gpu) = &self.gpu {
            let pool = crate::gpu::pool::GpuPoolManager::new(gpu.clone(), self.gpu_pool_limits);
            self.gpu_transfer = Some(Arc::new(crate::gpu::transfer::GpuTransfer::new(
                gpu.clone(),
                pool,
                crate::statistics::StatsContext::new(stats),
            )?));
        }
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    pub(crate) fn freeze_wgpu_context(&mut self) {
        self.gpu_frozen = true;
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
            Entry::Occupied(entry) => entry.into_mut().downcast_mut::<T>().unwrap_or_else(|| {
                panic!(
                    "Could not downcast metadata value to requested type '{}'",
                    core::any::type_name::<T>()
                )
            }),
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

    pub fn processor_stages(&self) -> &[StageConfig] {
        &self.stages
    }

    pub fn source_stage(&self) -> &StageConfig {
        &self.source
    }

    pub fn sink_stage(&self) -> &StageConfig {
        &self.sink
    }
}

#[derive(Deserialize, Debug)]
pub struct StageConfig {
    pub name: Option<String>,
    pub stage_type: String,
    #[serde(default)]
    pub dump: bool,
    pub replay: Option<String>,

    #[serde(flatten)]
    pub params: Option<Value>,
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
    ($source_type:ty, $public_name:literal) => {
        $crate::_register_vyd_stage_impl!($public_name, $source_type, FrameSource);
    };
    ($source_type:ty) => {
        $crate::_register_vyd_stage_impl!(env!("CARGO_PKG_NAME"), $source_type, FrameSource);
    };
}

#[macro_export]
macro_rules! register_vyd_frame_processor {
    ($processor_type:ty, $public_name:literal) => {
        $crate::_register_vyd_stage_impl!($public_name, $processor_type, FrameProcessor);
    };
    ($processor_type:ty) => {
        $crate::_register_vyd_stage_impl!(env!("CARGO_PKG_NAME"), $processor_type, FrameProcessor);
    };
}

#[macro_export]
macro_rules! register_vyd_frame_sink {
    ($sink_type:ty, $public_name:literal) => {
        $crate::_register_vyd_stage_impl!($public_name, $sink_type, FrameSink);
    };
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
                    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
                    NAME.get_or_init(|| {
                        format!("{}::{}", $crate_name.replace('-', "_"), stringify!($stage_type))
                    })
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
    if config
        .stages
        .iter()
        .any(|stage| stage.dump || stage.replay.is_some())
    {
        return Err(VydError::GeneralPluginError(
            "stage dump/replay requires the vyd PipelineBuilder".into(),
        ));
    }
    build_pipeline_with_processor_resolver(config, |_, _| Ok(None))
}

/// Build a graph while allowing the application facade to substitute a
/// processor (for example, a replay reader) before its normal constructor runs.
pub fn build_pipeline_with_processor_resolver(
    config: &PipelineConfig,
    mut resolve: impl FnMut(usize, &StageConfig) -> Result<Option<Box<dyn FrameProcessor>>, VydError>,
) -> Result<Pipeline, VydError> {
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
    for (index, stage) in config.stages.iter().enumerate() {
        if let Some(inner) = resolve(index, stage)? {
            let registered = inventory::iter::<&dyn StageFactory>()
                .find(|factory| factory.stage_type() == stage.stage_type)
                .ok_or_else(|| VydError::UnknownStage {
                    kind: "processor",
                    stage_type: stage.stage_type.clone(),
                })?;
            stages.push(FrameProcessorWrapper::new(inner, registered.stage_type()));
        } else {
            stages.push(find_processor_factory(
                &stage.stage_type,
                stage.params.as_ref(),
            )?);
        }
    }

    Ok(Pipeline {
        source,
        stages,
        sink,
    })
}

pub fn run_pipeline(
    pipeline_context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
) -> Result<PipelineContext, VydError> {
    let stats = stats.unwrap_or_else(|| Arc::new(PipelineStatistics::new()));
    let mut session = PipelineSession::new(pipeline_context, pipeline, stats);
    let result = (|| {
        session.start()?;
        while let Some(mut frame) = session.next_frame()? {
            let started = Instant::now();
            let mut emit = true;
            for index in 0..session.processor_count() {
                if matches!(
                    session.process(index, &mut frame)?,
                    FrameProcessorResult::Skip
                ) {
                    emit = false;
                    break;
                }
            }
            if emit {
                session.consume(&mut frame)?;
            }
            session.complete_frame(&frame, started)?;
        }
        Ok(())
    })();
    session.finish(result)
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
