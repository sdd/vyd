//! Application-facing Vyd pipeline builder and the core frame/stage API.
pub use vyd_core::*;

use std::path::Path;
use std::sync::Arc;
use vyd_core::pipeline::{Pipeline as StagePipeline, PipelineContext};
use vyd_core::statistics::PipelineStatistics;
use vyd_gstreamer::runtime::NativeNode;

/// Builds a configured serial GStreamer pipeline and owns all run state.
pub struct PipelineBuilder {
    raw_config: String,
    config: toml::Value,
    context: PipelineContext,
    stages: StagePipeline,
    statistics: Arc<PipelineStatistics>,
    nodes: Vec<NativeNode>,
}

impl PipelineBuilder {
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self, VydError> {
        let raw_config = std::fs::read_to_string(path)?;
        let config: toml::Value = toml::from_str(&raw_config)?;
        if config.get("native_scale").is_some() {
            return Err(VydError::GeneralPluginError(
                "[native_scale] is obsolete; describe the section with [[gstreamer_nodes]] and explicit geometry and annotation_policy".into(),
            ));
        }
        let stages: vyd_core::pipeline::PipelineConfig = toml::from_str(&raw_config)?;
        let nodes = vyd_gstreamer::pipeline_config::nodes_from_config(&config)?;
        let mut context = PipelineContext::new();
        vyd_gstreamer::pipeline_config::configure_gpu(&config, &mut context)?;
        Ok(Self {
            raw_config,
            config,
            context,
            stages: vyd_core::pipeline::build_pipeline(&stages)?,
            statistics: Arc::new(PipelineStatistics::new()),
            nodes,
        })
    }

    /// Insert a GStreamer chain after the source (boundary 0) or after the
    /// processor at index `boundary - 1`. For filters that change geometry or
    /// strip context, use `[[gstreamer_nodes]]` with explicit contracts.
    pub fn with_gstreamer_nodes(
        mut self,
        boundary: usize,
        pipeline_description: impl Into<String>,
    ) -> Self {
        self.nodes
            .push(NativeNode::description(boundary, pipeline_description));
        self
    }

    pub fn with_dump_dir(mut self, directory: impl AsRef<Path>) -> Result<Self, VydError> {
        let manager = vyd_telemetry::DumpManager::new(directory).map_err(|error| {
            VydError::GeneralPluginError(format!("telemetry setup failed: {error}"))
        })?;
        let manager = manager.with_config(self.raw_config.clone());
        let source = self
            .config
            .get("source")
            .and_then(|source| source.get("stage_type"))
            .and_then(toml::Value::as_str)
            .map(str::to_owned);
        let manager = if let Some(source) = source {
            manager.with_source(source)
        } else {
            manager
        };
        manager.attach(&mut self.context);
        Ok(self)
    }

    pub fn context_mut(&mut self) -> &mut PipelineContext {
        &mut self.context
    }

    pub fn statistics(&self) -> Arc<PipelineStatistics> {
        self.statistics.clone()
    }

    pub fn build(self) -> Pipeline {
        Pipeline {
            context: self.context,
            stages: self.stages,
            statistics: self.statistics,
            nodes: self.nodes,
        }
    }
}

/// An owned run; its context and statistics can be accessed before `run`.
pub struct Pipeline {
    context: PipelineContext,
    stages: StagePipeline,
    statistics: Arc<PipelineStatistics>,
    nodes: Vec<NativeNode>,
}

impl Pipeline {
    pub fn context_mut(&mut self) -> &mut PipelineContext {
        &mut self.context
    }

    pub fn statistics(&self) -> Arc<PipelineStatistics> {
        self.statistics.clone()
    }

    pub fn run(self) -> Result<PipelineContext, VydError> {
        vyd_gstreamer::runtime::run_pipeline_with_gstreamer_nodes(
            self.context,
            self.stages,
            Some(self.statistics),
            self.nodes,
        )
    }
}
