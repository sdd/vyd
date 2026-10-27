//! Application-facing Vyd pipeline builder and the core frame/stage API.
pub use vyd_core::*;

pub mod cli;
mod config;
mod recording;
pub mod telemetry {
    pub use vyd_telemetry::*;
}
pub use vyd_telemetry::register_vyd_stage_codec;

use anyhow::{Context, anyhow};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::pool::{BufferPoolManager, PoolLimits};
use vyd_core::pipeline::{Pipeline as StagePipeline, PipelineContext};
use vyd_core::statistics::PipelineStatistics;
use vyd_gstreamer::runtime::NativeNode;
use vyd_telemetry::{DumpManager, DumpRunOptions, PublishedRun, RunStatus};

/// Builds a configured serial GStreamer pipeline and owns all run state.
pub struct PipelineBuilder {
    config_path: PathBuf,
    raw_config: String,
    config: toml::Value,
    context: PipelineContext,
    stage_config: vyd_core::pipeline::PipelineConfig,
    statistics: Arc<PipelineStatistics>,
    nodes: Vec<NativeNode>,
    dump_options: DumpRunOptions,
}

impl PipelineBuilder {
    pub fn from_config_file(path: impl AsRef<Path>) -> Result<Self, VydError> {
        let config_path = path.as_ref().to_path_buf();
        let raw_config = std::fs::read_to_string(&config_path)?;
        let config: toml::Value = toml::from_str(&raw_config)?;
        if config.get("native_scale").is_some() {
            return Err(VydError::GeneralPluginError(
                "[native_scale] is obsolete; describe the section with [[gstreamer_nodes]] and explicit geometry and annotation_policy".into(),
            ));
        }
        let stage_config: vyd_core::pipeline::PipelineConfig = toml::from_str(&raw_config)?;
        let runtime_config: config::Config = toml::from_str(&raw_config)?;
        let dump_options = runtime_config.recording.try_into()?;
        let nodes = vyd_gstreamer::pipeline_config::nodes_from_config(&config)?;
        let mut context = PipelineContext::new();
        configure_cpu_pool(runtime_config.cpu_pool, &mut context)?;
        vyd_gstreamer::pipeline_config::configure_gpu(&config, &mut context)?;
        Ok(Self {
            config_path,
            raw_config,
            config,
            context,
            stage_config,
            statistics: Arc::new(PipelineStatistics::new()),
            nodes,
            dump_options,
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
        self.dump_options.root = directory.as_ref().to_path_buf();
        Ok(self)
    }

    pub fn with_dump_options(mut self, options: DumpRunOptions) -> Self {
        self.dump_options = options;
        self
    }

    pub(crate) fn with_run_overrides(
        mut self,
        dump_dir: Option<PathBuf>,
        run_name: Option<String>,
        replace_run: bool,
    ) -> Result<Self, VydError> {
        if let Some(directory) = dump_dir {
            self.dump_options.root = directory;
        }
        if let Some(name) = run_name {
            self.dump_options.run_name = Some(name);
        }
        if replace_run && self.dump_options.run_name.is_none() {
            return Err(VydError::PersistenceError(anyhow!(
                "--replace-run requires --run-name or recording.run_name"
            )));
        }
        self.dump_options.replace = replace_run;
        Ok(self)
    }

    pub fn context_mut(&mut self) -> &mut PipelineContext {
        &mut self.context
    }

    pub fn statistics(&self) -> Arc<PipelineStatistics> {
        self.statistics.clone()
    }

    /// A cloneable handle for requesting graceful pipeline shutdown.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.statistics.cancellation_token()
    }

    pub fn build(mut self) -> Result<Pipeline, VydError> {
        self.validate_recording_stages()?;
        let fingerprint = recording::source_fingerprint(&self.config)?;
        let manager = self.create_dump_manager(fingerprint.clone())?;
        let mut replacements = self.replay_processors(fingerprint.as_ref())?;
        let mut stages = vyd_core::pipeline::build_pipeline_with_processor_resolver(
            &self.stage_config,
            |index, _| Ok(replacements.remove(&index)),
        )?;
        self.wrap_recording_stages(&mut stages, manager.as_ref())?;
        if let Some(manager) = &manager {
            manager.attach(&mut self.context);
        }
        Ok(Pipeline {
            context: self.context,
            stages,
            statistics: self.statistics,
            nodes: self.nodes,
            dump_manager: manager,
        })
    }
}

fn configure_cpu_pool(
    config: Option<config::CpuPoolConfig>,
    context: &mut PipelineContext,
) -> Result<(), VydError> {
    let Some(config) = config else {
        return Ok(());
    };
    let limits = PoolLimits::try_from(config).map_err(vyd_core::frame::FrameError::from)?;
    context.set_frame_factory(FrameFactory::new(BufferPoolManager::new(limits)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::upstream_signature;

    fn dump_options_from_config(value: &toml::Value) -> Result<DumpRunOptions, VydError> {
        let config: config::Config = value.clone().try_into()?;
        Ok(config.recording.try_into()?)
    }

    fn configure_cpu_pool(
        value: &toml::Value,
        context: &mut PipelineContext,
    ) -> Result<(), VydError> {
        let config: config::Config = value.clone().try_into()?;
        super::configure_cpu_pool(config.cpu_pool, context)
    }

    #[test]
    fn recording_config_defaults_and_cli_precedence() {
        let empty: toml::Value = toml::from_str("").unwrap();
        let defaults = dump_options_from_config(&empty).unwrap();
        assert_eq!(defaults.root, PathBuf::from("./dumps"));
        assert_eq!(defaults.run_name, None);

        let path = std::env::temp_dir().join(format!(
            "vyd-recording-config-{}-{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let raw = format!(
            "{}\n[recording]\ndump_dir = './configured-dumps'\nrun_name = 'configured-run'\n",
            include_str!("../../../examples/basic.toml")
        );
        std::fs::write(&path, raw).unwrap();
        let builder = PipelineBuilder::from_config_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            builder.dump_options.root,
            PathBuf::from("./configured-dumps")
        );
        assert_eq!(
            builder.dump_options.run_name.as_deref(),
            Some("configured-run")
        );

        let builder = crate::cli::RunArgs {
            dump_dir: Some(PathBuf::from("./cli-dumps")),
            run_name: None,
            replace_run: true,
        }
        .apply_to(builder)
        .unwrap();
        assert_eq!(builder.dump_options.root, PathBuf::from("./cli-dumps"));
        assert_eq!(
            builder.dump_options.run_name.as_deref(),
            Some("configured-run")
        );
        assert!(builder.dump_options.replace);

        let config_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/basic.toml");
        let builder = PipelineBuilder::from_config_file(config_path).unwrap();
        let builder = crate::cli::RunArgs {
            run_name: Some("cli-run".into()),
            ..Default::default()
        }
        .apply_to(builder)
        .unwrap();
        assert_eq!(builder.dump_options.root, PathBuf::from("./dumps"));
        assert_eq!(builder.dump_options.run_name.as_deref(), Some("cli-run"));
    }

    #[test]
    fn recording_config_rejects_invalid_values_and_replace_without_name() {
        for section in [
            "[recording]\ndump_dir = ''",
            "[recording]\ndump_dir = 42",
            "[recording]\nrun_name = false",
            "[recording]\nother = 'value'",
        ] {
            let config: toml::Value = toml::from_str(section).unwrap();
            assert!(dump_options_from_config(&config).is_err(), "{section}");
        }
        let config_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/basic.toml");
        let builder = PipelineBuilder::from_config_file(config_path).unwrap();
        assert!(
            crate::cli::RunArgs {
                replace_run: true,
                ..Default::default()
            }
            .apply_to(builder)
            .is_err()
        );
    }

    #[test]
    fn recording_settings_do_not_change_replay_upstream_signature() {
        let original: toml::Value =
            toml::from_str(include_str!("../../../examples/basic.toml")).unwrap();
        let with_recording: toml::Value = toml::from_str(&format!(
            "{}\n[recording]\ndump_dir = './another-directory'\nrun_name = 'another-run'\n",
            include_str!("../../../examples/basic.toml")
        ))
        .unwrap();
        assert_eq!(
            upstream_signature(&original, 0).unwrap(),
            upstream_signature(&with_recording, 0).unwrap()
        );
    }

    #[test]
    fn cpu_pool_config_sets_bounds_without_allocating() {
        let config: toml::Value =
            toml::from_str("[cpu_pool]\nmax_buffers = 96\nmax_bytes = 1610612736\n").unwrap();
        let mut context = PipelineContext::new();
        configure_cpu_pool(&config, &mut context).unwrap();
        let snapshot = context.frame_factory().cpu_pool().snapshot().unwrap();
        assert_eq!(snapshot.limits.max_buffers(), 96);
        assert_eq!(snapshot.limits.max_bytes(), 1_610_612_736);
        assert_eq!(snapshot.allocated_buffers, 0);
        assert_eq!(snapshot.allocated_bytes, 0);
    }

    #[test]
    fn cpu_pool_config_accepts_decimal_and_binary_byte_sizes() {
        for (value, expected) in [
            ("'500 MB'", 500_000_000),
            ("'1 GB'", 1_000_000_000),
            ("'512 MiB'", 536_870_912),
            ("'1.5 GiB'", 1_610_612_736),
        ] {
            let config: toml::Value =
                toml::from_str(&format!("[cpu_pool]\nmax_bytes = {value}\n")).unwrap();
            let mut context = PipelineContext::new();
            configure_cpu_pool(&config, &mut context).unwrap();
            let snapshot = context.frame_factory().cpu_pool().snapshot().unwrap();
            assert_eq!(snapshot.limits.max_bytes(), expected, "{value}");
            assert_eq!(snapshot.allocated_bytes, 0);
        }
    }

    #[test]
    fn cpu_pool_config_rejects_invalid_bounds() {
        for section in [
            "[cpu_pool]\nmax_buffers = 0\n",
            "[cpu_pool]\nmax_buffers = -1\n",
            "[cpu_pool]\nmax_buffers = '96'\n",
            "[cpu_pool]\nmax_bytes = 0\n",
            "[cpu_pool]\nmax_bytes = 'large'\n",
            "[cpu_pool]\nmax_bytes = '0 MiB'\n",
            "[cpu_pool]\nmax_bytes = '-1 GiB'\n",
            "[cpu_pool]\nmax_bytes = '18446744073709551616 B'\n",
            "[cpu_pool]\nother = 2\n",
        ] {
            let config: toml::Value = toml::from_str(section).unwrap();
            assert!(configure_cpu_pool(&config, &mut PipelineContext::new()).is_err());
        }
    }
}

/// An owned run; its context and statistics can be accessed before `run`.
pub struct Pipeline {
    context: PipelineContext,
    stages: StagePipeline,
    statistics: Arc<PipelineStatistics>,
    nodes: Vec<NativeNode>,
    dump_manager: Option<DumpManager>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunTermination {
    EndOfStream,
    Cancelled,
}

pub struct PipelineRun {
    pub context: PipelineContext,
    pub termination: RunTermination,
    pub recording: Option<PublishedRun>,
}

impl Pipeline {
    pub fn context_mut(&mut self) -> &mut PipelineContext {
        &mut self.context
    }

    pub fn statistics(&self) -> Arc<PipelineStatistics> {
        self.statistics.clone()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.statistics.cancellation_token()
    }

    pub fn run(self) -> Result<PipelineContext, VydError> {
        self.run_with_outcome().map(|run| run.context)
    }

    pub fn run_with_outcome(self) -> Result<PipelineRun, VydError> {
        let result = vyd_gstreamer::runtime::run_pipeline_with_gstreamer_nodes_outcome(
            self.context,
            self.stages,
            Some(self.statistics),
            self.nodes,
        );
        match result {
            Ok(run) => {
                let termination = if run.end_of_stream {
                    RunTermination::EndOfStream
                } else {
                    RunTermination::Cancelled
                };
                let recording = self
                    .dump_manager
                    .map(|manager| {
                        manager
                            .publish(match termination {
                                RunTermination::EndOfStream => RunStatus::Completed,
                                RunTermination::Cancelled => RunStatus::Cancelled,
                            })
                            .context("publish pipeline recording")
                    })
                    .transpose()?;
                Ok(PipelineRun {
                    context: run.context,
                    termination,
                    recording,
                })
            }
            Err(error) => {
                if let Some(manager) = self.dump_manager
                    && !matches!(error, VydError::ShutdownTimeout { .. })
                    && let Err(mark_error) = manager.mark_failed()
                {
                    tracing::error!(?mark_error, "Could not mark failed Vyd run");
                }
                Err(error)
            }
        }
    }
}
