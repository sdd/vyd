//! Application-facing Vyd pipeline builder and the core frame/stage API.
pub use vyd_core::*;

pub mod cli;
pub mod telemetry {
    pub use vyd_telemetry::*;
}
pub use vyd_telemetry::register_vyd_stage_codec;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::pool::{BufferPoolManager, PoolLimits};
use vyd_core::pipeline::{Pipeline as StagePipeline, PipelineContext};
use vyd_core::statistics::PipelineStatistics;
use vyd_gstreamer::runtime::NativeNode;
use vyd_telemetry::{
    DumpManager, DumpRunOptions, DumpingProcessor, PublishedRun, RECORDING_FORMAT_VERSION,
    ReplayProcessor, RunStatus, SourceFingerprint, StageRecording, VortexReader, codec_for,
    read_manifest, schema_for,
};

fn persistence_error(error: impl std::fmt::Display) -> VydError {
    VydError::GeneralPluginError(format!("pipeline persistence: {error}"))
}

fn dump_options_from_config(config: &toml::Value) -> Result<DumpRunOptions, VydError> {
    let mut options = DumpRunOptions {
        root: PathBuf::from("./dumps"),
        run_name: None,
        replace: false,
    };
    let Some(recording) = config.get("recording") else {
        return Ok(options);
    };
    let table = recording
        .as_table()
        .ok_or_else(|| persistence_error("recording must be a table"))?;
    for key in table.keys() {
        if key != "dump_dir" && key != "run_name" {
            return Err(persistence_error(format!("unknown recording field: {key}")));
        }
    }
    if let Some(directory) = table.get("dump_dir") {
        let directory = directory
            .as_str()
            .filter(|directory| !directory.is_empty())
            .ok_or_else(|| persistence_error("recording.dump_dir must be a nonempty string"))?;
        options.root = PathBuf::from(directory);
    }
    if let Some(name) = table.get("run_name") {
        options.run_name = Some(
            name.as_str()
                .ok_or_else(|| persistence_error("recording.run_name must be a string"))?
                .to_string(),
        );
    }
    Ok(options)
}

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
        let dump_options = dump_options_from_config(&config)?;
        let nodes = vyd_gstreamer::pipeline_config::nodes_from_config(&config)?;
        let mut context = PipelineContext::new();
        configure_cpu_pool(&config, &mut context)?;
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
            return Err(persistence_error(
                "--replace-run requires --run-name or recording.run_name",
            ));
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
        for stage in [
            self.stage_config.source_stage(),
            self.stage_config.sink_stage(),
        ] {
            if stage.dump || stage.replay.is_some() {
                return Err(persistence_error(
                    "dump/replay currently supports processors only",
                ));
            }
        }
        let has_dump = self
            .stage_config
            .processor_stages()
            .iter()
            .any(|stage| stage.dump);
        for (index, stage) in self.stage_config.processor_stages().iter().enumerate() {
            if stage.dump && stage.replay.is_some() {
                return Err(persistence_error(format!(
                    "stage {index} cannot set both dump and replay"
                )));
            }
        }
        let source_fingerprint = source_fingerprint(&self.config)?;
        let manager = has_dump
            .then(|| {
                DumpManager::new(
                    self.dump_options.clone(),
                    self.raw_config.clone(),
                    self.stage_config.source_type().to_string(),
                )
                .map_err(persistence_error)
            })
            .transpose()?;
        if let Some(manager) = &manager {
            manager
                .set_source_fingerprint(source_fingerprint.clone())
                .map_err(persistence_error)?;
        }
        let mut replacements: HashMap<usize, Box<dyn FrameProcessor>> = HashMap::new();
        for (index, stage) in self.stage_config.processor_stages().iter().enumerate() {
            let Some(recording_path) = &stage.replay else {
                continue;
            };
            let path = Path::new(recording_path);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.config_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(path)
            };
            let directory = path
                .parent()
                .ok_or_else(|| persistence_error("invalid replay path"))?;
            let manifest = read_manifest(directory).map_err(persistence_error)?;
            if manifest.format_version != RECORDING_FORMAT_VERSION {
                return Err(persistence_error(format!(
                    "unsupported recording format version {}",
                    manifest.format_version
                )));
            }
            if !manifest.status.is_replayable() {
                return Err(persistence_error("replay recording was not finalized"));
            }
            if manifest.source_fingerprint != source_fingerprint {
                return Err(persistence_error(format!(
                    "replay source file changed at stage {index}"
                )));
            }
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| persistence_error("invalid replay file name"))?;
            let entry = manifest
                .stages
                .iter()
                .find(|entry| entry.file == file_name)
                .ok_or_else(|| persistence_error("replay stage not found in manifest"))?;
            let codec = codec_for(&stage.stage_type)?.ok_or_else(|| {
                persistence_error(format!("stage {} has no replay codec", stage.stage_type))
            })?;
            if entry.stage_type != stage.stage_type || entry.codec_version != codec.version() {
                return Err(persistence_error(format!(
                    "replay stage type or codec version mismatch at stage {index}"
                )));
            }
            let expected_upstream = upstream_signature(&self.config, index)?;
            if entry.upstream_signature != expected_upstream {
                // Older recordings included [recording] in this signature.
                // Recompute it from their stored config so run location and name
                // cannot affect replay compatibility.
                let recorded: toml::Value =
                    toml::from_str(&manifest.config).map_err(persistence_error)?;
                if upstream_signature(&recorded, index)? != expected_upstream {
                    return Err(persistence_error(format!(
                        "replay upstream configuration mismatch at stage {index}"
                    )));
                }
            }
            if !path.is_file() {
                return Err(persistence_error(format!(
                    "replay file {} is missing",
                    path.display()
                )));
            }
            let reader = VortexReader::new(path).map_err(persistence_error)?;
            replacements.insert(
                index,
                Box::new(ReplayProcessor::new(
                    codec,
                    reader,
                    manifest.status == RunStatus::Cancelled,
                )),
            );
        }
        let mut stages = vyd_core::pipeline::build_pipeline_with_processor_resolver(
            &self.stage_config,
            |index, _| Ok(replacements.remove(&index)),
        )?;
        let mut wrapped = Vec::with_capacity(stages.stages.len());
        for (index, stage) in std::mem::take(&mut stages.stages).into_iter().enumerate() {
            let config = &self.stage_config.processor_stages()[index];
            let inner: Box<dyn FrameProcessor> = if config.dump {
                let manager = manager.as_ref().expect("dump directory was checked");
                let codec = codec_for(&config.stage_type)?.ok_or_else(|| {
                    persistence_error(format!("stage {} has no dump codec", config.stage_type))
                })?;
                let writer = manager
                    .register_stage(
                        StageRecording {
                            index,
                            name: config.name.clone(),
                            stage_type: config.stage_type.clone(),
                            codec_version: codec.version(),
                            file: String::new(),
                            upstream_signature: upstream_signature(&self.config, index)?,
                        },
                        schema_for(codec.as_ref())?,
                    )
                    .map_err(persistence_error)?;
                Box::new(DumpingProcessor::new(stage.inner, codec, writer))
            } else {
                stage.inner
            };
            wrapped.push(vyd_core::pipeline::Stage::new(inner, stage.stage_type));
        }
        stages.stages = wrapped;
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

fn source_fingerprint(config: &toml::Value) -> Result<Option<SourceFingerprint>, VydError> {
    let Some(path) = config
        .get("source")
        .and_then(|source| source.get("file_path"))
        .and_then(toml::Value::as_str)
    else {
        return Ok(None);
    };
    let path = std::fs::canonicalize(path).map_err(persistence_error)?;
    let metadata = std::fs::metadata(&path).map_err(persistence_error)?;
    let modified_ns = metadata
        .modified()
        .map_err(persistence_error)?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(persistence_error)?
        .as_nanos();
    Ok(Some(SourceFingerprint {
        path,
        size: metadata.len(),
        modified_ns,
    }))
}

fn upstream_signature(config: &toml::Value, index: usize) -> Result<String, VydError> {
    let mut prefix = config.clone();
    let table = prefix
        .as_table_mut()
        .ok_or_else(|| persistence_error("pipeline config must be a table"))?;
    table.remove("sink");
    table.remove("recording");
    if let Some(stages) = table.get_mut("stages").and_then(toml::Value::as_array_mut) {
        stages.truncate(index);
        for stage in stages {
            if let Some(stage) = stage.as_table_mut() {
                stage.remove("dump");
                stage.remove("replay");
            }
        }
    }
    toml::to_string(&prefix).map_err(persistence_error)
}

fn configure_cpu_pool(config: &toml::Value, context: &mut PipelineContext) -> Result<(), VydError> {
    let Some(value) = config.get("cpu_pool") else {
        return Ok(());
    };
    let table = value
        .as_table()
        .ok_or_else(|| VydError::GeneralPluginError("cpu_pool must be a table".into()))?;
    for key in table.keys() {
        if key != "max_buffers" && key != "max_bytes" {
            return Err(VydError::GeneralPluginError(format!(
                "unknown cpu_pool field: {key}"
            )));
        }
    }
    let read_buffer_limit = |key: &str| -> Result<Option<usize>, VydError> {
        table
            .get(key)
            .map(|value| {
                value
                    .as_integer()
                    .and_then(|number| usize::try_from(number).ok())
                    .ok_or_else(|| {
                        VydError::GeneralPluginError(format!(
                            "cpu_pool.{key} must be a positive integer"
                        ))
                    })
            })
            .transpose()
    };
    let max_bytes = table
        .get("max_bytes")
        .map(|value| match value {
            toml::Value::Integer(number) => usize::try_from(*number).map_err(|_| {
                VydError::GeneralPluginError(
                    "cpu_pool.max_bytes must be a positive integer or byte size".into(),
                )
            }),
            toml::Value::String(size) => parse_size::parse_size(size)
                .map_err(|error| {
                    VydError::GeneralPluginError(format!(
                        "invalid cpu_pool.max_bytes size {size:?}: {error}"
                    ))
                })
                .and_then(|bytes| {
                    usize::try_from(bytes).map_err(|_| {
                        VydError::GeneralPluginError(
                            "cpu_pool.max_bytes exceeds the platform's addressable size".into(),
                        )
                    })
                }),
            _ => Err(VydError::GeneralPluginError(
                "cpu_pool.max_bytes must be a positive integer or byte size".into(),
            )),
        })
        .transpose()?;
    let limits = PoolLimits::new(read_buffer_limit("max_buffers")?, max_bytes)
        .map_err(|error| VydError::GeneralPluginError(format!("invalid cpu_pool: {error}")))?;
    context.set_frame_factory(FrameFactory::new(BufferPoolManager::new(limits)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                            .map_err(persistence_error)
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
