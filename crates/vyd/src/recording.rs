use crate::{FrameProcessor, PipelineBuilder, VydError};
use anyhow::{Context, anyhow};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use vyd_core::pipeline::{Pipeline as StagePipeline, Stage, StageConfig};
use vyd_telemetry::{
    DumpManager, DumpingProcessor, RECORDING_FORMAT_VERSION, ReplayProcessor, RunManifest,
    RunStatus, SourceFingerprint, StageCodec, StageRecording, VortexReader, codec_for,
    read_manifest, schema_for,
};

impl PipelineBuilder {
    pub(super) fn validate_recording_stages(&self) -> Result<(), VydError> {
        for stage in [
            self.stage_config.source_stage(),
            self.stage_config.sink_stage(),
        ] {
            if stage.dump || stage.replay.is_some() {
                return Err(VydError::PersistenceError(anyhow!(
                    "dump/replay currently supports processors only"
                )));
            }
        }
        for (index, stage) in self.stage_config.processor_stages().iter().enumerate() {
            if stage.dump && stage.replay.is_some() {
                return Err(VydError::PersistenceError(anyhow!(
                    "stage {index} cannot set both dump and replay"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn create_dump_manager(
        &self,
        fingerprint: Option<SourceFingerprint>,
    ) -> Result<Option<DumpManager>, VydError> {
        if !self
            .stage_config
            .processor_stages()
            .iter()
            .any(|stage| stage.dump)
        {
            return Ok(None);
        }
        let manager = DumpManager::new(
            self.dump_options.clone(),
            self.raw_config.clone(),
            self.stage_config.source_type().to_string(),
        )
        .context("create pipeline recording")?;
        manager
            .set_source_fingerprint(fingerprint)
            .context("record source fingerprint")?;
        Ok(Some(manager))
    }

    pub(super) fn replay_processors(
        &self,
        fingerprint: Option<&SourceFingerprint>,
    ) -> Result<HashMap<usize, Box<dyn FrameProcessor>>, VydError> {
        let mut processors = HashMap::new();
        for (index, stage) in self.stage_config.processor_stages().iter().enumerate() {
            if let Some(path) = &stage.replay {
                processors.insert(
                    index,
                    self.replay_processor(index, stage, path, fingerprint)?,
                );
            }
        }
        Ok(processors)
    }

    fn replay_processor(
        &self,
        index: usize,
        stage: &StageConfig,
        path: &str,
        fingerprint: Option<&SourceFingerprint>,
    ) -> Result<Box<dyn FrameProcessor>, VydError> {
        let path = self.replay_path(path);
        let manifest = replay_manifest(&path, index, fingerprint)?;
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("invalid replay file name")?;
        let entry = manifest
            .stages
            .iter()
            .find(|entry| entry.file == file_name)
            .context("replay stage not found in manifest")?;
        let codec = codec_for(&stage.stage_type)?
            .with_context(|| format!("stage {} has no replay codec", stage.stage_type))?;
        self.validate_replay_stage(index, stage, entry, codec.as_ref(), &manifest.config)?;
        if !path.is_file() {
            return Err(VydError::PersistenceError(anyhow!(
                "replay file {} is missing",
                path.display()
            )));
        }
        let reader = VortexReader::new(&path)
            .with_context(|| format!("start Vortex reader for {}", path.display()))?;
        Ok(Box::new(ReplayProcessor::new(
            codec,
            reader,
            manifest.status == RunStatus::Cancelled,
        )))
    }

    fn replay_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.config_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(path)
        }
    }

    fn validate_replay_stage(
        &self,
        index: usize,
        stage: &StageConfig,
        entry: &StageRecording,
        codec: &dyn StageCodec,
        recorded_config: &str,
    ) -> Result<(), VydError> {
        if entry.stage_type != stage.stage_type || entry.codec_version != codec.version() {
            return Err(VydError::PersistenceError(anyhow!(
                "replay stage type or codec version mismatch at stage {index}"
            )));
        }
        let expected = upstream_signature(&self.config, index)?;
        if entry.upstream_signature != expected {
            // Older signatures included [recording]; recompute them without run location/name.
            let recorded: toml::Value =
                toml::from_str(recorded_config).context("parse recorded pipeline config")?;
            if upstream_signature(&recorded, index)? != expected {
                return Err(VydError::PersistenceError(anyhow!(
                    "replay upstream configuration mismatch at stage {index}"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn wrap_recording_stages(
        &self,
        stages: &mut StagePipeline,
        manager: Option<&DumpManager>,
    ) -> Result<(), VydError> {
        let mut wrapped = Vec::with_capacity(stages.stages.len());
        for (index, stage) in std::mem::take(&mut stages.stages).into_iter().enumerate() {
            let config = &self.stage_config.processor_stages()[index];
            let inner = if config.dump {
                self.dumping_processor(
                    index,
                    config,
                    stage.inner,
                    manager.context("dump manager missing")?,
                )?
            } else {
                stage.inner
            };
            wrapped.push(Stage::new(inner, stage.stage_type));
        }
        stages.stages = wrapped;
        Ok(())
    }

    fn dumping_processor(
        &self,
        index: usize,
        config: &StageConfig,
        processor: Box<dyn FrameProcessor>,
        manager: &DumpManager,
    ) -> Result<Box<dyn FrameProcessor>, VydError> {
        let codec = codec_for(&config.stage_type)?
            .with_context(|| format!("stage {} has no dump codec", config.stage_type))?;
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
            .with_context(|| format!("register recording for stage {index}"))?;
        Ok(Box::new(DumpingProcessor::new(processor, codec, writer)))
    }
}

fn replay_manifest(
    path: &Path,
    index: usize,
    fingerprint: Option<&SourceFingerprint>,
) -> Result<RunManifest, VydError> {
    let directory = path.parent().context("invalid replay path")?;
    let manifest = read_manifest(directory)
        .with_context(|| format!("read replay manifest in {}", directory.display()))?;
    if manifest.format_version != RECORDING_FORMAT_VERSION {
        return Err(VydError::PersistenceError(anyhow!(
            "unsupported recording format version {}",
            manifest.format_version
        )));
    }
    if !manifest.status.is_replayable() {
        return Err(VydError::PersistenceError(anyhow!(
            "replay recording was not finalized"
        )));
    }
    if manifest.source_fingerprint.as_ref() != fingerprint {
        return Err(VydError::PersistenceError(anyhow!(
            "replay source file changed at stage {index}"
        )));
    }
    Ok(manifest)
}

pub(super) fn source_fingerprint(
    config: &toml::Value,
) -> Result<Option<SourceFingerprint>, VydError> {
    let Some(path) = config
        .get("source")
        .and_then(|source| source.get("file_path"))
        .and_then(toml::Value::as_str)
    else {
        return Ok(None);
    };
    let path =
        std::fs::canonicalize(path).with_context(|| format!("resolve source file {path}"))?;
    let metadata = std::fs::metadata(&path)
        .with_context(|| format!("read source metadata for {}", path.display()))?;
    let modified_ns = metadata
        .modified()
        .with_context(|| format!("read source modification time for {}", path.display()))?
        .duration_since(std::time::UNIX_EPOCH)
        .context("source modification time predates the Unix epoch")?
        .as_nanos();
    Ok(Some(SourceFingerprint {
        path,
        size: metadata.len(),
        modified_ns,
    }))
}

pub(super) fn upstream_signature(config: &toml::Value, index: usize) -> Result<String, VydError> {
    let mut prefix = config.clone();
    let table = prefix
        .as_table_mut()
        .context("pipeline config must be a table")?;
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
    Ok(toml::to_string(&prefix).context("serialize upstream pipeline config")?)
}
