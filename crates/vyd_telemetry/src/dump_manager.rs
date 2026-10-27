use crate::vortex_file::VortexWriter;
use anyhow::{Context, Result, anyhow};
use arrow_schema::SchemaRef;
use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use vyd_core::VydError;
use vyd_core::pipeline::{PipelineContext, PipelineObserver};
use vyd_core::statistics::PipelineStatistics;

#[derive(Clone, Debug)]
pub struct DumpRunOptions {
    pub root: PathBuf,
    pub run_name: Option<String>,
    pub replace: bool,
}

#[derive(Clone, Debug)]
pub struct PublishedRun {
    pub directory: PathBuf,
    pub status: RunStatus,
    pub stages: Vec<PublishedStage>,
}

#[derive(Clone, Debug)]
pub struct PublishedStage {
    pub file: String,
    pub rows: u64,
    pub bytes: u64,
}

impl fmt::Display for PublishedRun {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Vortex dump: {}", self.directory.display())?;
        writeln!(
            formatter,
            "  status: {}",
            match self.status {
                RunStatus::Recording => "recording",
                RunStatus::Completed => "completed",
                RunStatus::Cancelled => "cancelled",
                RunStatus::Failed => "failed",
            }
        )?;
        for stage in &self.stages {
            writeln!(
                formatter,
                "  {}: {} frames, {} bytes ({:.2} MiB)",
                stage.file,
                stage.rows,
                stage.bytes,
                stage.bytes as f64 / (1024.0 * 1024.0)
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Recording,
    Completed,
    Cancelled,
    Failed,
}

impl RunStatus {
    pub fn is_replayable(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StageRecording {
    pub index: usize,
    pub name: Option<String>,
    pub stage_type: String,
    pub codec_version: u32,
    pub file: String,
    pub upstream_signature: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunManifest {
    #[serde(default)]
    pub format_version: u32,
    pub run_id: String,
    pub run_name: String,
    pub start_time: DateTime<Utc>,
    pub status: RunStatus,
    pub source: String,
    #[serde(default)]
    pub source_fingerprint: Option<SourceFingerprint>,
    pub config: String,
    pub stages: Vec<StageRecording>,
}

pub const RECORDING_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SourceFingerprint {
    pub path: PathBuf,
    pub size: u64,
    pub modified_ns: u128,
}

pub fn read_manifest(directory: impl AsRef<Path>) -> Result<RunManifest> {
    let path = directory.as_ref().join("manifest.json");
    serde_json::from_slice(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

struct Inner {
    lock: Option<File>,
    options: DumpRunOptions,
    staging: PathBuf,
    target: PathBuf,
    manifest: RunManifest,
    writers: Vec<Arc<Mutex<VortexWriter>>>,
    closed: bool,
    published: bool,
}

/// Owns a named run while stage wrappers borrow its per-stage writers.
#[derive(Clone)]
pub struct DumpManager(Arc<Mutex<Inner>>);

impl DumpManager {
    pub fn new(mut options: DumpRunOptions, config: String, source: String) -> Result<Self> {
        fs::create_dir_all(&options.root)?;
        options.root = fs::canonicalize(&options.root)?;
        fs::create_dir_all(options.root.join(".locks"))?;
        fs::create_dir_all(options.root.join(".staging"))?;
        fs::create_dir_all(options.root.join(".backup"))?;
        let run_id = format!("{:016x}", rand::random::<u64>());
        let name = options
            .run_name
            .clone()
            .unwrap_or_else(|| format!("{}-{run_id}", Utc::now().format("%Y%m%dT%H%M%S")));
        validate_name(&name)?;
        if options.replace && options.run_name.is_none() {
            return Err(anyhow!("replace mode requires an explicit run name"));
        }
        let lock_path = options.root.join(".locks").join(format!("{name}.lock"));
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;
        lock.try_lock_exclusive()
            .with_context(|| format!("run {name:?} is already in use"))?;
        let target = options.root.join(&name);
        recover_backup(&options.root, &name, &target)?;
        if target.exists() && !options.replace {
            return Err(anyhow!("run {name:?} already exists; use --replace-run"));
        }
        let staging = options
            .root
            .join(".staging")
            .join(format!("{name}.{run_id}"));
        fs::create_dir(&staging)?;
        let manifest = RunManifest {
            format_version: RECORDING_FORMAT_VERSION,
            run_id,
            run_name: name,
            start_time: Utc::now(),
            status: RunStatus::Recording,
            source,
            source_fingerprint: None,
            config,
            stages: Vec::new(),
        };
        write_manifest(&staging, &manifest)?;
        Ok(Self(Arc::new(Mutex::new(Inner {
            lock: Some(lock),
            options,
            staging,
            target,
            manifest,
            writers: Vec::new(),
            closed: false,
            published: false,
        }))))
    }

    pub fn register_stage(
        &self,
        mut stage: StageRecording,
        schema: SchemaRef,
    ) -> Result<Arc<Mutex<VortexWriter>>> {
        let mut inner = self
            .0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?;
        anyhow::ensure!(!inner.closed, "cannot register a stage after finalization");
        if inner
            .manifest
            .stages
            .iter()
            .any(|entry| entry.index == stage.index)
        {
            return Err(anyhow!("duplicate stage dump index {}", stage.index));
        }
        let label = stage
            .name
            .as_deref()
            .unwrap_or_else(|| stage.stage_type.rsplit("::").next().unwrap_or("stage"));
        let label: String = label
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        stage.file = format!("{:02}-{label}.vortex", stage.index);
        let writer = Arc::new(Mutex::new(VortexWriter::new(
            inner.staging.join(&stage.file),
            schema,
        )?));
        inner.writers.push(writer.clone());
        inner.manifest.stages.push(stage);
        write_manifest(&inner.staging, &inner.manifest)?;
        Ok(writer)
    }

    pub fn set_source_fingerprint(&self, fingerprint: Option<SourceFingerprint>) -> Result<()> {
        let mut inner = self
            .0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?;
        inner.manifest.source_fingerprint = fingerprint;
        write_manifest(&inner.staging, &inner.manifest)
    }

    pub fn attach(&self, context: &mut PipelineContext) {
        context.add_observer(DumpObserver(self.clone()));
    }

    pub fn staging_dir(&self) -> Result<PathBuf> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?
            .staging
            .clone())
    }

    pub fn close_all(&self) -> Result<()> {
        let writers = self
            .0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?
            .writers
            .clone();
        let mut first = None;
        for writer in writers {
            if let Err(error) = writer
                .lock()
                .map_err(|_| anyhow!("Vortex writer lock poisoned"))?
                .close()
                && first.is_none()
            {
                first = Some(error);
            }
        }
        if let Some(error) = first {
            return Err(error);
        }
        self.0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?
            .closed = true;
        Ok(())
    }

    pub fn publish(&self, status: RunStatus) -> Result<PublishedRun> {
        anyhow::ensure!(status.is_replayable(), "cannot publish an incomplete run");
        let mut inner = self
            .0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?;
        anyhow::ensure!(!inner.published, "run already published");
        anyhow::ensure!(inner.closed, "Vortex writers have not been finalized");
        anyhow::ensure!(
            inner.manifest.stages.len() == inner.writers.len(),
            "recording stage/writer count mismatch"
        );
        let stages = inner
            .manifest
            .stages
            .iter()
            .zip(&inner.writers)
            .map(|(stage, writer)| {
                let rows = writer
                    .lock()
                    .map_err(|_| anyhow!("Vortex writer lock poisoned"))?
                    .rows_written();
                let bytes = fs::metadata(inner.staging.join(&stage.file))?.len();
                Ok(PublishedStage {
                    file: stage.file.clone(),
                    rows,
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        inner.manifest.status = status;
        write_manifest(&inner.staging, &inner.manifest)?;
        if inner.target.exists() {
            anyhow::ensure!(inner.options.replace, "run target already exists");
            let backup = inner.options.root.join(".backup").join(format!(
                "{}.{}",
                inner.manifest.run_name, inner.manifest.run_id
            ));
            fs::rename(&inner.target, &backup)?;
            if let Err(error) = fs::rename(&inner.staging, &inner.target) {
                let rollback = fs::rename(&backup, &inner.target);
                return Err(error)
                    .with_context(|| format!("publish failed; rollback: {rollback:?}"));
            }
            if let Err(error) = fs::remove_dir_all(backup) {
                tracing::warn!(?error, "Published run but could not remove its backup");
            }
        } else {
            fs::rename(&inner.staging, &inner.target)?;
        }
        inner.published = true;
        inner.lock.take();
        Ok(PublishedRun {
            directory: inner.target.clone(),
            status,
            stages,
        })
    }

    pub fn mark_failed(&self) -> Result<()> {
        let mut inner = self
            .0
            .lock()
            .map_err(|_| anyhow!("dump manager lock poisoned"))?;
        inner.manifest.status = RunStatus::Failed;
        write_manifest(&inner.staging, &inner.manifest)
    }
}

fn validate_name(name: &str) -> Result<()> {
    anyhow::ensure!(
        !name.is_empty() && name.len() <= 100,
        "invalid run name length"
    );
    anyhow::ensure!(name != "." && name != "..", "invalid run name");
    anyhow::ensure!(
        name.chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')),
        "run name must use ASCII letters, digits, '.', '-' or '_'"
    );
    Ok(())
}

fn write_manifest(dir: &Path, manifest: &RunManifest) -> Result<()> {
    let temporary = dir.join("manifest.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(manifest)?)?;
    fs::rename(temporary, dir.join("manifest.json"))?;
    Ok(())
}

fn recover_backup(root: &Path, name: &str, target: &Path) -> Result<()> {
    let prefix = format!("{name}.");
    let mut backups = fs::read_dir(root.join(".backup"))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|file| file.starts_with(&prefix))
        })
        .collect::<Vec<_>>();
    if target.exists() {
        if read_manifest(target).is_ok_and(|manifest| manifest.status.is_replayable()) {
            for backup in backups {
                fs::remove_dir_all(backup)?;
            }
            return Ok(());
        }
        anyhow::ensure!(
            backups.is_empty(),
            "target run is invalid while a backup exists"
        );
    } else if backups.len() == 1 {
        fs::rename(backups.remove(0), target)?;
    } else {
        anyhow::ensure!(backups.is_empty(), "multiple backups for run {name:?}");
    }
    Ok(())
}

struct DumpObserver(DumpManager);

impl PipelineObserver for DumpObserver {
    fn pipeline_finished(&mut self, context: &mut PipelineContext) -> Result<(), VydError> {
        let closed = self.0.close_all();
        let statistics = context
            .try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics")
            .map(|stats| stats.snapshot())
            .map_err(anyhow::Error::new)
            .and_then(|snapshot| {
                let staging = self
                    .0
                    .0
                    .lock()
                    .map_err(|_| anyhow!("dump manager lock poisoned"))?
                    .staging
                    .clone();
                fs::write(
                    staging.join("run_statistics.json"),
                    serde_json::to_vec_pretty(&snapshot)?,
                )?;
                Ok(())
            });
        Ok(closed.and(statistics).context("finalize telemetry")?)
    }
}
