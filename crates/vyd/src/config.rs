use anyhow::{Context, ensure};
use serde::Deserialize;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use vyd_core::frame::pool::{PoolError, PoolLimits};
use vyd_telemetry::DumpRunOptions;

/// Runtime-owned sections. Stage-specific fields are deserialized by their crates.
#[derive(Default, Deserialize)]
pub(crate) struct Config {
    #[serde(default)]
    pub recording: RecordingConfig,
    pub cpu_pool: Option<CpuPoolConfig>,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RecordingConfig {
    dump_dir: PathBuf,
    run_name: Option<String>,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            dump_dir: PathBuf::from("./dumps"),
            run_name: None,
        }
    }
}

impl TryFrom<RecordingConfig> for DumpRunOptions {
    type Error = anyhow::Error;

    fn try_from(config: RecordingConfig) -> Result<Self, Self::Error> {
        ensure!(
            !config.dump_dir.as_os_str().is_empty(),
            "recording.dump_dir must not be empty"
        );
        Ok(Self {
            root: config.dump_dir,
            run_name: config.run_name,
            replace: false,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CpuPoolConfig {
    max_buffers: Option<NonZeroUsize>,
    max_bytes: Option<ByteSize>,
}

impl TryFrom<CpuPoolConfig> for PoolLimits {
    type Error = PoolError;

    fn try_from(config: CpuPoolConfig) -> Result<Self, Self::Error> {
        Self::new(
            config.max_buffers.map(NonZeroUsize::get),
            config.max_bytes.map(|size| size.0.get()),
        )
    }
}

#[derive(Deserialize)]
#[serde(try_from = "ByteSizeValue")]
struct ByteSize(NonZeroUsize);

#[derive(Deserialize)]
#[serde(untagged)]
enum ByteSizeValue {
    Bytes(NonZeroUsize),
    HumanReadable(String),
}

impl TryFrom<ByteSizeValue> for ByteSize {
    type Error = anyhow::Error;

    fn try_from(value: ByteSizeValue) -> Result<Self, Self::Error> {
        let bytes = match value {
            ByteSizeValue::Bytes(bytes) => bytes,
            ByteSizeValue::HumanReadable(size) => {
                let bytes = parse_size::parse_size(&size)
                    .with_context(|| format!("invalid byte size {size:?}"))?;
                let bytes =
                    usize::try_from(bytes).context("byte size exceeds addressable memory")?;
                NonZeroUsize::new(bytes).context("byte size must be greater than zero")?
            }
        };
        Ok(Self(bytes))
    }
}
