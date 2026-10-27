use thiserror::Error;

#[derive(Error, Debug)]
pub enum VydError {
    /// Recording or replay failure, retaining its context and original source.
    #[error("Pipeline persistence error: {0}")]
    PersistenceError(#[from] anyhow::Error),

    #[error(transparent)]
    Frame(#[from] crate::frame::FrameError),
    #[cfg(feature = "gpu-wgpu")]
    #[error(transparent)]
    GpuTransfer(#[from] crate::gpu::transfer::GpuTransferError),
    #[cfg(feature = "gpu-wgpu")]
    #[error(
        "GPU processor {stage_index} declared {expected:?} output but returned {memory} {actual:?}"
    )]
    GpuStageOutput {
        stage_index: usize,
        expected: crate::frame::PixelFormat,
        actual: crate::frame::PixelFormat,
        memory: &'static str,
    },
    #[error("{feature} is not yet implemented: {detail}")]
    NotYetImplemented {
        feature: &'static str,
        detail: String,
    },
    #[error(
        "Pipeline shutdown exceeded {timeout:?}; cleanup retains live resources until active work returns"
    )]
    ShutdownTimeout { timeout: std::time::Duration },
    #[error("Pipeline services lock poisoned")]
    PipelineServicesPoisoned,

    #[error("Stage {stage} failed during {operation}: {source}")]
    StageFailed {
        stage: &'static str,
        operation: &'static str,
        #[source]
        source: Box<VydError>,
    },

    #[error("Unknown {kind} stage: {stage_type}")]
    UnknownStage {
        kind: &'static str,
        stage_type: String,
    },

    #[error("Factory for {stage_type} returned the wrong stage kind (expected {kind})")]
    StageTypeMismatch {
        kind: &'static str,
        stage_type: String,
    },

    #[error("Failed to read pipeline config: {0}")]
    ConfigRead(#[from] std::io::Error),

    #[error("Failed to parse pipeline config: {0}")]
    ConfigParse(#[from] toml::de::Error),
    #[error("Plugin error: {0}")]
    GeneralPluginError(String),

    #[error("Plugin config missing")]
    PluginMissingConfigError,

    #[error("Plugin config invalid: {0}")]
    PluginInvalidConfigError(String),

    #[error("Frame metadata not found")]
    FrameMetadataNotFoundError,

    #[error("Frame metadata downcast type error")]
    FrameMetadataTypeError,
}
