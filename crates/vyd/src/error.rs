use thiserror::Error;

#[derive(Error, Debug)]
pub enum VydError {
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
