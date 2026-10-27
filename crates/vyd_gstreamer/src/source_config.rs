//! Shared file/RTSP input settings for the CPU and GPU source adapters.
use http::Uri;
use serde::Deserialize;
use thiserror::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(try_from = "InputOptions")]
pub enum InputConfig {
    Rtsp { uri: Uri },
    File { path: String, pseudo_live: bool },
}

impl InputConfig {
    pub fn is_live_mode(&self) -> bool {
        matches!(
            self,
            Self::Rtsp { .. }
                | Self::File {
                    pseudo_live: true,
                    ..
                }
        )
    }
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct InputOptions {
    rtsp_url: Option<String>,
    file_path: Option<String>,
    pseudo_live: bool,
}

#[derive(Debug, Error)]
enum InputConfigError {
    #[error("either rtsp_url or file_path must be specified")]
    MissingInput,
    #[error("only one of rtsp_url or file_path can be specified")]
    ConflictingInputs,
    #[error("invalid rtsp_url: {0}")]
    InvalidUri(#[from] http::uri::InvalidUri),
}

impl TryFrom<InputOptions> for InputConfig {
    type Error = InputConfigError;

    fn try_from(options: InputOptions) -> Result<Self, Self::Error> {
        match (options.rtsp_url, options.file_path) {
            (Some(url), None) => Ok(Self::Rtsp { uri: url.parse()? }),
            (None, Some(path)) => Ok(Self::File {
                path,
                pseudo_live: options.pseudo_live,
            }),
            (None, None) => Err(InputConfigError::MissingInput),
            (Some(_), Some(_)) => Err(InputConfigError::ConflictingInputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_validation_rejects_missing_conflicting_and_malformed_settings() {
        for config in [
            "",
            "file_path = 'clip.mkv'\nrtsp_url = 'rtsp://example.test/live'",
            "rtsp_url = 'not a valid uri'",
            "file_path = 42",
            "file_path = 'clip.mkv'\npseudo_live = 'true'",
        ] {
            assert!(toml::from_str::<InputConfig>(config).is_err(), "{config}");
        }
    }

    #[test]
    fn file_defaults_and_rtsp_live_mode_are_preserved() {
        let file: InputConfig = toml::from_str("file_path = 'clip.mkv'").unwrap();
        assert!(!file.is_live_mode());
        let rtsp: InputConfig = toml::from_str("rtsp_url = 'rtsp://example.test/live'").unwrap();
        assert!(rtsp.is_live_mode());
    }
}
