pub(crate) use crate::source_config::InputConfig;
use serde::Deserialize;
use std::num::NonZeroUsize;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    #[serde(flatten)]
    pub(crate) input: InputConfig,
    #[serde(default)]
    pub(crate) mean_frame_latency_ms: u64,
    /// Explicit capture UTC of frame zero, as Unix microseconds.
    pub(crate) frame_zero_capture_unix_us: Option<u64>,
    #[serde(default = "default_slot_count")]
    pub(crate) buffer_pool_slot_count: NonZeroUsize,
    #[serde(default = "default_zero_copy")]
    pub(crate) zc_enabled: bool,
}

fn default_slot_count() -> NonZeroUsize {
    NonZeroUsize::new(16).unwrap()
}

fn default_zero_copy() -> bool {
    true
}

impl Config {
    pub(crate) fn frame_zero_capture_ts(&self) -> Option<std::time::SystemTime> {
        self.frame_zero_capture_unix_us
            .and_then(|us| std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_micros(us)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_capture_settings_are_deserialized() {
        let config: Config = toml::from_str("file_path = 'clip.mkv'\nmean_frame_latency_ms = 125\nframe_zero_capture_unix_us = 100000000").unwrap();
        assert_eq!(config.mean_frame_latency_ms, 125);
        assert_eq!(
            config.frame_zero_capture_ts(),
            Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(100))
        );
        assert!(
            toml::from_str::<Config>("file_path = 'clip.mkv'\nmean_frame_latency_ms = -1").is_err()
        );
    }

    #[test]
    fn source_defaults_and_pool_bounds_are_deserialized() {
        let config: Config = toml::from_str("file_path = 'clip.mkv'").unwrap();
        assert!(config.zc_enabled);
        assert_eq!(config.mean_frame_latency_ms, 0);
        assert_eq!(config.frame_zero_capture_ts(), None);
        assert_eq!(config.buffer_pool_slot_count.get(), 16);
        for value in ["0", "-1", "'16'", "true"] {
            let raw = format!("file_path = 'clip.mkv'\nbuffer_pool_slot_count = {value}");
            assert!(toml::from_str::<Config>(&raw).is_err(), "{raw}");
        }
    }
}
