pub(crate) use crate::source_config::InputConfig;
use serde::Deserialize;
use std::num::NonZeroUsize;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    #[serde(flatten)]
    pub(crate) input: InputConfig,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_defaults_and_pool_bounds_are_deserialized() {
        let config: Config = toml::from_str("file_path = 'clip.mkv'").unwrap();
        assert!(config.zc_enabled);
        assert_eq!(config.buffer_pool_slot_count.get(), 16);
        for value in ["0", "-1", "'16'", "true"] {
            let raw = format!("file_path = 'clip.mkv'\nbuffer_pool_slot_count = {value}");
            assert!(toml::from_str::<Config>(&raw).is_err(), "{raw}");
        }
    }
}
