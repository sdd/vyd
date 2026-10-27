use serde::Deserialize;
pub(crate) use vyd_gstreamer::source_config::InputConfig;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    #[serde(flatten)]
    pub(crate) input: InputConfig,
}
