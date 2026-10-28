use serde::Deserialize;
pub(crate) use vyd_gstreamer::source_config::InputConfig;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    #[serde(default)]
    pub(crate) mean_frame_latency_ms: u64,
    pub(crate) frame_zero_capture_unix_us: Option<u64>,
    #[serde(flatten)]
    pub(crate) input: InputConfig,
}
