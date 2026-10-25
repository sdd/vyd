use http::Uri;
use toml::Value;

#[derive(Clone, Debug)]
pub(crate) enum InputConfig {
    Rtsp { uri: Uri },
    File { path: String, pseudo_live: bool },
}

impl InputConfig {
    pub(crate) fn is_live_mode(&self) -> bool {
        match self {
            InputConfig::Rtsp { .. }
            | InputConfig::File {
                pseudo_live: true, ..
            } => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub(crate) input: InputConfig,
    pub(crate) buffer_pool_slot_count: usize,
    pub(crate) zc_enabled: bool,
    pub(crate) max_frame_buffers: Option<usize>,
    pub(crate) max_frame_buffer_bytes: Option<usize>,
}

impl TryFrom<&Value> for Config {
    type Error = String;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let Some(table) = value.as_table() else {
            return Err("Expected config to be a toml table".to_string());
        };

        let zc_enabled = if let Some(zc_enabled) = table.get("zc_enabled") {
            if let Some(zc_enabled) = zc_enabled.as_bool() {
                zc_enabled
            } else {
                return Err("zc_enabled must be a boolean".to_string());
            }
        } else {
            true
        };

        let buffer_pool_slot_count =
            if let Some(buffer_pool_slot_count) = table.get("buffer_pool_slot_count") {
                if let Some(buffer_pool_slot_count) = buffer_pool_slot_count.as_integer() {
                    buffer_pool_slot_count as usize
                } else {
                    return Err("buffer_pool_slot_count must be an integer".to_string());
                }
            } else {
                16
            };

        let max_frame_buffers = table
            .get("max_frame_buffers")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);
        let max_frame_buffer_bytes = table
            .get("max_frame_buffer_bytes")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let rtsp_url = table.get("rtsp_url");
        let file_path = table.get("file_path");
        let input = match (rtsp_url, file_path) {
            (None, None) => {
                return Err("Either rtsp_url or file_path must be specified in config".to_string());
            }
            (Some(rtsp_url), None) => {
                if let Some(rtsp_url) = rtsp_url.as_str() {
                    InputConfig::Rtsp {
                        uri: rtsp_url.to_string().try_into().expect("Invalid rtsp_url"),
                    }
                } else {
                    return Err("rtsp_url must be a string".to_string());
                }
            }
            (None, Some(file_path)) => {
                if let Some(file_path) = file_path.as_str() {
                    let pseudo_live = if let Some(pseudo_live) = table.get("pseudo_live") {
                        if let Some(pseudo_live) = pseudo_live.as_bool() {
                            pseudo_live
                        } else {
                            return Err("pseudo_live must be a boolean".to_string());
                        }
                    } else {
                        false
                    };

                    InputConfig::File {
                        path: file_path.to_string(),
                        pseudo_live,
                    }
                } else {
                    return Err("file_path must be a string".to_string());
                }
            }
            _ => {
                return Err(
                    "only one of rtsp_url or file_path can be specified in config".to_string(),
                );
            }
        };

        Ok(Self {
            input,
            buffer_pool_slot_count,
            zc_enabled,
            max_frame_buffers,
            max_frame_buffer_bytes,
        })
    }
}
