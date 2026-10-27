use http::Uri;
use toml::Value;

#[derive(Clone, Debug)]
pub(crate) enum InputConfig {
    Rtsp { uri: Uri },
    File { path: String, pseudo_live: bool },
}

impl InputConfig {
    pub(crate) fn is_live_mode(&self) -> bool {
        matches!(
            self,
            InputConfig::Rtsp { .. }
                | InputConfig::File {
                    pseudo_live: true,
                    ..
                }
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub(crate) input: InputConfig,
}

impl TryFrom<&Value> for Config {
    type Error = String;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let Some(table) = value.as_table() else {
            return Err("Expected config to be a toml table".to_string());
        };

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

        Ok(Self { input })
    }
}
