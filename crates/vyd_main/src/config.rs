use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "vyd_main")]
#[command(about = "Vyd video processing pipeline")]
#[command(version)]
pub struct Config {
    /// Path to the configuration TOML file
    #[arg(short, long, default_value = "examples/basic.toml")]
    pub config: PathBuf,
}

impl Config {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
