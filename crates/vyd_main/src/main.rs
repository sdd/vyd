#![allow(unused_imports)]

mod config;

use std::sync::Arc;

use vyd::statistics::PipelineStatistics;
// Link the built-in stages so inventory can register them for TOML configs.
use vyd::stages as _;

use crate::config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(false)
        .compact()
        .init();

    // Parse command line arguments
    let config = Config::parse_args();

    // Create shared statistics
    let stats = Arc::new(PipelineStatistics::new());
    let stats_for_signal = stats.clone();

    // Set up Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\nShutting down...");
        stats_for_signal.stop();
    })
    .expect("Error setting Ctrl-C handler");

    vyd_gstreamer::runtime::run_pipeline_with_config_file_path(&config.config, stats.clone())?;

    // If we reach here, the pipeline completed naturally
    stats.print_final_stats();
    Ok(())
}
