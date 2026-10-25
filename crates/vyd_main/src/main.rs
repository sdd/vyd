#![allow(unused_imports)]

mod config;

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

    let pipeline = vyd::PipelineBuilder::from_config_file(&config.config_path)?.build();
    let stats = pipeline.statistics();
    let stats_for_signal = stats.clone();

    // Set up Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\nShutting down...");
        stats_for_signal.stop();
    })
    .expect("Error setting Ctrl-C handler");

    pipeline.run()?;

    // If we reach here, the pipeline completed naturally
    stats.print_final_stats();
    Ok(())
}
