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

    let builder = vyd::PipelineBuilder::from_config_file(&config.config_path)?;
    let pipeline = config.run.apply_to(builder)?.build()?;
    let stats = pipeline.statistics();
    let cancellation = pipeline.cancellation_token();

    // Set up Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\nShutting down...");
        cancellation.cancel();
    })
    .expect("Error setting Ctrl-C handler");

    let run = pipeline.run_with_outcome()?;

    // A clean end or cancellation has finalized the pipeline.
    stats.print_final_stats();
    if let Some(recording) = run.recording {
        println!("\n{recording}");
    }
    Ok(())
}
