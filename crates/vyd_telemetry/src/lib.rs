pub mod dump_manager;
mod parquet_dumper;
mod traits;

pub use dump_manager::{DumpManager, RunMetadata};
pub use parquet_dumper::ParquetDumper;
pub use traits::Dumpable;

/// Run a configured pipeline with optional run metadata, Parquet dumps, and statistics export.
pub fn run_pipeline_with_config_file_path(
    path: &std::path::Path,
    stats: std::sync::Arc<vyd::statistics::PipelineStatistics>,
    dump_dir: Option<std::path::PathBuf>,
) -> Result<vyd::pipeline::PipelineContext, vyd::VydError> {
    let config_raw = std::fs::read_to_string(path)?;
    let config: vyd::pipeline::PipelineConfig = toml::from_str(&config_raw)?;
    let pipeline = vyd::pipeline::build_pipeline(&config)?;
    let mut context = vyd::pipeline::PipelineContext::new();
    if let Some(directory) = dump_dir {
        DumpManager::new(directory)
            .map_err(|error| {
                vyd::VydError::GeneralPluginError(format!("telemetry setup failed: {error}"))
            })?
            .with_config(config_raw)
            .with_source(config.source_type().to_string())
            .attach(&mut context);
    }
    vyd::pipeline::run_pipeline(context, pipeline, Some(stats))
}
