use crate::PipelineBuilder;
use clap::Args;
use std::path::PathBuf;
use vyd_core::VydError;

/// Reusable command-line options for a Vyd recording run.
#[derive(Args, Clone, Debug, Default)]
pub struct RunArgs {
    /// Override recording.dump_dir from TOML (default: ./dumps).
    #[arg(long)]
    pub dump_dir: Option<PathBuf>,
    /// Override recording.run_name from TOML.
    #[arg(long)]
    pub run_name: Option<String>,
    /// Replace a finalized run with the same name after this run finalizes.
    #[arg(long)]
    pub replace_run: bool,
}

impl RunArgs {
    pub fn apply_to(self, builder: PipelineBuilder) -> Result<PipelineBuilder, VydError> {
        builder.with_run_overrides(self.dump_dir, self.run_name, self.replace_run)
    }
}
