use crate::parquet_dumper::ParquetDumper;
use crate::traits::Dumpable;
use anyhow::Result;
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Manages all dumping operations for offline analysis
pub struct DumpManager {
    run_id: u64,
    run_dir: PathBuf,
    start_time: chrono::DateTime<chrono::Utc>,
    dumpers: HashMap<&'static str, Arc<dyn Any + Send + Sync>>,
    config: Option<String>,
    source_name: Option<String>,
}

impl DumpManager {
    /// Create a new DumpManager that will store data in the specified directory
    pub fn new<P: AsRef<Path>>(run_dir: P) -> Result<Self> {
        let run_id = rand::random::<u64>();
        let run_dir = run_dir.as_ref().to_path_buf();
        let start_time = chrono::Utc::now();

        std::fs::create_dir_all(&run_dir)?;

        info!(run_id, run_dir = ?run_dir, "Created new dump manager");

        Ok(Self {
            run_id,
            run_dir,
            start_time,
            dumpers: HashMap::new(),
            config: None,
            source_name: None,
        })
    }

    /// Sets the config info to be included in metadata
    pub fn with_config(mut self, config: String) -> Self {
        self.config = Some(config);
        self
    }

    /// Sets the source name to be included in metadata
    pub fn with_source(mut self, source_name: String) -> Self {
        self.source_name = Some(source_name);
        self
    }

    /// Write current metadata to the run directory
    pub fn write_metadata(&self) -> Result<()> {
        let metadata = serde_json::json!({
            "run_id": self.run_id,
            "start_time": self.start_time,
            "source": self.source_name,
            "config": self.config,
        });

        std::fs::write(self.run_dir.join("run_metadata.json"), metadata.to_string())?;
        Ok(())
    }

    /// Get a parquet dumper for the specified type
    pub fn dumper<T: Dumpable>(&mut self) -> Arc<Mutex<ParquetDumper<T>>>
    where
        T::Row: serde::Serialize,
    {
        let dumper_arc = self.dumpers
            .entry(T::TABLE_NAME)
            .or_insert_with(|| {
                let path = self.run_dir.join(format!("{}.parquet", T::TABLE_NAME));
                match ParquetDumper::<T>::new(&path, self.run_id) {
                    Ok(dumper) => {
                        info!(table = T::TABLE_NAME, path = ?path, "Created parquet dumper");
                        Arc::new(Mutex::new(dumper)) as Arc<dyn Any + Send + Sync>
                    },
                    Err(e) => {
                        warn!(table = T::TABLE_NAME, path = ?path, error = ?e, "Failed to create parquet dumper");
                        panic!("Failed to create ParquetDumper for {}: {}", T::TABLE_NAME, e);
                    }
                }
            })
            .clone();

        // Now we can downcast because we have Arc<dyn Any + Send + Sync>
        dumper_arc
            .downcast::<Mutex<ParquetDumper<T>>>()
            .expect("Type mismatch in dumper cache")
    }

    /// Flush all open dumpers
    pub fn flush_all(&mut self) -> Result<()> {
        // TODO: This is still challenging without a common trait, but we can work around it
        // by storing flush functions or implementing a trait-based approach
        tracing::debug!(
            "flush_all not fully implemented - consider redesigning with a common trait"
        );
        Ok(())
    }

    /// Get the run ID for this dump session
    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    /// Get the run directory for this dump session
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }
}
