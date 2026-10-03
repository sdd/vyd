use crate::parquet_dumper::ParquetDumper;
use crate::traits::Dumpable;
use anyhow::Result;
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RunMetadata {
    pub run_id: u64,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub source: Option<String>,
    pub config: Option<String>,
}

struct DumperHandle {
    value: Arc<dyn Any + Send + Sync>,
    flush: Box<dyn Fn() -> Result<()> + Send + Sync>,
}

/// Manages all dumping operations for offline analysis
pub struct DumpManager {
    run_id: u64,
    run_dir: PathBuf,
    start_time: chrono::DateTime<chrono::Utc>,
    dumpers: HashMap<&'static str, DumperHandle>,
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
        let metadata = self.metadata();
        std::fs::write(
            self.run_dir.join("run_metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }

    pub fn metadata(&self) -> RunMetadata {
        RunMetadata {
            run_id: self.run_id,
            start_time: self.start_time,
            source: self.source_name.clone(),
            config: self.config.clone(),
        }
    }

    /// Get a parquet dumper for the specified type
    pub fn dumper<T: Dumpable>(&mut self) -> Result<Arc<Mutex<ParquetDumper<T>>>>
    where
        T::Row: serde::Serialize,
    {
        if !self.dumpers.contains_key(T::TABLE_NAME) {
            let path = self.run_dir.join(format!("{}.parquet", T::TABLE_NAME));
            let dumper = Arc::new(Mutex::new(ParquetDumper::<T>::new(&path, self.run_id)?));
            let for_flush = Arc::clone(&dumper);
            self.dumpers.insert(
                T::TABLE_NAME,
                DumperHandle {
                    value: dumper,
                    flush: Box::new(move || {
                        for_flush
                            .lock()
                            .map_err(|e| anyhow::anyhow!("dumper lock poisoned: {e}"))?
                            .flush()
                    }),
                },
            );
            info!(table = T::TABLE_NAME, path = ?path, "Created parquet dumper");
        }
        Arc::clone(&self.dumpers[T::TABLE_NAME].value)
            .downcast::<Mutex<ParquetDumper<T>>>()
            .map_err(|_| anyhow::anyhow!("dumper type mismatch for {}", T::TABLE_NAME))
    }

    /// Flush all open dumpers
    pub fn flush_all(&mut self) -> Result<()> {
        for dumper in self.dumpers.values() {
            (dumper.flush)()?;
        }
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
