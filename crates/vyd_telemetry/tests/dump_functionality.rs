use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use toml::Value;
use vyd::pipeline::{PipelineConfig, PipelineContext, build_pipeline, run_pipeline};
use vyd::{VydError, register_vyd_frame_processor};
use vyd_telemetry::{DumpManager, Dumpable};

#[derive(Debug, Deserialize, Serialize)]
struct TestDataRow {
    run_id: u64,
    frame_index: usize,
    value: i32,
}

/// A simple struct that implements Dumpable for testing
struct TestData {
    value: i32,
}

impl Dumpable for TestData {
    type Row = TestDataRow;

    const TABLE_NAME: &'static str = "test_data";
    const VERSION: u32 = 1;

    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row {
        TestDataRow {
            run_id,
            frame_index,
            value: self.value,
        }
    }
}

/// A processor that dumps test data
struct DumpTestProcessor;

impl DumpTestProcessor {
    pub fn new(_config: Option<&Value>) -> Result<Self, VydError> {
        Ok(Self {})
    }
}

impl vyd::traits::FrameProcessor for DumpTestProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut vyd::FrameContext,
        pipeline_ctx: &mut PipelineContext,
    ) -> vyd::FrameProcessorResult {
        // Create some test data
        let test_data = TestData {
            value: frame_ctx.frame_index as i32 * 10,
        };

        // Get the dump manager from the pipeline context
        let entry = pipeline_ctx.entry("dump_manager");
        if let std::collections::hash_map::Entry::Occupied(mut entry) = entry {
            let dump_manager = entry
                .get_mut()
                .downcast_mut::<DumpManager>()
                .expect("dump_manager is not a DumpManager");

            // Dump the data for this frame
            let result = dump_manager.dumper::<TestData>().and_then(|dumper| {
                dumper
                    .lock()
                    .map_err(|e| anyhow::anyhow!("dumper lock poisoned: {e}"))?
                    .dump(&test_data, frame_ctx.frame_index)
            });
            if let Err(e) = result {
                tracing::warn!(error = ?e, "Failed to dump test data");
            }
        }

        vyd::FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "dump_test_processor"
    }

    fn processing_type(&self) -> vyd::statistics::ProcessingType {
        vyd::statistics::ProcessingType::Cpu
    }
}

register_vyd_frame_processor!(DumpTestProcessor);

#[test]
fn empty_dumps_close_to_readable_files_and_reject_later_writes() {
    let directory = std::env::temp_dir().join(format!("vyd_empty_dump_{}", rand::random::<u64>()));
    let mut manager = DumpManager::new(&directory).unwrap();
    let dumper = manager.dumper::<TestData>().unwrap();
    manager.close_all().unwrap();
    manager.close_all().unwrap();
    let reader =
        SerializedFileReader::new(fs::File::open(directory.join("test_data.parquet")).unwrap())
            .unwrap();
    assert_eq!(reader.metadata().file_metadata().num_rows(), 0);
    assert!(
        dumper
            .lock()
            .unwrap()
            .dump(&TestData { value: 1 }, 0)
            .is_err()
    );
    drop(reader);
    drop(dumper);
    drop(manager);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn test_dump_functionality() {
    // Create a test pipeline configuration
    let toml_config = r#"
[source]
stage_type = "vyd::DummySource"
frame_count = 10

[[stages]]
stage_type = "vyd_telemetry::DumpTestProcessor"

[sink]
stage_type = "vyd::DummySink"
"#;

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline
    let pipeline = build_pipeline(&config).unwrap();

    // Create pipeline context
    let mut pipeline_context = PipelineContext::new();

    // Create a temporary directory for the dump files
    let temp_dir = std::env::temp_dir().join(format!("vyd_dump_test_{}", rand::random::<u64>()));
    fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");

    // Create and initialize DumpManager in the pipeline context
    let dump_manager = DumpManager::new(&temp_dir)
        .expect("Failed to create DumpManager")
        .with_config("test_config".to_string())
        .with_source("test_source".to_string());

    dump_manager
        .write_metadata()
        .expect("Failed to write metadata");

    // Store the dump manager in the pipeline context
    dump_manager.attach(&mut pipeline_context);

    // Run the pipeline
    let final_context = run_pipeline(pipeline_context, pipeline, None).unwrap();

    // Verify stats
    let frames_sourced = final_context
        .try_get_as::<AtomicUsize>("frames_sourced")
        .expect("frames_sourced should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_sunk = final_context
        .try_get_as::<AtomicUsize>("frames_sunk")
        .expect("frames_sunk should be AtomicUsize")
        .load(Ordering::SeqCst);

    // Expect 10 frames
    assert_eq!(frames_sourced, 5, "Expected 5 frames to be generated");
    assert_eq!(frames_sunk, 5, "Expected 5 frames to be consumed");

    // Verify that the parquet file was created
    let dump_manager = final_context
        .try_get_as::<DumpManager>("dump_manager")
        .expect("dump_manager should be in context");

    let parquet_path = dump_manager.run_dir().join("test_data.parquet");
    assert!(parquet_path.exists(), "Parquet file was not created");
    // The footer must be readable while the manager and its writer handles are alive.
    let reader = SerializedFileReader::new(fs::File::open(&parquet_path).unwrap()).unwrap();
    assert_eq!(reader.metadata().file_metadata().num_rows(), 5);

    let snapshot: serde_json::Value = serde_json::from_slice(
        &fs::read(dump_manager.run_dir().join("run_statistics.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(snapshot["frame_count"], 5);

    // Check that the metadata file exists
    let metadata_path = dump_manager.run_dir().join("run_metadata.json");
    assert!(metadata_path.exists(), "Metadata file was not created");

    println!("Parquet file path: {}", parquet_path.display());

    drop(reader);
    drop(final_context);
    fs::remove_dir_all(&temp_dir).unwrap();

    println!("Dump functionality test completed successfully!");
    println!("  Frames sourced: {}", frames_sourced);
    println!("  Frames sunk: {}", frames_sunk);
}
