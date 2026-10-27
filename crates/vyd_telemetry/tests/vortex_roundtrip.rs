use arrow_schema::{FieldRef, Schema};
use serde::{Deserialize, Serialize};
use serde_arrow::schema::{SchemaLike, TracingOptions};
use std::sync::Arc;
use vyd_telemetry::{VortexReader, VortexWriter};

#[test]
fn writer_failure_preserves_io_source_across_worker_and_pipeline_errors() {
    let path = std::env::temp_dir()
        .join(format!("vyd-missing-{}", rand::random::<u64>()))
        .join("frames.vortex");
    let fields = Vec::<FieldRef>::from_type::<FrameRow>(TracingOptions::default()).unwrap();
    let mut writer = VortexWriter::new(path, Arc::new(Schema::new(fields))).unwrap();
    let error: vyd_core::VydError = writer.close().unwrap_err().into();
    assert!(std::error::Error::source(&error).is_some());
    let vyd_core::VydError::PersistenceError(source) = error else {
        panic!("expected persistence error");
    };
    assert_eq!(
        source.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::NotFound
    );
}

#[test]
fn reader_failure_preserves_source_chain_across_worker() {
    let path = std::env::temp_dir().join(format!("vyd-missing-{}.vortex", rand::random::<u64>()));
    let reader = VortexReader::new(&path).unwrap();
    let error = reader.next_row().unwrap_err();
    assert!(error.to_string().contains(&path.display().to_string()));
    assert!(error.chain().nth(1).is_some());
    reader.close().unwrap();
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct Feature {
    x: f32,
    descriptor: Vec<u8>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct FrameRow {
    frame_index: u64,
    features: Vec<Feature>,
}

#[test]
fn vortex_preserves_nested_features_and_empty_frames() {
    let directory = std::env::temp_dir().join(format!(
        "vyd_vortex_roundtrip_{}_{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("features.vortex");
    let fields = Vec::<FieldRef>::from_type::<FrameRow>(TracingOptions::default()).unwrap();
    let schema = Arc::new(Schema::new(fields.clone()));
    let expected = vec![
        FrameRow {
            frame_index: 0,
            features: vec![],
        },
        FrameRow {
            frame_index: 1,
            features: vec![Feature {
                x: 3.5,
                descriptor: vec![1, 2, 3],
            }],
        },
    ];
    let mut writer = VortexWriter::new(&path, schema).unwrap();
    for row in &expected {
        writer
            .write(serde_arrow::to_record_batch(&fields, &vec![row]).unwrap())
            .unwrap();
    }
    writer.close().unwrap();
    let reader = VortexReader::new(&path).unwrap();
    let mut actual = Vec::new();
    while let Some(batch) = reader.next_row().unwrap() {
        let mut rows: Vec<FrameRow> = serde_arrow::from_record_batch(&batch).unwrap();
        actual.append(&mut rows);
    }
    reader.close().unwrap();
    assert_eq!(actual, expected);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn empty_recording_is_finalized_and_readable() {
    let directory = std::env::temp_dir().join(format!(
        "vyd_vortex_empty_{}_{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("empty.vortex");
    let fields = Vec::<FieldRef>::from_type::<FrameRow>(TracingOptions::default()).unwrap();
    let mut writer = VortexWriter::new(&path, Arc::new(Schema::new(fields))).unwrap();
    writer.close().unwrap();
    let reader = VortexReader::new(&path).unwrap();
    assert!(reader.next_row().unwrap().is_none());
    reader.close().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}
