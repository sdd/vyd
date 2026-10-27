use arrow_schema::{FieldRef, Schema};
use serde::{Deserialize, Serialize};
use serde_arrow::schema::{SchemaLike, TracingOptions};
use std::path::Path;
use std::sync::Arc;
use vyd_telemetry::{
    DumpManager, DumpRunOptions, RECORDING_FORMAT_VERSION, RunStatus, StageRecording, read_manifest,
};

fn options(root: &Path, replace: bool) -> DumpRunOptions {
    DumpRunOptions {
        root: root.to_path_buf(),
        run_name: Some("orb_baseline".into()),
        replace,
    }
}

#[test]
fn named_run_replacement_publishes_a_finalized_cancelled_prefix() {
    let root = std::env::temp_dir().join(format!(
        "vyd_named_run_{}_{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    let first = DumpManager::new(options(&root, false), "first".into(), "source".into()).unwrap();
    first.close_all().unwrap();
    let target = first.publish(RunStatus::Completed).unwrap().directory;
    let first_id = read_manifest(&target).unwrap().run_id;
    assert_eq!(
        read_manifest(&target).unwrap().format_version,
        RECORDING_FORMAT_VERSION
    );
    drop(first);
    assert!(DumpManager::new(options(&root, false), "again".into(), "source".into()).is_err());

    let replacement =
        DumpManager::new(options(&root, true), "second".into(), "source".into()).unwrap();
    assert_eq!(read_manifest(&target).unwrap().run_id, first_id);
    assert!(replacement.publish(RunStatus::Cancelled).is_err());
    replacement.close_all().unwrap();
    replacement.publish(RunStatus::Cancelled).unwrap();
    let manifest = read_manifest(&target).unwrap();
    assert_eq!(manifest.status, RunStatus::Cancelled);
    assert_ne!(manifest.run_id, first_id);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_replacement_keeps_the_old_run() {
    let root = std::env::temp_dir().join(format!(
        "vyd_failed_replace_{}_{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    let first = DumpManager::new(options(&root, false), "first".into(), "source".into()).unwrap();
    first.close_all().unwrap();
    let target = first.publish(RunStatus::Completed).unwrap().directory;
    let first_id = read_manifest(&target).unwrap().run_id;
    drop(first);
    let replacement =
        DumpManager::new(options(&root, true), "second".into(), "source".into()).unwrap();
    replacement.mark_failed().unwrap();
    assert_eq!(read_manifest(&target).unwrap().run_id, first_id);
    std::fs::remove_dir_all(root).unwrap();
}

#[derive(Deserialize, Serialize)]
struct Row {
    value: u32,
}

#[test]
fn published_summary_reports_stage_rows_and_file_size() {
    let root = std::env::temp_dir().join(format!(
        "vyd_dump_summary_{}_{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    let manager =
        DumpManager::new(options(&root, false), "config".into(), "source".into()).unwrap();
    let fields = Vec::<FieldRef>::from_type::<Row>(TracingOptions::default()).unwrap();
    let writer = manager
        .register_stage(
            StageRecording {
                index: 0,
                name: Some("features".into()),
                stage_type: "test::Features".into(),
                codec_version: 1,
                file: String::new(),
                upstream_signature: String::new(),
            },
            Arc::new(Schema::new(fields.clone())),
        )
        .unwrap();
    writer
        .lock()
        .unwrap()
        .write(
            serde_arrow::to_record_batch(&fields, &[Row { value: 1 }, Row { value: 2 }]).unwrap(),
        )
        .unwrap();
    manager.close_all().unwrap();
    let summary = manager.publish(RunStatus::Completed).unwrap();
    assert!(summary.directory.is_absolute());
    assert_eq!(summary.stages.len(), 1);
    assert_eq!(summary.stages[0].rows, 2);
    assert!(summary.stages[0].bytes > 0);
    assert!(format!("{summary}").contains("2 frames"));
    std::fs::remove_dir_all(root).unwrap();
}
