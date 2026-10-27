use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn facade_builds_and_runs_gstreamer_pipeline() {
    let config = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/basic.toml");
    let pipeline = vyd::PipelineBuilder::from_config_file(config)
        .unwrap()
        .with_gstreamer_nodes(0, "identity")
        .build()
        .unwrap();
    let stats = pipeline.statistics();
    let run = pipeline.run_with_outcome().unwrap();
    assert_eq!(run.termination, vyd::RunTermination::EndOfStream);
    let context = run.context;
    assert_eq!(stats.snapshot().frame_count, 5);
    assert_eq!(
        context
            .get_as::<AtomicUsize>("frames_sourced")
            .load(Ordering::SeqCst),
        5
    );
}

#[test]
fn facade_reports_requested_cancellation() {
    let config = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/basic.toml");
    let pipeline = vyd::PipelineBuilder::from_config_file(config)
        .unwrap()
        .build()
        .unwrap();
    pipeline.cancellation_token().cancel();
    let run = pipeline.run_with_outcome().unwrap();
    assert_eq!(run.termination, vyd::RunTermination::Cancelled);
}
