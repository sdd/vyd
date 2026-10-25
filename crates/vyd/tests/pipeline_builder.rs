use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn facade_builds_and_runs_gstreamer_pipeline() {
    let config = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/basic.toml");
    let pipeline = vyd::PipelineBuilder::from_config_file(config)
        .unwrap()
        .with_gstreamer_nodes(0, "identity")
        .build();
    let stats = pipeline.statistics();
    let context = pipeline.run().unwrap();
    assert_eq!(stats.snapshot().frame_count, 5);
    assert_eq!(
        context
            .get_as::<AtomicUsize>("frames_sourced")
            .load(Ordering::SeqCst),
        5
    );
}
