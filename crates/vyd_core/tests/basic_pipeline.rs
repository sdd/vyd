use vyd_core::VydError;
use vyd_core::pipeline::{PipelineConfig, PipelineContext, build_pipeline, run_pipeline};

#[test]
fn unknown_stage_is_a_config_error() {
    let config: PipelineConfig = toml::from_str(
        "[source]\nstage_type = 'missing::Source'\n[sink]\nstage_type = 'vyd::DummySink'\n",
    )
    .unwrap();

    assert!(matches!(
        build_pipeline(&config),
        Err(VydError::UnknownStage { kind: "source", stage_type }) if stage_type == "missing::Source"
    ));
}
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_build_simplest_pipeline_from_toml() {
    // Create a TOML configuration for the test pipeline
    let toml_config = r#"
[source]
stage_type = "vyd::DummySource"

[[stages]]
stage_type = "vyd::DummyProcessor"

[sink]
stage_type = "vyd::DummySink"
"#;

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline from the configuration
    let pipeline = build_pipeline(&config).unwrap();

    // Verify that the pipeline components were created correctly
    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"vyd::DummySource\", stages: [\"vyd::DummyProcessor\"], sink: \"vyd::DummySink\" }"
    );
}

#[test]
fn test_build_pipeline_with_multiple_processors() {
    let toml_config = r#"
[source]
stage_type = "vyd::DummySource"

[[stages]]
stage_type = "vyd::DummyProcessor"

[[stages]]
stage_type = "vyd::DummyProcessor"

[sink]
stage_type = "vyd::DummySink"
"#;

    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    let pipeline = build_pipeline(&config).unwrap();

    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"vyd::DummySource\", stages: [\"vyd::DummyProcessor\", \"vyd::DummyProcessor\"], sink: \"vyd::DummySink\" }"
    );
}

#[test]
fn test_run_pipeline_integration() {
    // Create a simple pipeline configuration
    let toml_config = r#"
[source]
stage_type = "vyd::DummySource"

[[stages]]
stage_type = "vyd::DummyProcessor"

[sink]
stage_type = "vyd::DummySink"
"#;

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline
    let pipeline = build_pipeline(&config).unwrap();

    // Create pipeline context
    let pipeline_context = PipelineContext::new();

    // Verify initial state
    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"vyd::DummySource\", stages: [\"vyd::DummyProcessor\"], sink: \"vyd::DummySink\" }"
    );

    // Run the pipeline - this should process all frames from the dummy source
    // The dummy source produces 5 frames, so this should process all of them
    let final_context = run_pipeline(pipeline_context, pipeline, None).unwrap();

    println!("Pipeline execution completed successfully!");

    // Assert on the statistics recorded in the pipeline context
    let frames_sourced = final_context
        .try_get_as::<AtomicUsize>("frames_sourced")
        .expect("frames_sourced should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_processed = final_context
        .try_get_as::<AtomicUsize>("frames_processed")
        .expect("frames_processed should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_sunk = final_context
        .try_get_as::<AtomicUsize>("frames_sunk")
        .expect("frames_sunk should be AtomicUsize")
        .load(Ordering::SeqCst);

    // Verify expected counts - dummy source generates 5 frames
    assert_eq!(frames_sourced, 5, "Expected 5 frames to be generated");
    assert_eq!(frames_processed, 5, "Expected 5 frames to be processed");
    assert_eq!(frames_sunk, 5, "Expected 5 frames to be sunk");

    println!("Statistics verification passed:");
    println!("  Frames sourced: {}", frames_sourced);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);
}

#[test]
fn test_run_pipeline_with_multiple_processors() {
    let toml_config = r#"
[source]
stage_type = "vyd::DummySource"

[[stages]]
stage_type = "vyd::DummyProcessor"

[[stages]]
stage_type = "vyd::DummyProcessor"

[sink]
stage_type = "vyd::DummySink"
"#;

    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    let pipeline = build_pipeline(&config).unwrap();
    let pipeline_context = PipelineContext::new();

    // Run the pipeline
    let final_context = run_pipeline(pipeline_context, pipeline, None).unwrap();

    // With 2 processors, frames_processed should be 10 (5 frames × 2 processors)
    let frames_sourced = final_context
        .try_get_as::<AtomicUsize>("frames_sourced")
        .expect("frames_sourced should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_processed = final_context
        .try_get_as::<AtomicUsize>("frames_processed")
        .expect("frames_processed should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_sunk = final_context
        .try_get_as::<AtomicUsize>("frames_sunk")
        .expect("frames_sunk should be AtomicUsize")
        .load(Ordering::SeqCst);

    assert_eq!(frames_sourced, 5, "Expected 5 frames to be generated");
    assert_eq!(
        frames_processed, 10,
        "Expected 10 total processing operations (5 frames × 2 processors)"
    );
    assert_eq!(frames_sunk, 5, "Expected 5 frames to be consumed");

    println!("Multi-processor pipeline statistics:");
    println!("  Frames sourced: {}", frames_sourced);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);
}
