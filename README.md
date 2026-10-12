# Vyd

Vyd provides a reusable Rust video processing pipeline. A `FrameSource` produces frames, `FrameProcessor` stages transform or skip them, and a `FrameSink` consumes them. Stages can be registered and assembled from TOML; the library also provides frame storage, timing statistics, and optional Parquet dumps.

## Run the example

```sh
cargo run -p vyd-main -- --config examples/basic.toml
```

The example uses the built-in dummy source, processor, and sink. It processes five small grayscale frames without a camera or video file. To use your own stages, implement the relevant traits in `vyd::traits`, register them with `register_vyd_frame_source!`, `register_vyd_frame_processor!`, or `register_vyd_frame_sink!`, and link those stage crates into your application. The TOML `stage_type` is `<rust_crate_name>::<TypeName>`; package name hyphens become underscores in Rust crate names.

## Verify

```sh
cargo test --workspace
cargo test --workspace --all-features
```

The workspace contains:

- `vyd`: pipeline orchestration, lifecycle observers, stage interfaces, managed Luma8 and full-range NV12 CPU frames, optional wgpu frame access, and statistics.
- `vyd-telemetry`: run metadata, statistics export, and Parquet dumping.
- `vyd-gstreamer`: generic file and RTSP frame sources with pooled CPU buffers.
- `vyd-gstreamer-gpu`: experimental GPU buffer path. Its current source still assumes 1920×1080 Luma8 frames.
- `vyd-main`: example CLI.

The GStreamer crates require the matching system GStreamer libraries.

The `gpu-wgpu` feature enables managed GPU transfers and compute stages. Adjacent GPU stages can retain frames on the device; the current GStreamer source and sink boundaries use CPU GRAY8 buffers. NV12 conversion is defined for full-range grayscale Y data, with neutral chroma when converting from Luma8.

See [the design principles](https://github.com/sdd/vyd/blob/master/DESIGN_PRINCIPLES.md) for the API and runtime
guidelines behind Vyd.

## Run telemetry

Applications can attach a `vyd_telemetry::DumpManager` to their `PipelineContext` before calling `run_pipeline`. Stages access that manager to dump application data implementing `Dumpable`. The telemetry observer periodically flushes writers, closes them when the run finishes, and writes `run_metadata.json` and `run_statistics.json`. Finalization also runs after initialization and observer errors.

`vyd::pipeline::run_pipeline_with_config_file_path(path, stats)` runs a configuration without dumping. `vyd_telemetry::run_pipeline_with_config_file_path(path, stats, dump_dir)` adds the telemetry integration. The core `vyd` crate has no Arrow or Parquet dependencies.

## Release automation

Nightly CI checks formatting, Rust and workflow lints, tests, and coverage. Pushes to `master` prepare a release-plz PR after the all-feature tests pass. Merging a `release-plz-` PR publishes the workspace crates and creates one `crate-name/version` tag and GitHub release per crate.

Configure the repository secrets `RELEASE_PLZ_TOKEN` (a GitHub token that can open release PRs and trigger their CI) and `CARGO_REGISTRY_TOKEN` (a crates.io token with permission to publish the crates) before enabling the release workflows. `CODECOV_TOKEN` is optional; coverage is always saved as a workflow artifact.
