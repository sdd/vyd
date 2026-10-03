# Vidd

Vidd provides a reusable Rust video processing pipeline. A `FrameSource` produces frames, `FrameProcessor` stages transform or skip them, and a `FrameSink` consumes them. Stages can be registered and assembled from TOML; the library also provides frame storage, timing statistics, and optional Parquet dumps.

## Run the example

```sh
cargo run -p vyd_main -- --config examples/basic.toml
```

The example uses the built-in dummy source, processor, and sink. It processes five small grayscale frames without a camera or video file. To use your own stages, implement the relevant traits in `vyd::traits`, register them with `register_vyd_frame_source!`, `register_vyd_frame_processor!`, or `register_vyd_frame_sink!`, and link those stage crates into your application. The TOML `stage_type` is `<crate_name>::<TypeName>`.

## Verify

```sh
cargo test --workspace
```

The workspace contains:

- `vyd`: pipeline orchestration, lifecycle observers, stage interfaces, Luma8 CPU frames, GPU frame handles, and statistics.
- `vyd_telemetry`: run metadata, statistics export, and Parquet dumping.
- `vyd_gstreamer`: generic file and RTSP frame sources with pooled CPU buffers.
- `vyd_gstreamer_gpu`: experimental GPU buffer path. Its current source still assumes 1920×1080 Luma8 frames.
- `vyd_main`: example CLI.

The GStreamer crates require the matching system GStreamer libraries. Vidd is not published to crates.io yet.

## Run telemetry

Applications can attach a `vyd_telemetry::DumpManager` to their `PipelineContext` before calling `run_pipeline`. Stages access that manager to dump application data implementing `Dumpable`. The telemetry observer periodically flushes writers, closes them when the run finishes, and writes `run_metadata.json` and `run_statistics.json`. Finalization also runs after initialization and observer errors.

`vyd::pipeline::run_pipeline_with_config_file_path(path, stats)` runs a configuration without dumping. `vyd_telemetry::run_pipeline_with_config_file_path(path, stats, dump_dir)` adds the telemetry integration. The core `vyd` crate has no Arrow or Parquet dependencies.
