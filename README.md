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

- `vyd-core`: pipeline orchestration, lifecycle observers, stage interfaces, managed Luma8 and full-range NV12 CPU frames, optional wgpu frame access, and statistics. It has no GStreamer or Parquet dependency.
- `vyd`: application-facing builder and owned pipeline. It combines the core, GStreamer runtime, and optional telemetry setup.
- `vyd-telemetry`: run metadata, statistics export, and Parquet dumping.
- `vyd-gstreamer`: generic file and RTSP frame sources with pooled CPU buffers.
- `vyd-gstreamer-gpu`: experimental GPU buffer path. Its current source still assumes 1920×1080 Luma8 frames.
- `vyd-main`: example CLI.

The GStreamer crates require the matching system GStreamer libraries.

The `gpu-wgpu` feature enables managed GPU transfers and compute stages. Adjacent GPU stages can retain frames on the device; the current GStreamer source and sink boundaries use CPU GRAY8 buffers. NV12 conversion is defined for full-range grayscale Y data, with neutral chroma when converting from Luma8.

See [the design principles](https://github.com/sdd/vyd/blob/master/DESIGN_PRINCIPLES.md) for the API and runtime
guidelines behind Vyd.

## Run telemetry

Applications can construct a run with `vyd::PipelineBuilder::from_config_file(path)`, optionally call `with_dump_dir(directory)`, then call `build()`. The returned `vyd::Pipeline` exposes `context_mut()` and a cloneable `statistics()` handle before `run()` consumes it. The builder configures a GPU when any configured processor has `gpu = true`; this temporary convention does not depend on processor names. A build without `gpu-wgpu` reports that feature as required.

```rust,no_run
let pipeline = vyd::PipelineBuilder::from_config_file("examples/basic.toml")?
    .with_dump_dir("./dumps")?
    .build();
let stats = pipeline.statistics();
pipeline.run()?;
stats.print_final_stats();
# Ok::<(), Box<dyn std::error::Error>>(())
```

The `vyd-telemetry` crate still offers `DumpManager` for custom `Dumpable` stage data. Its observer flushes writers and writes `run_metadata.json` and `run_statistics.json`, including on pipeline errors. Direct users of `vyd-core` can keep constructing a `PipelineContext` and calling its serial runner without a GStreamer dependency.

Pipelines that retain many CPU frames can set pool limits in the same TOML file:

```toml
[cpu_pool]
max_buffers = 96
max_bytes = "1.5 GiB"
```

Both fields are optional. `max_bytes` accepts an integer byte count or a quoted size such as `500 MB`, `1 GB`, or `1.5 GiB`. Decimal units (`MB`, `GB`) use powers of 1000; binary units (`MiB`, `GiB`) use powers of 1024. The defaults are 32 buffers and 256 MiB. These values cap pooled allocation; they do not preallocate buffers. Choose limits based on the frames retained by your stages and the expected frame size.

GStreamer sections can be given as strings through `PipelineBuilder::with_gstreamer_nodes(boundary, "identity")`, or configured in TOML:

```toml
[[gstreamer_nodes]]
boundary = 0
pipeline = "identity"
```

Boundary 0 follows the source; boundary 1 follows the first Rust processor. A section that strips frame context requires `bridge_capacity` and `correspondence = "one_to_one_ordered"`. A geometry change also requires `input_size`, `output_size`, and `annotation_policy = "drop_all"`; these declarations are checked against negotiated packed GRAY8 buffers. The bridge does not support dropping, duplicating, reordering, or aggregation. The old `[native_scale]` app-specific section is rejected with a migration error.

## Release automation

Nightly CI checks formatting, Rust and workflow lints, tests, and coverage. Pushes to `master` prepare a release-plz PR after the all-feature tests pass. Merging a `release-plz-` PR publishes the workspace crates and creates one `crate-name/version` tag and GitHub release per crate.

Configure the repository secrets `RELEASE_PLZ_TOKEN` (a GitHub token that can open release PRs and trigger their CI) and `CARGO_REGISTRY_TOKEN` (a crates.io token with permission to publish the crates) before enabling the release workflows. `CODECOV_TOKEN` is optional; coverage is always saved as a workflow artifact.
