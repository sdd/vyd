# Vyd

Vyd provides a reusable Rust video processing pipeline. A `FrameSource` produces frames, `FrameProcessor` stages transform or skip them, and a `FrameSink` consumes them. Stages can be registered and assembled from TOML; the library also provides frame storage, timing statistics, and optional Vortex recordings.

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

- `vyd-core`: pipeline orchestration, lifecycle observers, stage interfaces, managed Luma8 and full-range NV12 CPU frames, optional wgpu frame access, and statistics. It has no GStreamer or Vortex dependency.
- `vyd`: application-facing builder and owned pipeline. It combines the core, GStreamer runtime, and optional telemetry setup.
- `vyd-telemetry`: run metadata, statistics export, and per-stage Vortex recording and replay.
- `vyd-gstreamer`: generic file and RTSP frame sources with pooled CPU buffers.
- `vyd-gstreamer-gpu`: experimental GPU buffer path. Its current source still assumes 1920×1080 Luma8 frames.
- `vyd-main`: example CLI.

The GStreamer crates require the matching system GStreamer libraries.

The `gpu-wgpu` feature enables managed GPU transfers and compute stages. Adjacent GPU stages can retain frames on the device; the current GStreamer source and sink boundaries use CPU GRAY8 buffers. NV12 conversion is defined for full-range grayscale Y data, with neutral chroma when converting from Luma8.

See [the design principles](https://github.com/sdd/vyd/blob/master/DESIGN_PRINCIPLES.md) for the API and runtime
guidelines behind Vyd.

## Run telemetry

Applications can construct a run with `vyd::PipelineBuilder::from_config_file(path)`, optionally override its recording options, then call `build()?`. The returned `vyd::Pipeline` exposes `context_mut()`, a cloneable `statistics()` handle, and `cancellation_token()` before `run()` consumes it. The builder configures a GPU when any configured processor has `gpu = true`; this temporary convention does not depend on processor names. A build without `gpu-wgpu` reports that feature as required.

```rust,no_run
let pipeline = vyd::PipelineBuilder::from_config_file("examples/basic.toml")?
    .build()?;
let stats = pipeline.statistics();
let cancellation = pipeline.cancellation_token();
// Call cancellation.cancel() from a signal handler to finalize a partial run.
pipeline.run()?;
stats.print_final_stats();
# Ok::<(), Box<dyn std::error::Error>>(())
```

To record a processor, add `dump = true` to its `[[stages]]` section. Recordings default to `./dumps` relative to the working directory, with a generated run name. An optional top-level section selects a stable name and directory:

```toml
[[stages]]
stage_type = "my_crate::FeatureProcessor"
dump = true

[recording]
dump_dir = "./dumps"
run_name = "feature-test"
```

Relative `recording.dump_dir` values resolve from the working directory. `--dump-dir` and `--run-name` override their corresponding TOML values independently. `--replace-run` replaces the named run only after the new run completes or is cancelled cleanly; it accepts a run name from TOML or the CLI. Each recorded stage writes its own `.vortex` file under `dump-dir/NAME`; the same directory contains a versioned `manifest.json` and `run_statistics.json`. A failed or unfinished run remains in `.staging` and cannot be replayed. A clean cancellation publishes a finalized prefix.

After the pipeline statistics, the example CLI prints the published directory and each stage file's frame count and size. Library callers can inspect the same data in `Pipeline::run_with_outcome()?.recording`.

The processor crate registers a `TypedStageCodec` for its stage type. Its row struct becomes separate Vortex columns for each named output. Every row also carries frame index, timestamps, dimensions, and continue/skip disposition. To replay, replace `dump = true` with `replay = "dumps/NAME/00-FeatureProcessor.vortex"` (relative paths resolve from the config file). Replay substitutes the processor, validates the finalized manifest and source file metadata, and checks each incoming frame's identity and geometry. Recording and replay currently support processor metadata only. Direct users of `vyd-core` can keep constructing a `PipelineContext` and calling its serial runner without a GStreamer dependency.

The writer has a one-chunk queue and a 64 MiB in-flight byte budget. One frame larger than the budget is allowed through alone.

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

## Frame capture timing

`FrameContext::frame_index` is a `u64` assigned at ingestion, before Vyd's live
queue drops frames. Consumers see gaps rather than renumbered frames. Processors
should preserve the context when replacing pixels. Custom sources can use
`vyd_core::capture_timing::CaptureClock` to assign indices and timing together.
`frame_capture_ts: Option<SystemTime>` carries capture UTC; `us_since(&previous)`
returns relative capture microseconds using wrapping subtraction. The private
32-bit counter wraps every 71.6 minutes, so differences must span less than one
wrap.

Relative timing uses consecutive genuine capture UTC deltas, then media PTS,
then monotonic ingestion time. Backwards PTS steps up to two frame durations (100 ms when duration is unknown)
are clamped without double-counting the recovery. Larger steps clear the UTC
estimation window and use monotonic ingestion time for the transition. Capture
UTC deltas that disagree with PTS by more than that same tolerance fall back to
PTS, in either direction. Offline decoding therefore preserves media time
regardless of processing speed. Missing source timestamps require the ingestion
fallback; original timing cannot be recovered from an untimed file. The GStreamer sinks rebase clocks on FlushStop, StreamStart, and Segment events;
indices and the relative timeline remain continuous. Custom sources can call
`CaptureClock::reset()` at equivalent boundaries. Offline UTC anchors follow
media PTS across seeks; the relative transition interval uses ingestion time.

Both GStreamer source adapters accept `mean_frame_latency_ms` (unsigned integer,
default `0`). For live streams, UTC is estimated from the mean of the latest 32
ingestion UTC-minus-PTS samples, then this latency is subtracted. Until the window
fills, the available samples are used. Without PTS, ingestion UTC minus latency is
used. Latency never changes relative timing or genuine source capture UTC.
Network jitter, sender clock errors, receiver UTC adjustments, and an inaccurate
latency setting limit the accuracy of these estimates. RTSP reference timestamp
metadata is requested on GStreamer versions that support it; Unix and NTP
reference domains are accepted, but the sender must actually provide a valid
capture clock mapping. Upstream camera/decoder losses cannot be counted by Vyd.

Offline files have no inferred UTC unless genuine capture UTC or an explicit
frame-zero capture timestamp is available. The supported container convention is
an `ExtendedComment` tag `frame-zero-capture-unix-us=<Unix microseconds>`. This
works only when the demuxer forwards that tag to the video sink before frames;
it is not general MP4/MKV metadata discovery. Alternatively, supply
`frame_zero_capture_unix_us` in the source configuration after extracting a known
capture timestamp externally. The configured value takes precedence over tags.
Generic container creation dates are deliberately ignored. UTC is frame-zero UTC
plus relative media time; latency does not adjust this offline anchor. These
fields survive Rust stages and supported GStreamer context handoffs, including
ordered correspondence bridges. They are not added to the Vortex recording
schema by this change.

API migration: this change is breaking and must be released as such. `frame_index`
and its constructor argument are now `u64`. Use `FrameContext::new` rather than
struct literals, and `CaptureClock` when assigning source timing. Queued frames
now carry a context and no longer implement `Debug`/`Clone`; contexts contain
non-cloneable application metadata. The `u32` wrapping interval API is intentional:
callers needing longer intervals must accumulate successive `us_since` values
into a wider counter, sampling at least once per wrap.
