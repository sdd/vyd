# Vyd design principles

These principles guide Vyd's APIs and runtime. Some describe future extensions;
the README and crate APIs describe what is currently implemented.

The settled direction is one GStreamer pipeline runtime, GStreamer negotiation
of declared format alternatives, and wgpu for application-defined GPU processing.
Keep the older runner only for migration comparison and retire it after paired
validation. A serial debugging configuration uses the same runtime.

## Keep infrastructure application independent

Vyd owns stage interfaces, frames, contexts, adapters, graph construction and
generic telemetry. Astronomy models, annotations and algorithms stay in Astrocap.
An extension should express a reusable capability without depending on one
application's data model. Typed context metadata lets applications carry domain
values without adding them to Vyd's core.

## Make Rust processors straightforward to write

Processor authors should need little or no GStreamer knowledge. Use ordinary
Rust methods and results for frame processing, skipping and errors. Put pads,
events, caps plumbing, metadata registration and state transitions in adapters.
Keep a convenient linear builder that can later extend to branches in the same
graph model. Branching is deferred from the initial release.
Native elements enter through one generic integration; specialised builders
should simplify that integration rather than create another runtime.

## Reuse mature media infrastructure and compatible Rust types

Use GStreamer for graph execution, negotiation, queues, lifecycle and native
allocation/pool negotiation. Express port relationships faithfully and validate
layouts where caps are insufficient. Conversion and memory transfer need
explicit operations; negotiation chooses a supported format. The primary
[negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html)
and [allocation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html)
guides describe the mechanisms to build upon.

Reuse `image` pixel descriptions and borrowed views where they fit. Do not force
padded or multiplane images into `ImageBuffer`, or build a parallel image library
to avoid a small validated adapter. Dependency upgrades and their consumer
impact remain explicit implementation decisions.

## Make sharing, pooling and COW the default

Frame clones share contents. A write detaches when readers, clones or native
users still need the old version; unique safe storage can be reused in place.
Retain coherent CPU/GPU replicas until mutation invalidates them. Create ordinary
frames through the pipeline's `FrameFactory`, backed by bounded pools and native
leases. Metadata values share immutably while branch-local map updates use COW.
Cheap sharing must preserve version and branch isolation.

## Measure the costs that affect performance

Measure copies, CPU/GPU transfers, allocations, pool reuse/growth, queue stalls
and access waits with run/stage attribution. Distinguish attempts and failures
from completed operations and count actual bytes. Call a path zero-copy only
when it demonstrably shares storage; a GPU copy is still a copy. A benchmark
needs semantic and lifetime checks as well as throughput/latency measurements.
Compare representative consumer workloads before expanding performance claims.

## Use types and checked construction to prevent invalid states

Separate logical format, physical layout, access permissions and storage device.
Validate dimensions, arithmetic, strides, bounds and plane non-aliasing before
exposing views. Keep validated fields private, represent absent frames outside
`Frame`, and reject unsupported format/layout/device combinations clearly.
Never expose uninitialized memory, including padding in safe slices. Types
should encode proven invariants; caller declarations about opaque native frame
correspondence must remain explicit assumptions, not be presented as proofs.

## Keep lifetime management below processor APIs

Guards retain mappings and backing leases; issuers reclaim storage only after
the final relevant reference and GPU use end. Processors should not manually
return buffers or manage native fences. Managed GPU recordings register access
and retirement, without safe raw-handle escapes for pooled frame resources.
A submission receipt is not completion, and dropping it does not stop accepted
work; see [wgpu queue submission and completion](https://docs.rs/wgpu/latest/wgpu/struct.Queue.html).
Diagnostic records must not prolong resource lifetimes or create ownership cycles.

## Prefer explicit bounds and fallible operations

Bound queues, sidecars, in-flight work and pool count/bytes. Preallocate where
practical and make growth observable. Use `WaitPolicy::Immediate | Timeout`
consistently, with one operation deadline across nested steps. Propagate capacity,
layout, device and shutdown failures instead of silently copying or waiting
forever. Finalize started stages and telemetry on managed termination paths.
Report shutdown timeouts honestly: bounded waiting cannot safely preempt arbitrary
CPU code, nor permit early reclamation of resources still in use.

## Preserve frame correspondence and future branch isolation

Transport frame identity/timing and metadata independently of pixels. Rebuild
frames from actual native output buffers. Use FIFO context bridges only across
declared count-and-order-preserving sections, and keep leaky queues outside
them. A queue fullness signal cannot identify which buffer was dropped; the
[GST queue reference](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html)
defines that signal. Reject declared incompatible correspondence; require
specialised integration for drops, duplication, reordering or aggregation.
Runtime checks cannot detect every false declaration.

Update or discard annotations invalidated by transforms such as scaling/cropping.
Keep branch metadata updates isolated. Put explicitly shared mutable services
and latest-value feedback in `PipelineContext`; separate reads do not constitute
an atomic pipeline snapshot. Defer joins until matching and merge policies have
a concrete application and tests.

## Keep portable GPU algorithms behind wgpu

Start with one shared wgpu device/queue per pipeline and managed compute access.
Keep native backend imports/exports in interoperability adapters, with verified
ownership, synchronization, format and device compatibility. Report whether each
path shares storage, copies on the GPU or transfers through the CPU. Do not
promise universal native zero-copy interoperability. CubeCL requires later
resource/submission integration and workload evidence.

## Extend for demonstrated needs

Provide narrow pool-provider and native-node extension points around concrete
implementations. Prefer synchronous processors on suitable workers; add async
integration where needed. Scope managed child work to an invocation by default;
persistent tasks need pipeline ownership, cancellation and joining. Express
skip-while-busy/latest-result behaviour within this lifecycle, without a second
background-pipeline framework.

Defer general joins, global resource scheduling and speculative backend systems.
GST threads and third-party pools are not automatically governed by a Vyd worker
budget. Add controls only where enforceable and measured. A proposed extension
should identify its consumer, contract, cost and acceptance test.

## Verify consumers at every public API checkpoint

Vyd and Astrocap evolve together. Build and test both, then exercise a
representative application run with telemetry. Preserve frame semantics and
inspect real artifacts, rather than treating compilation as end-to-end evidence.
Separate implemented functionality, agreed design, experimental paths and
deferred work in documentation. Record paired revisions and validation results
before retiring migration compatibility or claiming a milestone complete.
