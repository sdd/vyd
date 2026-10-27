#![recursion_limit = "256"]

use gst::prelude::*;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use vyd_core::caps::{
    FrameFormatRelation, FrameMemory, FramePortSpec, FramePortSpecs, ProcessorPortSpecs,
};
use vyd_core::frame::PixelFormat;
use vyd_core::pipeline::{Pipeline, PipelineContext, PipelineObserver, Stage};
use vyd_core::statistics::{PipelineStatistics, ProcessingType};
use vyd_core::{
    Frame, FrameContext, FrameProcessor, FrameProcessorResult, FrameSink, FrameSource, FrameTiming,
    VydError,
};
use vyd_gstreamer::runtime::{
    run_pipeline, run_pipeline_with_native_nodes, run_pipeline_with_native_nodes_and_options,
    run_pipeline_with_options, FrameCorrespondence, NativeGeometry, NativeNode, NativeProperty,
    NativePropertyValue, RuntimeOptions,
};

type Events = Arc<Mutex<Vec<String>>>;
type Frames = Arc<Mutex<Vec<(usize, Vec<u8>, String)>>>;
struct TypedTag(usize);
fn record(events: &Events, event: &str) {
    events.lock().unwrap().push(event.into());
}
fn failure(message: &str) -> VydError {
    VydError::GeneralPluginError(message.into())
}
fn nv12() -> FramePortSpec {
    FramePortSpec::new(PixelFormat::Nv12, FrameMemory::CpuPacked)
}
fn luma8() -> FramePortSpec {
    FramePortSpec::luma8_cpu_packed()
}
fn strided_luma8() -> FramePortSpec {
    FramePortSpec::new(PixelFormat::Luma8, FrameMemory::CpuRowStrided)
}
fn gpu_luma8() -> FramePortSpec {
    FramePortSpec::new(PixelFormat::Luma8, FrameMemory::Wgpu)
}

struct Source {
    events: Events,
    remaining: usize,
    index: usize,
    resize: bool,
    packed: bool,
    fail: bool,
    output_port: Option<FramePortSpecs>,
}
impl FrameSource for Source {
    fn output_port(&self) -> FramePortSpecs {
        self.output_port.clone().unwrap_or_default()
    }
    fn next_frame(&mut self, _: &mut PipelineContext) -> Option<FrameContext> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let width = if (self.resize && self.index > 0) || self.packed {
            4
        } else {
            3
        };
        let mut frame = FrameContext::new(
            Frame::from_raw(width, 2, vec![self.index as u8; width as usize * 2]).unwrap(),
            self.index,
        );
        frame.put("tag", format!("frame-{}", self.index));
        frame.put_typed(TypedTag(self.index));
        frame.timing = FrameTiming {
            presentation_timestamp: Some(Duration::from_millis(self.index as u64 * 40)),
            decode_timestamp: None,
            duration: Some(Duration::from_millis(40)),
        };
        self.index += 1;
        Some(frame)
    }
    fn try_next_frame(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Result<Option<FrameContext>, VydError> {
        if self.fail {
            Err(failure("source error"))
        } else {
            Ok(self.next_frame(ctx))
        }
    }
    fn pipeline_ctx_init(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "source.init");
        Ok(())
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "source.finish");
        Ok(())
    }
    fn name(&self) -> &str {
        "source"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
struct Processor {
    events: Events,
    fail_init: bool,
    fail: bool,
    packed: bool,
    scaled: bool,
    scale_actual: bool,
    ports: Option<ProcessorPortSpecs>,
}
impl FrameProcessor for Processor {
    fn ports(&self) -> ProcessorPortSpecs {
        self.ports.clone().unwrap_or_default()
    }
    fn process(
        &mut self,
        frame: &mut FrameContext,
        _: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if frame.frame_index == 1 {
            return FrameProcessorResult::Skip;
        }
        // Replace pixels and metadata: downstream must receive actual output.
        let Frame::Cpu(cpu) = &frame.frame else {
            panic!("expected CPU frame")
        };
        let (width, height) = cpu.dimensions();
        if self.packed {
            assert_eq!(&cpu.img.as_raw()[..], &[9; 8]);
        }
        if self.scaled {
            assert_eq!((width, height), (8, 2));
            assert_eq!(&cpu.img.as_raw()[..], &[9; 16]);
        }
        if self.scale_actual {
            assert_eq!((width, height), (8, 2));
            assert_eq!(&cpu.img.as_raw()[..], &[frame.frame_index as u8; 16]);
        }
        frame.frame =
            Frame::from_raw(width, height, vec![42; width as usize * height as usize]).unwrap();
        frame.put("tag", format!("processed-{}", frame.frame_index));
        frame.put_typed(TypedTag(frame.frame_index + 100));
        FrameProcessorResult::Continue
    }
    fn try_process(
        &mut self,
        frame: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, VydError> {
        if self.fail {
            Err(failure("processor error"))
        } else if self.ports.as_ref().is_some_and(|ports| {
            ports.relations().len() == 1 && ports.relations()[0].input.memory() == FrameMemory::Wgpu
        }) {
            // Deliberately violate a GPU-only output declaration so the
            // adapter's stage contract error can be checked below.
            let (width, height) = frame.frame.dimensions();
            frame.frame = Frame::from_raw(width, height, vec![7; (width * height) as usize])?;
            Ok(FrameProcessorResult::Continue)
        } else {
            Ok(self.process(frame, ctx))
        }
    }
    fn pipeline_ctx_init(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "processor.init");
        if self.fail_init {
            Err(failure("init error"))
        } else {
            Ok(())
        }
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "processor.finish");
        Ok(())
    }
    fn name(&self) -> &str {
        "processor"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
struct Sink {
    events: Events,
    frames: Frames,
    fail: bool,
    input_port: Option<FramePortSpecs>,
}
impl FrameSink for Sink {
    fn input_port(&self) -> FramePortSpecs {
        self.input_port.clone().unwrap_or_default()
    }
    fn consume(&mut self, frame: &mut FrameContext, _: &mut PipelineContext) {
        assert_eq!(
            frame.get_typed::<TypedTag>().unwrap().0,
            frame.frame_index + 100
        );
        assert_eq!(
            frame.timing.presentation_timestamp,
            Some(Duration::from_millis(frame.frame_index as u64 * 40))
        );
        assert_eq!(frame.timing.duration, Some(Duration::from_millis(40)));
        let Frame::Cpu(cpu) = &frame.frame else {
            panic!("expected CPU frame")
        };
        self.frames.lock().unwrap().push((
            frame.frame_index,
            cpu.img.as_raw().to_vec(),
            frame.get_as::<String>("tag").clone(),
        ));
    }
    fn try_consume(
        &mut self,
        frame: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<(), VydError> {
        if self.fail {
            return Err(failure("sink error"));
        }
        self.consume(frame, ctx);
        Ok(())
    }
    fn pipeline_ctx_init(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "sink.init");
        Ok(())
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "sink.finish");
        Ok(())
    }
    fn name(&self) -> &str {
        "sink"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
struct Observer(Events);
impl PipelineObserver for Observer {
    fn pipeline_started(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.0, "observer.start");
        Ok(())
    }
    fn frame_processed(
        &mut self,
        frame: &FrameContext,
        _: &mut PipelineContext,
    ) -> Result<(), VydError> {
        record(&self.0, &format!("observer.frame-{}", frame.frame_index));
        Ok(())
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.0, "observer.finish");
        Ok(())
    }
}
fn fixture(events: &Events, frames: &Frames, mode: &str) -> (PipelineContext, Pipeline) {
    let mut context = PipelineContext::new();
    context.add_observer(Observer(events.clone()));
    let pipeline = Pipeline {
        source: Stage::new(
            Box::new(Source {
                events: events.clone(),
                remaining: if mode == "empty" { 0 } else { 3 },
                index: 0,
                resize: mode == "resize",
                packed: mode == "packed" || mode == "scaled" || mode == "scale_actual",
                fail: mode == "source",
                output_port: match mode {
                    "caps_alternatives" => Some(
                        FramePortSpecs::one(gpu_luma8())
                            .with_alternative(nv12())
                            .with_alternative(luma8()),
                    ),
                    "source_nv12" => Some(FramePortSpecs::one(nv12())),
                    "source_strided" => Some(FramePortSpecs::one(strided_luma8())),
                    _ => None,
                },
            }),
            "source",
        ),
        stages: vec![Stage::new(
            Box::new(Processor {
                events: events.clone(),
                fail_init: mode == "init",
                fail: mode == "processor",
                packed: mode == "packed",
                scaled: mode == "scaled",
                scale_actual: mode == "scale_actual",
                ports: match mode {
                    "caps_alternatives" => Some(ProcessorPortSpecs::passthrough(
                        FramePortSpecs::one(gpu_luma8())
                            .with_alternative(nv12())
                            .with_alternative(luma8()),
                    )),
                    "format_change" => Some(
                        ProcessorPortSpecs::transforming(vec![FrameFormatRelation {
                            input: luma8(),
                            output: nv12(),
                        }])
                        .unwrap(),
                    ),
                    "processor_strided" => Some(ProcessorPortSpecs::passthrough(
                        FramePortSpecs::one(strided_luma8()),
                    )),
                    "processor_wgpu" => Some(ProcessorPortSpecs::passthrough(FramePortSpecs::one(
                        gpu_luma8(),
                    ))),
                    _ => None,
                },
            }),
            "processor",
        )],
        sink: Stage::new(
            Box::new(Sink {
                events: events.clone(),
                frames: frames.clone(),
                fail: mode == "sink",
                input_port: match mode {
                    "caps_alternatives" => Some(
                        FramePortSpecs::one(gpu_luma8())
                            .with_alternative(nv12())
                            .with_alternative(luma8()),
                    ),
                    "sink_nv12" => Some(FramePortSpecs::one(nv12())),
                    "sink_wgpu" => Some(FramePortSpecs::one(gpu_luma8())),
                    _ => None,
                },
            }),
            "sink",
        ),
    };
    (context, pipeline)
}

#[test]
fn gst_matches_serial_pixels_metadata_skips_eos_and_counts_real_copies() {
    let expected_events = Events::default();
    let expected_frames = Frames::default();
    let (ctx, pipeline) = fixture(&expected_events, &expected_frames, "normal");
    vyd_core::pipeline::run_pipeline(ctx, pipeline, None).unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let stats = Arc::new(PipelineStatistics::new());
    run_pipeline(ctx, pipeline, Some(stats.clone())).unwrap();
    assert_eq!(*frames.lock().unwrap(), *expected_frames.lock().unwrap());
    assert_eq!(*events.lock().unwrap(), *expected_events.lock().unwrap());
    assert_eq!(frames.lock().unwrap().len(), 2);
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.frame_count, 3);
    // Three source buffers and two processor outputs, all 3x2 pixels. GST
    // padding makes each allocation 8 bytes; copies move 6 visible bytes.
    assert_eq!(
        snapshot.memory_operations["gst-adapter/pack_cpu_copy_bytes"],
        30
    );
    assert_eq!(
        snapshot.memory_operations["gst-adapter/pack_cpu_alloc_bytes"],
        40
    );
    assert_eq!(
        snapshot.memory_operations["gst-adapter/unpack_cpu_copy_bytes"],
        30
    );
}

#[test]
fn caps_alternatives_are_not_reduced_to_the_first_entry() {
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "caps_alternatives");
    run_pipeline(ctx, pipeline, None).unwrap();
    assert_eq!(frames.lock().unwrap().len(), 2);
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn incompatible_caps_and_implicit_conversion_are_rejected() {
    for mode in ["source_nv12", "sink_nv12", "sink_wgpu", "format_change"] {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, mode);
        let error = run_pipeline(ctx, pipeline, None).unwrap_err();
        assert!(frames.lock().unwrap().is_empty(), "mode {mode}");
        assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
        assert!(
            error.to_string().contains(match mode {
                "format_change" => "explicit conversion",
                "sink_wgpu" => "GPU adapter",
                _ => "packed CPU Luma8",
            }),
            "{mode}: {error}"
        );
    }
}

#[test]
fn gpu_only_processor_requires_a_configured_pipeline_context() {
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "processor_wgpu");
    let error = run_pipeline(ctx, pipeline, None).unwrap_err();
    assert!(error.to_string().contains("no GPU context"), "{error}");
    assert!(frames.lock().unwrap().is_empty());
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn selected_gpu_segment_rejects_cpu_output_from_a_gpu_only_stage() {
    use vyd_core::gpu::{WgpuContext, WgpuContextOptions};

    let events = Events::default();
    let frames = Frames::default();
    let (mut ctx, pipeline) = fixture(&events, &frames, "processor_wgpu");
    ctx.set_wgpu_context(Arc::new(
        WgpuContext::request_blocking(WgpuContextOptions::default()).unwrap(),
    ))
    .unwrap();
    let error = run_pipeline(ctx, pipeline, None).unwrap_err();
    assert!(
        matches!(error, VydError::GpuStageOutput { stage_index: 0, .. }),
        "{error}"
    );
    assert!(frames.lock().unwrap().is_empty());
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn strided_ports_fail_during_graph_construction_until_explicit_adapter_exists() {
    for mode in ["source_strided", "processor_strided"] {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, mode);
        let error = run_pipeline(ctx, pipeline, None).unwrap_err();
        assert!(matches!(
            error,
            VydError::NotYetImplemented {
                feature: "packed/padded CPU layout negotiation",
                ..
            }
        ));
        assert!(frames.lock().unwrap().is_empty());
        assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
    }
}

#[test]
fn native_factory_element_and_description_share_the_graph_and_preserve_context() {
    gst::init().unwrap();
    let expected_events = Events::default();
    let expected_frames = Frames::default();
    let (ctx, pipeline) = fixture(&expected_events, &expected_frames, "normal");
    run_pipeline(ctx, pipeline, None).unwrap();

    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let existing = gst::ElementFactory::make("identity").build().unwrap();
    let nodes = vec![
        NativeNode::factory(
            0,
            "identity",
            vec![NativeProperty::new(
                "silent",
                NativePropertyValue::Bool(true),
            )],
        ),
        NativeNode::description(0, "identity ! identity"),
        NativeNode::element(1, existing),
    ];
    run_pipeline_with_native_nodes(ctx, pipeline, None, nodes).unwrap();
    assert_eq!(*frames.lock().unwrap(), *expected_frames.lock().unwrap());
    assert_eq!(*events.lock().unwrap(), *expected_events.lock().unwrap());
}

fn replacing_native_node(boundary: usize, value: u8) -> NativeNode {
    let element = gst::ElementFactory::make("identity").build().unwrap();
    element
        .static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, move |_, info| {
            let buffer = info.buffer_mut().unwrap().make_mut();
            buffer.replace_all_memory(gst::Memory::from_mut_slice(vec![value; 8]));
            gst::PadProbeReturn::Ok
        })
        .unwrap();
    NativeNode::element(boundary, element)
}

#[test]
fn packed_native_outputs_are_read_from_replaced_gst_memory() {
    gst::init().unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "packed");
    let stats = Arc::new(PipelineStatistics::new());
    run_pipeline_with_native_nodes(
        ctx,
        pipeline,
        Some(stats.clone()),
        vec![replacing_native_node(0, 9), replacing_native_node(1, 7)],
    )
    .unwrap();
    let got = frames.lock().unwrap();
    assert_eq!(got.len(), 2);
    assert!(got.iter().all(|(_, pixels, _)| pixels == &[7; 8]));
    let memory = stats.snapshot().memory_operations;
    assert_eq!(memory["gst-adapter/pack_zero_copy"], 5);
    assert_eq!(memory["gst-adapter/unpack_zero_copy"], 5);
    assert!(!memory.contains_key("gst-adapter/pack_cpu_copy_bytes"));
    assert!(!memory.contains_key("gst-adapter/unpack_cpu_copy_bytes"));
}

fn scaling_node(boundary: usize, bridge: bool, replace: Option<u8>) -> NativeNode {
    let description = if bridge {
        "identity name=stripper ! videoscale method=0 ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2"
    } else {
        "videoscale method=0 ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2"
    };
    let bin = gst::parse_bin_from_description(description, true).unwrap();
    if bridge {
        bin.by_name("stripper")
            .unwrap()
            .static_pad("src")
            .unwrap()
            .add_probe(gst::PadProbeType::BUFFER, |_, info| {
                if let Some(buffer) = info.buffer_mut() {
                    if let Ok(meta) = gst::meta::CustomMeta::from_mut_buffer(
                        buffer.make_mut(),
                        "VydSerialFrameContextMeta",
                    ) {
                        meta.remove().unwrap();
                    }
                }
                gst::PadProbeReturn::Ok
            })
            .unwrap();
    }
    if let Some(value) = replace {
        bin.static_pad("src")
            .unwrap()
            .add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                info.buffer_mut()
                    .unwrap()
                    .make_mut()
                    .replace_all_memory(gst::Memory::from_mut_slice(vec![value; 16]));
                gst::PadProbeReturn::Ok
            })
            .unwrap();
    }
    let node = NativeNode::element(boundary, bin.upcast())
        .with_packed_gray8_geometry(NativeGeometry::new((4, 2), (8, 2)).unwrap());
    if bridge {
        node.with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
            .unwrap()
    } else {
        node
    }
}

#[test]
fn declared_scale_rebuilds_actual_output_before_processor_and_sink() {
    gst::init().unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "scale_actual");
    run_pipeline_with_native_nodes(ctx, pipeline, None, vec![scaling_node(0, true, None)]).unwrap();
    assert!(frames
        .lock()
        .unwrap()
        .iter()
        .all(|(_, pixels, _)| pixels == &[42; 16]));

    for bridge in [false, true] {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, "scaled");
        let stats = Arc::new(PipelineStatistics::new());
        run_pipeline_with_native_nodes(
            ctx,
            pipeline,
            Some(stats.clone()),
            vec![scaling_node(0, bridge, Some(9))],
        )
        .unwrap();
        let got = frames.lock().unwrap();
        assert_eq!(got.len(), 2);
        assert!(got
            .iter()
            .all(|(_, pixels, tag)| pixels == &[42; 16] && tag.starts_with("processed-")));
        assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["gst-adapter/pack_zero_copy"], 5);
        assert_eq!(memory["gst-adapter/unpack_zero_copy"], 5);
        assert!(!memory.contains_key("gst-adapter/pack_cpu_copy_bytes"));
        assert!(!memory.contains_key("gst-adapter/unpack_cpu_copy_bytes"));
    }

    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "packed");
    run_pipeline_with_native_nodes(
        ctx,
        pipeline,
        None,
        vec![replacing_native_node(0, 9), scaling_node(1, true, Some(7))],
    )
    .unwrap();
    let got = frames.lock().unwrap();
    assert_eq!(got.len(), 2);
    assert!(got
        .iter()
        .all(|(_, pixels, tag)| pixels == &[7; 16] && tag.starts_with("processed-")));
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn declared_scale_rejects_wrong_output_size_and_dimensions_before_bridge_pairing() {
    gst::init().unwrap();
    for defect in ["size", "dimensions"] {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, "scaled");
        let node = if defect == "size" {
            let bin = gst::parse_bin_from_description(
                "videoscale ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2",
                true,
            )
            .unwrap();
            bin.static_pad("src")
                .unwrap()
                .add_probe(gst::PadProbeType::BUFFER, |_, info| {
                    info.buffer_mut()
                        .unwrap()
                        .make_mut()
                        .replace_all_memory(gst::Memory::from_mut_slice(vec![9; 15]));
                    gst::PadProbeReturn::Ok
                })
                .unwrap();
            NativeNode::element(0, bin.upcast())
                .with_packed_gray8_geometry(NativeGeometry::new((4, 2), (8, 2)).unwrap())
                .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
                .unwrap()
        } else {
            scaling_node(0, true, None)
                .with_packed_gray8_geometry(NativeGeometry::new((4, 2), (7, 2)).unwrap())
        };
        let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
        assert!(
            error.to_string().contains(if defect == "size" {
                "size"
            } else {
                "expected 7x2"
            }),
            "{defect}: {error}"
        );
        assert!(frames.lock().unwrap().is_empty());
        assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
    }
}

#[test]
fn native_geometry_changes_require_a_matching_declaration() {
    gst::init().unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "scale_actual");
    let node = NativeNode::description(
        0,
        "videoscale ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2",
    );
    let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
    assert!(error.to_string().contains("undeclared"), "{error}");
    assert!(frames.lock().unwrap().is_empty());
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");

    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "scaled");
    let second = NativeNode::factory(0, "identity", vec![])
        .with_packed_gray8_geometry(NativeGeometry::new((4, 2), (8, 2)).unwrap());
    let error = run_pipeline_with_native_nodes(
        ctx,
        pipeline,
        None,
        vec![scaling_node(0, false, None), second],
    )
    .unwrap_err();
    assert!(error.to_string().contains("contracts disagree"), "{error}");
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn declared_scale_rejects_midstream_caps_renegotiation() {
    gst::init().unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "scale_actual");
    let bin = gst::parse_bin_from_description(
        "videoscale ! capsfilter name=scaled caps=video/x-raw,format=GRAY8,width=8,height=2",
        true,
    )
    .unwrap();
    let filter = bin.by_name("scaled").unwrap();
    let observed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let on_output = observed.clone();
    bin.static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, move |_, _| {
            if on_output.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                let caps = gst::Caps::builder("video/x-raw")
                    .field("format", "GRAY8")
                    .field("width", 4i32)
                    .field("height", 2i32)
                    .build();
                filter.set_property("caps", caps);
            }
            gst::PadProbeReturn::Ok
        })
        .unwrap();
    let node = NativeNode::element(0, bin.upcast())
        .with_packed_gray8_geometry(NativeGeometry::new((4, 2), (8, 2)).unwrap())
        .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
        .unwrap();
    let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
    assert!(error.to_string().contains("expected 8x2"), "{error}");
    assert!(observed.load(std::sync::atomic::Ordering::SeqCst) >= 2);
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn invalid_native_sections_fail_before_processing_and_finalize() {
    gst::init().unwrap();
    let invalid = vec![
        ("boundary", NativeNode::factory(2, "identity", vec![])),
        (
            "factory",
            NativeNode::factory(0, "vyd-no-such-factory", vec![]),
        ),
        (
            "property",
            NativeNode::factory(
                0,
                "identity",
                vec![NativeProperty::new(
                    "vyd-no-such-property",
                    NativePropertyValue::Bool(true),
                )],
            ),
        ),
        (
            "type",
            NativeNode::factory(
                0,
                "identity",
                vec![NativeProperty::new(
                    "silent",
                    NativePropertyValue::String("true".into()),
                )],
            ),
        ),
        (
            "pads",
            NativeNode::element(0, gst::ElementFactory::make("fakesink").build().unwrap()),
        ),
    ];
    for (label, node) in invalid {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, "normal");
        let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
        assert!(frames.lock().unwrap().is_empty(), "{label}");
        assert_eq!(
            events.lock().unwrap().last().unwrap(),
            "observer.finish",
            "{label}: {error}"
        );
    }
}

#[test]
fn native_bus_error_reaches_caller_and_finalizes() {
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let node = NativeNode::factory(
        0,
        "identity",
        vec![NativeProperty::new(
            "error-after",
            NativePropertyValue::I32(1),
        )],
    );
    let result = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]);
    assert!(result.is_err(), "native error became successful EOS");
    let events = events.lock().unwrap();
    assert!(events.iter().any(|event| event == "source.finish"));
    assert!(events.iter().any(|event| event == "processor.finish"));
    assert_eq!(events.last().unwrap(), "observer.finish");
}

#[test]
fn fifo_bridge_restores_context_after_native_meta_stripping() {
    gst::init().unwrap();
    let expected_events = Events::default();
    let expected_frames = Frames::default();
    let (ctx, pipeline) = fixture(&expected_events, &expected_frames, "normal");
    run_pipeline(ctx, pipeline, None).unwrap();

    let bin = gst::parse_bin_from_description("identity name=stripper ! identity", true).unwrap();
    let stripper = bin.by_name("stripper").unwrap();
    stripper
        .static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, |_, info| {
            if let Some(buffer) = info.buffer_mut() {
                if let Ok(meta) = gst::meta::CustomMeta::from_mut_buffer(
                    buffer.make_mut(),
                    "VydSerialFrameContextMeta",
                ) {
                    meta.remove().unwrap();
                }
            }
            gst::PadProbeReturn::Ok
        })
        .unwrap();

    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let node = NativeNode::element(0, bin.upcast())
        .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
        .unwrap();
    run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap();
    assert_eq!(*frames.lock().unwrap(), *expected_frames.lock().unwrap());
    assert_eq!(*events.lock().unwrap(), *expected_events.lock().unwrap());
}

#[test]
fn fifo_bridge_rejects_unsupported_correspondence_and_detects_drops() {
    assert!(NativeNode::factory(0, "identity", vec![])
        .with_fifo_bridge(0, FrameCorrespondence::OneToOneOrdered)
        .is_err());
    for contract in [
        FrameCorrespondence::MayDrop,
        FrameCorrespondence::MayDuplicate,
        FrameCorrespondence::MayReorder,
        FrameCorrespondence::MayAggregate,
    ] {
        assert!(NativeNode::factory(0, "identity", vec![])
            .with_fifo_bridge(8, contract)
            .is_err());
    }

    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let node = NativeNode::factory(
        0,
        "identity",
        vec![NativeProperty::new(
            "drop-probability",
            NativePropertyValue::F32(1.0),
        )],
    )
    .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
    .unwrap();
    let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
    assert!(
        error.to_string().contains("unmatched input contexts"),
        "{error}"
    );
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn fifo_bridge_rejects_leaky_queue_inside_section() {
    gst::init().unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let node = NativeNode::description(0, "queue leaky=downstream ! identity")
        .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
        .unwrap();
    let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
    assert!(error.to_string().contains("dropping queue"), "{error}");
    assert!(frames.lock().unwrap().is_empty());
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");

    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let nodes = vec![
        NativeNode::description(0, "queue leaky=downstream"),
        NativeNode::factory(0, "identity", vec![])
            .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
            .unwrap(),
    ];
    run_pipeline_with_native_nodes(ctx, pipeline, None, nodes).unwrap();
    assert_eq!(frames.lock().unwrap().len(), 2);
}

#[test]
fn fifo_bridge_releases_context_on_native_bus_error_and_downstream_failure() {
    for (mode, node) in [
        (
            "normal",
            NativeNode::factory(
                0,
                "identity",
                vec![NativeProperty::new(
                    "error-after",
                    NativePropertyValue::I32(1),
                )],
            ),
        ),
        ("sink", NativeNode::factory(1, "identity", vec![])),
    ] {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, mode);
        let node = node
            .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
            .unwrap();
        let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
        if mode == "sink" {
            assert!(
                matches!(error, VydError::StageFailed { stage: "sink", .. }),
                "{error}"
            );
        }
        assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
    }
}

#[test]
fn fifo_bridge_cancellation_releases_admitted_context_and_finalizes() {
    gst::init().unwrap();
    let element = gst::ElementFactory::make("identity").build().unwrap();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (entered_tx, entered_rx) = mpsc::channel();
    let callback_gate = gate.clone();
    element
        .static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, move |_, _| {
            entered_tx.send(()).unwrap();
            let (lock, changed) = &*callback_gate;
            let mut ready = lock.lock().unwrap();
            while !*ready {
                ready = changed.wait(ready).unwrap();
            }
            gst::PadProbeReturn::Ok
        })
        .unwrap();
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let node = NativeNode::element(0, element)
        .with_fifo_bridge(8, FrameCorrespondence::OneToOneOrdered)
        .unwrap();
    let stats = Arc::new(PipelineStatistics::new());
    let worker_stats = stats.clone();
    let (result_tx, result_rx) = mpsc::channel();
    let runner = std::thread::spawn(move || {
        result_tx
            .send(run_pipeline_with_native_nodes_and_options(
                ctx,
                pipeline,
                Some(worker_stats),
                vec![node],
                RuntimeOptions {
                    shutdown_timeout: Duration::from_secs(5),
                },
            ))
            .unwrap();
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    stats.cancellation_token().cancel();
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    let result = result_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    runner.join().unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn fifo_bridge_capacity_is_bounded_with_a_non_dropping_native_queue() {
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "normal");
    let node = NativeNode::description(0, "queue ! identity sleep-time=1000000")
        .with_fifo_bridge(1, FrameCorrespondence::OneToOneOrdered)
        .unwrap();
    let error = run_pipeline_with_native_nodes(ctx, pipeline, None, vec![node]).unwrap_err();
    assert!(
        error.to_string().contains("capacity 1 exhausted"),
        "{error}"
    );
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn empty_source_reaches_eos_and_finalizes() {
    let events = Events::default();
    let frames = Frames::default();
    let (ctx, pipeline) = fixture(&events, &frames, "empty");
    run_pipeline(ctx, pipeline, None).unwrap();
    assert!(frames.lock().unwrap().is_empty());
    assert_eq!(events.lock().unwrap().last().unwrap(), "observer.finish");
}

#[test]
fn errors_reach_caller_and_partial_initialization_is_unwound() {
    for mode in ["source", "processor", "sink", "init", "resize"] {
        let events = Events::default();
        let frames = Frames::default();
        let (ctx, pipeline) = fixture(&events, &frames, mode);
        let result = run_pipeline(ctx, pipeline, None).unwrap_err();
        if ["source", "processor", "sink"].contains(&mode) {
            assert!(matches!(result, VydError::StageFailed { stage, .. } if stage==mode));
        } else {
            assert!(result.to_string().contains(if mode == "init" {
                "init error"
            } else {
                "midstream"
            }));
        }
        let events = events.lock().unwrap();
        assert!(events.iter().any(|e| e == "source.finish"));
        assert!(events.iter().any(|e| e == "processor.finish"));
        assert_eq!(events.iter().any(|e| e == "sink.finish"), mode != "init");
        assert_eq!(events.last().unwrap(), "observer.finish");
    }
}

#[test]
fn shutdown_timeout_retains_a_blocked_stage_until_it_can_be_finalized() {
    struct Blocker {
        entered: mpsc::Sender<()>,
        gate: Arc<(Mutex<bool>, Condvar)>,
        finished: mpsc::Sender<()>,
    }
    impl FrameProcessor for Blocker {
        fn process(
            &mut self,
            _: &mut FrameContext,
            _: &mut PipelineContext,
        ) -> FrameProcessorResult {
            self.entered.send(()).unwrap();
            let (lock, changed) = &*self.gate;
            let mut ready = lock.lock().unwrap();
            while !*ready {
                ready = changed.wait(ready).unwrap();
            }
            FrameProcessorResult::Continue
        }
        fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
            self.finished.send(()).unwrap();
            Ok(())
        }
        fn name(&self) -> &str {
            "blocker"
        }
        fn processing_type(&self) -> ProcessingType {
            ProcessingType::Cpu
        }
    }
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (entered_tx, entered) = mpsc::channel();
    let (finished_tx, finished) = mpsc::channel();
    let (ctx, mut pipeline) = fixture(&Events::default(), &Frames::default(), "normal");
    pipeline.stages = vec![Stage::new(
        Box::new(Blocker {
            entered: entered_tx,
            gate: gate.clone(),
            finished: finished_tx,
        }),
        "blocker",
    )];
    let stats = Arc::new(PipelineStatistics::new());
    let running_stats = stats.clone();
    let (result_tx, result_rx) = mpsc::channel();
    let runner = std::thread::spawn(move || {
        let result = run_pipeline_with_options(
            ctx,
            pipeline,
            Some(running_stats),
            RuntimeOptions {
                shutdown_timeout: Duration::from_millis(100),
            },
        );
        result_tx.send(result).unwrap();
    });
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    let start = Instant::now();
    stats.cancellation_token().cancel();
    let result = result_rx.recv_timeout(Duration::from_secs(2));
    let not_finalized = finished.try_recv().is_err();
    // Always release the callback before asserting, even when the deadline test fails.
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    assert!(matches!(
        result.unwrap(),
        Err(VydError::ShutdownTimeout { .. })
    ));
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(not_finalized);
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    runner.join().unwrap();
}

#[test]
fn native_source_bus_error_is_not_reported_as_eos() {
    let config =
        toml::from_str::<toml::Value>("file_path = '/tmp/vyd-v2-intentionally-missing-input.mkv'")
            .unwrap();
    let source = vyd_gstreamer::GstSource::new(Some(&config)).unwrap();
    let events = Events::default();
    let pipeline = Pipeline {
        source: Stage::new(Box::new(source), "native-source"),
        stages: Vec::new(),
        sink: Stage::new(
            Box::new(Sink {
                events,
                frames: Frames::default(),
                fail: false,
                input_port: None,
            }),
            "sink",
        ),
    };
    let result = run_pipeline_with_options(
        PipelineContext::new(),
        pipeline,
        None,
        RuntimeOptions {
            shutdown_timeout: Duration::from_secs(5),
        },
    );
    assert!(
        matches!(
            result,
            Err(VydError::StageFailed {
                stage: "native-source",
                operation: "next_frame",
                ..
            })
        ),
        "got {result:?}"
    );
}

#[test]
fn gst_stage_failure_finalizes_telemetry() {
    let directory = std::env::temp_dir().join(format!(
        "vyd_gst_failed_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let events = Events::default();
    let frames = Frames::default();
    let (mut ctx, pipeline) = fixture(&events, &frames, "processor");
    let manager = vyd_telemetry::DumpManager::new(
        vyd_telemetry::DumpRunOptions {
            root: directory.clone(),
            run_name: Some("failure".into()),
            replace: false,
        },
        "gst error case".into(),
        "fixture".into(),
    )
    .unwrap();
    let staging = manager.staging_dir().unwrap();
    manager.attach(&mut ctx);
    let error = run_pipeline(ctx, pipeline, None).unwrap_err();
    assert!(matches!(
        error,
        VydError::StageFailed {
            stage: "processor",
            ..
        }
    ));
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(staging.join("manifest.json")).unwrap()).unwrap();
    let statistics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(staging.join("run_statistics.json")).unwrap())
            .unwrap();
    assert!(metadata.is_object());
    assert_eq!(statistics["frame_count"], 0);
    assert_eq!(statistics["stages"]["processor"]["call_count"], 1);
    std::fs::remove_dir_all(directory).unwrap();
}
