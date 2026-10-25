#![cfg(feature = "gpu-wgpu")]

use std::sync::{Arc, Mutex};
use std::time::Duration;
use vyd_core::caps::{FrameFormatRelation, FrameMemory, FramePortSpec, ProcessorPortSpecs};
use vyd_core::frame::pool::WaitPolicy;
use vyd_core::frame::{Frame, PixelFormat};
use vyd_core::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};
use vyd_core::pipeline::{Pipeline, PipelineContext, PipelineObserver, Stage};
use vyd_core::statistics::{PipelineStatistics, ProcessingType};
use vyd_core::{
    FrameContext, FrameProcessor, FrameProcessorResult, FrameSink, FrameSource, VydError,
};
use vyd_gstreamer::runtime::{run_pipeline, run_pipeline_with_native_nodes, NativeNode};

type Output = Arc<Mutex<Vec<(usize, Vec<u8>, String)>>>;
type Events = Arc<Mutex<Vec<String>>>;

struct Source {
    next: usize,
    count: usize,
}

impl FrameSource for Source {
    fn next_frame(&mut self, _: &mut PipelineContext) -> Option<FrameContext> {
        if self.next == self.count {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let mut frame = FrameContext::new(
            Frame::from_raw(3, 3, (1..=9).map(|value| value + index as u8).collect()).unwrap(),
            index,
        );
        frame.put("route", String::new());
        Some(frame)
    }
    fn name(&self) -> &str {
        "source"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

enum Kind {
    Gpu {
        input: PixelFormat,
        output: PixelFormat,
        skip_index: Option<usize>,
    },
    Cpu,
}

struct Processor {
    name: &'static str,
    kind: Kind,
    events: Events,
    service_identity: Arc<Mutex<Option<usize>>>,
}

impl FrameProcessor for Processor {
    fn ports(&self) -> ProcessorPortSpecs {
        match self.kind {
            Kind::Cpu => ProcessorPortSpecs::default(),
            Kind::Gpu { input, output, .. } => {
                ProcessorPortSpecs::transforming(vec![FrameFormatRelation {
                    input: FramePortSpec::new(input, FrameMemory::Wgpu),
                    output: FramePortSpec::new(output, FrameMemory::Wgpu),
                }])
                .unwrap()
            }
        }
    }
    fn process(&mut self, _: &mut FrameContext, _: &mut PipelineContext) -> FrameProcessorResult {
        unreachable!("the fallible GPU test entry point is used")
    }
    fn try_process(
        &mut self,
        frame: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, VydError> {
        self.events.lock().unwrap().push(self.name.into());
        let mut route = frame.get_as::<String>("route").clone();
        route.push_str(self.name);
        frame.put("route", route);
        match self.kind {
            Kind::Gpu {
                input,
                output,
                skip_index,
            } => {
                assert!(matches!(frame.frame, Frame::ManagedGpu(_)));
                assert_eq!(frame.frame.pixel_format(), input);
                let factory = ctx.frame_factory().clone();
                let transfer = ctx
                    .gpu_transfer()
                    .map_err(|error| VydError::GeneralPluginError(error.to_string()))?
                    .clone();
                let pointer = Arc::as_ptr(&transfer) as usize;
                let mut identity = self.service_identity.lock().unwrap();
                if let Some(previous) = *identity {
                    assert_eq!(previous, pointer, "GPU stages received different services");
                } else {
                    *identity = Some(pointer);
                }
                drop(identity);
                if skip_index == Some(frame.frame_index) {
                    return Ok(FrameProcessorResult::Skip);
                }
                let read = frame.frame.try_read_gpu_as(
                    &factory,
                    &transfer,
                    output,
                    WaitPolicy::Timeout(Duration::from_secs(3)),
                )?;
                frame.frame = read.into_frame();
            }
            Kind::Cpu => {
                let Frame::Cpu(cpu) = &frame.frame else {
                    panic!("CPU neighbor did not receive packed Luma8")
                };
                let (width, height) = cpu.dimensions();
                frame.frame = Frame::from_raw(
                    width,
                    height,
                    cpu.img.as_raw().iter().map(|value| value + 1).collect(),
                )?;
            }
        }
        Ok(FrameProcessorResult::Continue)
    }
    fn name(&self) -> &str {
        self.name
    }
    fn processing_type(&self) -> ProcessingType {
        match self.kind {
            Kind::Gpu { .. } => ProcessingType::Gpu,
            Kind::Cpu => ProcessingType::Cpu,
        }
    }
}

struct Sink(Output);
impl FrameSink for Sink {
    fn consume(&mut self, frame: &mut FrameContext, _: &mut PipelineContext) {
        let Frame::Cpu(cpu) = &frame.frame else {
            panic!("GST sink did not receive packed Luma8")
        };
        self.0.lock().unwrap().push((
            frame.frame_index,
            cpu.img.as_raw().to_vec(),
            frame.get_as::<String>("route").clone(),
        ));
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
    fn frame_processed(
        &mut self,
        frame: &FrameContext,
        _: &mut PipelineContext,
    ) -> Result<(), VydError> {
        assert!(matches!(frame.frame, Frame::Cpu(_)));
        self.0
            .lock()
            .unwrap()
            .push(format!("observed-{}", frame.frame_index));
        Ok(())
    }
}

fn context() -> Option<PipelineContext> {
    let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
        Ok(gpu) => Arc::new(gpu),
        Err(GpuContextError::Unavailable) => return None,
        Err(error) => panic!("GPU setup failed: {error}"),
    };
    let mut ctx = PipelineContext::new();
    ctx.set_wgpu_context(gpu).unwrap();
    Some(ctx)
}

fn pipeline(
    steps: &[(&'static str, Kind)],
    frames: usize,
    events: Events,
    output: Output,
) -> Pipeline {
    let identity = Arc::new(Mutex::new(None));
    Pipeline {
        source: Stage::new(
            Box::new(Source {
                next: 0,
                count: frames,
            }),
            "source",
        ),
        stages: steps
            .iter()
            .map(|(name, kind)| {
                Stage::new(
                    Box::new(Processor {
                        name,
                        kind: match kind {
                            Kind::Cpu => Kind::Cpu,
                            Kind::Gpu {
                                input,
                                output,
                                skip_index,
                            } => Kind::Gpu {
                                input: *input,
                                output: *output,
                                skip_index: *skip_index,
                            },
                        },
                        events: events.clone(),
                        service_identity: identity.clone(),
                    }) as Box<dyn FrameProcessor>,
                    name,
                )
            })
            .collect(),
        sink: Stage::new(Box::new(Sink(output)), "sink"),
    }
}

fn completed(stats: &PipelineStatistics, prefix: &str) -> u64 {
    stats
        .snapshot()
        .frame_operations
        .values()
        .flat_map(|operations| operations.iter())
        .filter(|(label, _)| label.starts_with(prefix))
        .map(|(_, counters)| counters.completed)
        .sum()
}

#[test]
fn adjacent_gpu_stages_use_one_upload_and_one_final_readback() {
    let Some(mut ctx) = context() else { return };
    let events = Events::default();
    let output = Output::default();
    ctx.add_observer(Observer(events.clone()));
    let stats = Arc::new(PipelineStatistics::new());
    run_pipeline(
        ctx,
        pipeline(
            &[
                (
                    "a",
                    Kind::Gpu {
                        input: PixelFormat::Luma8,
                        output: PixelFormat::Luma8,
                        skip_index: None,
                    },
                ),
                (
                    "b",
                    Kind::Gpu {
                        input: PixelFormat::Luma8,
                        output: PixelFormat::Luma8,
                        skip_index: None,
                    },
                ),
            ],
            1,
            events.clone(),
            output.clone(),
        ),
        Some(stats.clone()),
    )
    .unwrap();
    assert_eq!(
        output.lock().unwrap().as_slice(),
        &[(0, (1..=9).collect(), "ab".into())]
    );
    assert_eq!(events.lock().unwrap().as_slice(), &["a", "b", "observed-0"]);
    assert_eq!(completed(&stats, "upload_luma8"), 1);
    assert_eq!(completed(&stats, "download_luma8"), 1);
    assert_eq!(completed(&stats, "download_nv12"), 0);
}

#[test]
fn gpu_nv12_internal_round_trip_keeps_y_exact_without_intermediate_cpu_transfer() {
    let Some(ctx) = context() else { return };
    let events = Events::default();
    let output = Output::default();
    let stats = Arc::new(PipelineStatistics::new());
    run_pipeline(
        ctx,
        pipeline(
            &[
                (
                    "to_nv12",
                    Kind::Gpu {
                        input: PixelFormat::Luma8,
                        output: PixelFormat::Nv12,
                        skip_index: None,
                    },
                ),
                (
                    "to_luma",
                    Kind::Gpu {
                        input: PixelFormat::Nv12,
                        output: PixelFormat::Luma8,
                        skip_index: None,
                    },
                ),
            ],
            1,
            events,
            output.clone(),
        ),
        Some(stats.clone()),
    )
    .unwrap();
    assert_eq!(
        output.lock().unwrap().as_slice(),
        &[(0, (1..=9).collect(), "to_nv12to_luma".into())]
    );
    assert_eq!(completed(&stats, "upload_luma8"), 1);
    assert_eq!(completed(&stats, "convert_gpu_luma8_to_nv12"), 1);
    assert_eq!(completed(&stats, "convert_gpu_nv12_to_luma8"), 1);
    let snapshot = stats.snapshot();
    assert!(snapshot.frame_operations["to_nv12"]
        .keys()
        .any(|key| key.starts_with("convert_gpu_luma8_to_nv12")));
    assert!(snapshot.frame_operations["to_luma"]
        .keys()
        .any(|key| key.starts_with("convert_gpu_nv12_to_luma8")));
    assert_eq!(completed(&stats, "download_luma8"), 1);
    assert_eq!(completed(&stats, "download_nv12"), 0);
    assert_eq!(completed(&stats, "upload_nv12"), 0);
}

#[test]
fn nv12_gpu_segment_exit_converts_on_device_before_luma8_download() {
    let Some(ctx) = context() else { return };
    let events = Events::default();
    let output = Output::default();
    let stats = Arc::new(PipelineStatistics::new());
    run_pipeline(
        ctx,
        pipeline(
            &[(
                "to_nv12",
                Kind::Gpu {
                    input: PixelFormat::Luma8,
                    output: PixelFormat::Nv12,
                    skip_index: None,
                },
            )],
            1,
            events,
            output.clone(),
        ),
        Some(stats.clone()),
    )
    .unwrap();
    assert_eq!(
        output.lock().unwrap().as_slice(),
        &[(0, (1..=9).collect(), "to_nv12".into())]
    );
    assert_eq!(completed(&stats, "upload_luma8"), 1);
    assert_eq!(completed(&stats, "convert_gpu_luma8_to_nv12"), 1);
    assert_eq!(completed(&stats, "convert_gpu_nv12_to_luma8"), 1);
    assert_eq!(completed(&stats, "download_luma8"), 1);
    assert_eq!(completed(&stats, "download_nv12"), 0);
}

#[test]
fn cpu_and_native_neighbors_split_gpu_segments_and_keep_skip_observers() {
    let Some(mut ctx) = context() else { return };
    let events = Events::default();
    let output = Output::default();
    ctx.add_observer(Observer(events.clone()));
    let stats = Arc::new(PipelineStatistics::new());
    run_pipeline_with_native_nodes(
        ctx,
        pipeline(
            &[
                ("cpu_before", Kind::Cpu),
                (
                    "gpu_a",
                    Kind::Gpu {
                        input: PixelFormat::Luma8,
                        output: PixelFormat::Luma8,
                        skip_index: None,
                    },
                ),
                (
                    "gpu_b",
                    Kind::Gpu {
                        input: PixelFormat::Luma8,
                        output: PixelFormat::Luma8,
                        skip_index: Some(1),
                    },
                ),
                ("cpu_after", Kind::Cpu),
            ],
            2,
            events.clone(),
            output.clone(),
        ),
        Some(stats.clone()),
        vec![NativeNode::factory(2, "identity", vec![])],
    )
    .unwrap();
    assert_eq!(
        output.lock().unwrap().as_slice(),
        &[(
            0,
            (3..=11).collect(),
            "cpu_beforegpu_agpu_bcpu_after".into()
        )]
    );
    assert_eq!(completed(&stats, "upload_luma8"), 4);
    assert_eq!(completed(&stats, "download_luma8"), 4);
    assert_eq!(stats.snapshot().frame_count, 2);
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with("observed-"))
            .count(),
        2
    );
}
