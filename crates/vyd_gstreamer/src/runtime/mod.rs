//! Serial GStreamer runner for Rust stages. Astrocap uses it by default;
//! branching and Frame v2 remain migration checkpoints.
mod bridge;
mod caps;
mod elements;
mod native;
mod transport;

pub use native::{
    FrameCorrespondence, NativeGeometry, NativeNode, NativeProperty, NativePropertyValue,
    NativeSection,
};
pub use transport::NativeFrameAnnotations;

use gst::prelude::*;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
#[cfg(feature = "gpu-wgpu")]
use vyd_core::frame::pool::WaitPolicy;
#[cfg(feature = "gpu-wgpu")]
use vyd_core::frame::{Frame, PixelFormat};
use vyd_core::pipeline::{Pipeline, PipelineContext, PipelineSession};
use vyd_core::statistics::PipelineStatistics;
#[cfg(feature = "gpu-wgpu")]
use vyd_core::statistics::StatsContext;
use vyd_core::{FrameProcessorResult, VydError};

#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    /// Maximum caller wait after cancellation or entry into finalization.
    /// On expiry the worker retains the pipeline and finishes cleanup once
    /// application code cooperates; no live storage is reclaimed early.
    pub shutdown_timeout: Duration,
}
impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            shutdown_timeout: Duration::from_secs(120),
        }
    }
}

pub(super) fn error(message: impl std::fmt::Display) -> VydError {
    VydError::GeneralPluginError(format!("GStreamer runtime: {message}"))
}

type Shared = Arc<Mutex<RuntimeState>>;
#[cfg(feature = "gpu-wgpu")]
const GPU_BOUNDARY_WAIT: Duration = Duration::from_secs(30);
struct RuntimeState {
    session: Option<PipelineSession>,
    stats: Arc<PipelineStatistics>,
    info: Option<gst_video::VideoInfo>,
    stage_dimensions: Vec<Option<(u32, u32)>>,
    #[cfg(feature = "gpu-wgpu")]
    selections: Vec<caps::ProcessorSelection>,
    failure: Option<VydError>,
}
impl RuntimeState {
    fn source(&mut self) -> Result<Option<(gst::Buffer, gst::Caps)>, VydError> {
        let Some(frame) = self.session.as_mut().unwrap().next_frame()? else {
            return Ok(None);
        };
        // A source may have been blocked when cancellation was requested.
        if !self.stats.is_running() {
            return Ok(None);
        }
        let started = Instant::now();
        let info = transport::info(&frame.frame)?;
        let expected = self.info.get_or_insert(info);
        let buffer = transport::pack(frame, started, expected, &self.stats)?;
        Ok(Some((buffer, expected.to_caps().map_err(error)?)))
    }

    fn process(
        &mut self,
        start: usize,
        end: usize,
        pad: &gst::Pad,
        buffer: &gst::BufferRef,
    ) -> Result<Option<gst::Buffer>, VydError> {
        if !self.stats.is_running() {
            return Ok(None);
        }
        let source = self
            .info
            .as_ref()
            .ok_or_else(|| error("frame arrived before caps"))?;
        let info = caps::negotiated_gray8(pad)?;
        let expected = self.stage_dimensions[start].unwrap_or((source.width(), source.height()));
        if (info.width(), info.height()) != expected {
            return Err(error(
                "unsupported undeclared or midstream native geometry change",
            ));
        }
        let (mut frame, started) = transport::unpack(buffer, &info, &self.stats)?;
        #[cfg(feature = "gpu-wgpu")]
        if matches!(self.selections[start], caps::ProcessorSelection::Gpu { .. }) {
            return self.process_gpu(start, end, frame, started, &info);
        }
        debug_assert_eq!(end, start + 1);
        let session = self.session.as_mut().unwrap();
        match session.process(start, &mut frame)? {
            FrameProcessorResult::Skip => {
                session.complete_frame(&frame, started)?;
                Ok(None)
            }
            FrameProcessorResult::Continue => {
                Ok(Some(transport::pack(frame, started, &info, &self.stats)?))
            }
        }
    }

    #[cfg(feature = "gpu-wgpu")]
    fn process_gpu(
        &mut self,
        start: usize,
        end: usize,
        mut frame: vyd_core::FrameContext,
        started: Instant,
        info: &gst_video::VideoInfo,
    ) -> Result<Option<gst::Buffer>, VydError> {
        let stats = self.stats.clone();
        let session = self.session.as_mut().unwrap();
        let factory = session.frame_factory().clone();
        let transfer = session.gpu_transfer().map_err(error)?.clone();
        let wait = WaitPolicy::Timeout(GPU_BOUNDARY_WAIT);
        for index in start..end {
            if !self.stats.is_running() {
                return Ok(None);
            }
            let caps::ProcessorSelection::Gpu { input, output } = self.selections[index] else {
                unreachable!("GPU segment was validated by CapsPlan")
            };
            // A requested GPU read performs and counts any needed conversion
            // or upload. Promotion makes the stage's declared memory contract
            // true before its callback runs.
            let uploading = !matches!(&frame.frame, Frame::ManagedGpu(_));
            let upload_started = Instant::now();
            let upload_scope = StatsContext::new(stats.clone())
                .for_stage("boundary/GPU upload")
                .enter();
            let read = frame
                .frame
                .try_read_gpu_as(&factory, &transfer, input, wait)?;
            drop(upload_scope);
            frame.frame = read.into_frame();
            if uploading {
                stats.record_stage_timing(
                    "boundary/GPU upload",
                    upload_started.elapsed().as_micros() as u64,
                );
            }
            match session.process(index, &mut frame)? {
                FrameProcessorResult::Skip => {
                    Self::materialize_gray8(&mut frame.frame, &factory, &transfer, wait, &stats)?;
                    session.complete_frame(&frame, started)?;
                    return Ok(None);
                }
                FrameProcessorResult::Continue => {}
            }
            if !matches!(&frame.frame, Frame::ManagedGpu(_)) || frame.frame.pixel_format() != output
            {
                return Err(VydError::GpuStageOutput {
                    stage_index: index,
                    expected: output,
                    actual: frame.frame.pixel_format(),
                    memory: if matches!(&frame.frame, Frame::ManagedGpu(_)) {
                        "Wgpu"
                    } else {
                        "CPU or unmanaged"
                    },
                });
            }
        }
        Self::materialize_gray8(&mut frame.frame, &factory, &transfer, wait, &stats)?;
        Ok(Some(transport::pack(frame, started, info, &self.stats)?))
    }

    #[cfg(feature = "gpu-wgpu")]
    fn materialize_gray8(
        frame: &mut Frame,
        factory: &vyd_core::frame::factory::FrameFactory,
        transfer: &vyd_core::gpu::transfer::GpuTransfer,
        wait: WaitPolicy,
        stats: &Arc<PipelineStatistics>,
    ) -> Result<(), VydError> {
        if matches!(frame, Frame::ManagedGpu(_)) {
            let download_started = Instant::now();
            let download_scope = StatsContext::new(stats.clone())
                .for_stage("boundary/GPU readback")
                .enter();
            // Convert while resident, so only the Luma8 plane crosses the
            // device boundary even when a segment emits NV12.
            let read = frame.try_read_gpu_as(factory, transfer, PixelFormat::Luma8, wait)?;
            *frame = read.into_frame();
            frame.ensure_cpu_as_with_transfer(factory, transfer, PixelFormat::Luma8, wait)?;
            drop(download_scope);
            stats.record_stage_timing(
                "boundary/GPU readback",
                download_started.elapsed().as_micros() as u64,
            );
            return Ok(());
        }
        frame.ensure_cpu_as_with_transfer(factory, transfer, PixelFormat::Luma8, wait)?;
        Ok(())
    }

    fn sink(&mut self, pad: &gst::Pad, buffer: &gst::BufferRef) -> Result<(), VydError> {
        if !self.stats.is_running() {
            return Ok(());
        }
        let source = self
            .info
            .as_ref()
            .ok_or_else(|| error("frame arrived before caps"))?;
        let info = caps::negotiated_gray8(pad)?;
        let expected = self
            .stage_dimensions
            .last()
            .copied()
            .flatten()
            .unwrap_or((source.width(), source.height()));
        if (info.width(), info.height()) != expected {
            return Err(error(
                "unsupported undeclared or midstream native geometry change",
            ));
        }
        let (mut frame, started) = transport::unpack(buffer, &info, &self.stats)?;
        let session = self.session.as_mut().unwrap();
        session.consume(&mut frame)?;
        session.complete_frame(&frame, started)
    }
}

fn invoke<T>(
    shared: &Shared,
    work: impl FnOnce(&mut RuntimeState) -> Result<T, VydError>,
) -> Result<T, gst::FlowError> {
    let mut state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if state.failure.is_some() {
        return Err(gst::FlowError::Error);
    }
    work(&mut state).map_err(|failure| {
        tracing::error!(?failure, "Rust stage adapter failed");
        state.failure = Some(failure);
        state.stats.stop();
        gst::FlowError::Error
    })
}

enum Progress {
    Finishing,
    Done(Result<PipelineContext, VydError>),
}

/// Run existing Rust stages as a serial GST source → processors → sink graph.
/// The context stays exclusively owned by the worker until cleanup completes.
pub fn run_pipeline(
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
) -> Result<PipelineContext, VydError> {
    run_pipeline_with_options(context, pipeline, stats, RuntimeOptions::default())
}

pub fn run_pipeline_with_options(
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
    options: RuntimeOptions,
) -> Result<PipelineContext, VydError> {
    run_pipeline_with_native_nodes_and_options(context, pipeline, stats, Vec::new(), options)
}

/// Run a linear Rust stage graph with native one-input/one-output sections.
/// Sections that strip the Vyd context meta need a declared FIFO bridge.
pub fn run_pipeline_with_native_nodes(
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
    nodes: Vec<NativeNode>,
) -> Result<PipelineContext, VydError> {
    run_pipeline_with_native_nodes_and_options(
        context,
        pipeline,
        stats,
        nodes,
        RuntimeOptions::default(),
    )
}

/// Low-level adapter used by the `vyd::Pipeline` facade. Most callers should
/// describe nodes through `vyd::PipelineBuilder` and call `pipeline.run()`.
pub fn run_pipeline_with_gstreamer_nodes(
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
    nodes: Vec<NativeNode>,
) -> Result<PipelineContext, VydError> {
    run_pipeline_with_native_nodes(context, pipeline, stats, nodes)
}

pub fn run_pipeline_with_native_nodes_and_options(
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
    nodes: Vec<NativeNode>,
    options: RuntimeOptions,
) -> Result<PipelineContext, VydError> {
    crate::initialize_gstreamer_tracing().map_err(error)?;

    transport::register();
    let stats = stats.unwrap_or_else(|| Arc::new(PipelineStatistics::new()));
    let worker_stats = stats.clone();
    let (send, receive) = mpsc::channel();
    std::thread::Builder::new()
        .name("vyd-gstreamer-runtime".into())
        .spawn(move || {
            let result = run_worker(context, pipeline, worker_stats, nodes, &send);
            if send.send(Progress::Done(result)).is_err() {
                tracing::info!("GStreamer cleanup finished after caller stopped waiting");
            }
        })
        .map_err(error)?;
    let mut closing: Option<Instant> = None;
    loop {
        if !stats.is_running() {
            closing.get_or_insert_with(Instant::now);
        }
        let wait = closing.map_or(Duration::from_millis(20), |since| {
            options
                .shutdown_timeout
                .saturating_sub(since.elapsed())
                .min(Duration::from_millis(20))
        });
        match receive.recv_timeout(wait) {
            Ok(Progress::Done(result)) => return result,
            Ok(Progress::Finishing) => {
                closing.get_or_insert_with(Instant::now);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(error("runtime worker exited without a result"))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if closing.is_some_and(|since| since.elapsed() >= options.shutdown_timeout) {
            stats.stop();
            return Err(VydError::ShutdownTimeout {
                timeout: options.shutdown_timeout,
            });
        }
    }
}

fn run_worker(
    context: PipelineContext,
    pipeline: Pipeline,
    stats: Arc<PipelineStatistics>,
    nodes: Vec<NativeNode>,
    progress: &mpsc::Sender<Progress>,
) -> Result<PipelineContext, VydError> {
    let mut session = PipelineSession::new(context, pipeline, stats.clone());
    if let Err(failure) = session.start() {
        let _ = progress.send(Progress::Finishing);
        return session.finish(Err(failure));
    }
    let count = session.processor_count();
    let mut native = match native::place(nodes, count) {
        Ok(nodes) => nodes,
        Err(failure) => {
            let _ = progress.send(Progress::Finishing);
            return session.finish(Err(failure));
        }
    };
    let native_boundaries = native
        .iter()
        .map(|nodes| !nodes.is_empty())
        .collect::<Vec<_>>();
    let caps = match caps::CapsPlan::new(
        session.port_declarations(),
        &native_boundaries,
        session.has_wgpu_context(),
    ) {
        Ok(caps) => caps,
        Err(failure) => {
            let _ = progress.send(Progress::Finishing);
            return session.finish(Err(failure));
        }
    };
    debug_assert_eq!(caps.selections.len(), count);
    let mut dimensions = None;
    let mut stage_dimensions = Vec::with_capacity(count + 1);
    for boundary in &native {
        for node in boundary {
            if let Some(geometry) = node.geometry() {
                if dimensions.is_some_and(|previous| previous != geometry.input) {
                    let _ = progress.send(Progress::Finishing);
                    return session
                        .finish(Err(error("adjacent native geometry contracts disagree")));
                }
                dimensions = Some(geometry.output);
            }
        }
        stage_dimensions.push(dimensions);
    }
    let shared = Arc::new(Mutex::new(RuntimeState {
        session: Some(session),
        stats: stats.clone(),
        info: None,
        stage_dimensions,
        #[cfg(feature = "gpu-wgpu")]
        selections: caps.selections.clone(),
        failure: None,
    }));
    let graph = gst::Pipeline::with_name("vyd-serial");
    let mut bridges = Vec::new();
    let mut normal_eos = false;
    let result = (|| {
        let source = elements::source(shared.clone());
        graph.add(&source).map_err(error)?;
        let mut previous = source;
        let source_filter = caps::filter("vyd-source-formats", &caps.source)?;
        graph.add(&source_filter).map_err(error)?;
        previous.link(&source_filter).map_err(error)?;
        previous = source_filter;
        native::append(
            &graph,
            &mut previous,
            std::mem::take(&mut native[0]),
            &shared,
            &mut bridges,
        )?;
        let mut index = 0;
        while index < count {
            let end = caps
                .gpu_segments
                .iter()
                .find(|segment| segment.start == index)
                .map_or(index + 1, |segment| segment.end);
            let input_filter = caps::filter(
                &format!("vyd-processor-{index}-input-formats"),
                &caps.processors[index].0,
            )?;
            graph.add(&input_filter).map_err(error)?;
            previous.link(&input_filter).map_err(error)?;
            previous = input_filter;
            let processor = elements::processor(shared.clone(), index, end);
            graph.add(&processor).map_err(error)?;
            previous.link(&processor).map_err(error)?;
            previous = processor;
            let output_filter = caps::filter(
                &format!("vyd-processor-{}-output-formats", end - 1),
                &caps.processors[end - 1].1,
            )?;
            graph.add(&output_filter).map_err(error)?;
            previous.link(&output_filter).map_err(error)?;
            previous = output_filter;
            native::append(
                &graph,
                &mut previous,
                std::mem::take(&mut native[end]),
                &shared,
                &mut bridges,
            )?;
            index = end;
        }
        let sink_filter = caps::filter("vyd-sink-formats", &caps.sink)?;
        graph.add(&sink_filter).map_err(error)?;
        previous.link(&sink_filter).map_err(error)?;
        previous = sink_filter;
        let sink = elements::sink(shared.clone());
        graph.add(&sink).map_err(error)?;
        previous.link(&sink).map_err(error)?;
        graph.set_state(gst::State::Playing).map_err(error)?;
        let bus = graph.bus().ok_or_else(|| error("pipeline has no bus"))?;
        loop {
            if !stats.is_running() {
                bridge::abort(&bridges);
                break;
            }
            if let Some(message) = bus.timed_pop(gst::ClockTime::from_mseconds(20)) {
                match message.view() {
                    gst::MessageView::Eos(..) => {
                        normal_eos = true;
                        break;
                    }
                    gst::MessageView::Error(failure) => {
                        bridge::abort(&bridges);
                        return Err(error(format!(
                            "{}: {} ({:?})",
                            failure.src().map(|s| s.path_string()).unwrap_or_default(),
                            failure.error(),
                            failure.debug()
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    tracing::debug!(?result, "GST graph reached a termination condition");
    let _ = progress.send(Progress::Finishing);
    // NULL can block while a user callback is still active. Only this owning
    // worker waits; the public caller has a deadline and never frees live state.
    let stopped = graph.set_state(gst::State::Null).map_err(error);
    tracing::debug!(?stopped, "GST graph stopped");
    drop(graph);
    let bridge_failure = bridge::finish(&bridges, normal_eos);
    let mut state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let failure = state
        .failure
        .take()
        .or_else(|| result.err())
        .or_else(|| stopped.err())
        .or(bridge_failure);
    let session = state.session.take().unwrap();
    drop(state);
    session.finish(failure.map_or(Ok(()), Err))
}

pub fn run_pipeline_with_config_file_path(
    path: &std::path::Path,
    stats: Arc<PipelineStatistics>,
) -> Result<PipelineContext, VydError> {
    let config = toml::from_str(&std::fs::read_to_string(path)?)?;
    run_pipeline(
        PipelineContext::new(),
        vyd_core::pipeline::build_pipeline(&config)?,
        Some(stats),
    )
}
