use std::sync::{
    atomic::{AtomicU8, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use gst::prelude::*;
use gst::{Message, MessageView};

use crate::config::*;
use crate::gst_buffer_timing_meta::instrument_pipeline_with_timing_meta;
use crate::gst_pipeline::*;
use crate::managed_pool::ManagedFramePool;
use crate::vyd_frame_queue::*;
use crate::vyd_gst_sink::VydGstSink;
use thiserror::Error;
use vyd_core::statistics::{PipelineStatistics, ProcessingType};
use vyd_core::traits::FrameSource;
use vyd_core::{pipeline::PipelineContext, FrameContext};

const FRAME_BLOCKING_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Error, Debug)]
pub enum GstSourceError {
    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("GStreamer init error: {0}")]
    GstInitError(#[from] gst::glib::Error),

    #[error("General init error: {0}")]
    GeneralInitError(String),

    #[error("GStreamer state change error: {0}")]
    GstStateChangeError(#[from] gst::StateChangeError),

    #[error("Bus error: {0}")]
    BusError(String),
}

impl From<GstSourceError> for vyd_core::VydError {
    fn from(err: GstSourceError) -> Self {
        vyd_core::VydError::GeneralPluginError(format!("{:?}", err))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
enum PlayState {
    Initializing = 0,
    Playing = 1,
    Stopped = 2,
    Eos = 3,
}

impl From<u8> for PlayState {
    fn from(value: u8) -> Self {
        match value {
            0 => PlayState::Initializing,
            1 => PlayState::Playing,
            2 => PlayState::Stopped,
            3 => PlayState::Eos,
            _ => PlayState::Initializing, // Default fallback
        }
    }
}

/// Vyd GstSource pipeline plugin.
///
/// Acts as a source of frames rendered via GStreamer.
/// Ultimately provides frames from either an RTSP stream
/// or from a file. When sourcing frames from a file,
/// pseudo_live flag can be set to mimic the frame dropping
/// that happens when using RTSP to process a live feed.
pub struct GstSource {
    #[allow(unused)]
    config: Config,
    frames_out_queue: Arc<VydFrameQueue>,
    pipeline: Arc<gst::Pipeline>,
    producer_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    flow_control_handle: Option<thread::JoinHandle<()>>,
    play_state: Arc<AtomicU8>, // Store as u8 for atomic operations
    producer_error: Arc<Mutex<Option<String>>>,
    sink: VydGstSink,
}

impl GstSource {
    pub fn new(params: Option<&toml::Value>) -> Result<Self, GstSourceError> {
        let Some(params) = params else {
            return Err(GstSourceError::ConfigError(
                "No config provided".to_string(),
            ));
        };

        let config: Config = params
            .clone()
            .try_into()
            .map_err(|e| GstSourceError::ConfigError(format!("Failed to parse config: {}", e)))?;

        // Use default frame info - will be updated when we get actual caps
        let frame_info = FrameInfo {
            width: 1920,  // Default - will be updated
            height: 1080, // Default - will be updated
            stride: 1920, // Default - will be updated
            timestamp: 0,
            frame_size: 1920 * 1080, // Default - will be updated
        };

        let live_mode = config.input.is_live_mode();
        let output_frame_queue = Arc::new(VydFrameQueue::new(frame_info, 10, live_mode));

        output_frame_queue
            .configure_timing(config.mean_frame_latency_ms, config.frame_zero_capture_ts());

        crate::initialize_gstreamer_tracing()?;

        let pool = if config.zc_enabled {
            Some(
                ManagedFramePool::new(Some(config.buffer_pool_slot_count.get()), None)
                    .map_err(|error| GstSourceError::ConfigError(error.to_string()))?,
            )
        } else {
            None
        };

        let pipeline = match &config.input {
            InputConfig::File { path, .. } => {
                build_file_pipeline(path, output_frame_queue.clone(), pool)
            }
            InputConfig::Rtsp { uri } => {
                // Convert Uri to string for the pipeline builder
                let uri_str = uri.to_string();
                build_rtsp_client_pipeline(&uri_str, output_frame_queue.clone(), pool)
            }
        }
        .map_err(|error| {
            GstSourceError::GeneralInitError(format!("Failed to build pipeline: {error}"))
        })?;

        let play_state = Arc::new(AtomicU8::new(PlayState::Initializing as u8));
        let sink = pipeline
            .iterate_elements()
            .into_iter()
            .filter_map(Result::ok)
            .find_map(|element| element.downcast::<VydGstSink>().ok())
            .ok_or_else(|| {
                GstSourceError::GeneralInitError("Vyd GST source has no VydGstSink".into())
            })?;
        let pipeline = Arc::new(pipeline);

        let flow_control_handle = if !live_mode {
            Some(Self::start_flow_control_thread(
                pipeline.clone(),
                output_frame_queue.clone(),
                play_state.clone(),
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            frames_out_queue: output_frame_queue,
            pipeline,
            producer_handle: Arc::new(Mutex::new(None)),
            flow_control_handle,
            play_state,
            producer_error: Arc::new(Mutex::new(None)),
            sink,
        })
    }

    fn shutdown(&mut self) -> Result<(), vyd_core::VydError> {
        self.play_state
            .store(PlayState::Stopped as u8, Ordering::SeqCst);
        self.frames_out_queue.clear_state_sender();
        let mut error = None;
        if let Some(handle) = self.flow_control_handle.take() {
            if handle.join().is_err() {
                error = Some(vyd_core::VydError::GeneralPluginError(
                    "GStreamer flow control thread panicked".into(),
                ));
            }
        }
        if let Err(failure) = self.pipeline.set_state(gst::State::Null) {
            error.get_or_insert_with(|| {
                vyd_core::VydError::GeneralPluginError(format!(
                    "GStreamer shutdown failed: {failure}"
                ))
            });
        }
        if let Some(handle) = self.producer_handle.lock().unwrap().take() {
            if handle.join().is_err() {
                error.get_or_insert_with(|| {
                    vyd_core::VydError::GeneralPluginError(
                        "GStreamer producer thread panicked".into(),
                    )
                });
            }
        }
        if let Some(error) = error {
            return Err(error);
        }
        Ok(())
    }

    fn start_flow_control_thread(
        pipeline: Arc<gst::Pipeline>,
        ring_buffer: Arc<VydFrameQueue>,
        play_state: Arc<AtomicU8>,
    ) -> thread::JoinHandle<()> {
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        let handle = thread::spawn(move || {
            while let Ok(is_full) = rx.recv() {
                Self::handle_buffer_occupancy_events(pipeline.clone(), is_full, play_state.clone());
            }
        });

        ring_buffer.set_state_sender(tx);
        handle
    }

    fn handle_buffer_occupancy_events(
        pipeline: Arc<gst::Pipeline>,
        is_full: bool,
        play_state: Arc<AtomicU8>,
    ) {
        if PlayState::from(play_state.load(Ordering::SeqCst)) != PlayState::Playing {
            return;
        }
        let (new_state, action_verb) = if is_full {
            (gst::State::Paused, "pause")
        } else {
            if PlayState::from(play_state.load(Ordering::SeqCst)) != PlayState::Playing {
                return;
            }
            (gst::State::Playing, "play")
        };

        tracing::debug!("attempting gst pipeline {action_verb}");
        let state_change_result = pipeline.set_state(new_state);

        match state_change_result {
            Ok(gst::StateChangeSuccess::Success) | Ok(gst::StateChangeSuccess::NoPreroll) => {
                tracing::debug!("gst pipeline {action_verb} succeeded");
            }
            Ok(gst::StateChangeSuccess::Async) => {
                tracing::debug!("gst pipeline {action_verb} in progress");
                let (result, current, pending) =
                    pipeline.state(Some(gst::ClockTime::from_seconds(1)));
                match result {
                    Ok(_) => {
                        if current == gst::State::Paused {
                            tracing::debug!("gst pipeline {action_verb} succeeded");
                        } else {
                            tracing::warn!(
                                ?current,
                                ?pending,
                                "gst pipeline {action_verb} initiation incomplete"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(?e, "gst pipeline {action_verb} initiation error");
                    }
                }
            }
            Err(error) => {
                tracing::error!(?error, "gst pipeline {action_verb} failed");
            }
        }
    }

    fn get_play_state(&self) -> PlayState {
        PlayState::from(self.play_state.load(Ordering::SeqCst))
    }

    fn start_producer_thread(&self) -> Result<(), GstSourceError> {
        if self
            .play_state
            .compare_exchange(
                PlayState::Initializing as u8,
                PlayState::Playing as u8,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            let play_state = self.play_state.clone();
            let pipeline = self.pipeline.clone();
            let producer_error = self.producer_error.clone();
            let handle = thread::spawn(move || {
                if let Err(err) = Self::run_pipeline(pipeline.clone(), play_state) {
                    tracing::error!(?err, "gst pipeline error");
                    *producer_error.lock().unwrap() = Some(err.to_string());
                }
            });

            *self
                .producer_handle
                .lock()
                .expect("producer_handle lock failed") = Some(handle);
        };

        Ok(())
    }

    fn run_pipeline(
        pipeline: Arc<gst::Pipeline>,
        play_state: Arc<AtomicU8>,
    ) -> Result<(), GstSourceError> {
        instrument_pipeline_with_timing_meta(pipeline.as_ref());

        let result = (|| {
            pipeline.set_state(gst::State::Playing)?;
            Self::process_gst_messages(pipeline.as_ref(), play_state.clone())
        })();

        play_state.store(PlayState::Stopped as u8, Ordering::SeqCst);

        // perform clean gst shutdown
        let stopped = pipeline.set_state(gst::State::Null);
        result?;
        stopped?;
        Ok(())
    }

    fn process_gst_messages(
        pipeline: &gst::Pipeline,
        play_state: Arc<AtomicU8>,
    ) -> Result<(), GstSourceError> {
        tracing::trace!("Starting gst msg processing loop");

        let bus = pipeline
            .bus()
            .ok_or_else(|| GstSourceError::BusError("pipeline has no bus".into()))?;
        loop {
            if PlayState::from(play_state.load(Ordering::SeqCst)) == PlayState::Stopped {
                break;
            }
            match bus.timed_pop(gst::ClockTime::from_mseconds(50)) {
                Some(msg) => match msg.view() {
                    MessageView::Eos(_) => {
                        log_gst_message(msg);
                        tracing::info!("gst Eos msg received (end of stream)");
                        play_state.store(PlayState::Eos as u8, Ordering::SeqCst);
                        break;
                    }
                    MessageView::Error(err) => {
                        let detail = format!(
                            "{}: {} ({:?})",
                            msg.src().map(|s| s.path_string()).unwrap_or_default(),
                            err.error(),
                            err.debug()
                        );
                        log_gst_message(msg);
                        return Err(GstSourceError::BusError(detail));
                    }
                    _ => log_gst_message(msg),
                },
                None => {
                    // Timeout polling for gst message from bus.
                    // Check if the pipeline is still running
                    let state = pipeline.current_state();

                    // If pipeline stopped unexpectedly, exit loop
                    if state == gst::State::Null {
                        tracing::warn!("Pipeline went to NULL state unexpectedly");
                        return Err(GstSourceError::BusError(
                            "pipeline entered NULL before EOS".into(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    fn create_frame_ctx(
        &self,
        frame_data: FrameWithTiming,
        ctx: &mut PipelineContext,
    ) -> Option<FrameContext> {
        let processing_start = Instant::now();
        // Delivered-frame statistics are independent of ingestion identity.
        ctx.get_as_or_insert("frames_sourced", || AtomicUsize::new(0))
            .fetch_add(1, Ordering::SeqCst);

        let FrameWithTiming {
            context: mut frame_ctx,
            timing_data,
        } = frame_data;

        if let Some(timing_data) = timing_data {
            if let Ok(stats) = ctx.try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
                crate::timing::record_gst_timing_data(stats, &timing_data);
            }
            frame_ctx.put("timing_data", timing_data);

            tracing::trace!(
                "Added timing metadata to frame context with {} fields",
                frame_ctx.get_as::<Vec<(u64, String)>>("timing_data").len()
            );
        }

        let processing_duration = processing_start.elapsed();
        let duration_us = processing_duration.as_micros() as u64;

        ctx.get_as::<Arc<PipelineStatistics>>("pipeline_statistics")
            .record_stage_timing("GstSource", duration_us);

        tracing::trace!(
            duration_us,
            "GstSource frame processing completed, emitting frame"
        );

        Some(frame_ctx)
    }
}

impl FrameSource for GstSource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        match self.try_next_frame(ctx) {
            Ok(frame) => frame,
            Err(error) => {
                tracing::error!(?error, "GStreamer source failed");
                None
            }
        }
    }

    fn try_next_frame(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Result<Option<FrameContext>, vyd_core::VydError> {
        if self.get_play_state() == PlayState::Initializing {
            if let Ok(stats) = ctx.try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
                self.sink.set_statistics(stats.clone());
            }
            self.start_producer_thread()
                .map_err(vyd_core::VydError::from)?;
        }
        loop {
            if ctx
                .try_get_as::<Arc<PipelineStatistics>>("pipeline_statistics")
                .is_ok_and(|stats| stats.cancellation_token().is_cancelled())
            {
                return Ok(None);
            }
            if let Some(frame_data) = self.frames_out_queue.try_read_frame() {
                return Ok(self.create_frame_ctx(frame_data, ctx));
            }
            match self.get_play_state() {
                PlayState::Playing => {
                    if let Some(frame_data) =
                        self.frames_out_queue.read_frame(FRAME_BLOCKING_TIMEOUT)
                    {
                        return Ok(self.create_frame_ctx(frame_data, ctx));
                    }
                }
                PlayState::Eos => return Ok(None),
                PlayState::Stopped => {
                    if let Some(failure) = self.producer_error.lock().unwrap().as_ref() {
                        return Err(vyd_core::VydError::GeneralPluginError(failure.clone()));
                    }
                    // The producer publishes its failure immediately after
                    // marking Stopped; wait for that publication via join.
                    if let Some(handle) = self.producer_handle.lock().unwrap().take() {
                        handle.join().map_err(|_| {
                            vyd_core::VydError::GeneralPluginError(
                                "GStreamer source thread panicked".into(),
                            )
                        })?;
                    }
                    if let Some(failure) = self.producer_error.lock().unwrap().as_ref() {
                        return Err(vyd_core::VydError::GeneralPluginError(failure.clone()));
                    }
                    return Ok(None);
                }
                PlayState::Initializing => self
                    .start_producer_thread()
                    .map_err(vyd_core::VydError::from)?,
            }
        }
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), vyd_core::VydError> {
        self.shutdown()
    }

    fn name(&self) -> &str {
        "gst_source"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

fn log_gst_message(msg: Message) {
    match msg.view() {
        MessageView::Eos(_) => {
            tracing::info!("gst Eos msg received (end of stream)");
        }
        MessageView::Error(err) => {
            tracing::error!(
                "Pipeline error: {} - {}",
                err.error(),
                err.debug().unwrap_or_default()
            );
        }
        MessageView::StateChanged(state_changed) => {
            tracing::trace!(
                "Pipeline state changed from {:?} to {:?}",
                state_changed.old(),
                state_changed.current()
            );
        }
        MessageView::StreamStart(_) => {
            tracing::debug!("Stream started");
        }

        MessageView::AsyncDone(_) => {
            tracing::trace!("gst async operation completed");
        }

        MessageView::Tag(tag) => {
            tracing::trace!("Received tag: {:?}", tag.tags());
        }

        MessageView::StreamStatus(status) => {
            tracing::trace!("Received stream status: {:?}", status);
        }

        MessageView::Latency(latency) => {
            tracing::trace!("Received Latency msg: {:?}", latency);
        }

        MessageView::DurationChanged(duration) => {
            tracing::trace!("Received DurationChanged msg: {:?}", duration);
        }

        MessageView::NewClock(new_clock) => {
            tracing::trace!("Received NewClock msg: {:?}", new_clock);
        }
        _msg_view => {
            tracing::trace!("Received other message: {:?}", msg.type_());
        }
    }
}

impl Drop for GstSource {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            tracing::error!(?error, "GStreamer source cleanup failed");
        }
    }
}
