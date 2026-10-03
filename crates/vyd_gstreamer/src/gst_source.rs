use std::sync::{
    atomic::{AtomicU8, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use gst::prelude::*;
use gst::{Message, MessageView};

use crate::config::*;
use crate::create_shared_pool;
use crate::gst_buffer_timing_meta::instrument_pipeline_with_timing_meta;
use crate::gst_pipeline::*;
use crate::vyd_frame_queue::*;
use thiserror::Error;
use vyd::statistics::{PipelineStatistics, ProcessingType};
use vyd::traits::FrameSource;
use vyd::{pipeline::PipelineContext, FrameContext, VydError};

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

    #[error("Bus error")]
    BusError,
}

impl From<GstSourceError> for vyd::VydError {
    fn from(err: GstSourceError) -> Self {
        vyd::VydError::GeneralPluginError(format!("{:?}", err))
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
    play_state: Arc<AtomicU8>, // Store as u8 for atomic operations
}

impl GstSource {
    pub fn new(params: Option<&toml::Value>) -> Result<Self, GstSourceError> {
        let Some(params) = params else {
            return Err(GstSourceError::ConfigError(
                "No config provided".to_string(),
            ));
        };

        let config: Config = params
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

        // See https://lib.rs/crates/tracing-gstreamer
        // See also https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html?gi-language=rust
        // tracing_gstreamer::integrate_events();
        // gst::debug_remove_default_log_function();
        gst::init()?;
        // tracing_gstreamer::integrate_spans();

        let pool = if config.zc_enabled {
            // Use default buffer size - the pool will handle different sizes
            let buffer_size = output_frame_queue.frame_info().frame_size;
            Some(create_shared_pool(
                buffer_size,
                config.buffer_pool_slot_count,
            ))
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
        .expect("Failed to build pipeline");

        let play_state = Arc::new(AtomicU8::new(PlayState::Initializing as u8));
        let pipeline = Arc::new(pipeline);

        if !live_mode {
            Self::start_flow_control_thread(
                pipeline.clone(),
                output_frame_queue.clone(),
                play_state.clone(),
            );
        }

        Ok(Self {
            config,
            frames_out_queue: output_frame_queue,
            pipeline,
            producer_handle: Arc::new(Mutex::new(None)),
            play_state,
        })
    }

    fn start_flow_control_thread(
        pipeline: Arc<gst::Pipeline>,
        ring_buffer: Arc<VydFrameQueue>,
        play_state: Arc<AtomicU8>,
    ) {
        let (tx, rx) = std::sync::mpsc::channel::<bool>();

        thread::spawn(move || {
            while let Ok(is_full) = rx.recv() {
                Self::handle_buffer_occupancy_events(pipeline.clone(), is_full, play_state.clone());
            }
        });

        ring_buffer.set_state_sender(tx);
    }

    fn handle_buffer_occupancy_events(
        pipeline: Arc<gst::Pipeline>,
        is_full: bool,
        play_state: Arc<AtomicU8>,
    ) {
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
            let handle = thread::spawn(move || {
                if let Err(err) = Self::run_pipeline(pipeline.clone(), play_state) {
                    tracing::error!(?err, "gst pipeline error");
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

        pipeline.set_state(gst::State::Playing)?;

        let result = Self::process_gst_messages(pipeline.as_ref(), play_state.clone());

        play_state.store(PlayState::Stopped as u8, Ordering::SeqCst);

        // perform clean gst shutdown
        pipeline.set_state(gst::State::Null)?;

        result
    }

    fn process_gst_messages(
        pipeline: &gst::Pipeline,
        play_state: Arc<AtomicU8>,
    ) -> Result<(), GstSourceError> {
        tracing::trace!("Starting gst msg processing loop");

        let bus = pipeline.bus().ok_or(GstSourceError::BusError)?;
        loop {
            match bus.timed_pop(gst::ClockTime::from_mseconds(50)) {
                Some(msg) => match msg.view() {
                    MessageView::Eos(_) => {
                        log_gst_message(msg);
                        tracing::info!("gst Eos msg received (end of stream)");
                        play_state.store(PlayState::Eos as u8, Ordering::SeqCst);
                        break;
                    }
                    MessageView::Error(_) => {
                        log_gst_message(msg);
                        break;
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
                        break;
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

        let FrameWithTiming { frame, timing_data } = frame_data;

        let frame_index = Self::inc_frame_counter(ctx);
        let mut frame_ctx = FrameContext::new(frame, frame_index);

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

    fn inc_frame_counter(ctx: &mut PipelineContext) -> usize {
        ctx.get_as_or_insert("frames_sourced", || AtomicUsize::new(0))
            .fetch_add(1, Ordering::SeqCst)
    }
}

impl FrameSource for GstSource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        let frame_data = match self.get_play_state() {
            PlayState::Initializing => {
                let _ = self.start_producer_thread();
                return self.next_frame(ctx);
            }

            PlayState::Playing => loop {
                if let Some(frame_data) = self.frames_out_queue.read_frame(FRAME_BLOCKING_TIMEOUT) {
                    break Some(frame_data);
                }

                if self.get_play_state() == PlayState::Stopped {
                    tracing::debug!("gst pipeline stopped whilst waiting");
                    return self.next_frame(ctx);
                }

                tracing::info!("timeout waiting for a frame");
            },

            _ => self.frames_out_queue.try_read_frame(),
        };

        let frame_ctx = frame_data.and_then(|frame_data| self.create_frame_ctx(frame_data, ctx));
        tracing::trace!("Emitting frame");
        frame_ctx
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
