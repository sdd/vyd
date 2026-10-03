use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use gst::glib::Cast;
use gst::prelude::GstBinExt;
use gst::prelude::*;
use gst::Pipeline;
use http::Uri;

use crate::frame_buffer::FrameBuffer;
use crate::ring_buffer_sink::RingBufferSink;

/// Global storage for timing data keyed by buffer pointer address
type TimingStorage = Arc<Mutex<HashMap<usize, Instant>>>;

/// Normalize element names by removing numeric suffixes that GStreamer adds
/// e.g., "decodebin0", "decodebin1" -> "decodebin"
fn normalize_stage_name(name: &str) -> String {
    // Find where the trailing digits start
    let mut end_pos = name.len();
    for (i, c) in name.char_indices().rev() {
        if c.is_ascii_digit() {
            end_pos = i;
        } else {
            break;
        }
    }

    name[..end_pos].to_string()
}

/// Instrument all static pads in the pipeline for latency measurement.
///
/// Attach this after pipeline is constructed but before set to Playing.
pub fn instrument_pipeline_latency(pipeline: &gst::Pipeline) {
    let elements: Result<Vec<_>, _> = pipeline.iterate_elements().into_iter().collect();
    let elements = elements.expect("Failed to iterate pipeline elements");

    // Shared storage for timing data
    let timing_storage: TimingStorage = Arc::new(Mutex::new(HashMap::new()));

    for element in elements {
        let elem_name = element.name();
        let normalized_name = normalize_stage_name(&elem_name);

        // Use iterate_pads() instead of static_pads()
        let pads: Result<Vec<_>, _> = element.iterate_pads().into_iter().collect();
        let pads = pads.expect("Failed to iterate element pads");

        for pad in pads {
            let stage_name = normalized_name.clone();

            match pad.direction() {
                gst::PadDirection::Src => {
                    let storage = timing_storage.clone();
                    // Start probe — attach timestamp on buffer entry
                    pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                        if let Some(gst::PadProbeData::Buffer(ref buffer)) = info.data {
                            let buffer_addr = buffer.as_ptr() as usize;
                            let start_time = Instant::now();

                            if let Ok(mut storage) = storage.lock() {
                                storage.insert(buffer_addr, start_time);
                            }

                            tracing::trace!(
                                stage_name = ?stage_name,
                                buffer_addr = %format!("0x{:x}", buffer_addr),
                                "Buffer entering src pad"
                            );
                        }
                        gst::PadProbeReturn::Ok
                    });
                }
                gst::PadDirection::Sink => {
                    let storage = timing_storage.clone();
                    // End probe — calculate delta and emit histogram
                    pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                        if let Some(gst::PadProbeData::Buffer(ref buffer)) = info.data {
                            let buffer_addr = buffer.as_ptr() as usize;
                            let end_time = Instant::now();

                            if let Ok(mut storage) = storage.lock() {
                                if let Some(start_time) = storage.remove(&buffer_addr) {
                                    let delta = end_time.duration_since(start_time);
                                    let delta_ns = delta.as_nanos() as u64;

                                    metrics::histogram!(
                                        "gst_pipeline.stage_latency_ns",
                                        "stage" => stage_name.to_string(),
                                    )
                                    .record(delta_ns as f64);

                                    tracing::debug!(
                                        stage_name = ?stage_name,
                                        buffer_addr = %format!("0x{:x}", buffer_addr),
                                        delta_ns,
                                        delta_us = delta.as_micros(),
                                        "gst_pipeline.stage_latency_ns"
                                    );
                                } else {
                                    tracing::trace!(
                                        stage_name = ?stage_name,
                                        buffer_addr = %format!("0x{:x}", buffer_addr),
                                        "Buffer arrived at sink pad but no start time found"
                                    );
                                }
                            }
                        }
                        gst::PadProbeReturn::Ok
                    });
                }
                _ => {}
            }
        }
    }
}

pub fn build_rtsp_client_pipeline(
    uri: &Uri,
    ring_buffer: Arc<FrameBuffer>,
) -> anyhow::Result<Pipeline> {
    let pipeline_str = format!(
        "rtspsrc location={} latency=0 ! queue ! rtph265depay ! h265parse ! avdec_h265",
        uri
    );

    build_generic_pipeline(&pipeline_str, ring_buffer)
}

pub fn build_file_pipeline(path: &str, ring_buffer: Arc<FrameBuffer>) -> anyhow::Result<Pipeline> {
    let pipeline_str = format!("filesrc location={} ! decodebin", path);

    build_generic_pipeline(&pipeline_str, ring_buffer)
}

fn build_generic_pipeline(
    input_description: &str,
    ring_buffer: Arc<FrameBuffer>,
) -> anyhow::Result<Pipeline> {
    gst::Element::register(
        None,
        "vydsink",
        gst::Rank::None,
        RingBufferSink::static_type(),
    )?;

    let pipeline_str = format!(
        "{} ! videoconvert ! video/x-raw,format=GRAY8 ! vydsink name=sink",
        input_description
    );

    let pipeline = gst::parse_launch(&pipeline_str)?
        .downcast::<Pipeline>()
        .expect("Expected a gst::Pipeline");

    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<RingBufferSink>()
        .unwrap();

    sink.set_buffer(ring_buffer);

    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_stage_name() {
        assert_eq!(normalize_stage_name("decodebin0"), "decodebin");
        assert_eq!(normalize_stage_name("decodebin1"), "decodebin");
        assert_eq!(normalize_stage_name("decodebin123"), "decodebin");
        assert_eq!(normalize_stage_name("queue"), "queue");
        assert_eq!(normalize_stage_name("videoconvert2"), "videoconvert");
        assert_eq!(normalize_stage_name("capsfilter3"), "capsfilter");
    }
}
