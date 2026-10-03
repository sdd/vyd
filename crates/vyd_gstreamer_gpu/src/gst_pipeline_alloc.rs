use std::sync::Arc;

use gst::glib::Cast;
use gst::prelude::GstBinExt;
use gst::prelude::*;
use gst::Pipeline;
use http::Uri;

use crate::frame_buffer::FrameBuffer;
use crate::ring_buffer_sink::RingBufferSink;

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
    ring_buffer_pool: Arc<RingBufferPool>,
) -> anyhow::Result<Pipeline> {
    gst::Element::register(
        None,
        "vydsink",
        gst::Rank::None,
        RingBufferSink::static_type(),
    )?;

    // Create custom allocator and buffer pool
    let allocator = RingBufferGstAllocator::new(ring_buffer_pool.clone());

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

    sink.set_buffer_pool(ring_buffer_pool);

    Ok(pipeline)
}
