use anyhow::Result;
use gst::prelude::*;
use gst::Pipeline;
use std::sync::Arc;

use crate::managed_pool::ManagedFramePool;
use crate::vyd_frame_queue::VydFrameQueue;
use crate::vyd_gst_sink::VydGstSink;

pub fn build_file_pipeline(
    path: &str,
    ring_buffer: Arc<VydFrameQueue>,
    pool: Option<Arc<ManagedFramePool>>,
) -> Result<Pipeline> {
    let pipeline_str = format!("filesrc location={} ! decodebin", path);
    build_generic_pipeline(&pipeline_str, ring_buffer, pool)
}

pub fn build_rtsp_client_pipeline(
    uri: &str, // Changed from &Uri to &str to match caller
    ring_buffer: Arc<VydFrameQueue>,
    pool: Option<Arc<ManagedFramePool>>,
) -> Result<Pipeline> {
    let pipeline_str = format!("rtspsrc location={} ! decodebin", uri);
    build_generic_pipeline(&pipeline_str, ring_buffer, pool)
}

// if pool is None, the vydsink will run in non-ZC mode
fn build_generic_pipeline(
    source_pipeline: &str,
    ring_buffer: Arc<VydFrameQueue>,
    pool: Option<Arc<ManagedFramePool>>,
) -> Result<Pipeline> {
    let pipeline = gst::parse_launch(source_pipeline)?;
    let pipeline = pipeline
        .downcast::<Pipeline>()
        .expect("Expected a gst::Pipeline");

    // Create videoconvert for format conversion
    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("convert")
        .build()?;

    let vydsink = VydGstSink::new();
    if let Some(pool) = pool {
        vydsink.set_managed_pool(pool);
    }

    // Set sync=false for maximum speed processing
    vydsink.set_property("sync", false);

    vydsink.set_vyd_frame_queue(ring_buffer.clone());

    // Add both elements to pipeline
    pipeline.add_many([&videoconvert, vydsink.upcast_ref()])?;

    // Link videoconvert to vydsink with explicit caps
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .build();

    videoconvert.link_filtered(&vydsink, &caps)?;

    // Find decodebin and connect its pad-added signal
    let mut decodebin = None;
    for element in pipeline.iterate_elements() {
        if let Ok(element) = element {
            if let Some(factory) = element.factory() {
                if factory.name().as_str().contains("decodebin") {
                    decodebin = Some(element);
                    break;
                }
            }
        }
    }

    let decodebin = decodebin.ok_or_else(|| anyhow::anyhow!("Could not find decodebin element"))?;

    // Connect decodebin's dynamic pad to videoconvert
    let videoconvert_weak = videoconvert.downgrade();
    decodebin.connect_pad_added(move |_element, src_pad| {
        let Some(videoconvert) = videoconvert_weak.upgrade() else {
            tracing::warn!("VideoConvert was dropped, cannot link pad");
            return;
        };

        let sink_pad = videoconvert
            .static_pad("sink")
            .expect("videoconvert should have sink pad");

        if sink_pad.is_linked() {
            tracing::debug!("VideoConvert sink pad already linked, ignoring");
            return;
        }

        let caps = src_pad
            .current_caps()
            .or_else(|| Some(src_pad.query_caps(None)));
        if let Some(caps) = caps {
            let structure = caps.structure(0).unwrap();
            let media_type = structure.name();

            tracing::info!(
                "Connecting decodebin pad with caps: {} to videoconvert",
                media_type
            );

            // Only link video pads
            if media_type.starts_with("video/") {
                if let Err(e) = src_pad.link(&sink_pad) {
                    tracing::error!("Failed to link decodebin to videoconvert: {}", e);
                } else {
                    tracing::info!("✅ Successfully linked decodebin → videoconvert → vydsink");
                }
            }
        } else {
            tracing::warn!("No caps available on decodebin src pad");
        }
    });

    Ok(pipeline)
}
