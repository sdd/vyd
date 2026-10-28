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

    // GStreamer >= 1.22 can retain the sender's NTP clock mapping.
    for element in pipeline.iterate_elements().into_iter().flatten() {
        if element
            .find_property("add-reference-timestamp-meta")
            .is_some_and(|p| {
                p.value_type() == bool::static_type()
                    && p.flags().contains(gst::glib::ParamFlags::WRITABLE)
            })
        {
            element.set_property("add-reference-timestamp-meta", true);
        }
    }

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
    for element in pipeline.iterate_elements().into_iter().flatten() {
        if let Some(factory) = element.factory() {
            if factory.name().as_str().contains("decodebin") {
                decodebin = Some(element);
                break;
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

#[cfg(test)]
mod capture_tests {
    use super::*;
    use crate::vyd_frame_queue::FrameInfo;
    use std::time::{Duration, UNIX_EPOCH};
    use vyd_core::statistics::PipelineStatistics;

    #[test]
    fn capture_meta_survives_decode_convert_and_source_handoff() {
        gst::init().unwrap();
        for (description, pooled) in [
            ("videotestsrc name=capture num-buffers=3 ! video/x-raw,format=GRAY8,width=8,height=8,framerate=25/1 ! decodebin", false),
            ("videotestsrc name=capture num-buffers=3 ! video/x-raw,format=GRAY8,width=8,height=8,framerate=25/1 ! decodebin", true),
            ("videotestsrc num-buffers=3 ! video/x-raw,format=I420,width=16,height=16,framerate=25/1 ! theoraenc name=capture ! decodebin", false),
            ("videotestsrc num-buffers=3 ! video/x-raw,format=I420,width=16,height=16,framerate=25/1 ! theoraenc name=capture ! decodebin", true),
        ] {
            let queue = Arc::new(VydFrameQueue::new(
                FrameInfo {
                    width: 8,
                    height: 8,
                    stride: 8,
                    timestamp: 0,
                    frame_size: 64,
                },
                10,
                false,
            ));
            let pool = pooled.then(|| ManagedFramePool::new(Some(8), None).unwrap());
            let pipeline = build_generic_pipeline(
                description, queue.clone(), pool,
            ).unwrap();
            let stats = Arc::new(PipelineStatistics::new());
            let sink = pipeline
                .iterate_elements()
                .into_iter()
                .flatten()
                .find_map(|e| e.downcast::<VydGstSink>().ok())
                .unwrap();
            sink.set_statistics(stats.clone());
            pipeline
                .by_name("capture")
                .unwrap()
                .static_pad("src")
                .unwrap()
                .add_probe(gst::PadProbeType::BUFFER, |_, info| {
                    if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                        let buffer = buffer.make_mut();
                        let Some(pts) = buffer.pts() else { return gst::PadProbeReturn::Ok; };
                        let utc = gst::ClockTime::from_seconds(100) + pts;
                        gst::ReferenceTimestampMeta::add(
                            buffer,
                            &gst::Caps::builder("timestamp/x-unix").build(),
                            utc,
                            None,
                        );
                    }
                    gst::PadProbeReturn::Ok
                });
            pipeline.set_state(gst::State::Playing).unwrap();
            let message = pipeline.bus().unwrap().timed_pop_filtered(
                gst::ClockTime::from_seconds(10),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            );
            pipeline.set_state(gst::State::Null).unwrap();
            assert!(
                matches!(
                    message.as_ref().map(|m| m.view()),
                    Some(gst::MessageView::Eos(_))
                ),
                "{message:?}"
            );
            let first = queue.try_read_frame().unwrap().context;
            assert_eq!(
                first.frame_capture_ts,
                Some(UNIX_EPOCH + Duration::from_secs(100))
            );
            for index in 1..3 {
                let next = queue.try_read_frame().unwrap().context;
                assert_eq!(next.frame_index, index);
                assert_eq!(
                    next.frame_capture_ts,
                    Some(UNIX_EPOCH + Duration::from_millis(100_000 + index * 40))
                );
                assert_eq!(next.us_since(&first), index * 40_000);
            }
            if pooled {
                assert_eq!(
                    stats
                        .snapshot()
                        .memory_operations
                        .get("gst-source/sink_zero_copy"),
                    Some(&3)
                );
            }
        }
    }
}
