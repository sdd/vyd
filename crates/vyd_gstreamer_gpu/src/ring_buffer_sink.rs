use std::sync::{Arc, Mutex};

use gst::glib;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use once_cell::sync::Lazy;

use crate::frame_buffer::FrameBuffer;
use crate::gst_buffer_timing_meta::TimingMeta;

glib::wrapper! {
    /// Vyd custom GStreamer sink element.
    ///
    /// Registers within GStreamer under the name "vydsink".
    /// Used as the last element in the gstreamer pipeline. Transfers frames coming
    /// from GStreamer into the Vyd pipeline via the Vyd GstSource plugin.
    pub(crate) struct RingBufferSink(ObjectSubclass<imp::RingBufferSink>) @extends gst_base::BaseSink, gst::Element, gst::Object;
}

impl RingBufferSink {
    #[allow(unused)]
    pub(crate) fn new() -> Self {
        glib::Object::builder().build()
    }

    pub(crate) fn set_buffer(&self, frame_buffer: Arc<FrameBuffer>) {
        let imp = self.imp();
        *imp.buffer.lock().unwrap() = Some(frame_buffer);
    }
}

mod imp {
    use super::*;

    #[derive(Default)]
    pub(crate) struct RingBufferSink {
        pub(crate) buffer: Mutex<Option<Arc<FrameBuffer>>>,
    }

    impl RingBufferSink {
        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            let ring_buffer_guard = self.buffer.lock().unwrap();
            let Some(ref ring_buffer) = *ring_buffer_guard else {
                return Err(gst::FlowError::Error);
            };

            // Map the buffer for reading
            let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
            let data = map.as_slice();

            // Extract timing metadata from the buffer
            let timing_data = TimingMeta::extract_timing_data(buffer);

            let timing_data_option = if timing_data.is_empty() {
                None
            } else {
                tracing::trace!(
                    timing_field_count = timing_data.len(),
                    ?timing_data,
                    "Extracted timing metadata from buffer"
                );

                // Print timing summary for debugging
                // TimingMeta::print_timing_summary(buffer);

                Some(timing_data)
            };

            tracing::trace!("calling write_frame with timing data");
            match ring_buffer.write_frame_with_timing(
                data,
                timing_data_option,
                vyd_core::FrameTiming {
                    presentation_timestamp: buffer
                        .pts()
                        .map(|t| std::time::Duration::from_nanos(t.nseconds())),
                    decode_timestamp: buffer
                        .dts()
                        .map(|t| std::time::Duration::from_nanos(t.nseconds())),
                    duration: buffer
                        .duration()
                        .map(|t| std::time::Duration::from_nanos(t.nseconds())),
                },
                vyd_gstreamer::timing::capture_utc(buffer),
            ) {
                Ok(()) => Ok(gst::FlowSuccess::Ok),
                Err("Buffer full") => {
                    // Pipeline should already be paused but just in case
                    tracing::warn!("frame rejected due to full buffer");
                    Err(gst::FlowError::Flushing) // Tell GStreamer to retry later
                }
                Err(_) => Err(gst::FlowError::Error),
            }
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for RingBufferSink {
        const NAME: &'static str = "RingBufferSink";
        type Type = super::RingBufferSink;
        type ParentType = gst_base::BaseSink;
    }

    impl ObjectImpl for RingBufferSink {}

    impl GstObjectImpl for RingBufferSink {}

    impl ElementImpl for RingBufferSink {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Ring Buffer Sink",
                    "Sink/Video",
                    "Writes video frames directly to ring buffer",
                    "Scott Donnelly <scott@donnel.ly>",
                )
            });

            Some(&*ELEMENT_METADATA)
        }

        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
                let caps = gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Gray8)
                    .build();

                vec![gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap()]
            });

            PAD_TEMPLATES.as_ref()
        }
    }

    impl BaseSinkImpl for RingBufferSink {
        fn event(&self, event: gst::Event) -> bool {
            if matches!(
                event.view(),
                gst::EventView::FlushStop(_)
                    | gst::EventView::StreamStart(_)
                    | gst::EventView::Segment(_)
            ) {
                if let Some(queue) = self.buffer.lock().unwrap().as_ref() {
                    queue.reset_timing();
                }
            }
            if let gst::EventView::Tag(tag) = event.view() {
                if let Some(utc) = vyd_gstreamer::timing::frame_zero_capture_utc(tag.tag()) {
                    if let Some(queue) = self.buffer.lock().unwrap().as_ref() {
                        queue.set_frame_zero(utc);
                    }
                }
            }
            self.parent_event(event)
        }

        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            self.render(buffer)
        }
    }
}

#[cfg(test)]
mod clock_event_tests {
    use super::*;
    use gst_base::subclass::prelude::BaseSinkImpl;
    use std::time::Duration;
    use vyd_core::FrameTiming;

    #[test]
    fn flush_stream_and_segment_events_rebase_the_clock() {
        gst::init().unwrap();
        let segment = gst::FormattedSegment::<gst::ClockTime>::new();
        for event in [
            gst::event::FlushStop::new(false),
            gst::event::StreamStart::new("restart"),
            gst::event::Segment::new(&segment),
        ] {
            let queue = Arc::new(FrameBuffer::new(
                crate::frame_buffer::FrameInfo {
                    width: 1,
                    height: 1,
                    stride: 1,
                    timestamp: 0,
                    frame_size: 1,
                },
                4,
                true,
            ));
            let sink = RingBufferSink::new();
            sink.set_buffer(queue.clone());
            let write = |us| {
                let timing = FrameTiming {
                    presentation_timestamp: Some(Duration::from_micros(us)),
                    ..Default::default()
                };
                queue
                    .write_frame_with_timing(&[0], None, timing, None)
                    .unwrap();
            };
            write(60_000_000);
            queue.try_read_frame().unwrap();
            sink.imp().event(event);
            write(0);
            let first = queue.try_read_frame().unwrap().context;
            write(40_000);
            let next = queue.try_read_frame().unwrap().context;
            assert_eq!(first.frame_index, 1);
            assert_eq!(next.frame_index, 2);
            assert_eq!(next.us_since(&first), 40_000);
        }
    }
}
