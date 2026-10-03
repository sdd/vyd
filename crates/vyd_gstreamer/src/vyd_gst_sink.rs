use std::sync::{Arc, Mutex};

use gst::glib;
use gst::prelude::BufferPoolExtManual;
use gst::prelude::{ElementExt, PadExt};
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use once_cell::sync::Lazy;

use crate::frame_buffer_pool::SharedFrameBufferPool;
use crate::gst_buffer_timing_meta::TimingMeta;
use crate::vyd_frame_queue::VydFrameQueue;
use crate::vyd_gst_buffer_pool::VydGstBufferPool;

glib::wrapper! {
    /// Vyd custom GStreamer sink element.
    ///
    /// Registers within GStreamer under the name "vydsink".
    /// Used as the last element in the gstreamer pipeline. Transfers frames coming
    /// from GStreamer into the Vyd pipeline via the Vyd GstSource plugin.
    pub struct VydGstSink(ObjectSubclass<imp::VydGstSink>) @extends gst_base::BaseSink, gst::Element, gst::Object;
}

impl VydGstSink {
    #[allow(unused)]
    pub fn new() -> Self {
        glib::Object::builder().build()
    }

    pub fn set_vyd_frame_queue(&self, frame_buffer: Arc<VydFrameQueue>) {
        let imp = self.imp();
        *imp.output_frame_queue.lock().unwrap() = Some(frame_buffer);
    }

    pub fn set_frame_buffer_pool(&self, pool: SharedFrameBufferPool) {
        let imp = self.imp();
        *imp.frame_buffer_pool.lock().unwrap() = Some(pool);
    }
}

mod imp {
    use super::*;
    use gst::glib::Cast;
    use vyd::Frame;

    #[derive(Default)]
    pub struct VydGstSink {
        pub(crate) output_frame_queue: Mutex<Option<Arc<VydFrameQueue>>>,
        pub(crate) frame_buffer_pool: Mutex<Option<SharedFrameBufferPool>>,
        pub(crate) gst_buffer_pool: Mutex<Option<VydGstBufferPool>>,
    }

    impl VydGstSink {
        /// Extract video dimensions from caps
        fn get_video_dimensions_from_caps(&self) -> Option<(u32, u32)> {
            let binding = self.obj();
            let element: &super::VydGstSink = binding.as_ref();
            let sink_pad = element.static_pad("sink")?;
            let caps = sink_pad.current_caps()?;
            let video_info = gst_video::VideoInfo::from_caps(&caps).ok()?;
            Some((video_info.width(), video_info.height()))
        }

        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            let mut timing_data = TimingMeta::extract_timing_data(buffer);
            timing_data.push((TimingMeta::now(), "gst_vydsink_entry".to_string()));

            // Get dimensions from caps instead of hardcoding
            let (width, height) = self.get_video_dimensions_from_caps().unwrap_or_else(|| {
                tracing::warn!(
                    "Could not determine video dimensions from caps, using fallback 1920x1080"
                );
                (1920, 1080)
            });

            // non-ZC fallback if memory count is not 1
            let memory_count = buffer.n_memory();
            if memory_count != 1 {
                return self.render_with_copy(
                    buffer,
                    timing_data,
                    "Buffer has unexpected memory count",
                    width,
                    height,
                );
            }

            // non-ZC fallback if we don't have a frame buffer pool
            let Some(frame_buffer_pool) = self.frame_buffer_pool.lock().unwrap().clone() else {
                return self.render_with_copy(
                    buffer,
                    timing_data,
                    "No frame buffer pool configured",
                    width,
                    height,
                );
            };

            let memory_ptr = {
                let Ok(map) = buffer.peek_memory(0).map_readable() else {
                    tracing::error!("Failed to map buffer memory for reading");
                    return Err(gst::FlowError::Error);
                };
                map.as_ptr()
            };

            // Where the ZC magic happens - acquire a Frame from our pool from
            // the slot that corresponds to the memory pointer
            let Ok(frame) = ({
                frame_buffer_pool
                    .lock()
                    .unwrap()
                    .acquire_frame_from_ptr(memory_ptr, width, height)
            }) else {
                // non-ZC fallback if we couldn't acquire a Frame from our pool using the ptr
                return self.render_with_copy(
                    buffer,
                    timing_data,
                    "Memory pointer not from our pool",
                    width,
                    height,
                );
            };

            tracing::trace!(
                memory_ptr = ?memory_ptr,
                "Zero-copy: acquired Frame from memory pointer"
            );

            self.write_frame_to_queue(frame, timing_data)
        }

        /// Fallback copy-based rendering for buffers not from our pool
        fn render_with_copy(
            &self,
            buffer: &gst::Buffer,
            timing_data: Vec<(u64, String)>,
            reason: &str,
            width: u32,
            height: u32,
        ) -> Result<gst::FlowSuccess, gst::FlowError> {
            tracing::trace!(?reason, "non-ZC render");

            let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
            let data = map.as_slice();

            let Ok(frame) = Frame::from_raw(width, height, data.to_vec()) else {
                tracing::error!("Failed to create Frame from buffer");
                return Err(gst::FlowError::Error);
            };

            self.write_frame_to_queue(frame, timing_data)
        }

        /// Extract timing metadata and write frame to output queue
        fn write_frame_to_queue(
            &self,
            frame: Frame,
            mut timing_data: Vec<(u64, String)>,
        ) -> Result<gst::FlowSuccess, gst::FlowError> {
            timing_data.push((TimingMeta::now(), "gst_vydsink_exit".to_string()));

            let Some(queue) = self.output_frame_queue.lock().unwrap().clone() else {
                tracing::error!("No frame queue configured");
                return Err(gst::FlowError::Error);
            };

            match queue.write_frame_with_timing(frame, Some(timing_data)) {
                Ok(()) => Ok(gst::FlowSuccess::Ok),
                Err("Buffer full") => {
                    tracing::warn!("Frame rejected due to full buffer");
                    Err(gst::FlowError::Flushing)
                }
                Err(e) => {
                    tracing::error!(error = %e, "Frame handoff failed");
                    Err(gst::FlowError::Error)
                }
            }
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for VydGstSink {
        const NAME: &'static str = "RingBufferSink";
        type Type = super::VydGstSink;
        type ParentType = gst_base::BaseSink;
    }

    impl ObjectImpl for VydGstSink {}

    impl GstObjectImpl for VydGstSink {}

    impl ElementImpl for VydGstSink {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Ring Buffer Sink",
                    "Sink/Video",
                    "Writes video frames directly to ring buffer with zero-copy when possible",
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

    impl BaseSinkImpl for VydGstSink {
        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            self.render(buffer)
        }

        fn propose_allocation(
            &self,
            query: &mut gst::query::Allocation,
        ) -> Result<(), gst::LoggableError> {
            tracing::debug!("propose_allocation called");

            // Get our frame buffer pool
            let Some(frame_pool) = self.frame_buffer_pool.lock().unwrap().clone() else {
                tracing::warn!(
                    "No frame buffer pool configured for allocation proposal. ZC-mode unavailable"
                );
                return Ok(()); // Let parent handle it - will fall back to copy mode
            };

            // Create or reuse our GStreamer buffer pool
            let gst_pool = {
                let mut pool_guard = self.gst_buffer_pool.lock().unwrap();
                if pool_guard.is_none() {
                    tracing::debug!("Creating new VydGstBufferPool for allocation");
                    *pool_guard = Some(VydGstBufferPool::new());
                }
                pool_guard.as_ref().unwrap().clone()
            };

            // TODO: why do we need to do this here? Can't we do it when constructing
            //       the VydGstBufferPool?

            // Configure the pool with our frame buffer pool
            gst_pool.set_frame_buffer_pool(frame_pool);

            // Get video info from the allocation query to configure pool
            let (Some(caps_ref), _) = query.get() else {
                tracing::warn!("No caps found in allocation query - cannot configure buffer pool");
                return Ok(());
            };

            tracing::debug!(
                ?caps_ref,
                "Configuring buffer pool with caps from allocation query"
            );

            let video_info = gst_video::VideoInfo::from_caps(caps_ref).map_err(|_| {
                gst::loggable_error!(gst::CAT_RUST, "Failed to parse video info from caps")
            })?;

            let size = video_info.size() as u32;
            let min_buffers = 4u32; // Reasonable minimum for video pipeline
            let max_buffers = 0u32; // 0 means unlimited

            // Configure the buffer pool
            let mut config = gst_pool.config();
            config.set_params(Some(&caps_ref.copy()), size, min_buffers, max_buffers);

            gst_pool.set_config(config).map_err(|_| {
                gst::loggable_error!(gst::CAT_RUST, "Failed to configure buffer pool")
            })?;

            // Add our pool to the allocation query with high priority
            let gst_buffer_pool: &gst::BufferPool = gst_pool.upcast_ref();
            query.add_allocation_pool(Some(gst_buffer_pool), size, min_buffers, max_buffers);

            tracing::info!(
                size,
                min_buffers,
                max_buffers,
                "Successfully proposed zero-copy buffer pool to upstream elements"
            );

            Ok(())
        }
    }
}
