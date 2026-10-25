use std::sync::{Arc, Mutex};
use std::time::Duration;

use gst::glib;
use gst::prelude::BufferPoolExtManual;
use gst::prelude::{ElementExt, PadExt};
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use once_cell::sync::Lazy;

use crate::frame_buffer_pool::SharedFrameBufferPool;
use crate::gst_buffer_timing_meta::TimingMeta;
use crate::managed_pool::ManagedFramePool;
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

    pub fn set_managed_pool(&self, pool: Arc<ManagedFramePool>) {
        *self.imp().managed_pool.lock().unwrap() = Some(pool);
    }

    pub fn set_statistics(&self, stats: Arc<vyd_core::statistics::PipelineStatistics>) {
        if let Some(pool) = self.imp().managed_pool.lock().unwrap().as_ref() {
            pool.set_statistics(stats.clone());
        }
        *self.imp().statistics.lock().unwrap() = Some(stats);
    }
}

mod imp {
    use super::*;
    use gst::glib::Cast;
    use vyd_core::frame::layout::{CpuLayout, CpuPlaneLayout, FrameSpec};
    use vyd_core::frame::PixelFormat;
    use vyd_core::statistics::{MemoryOperation, PipelineStatistics};
    use vyd_core::{Frame, FrameTiming};

    #[derive(Default)]
    pub struct VydGstSink {
        pub(crate) output_frame_queue: Mutex<Option<Arc<VydFrameQueue>>>,
        pub(crate) frame_buffer_pool: Mutex<Option<SharedFrameBufferPool>>,
        pub(crate) managed_pool: Mutex<Option<Arc<ManagedFramePool>>>,
        pub(crate) gst_buffer_pool: Mutex<Option<VydGstBufferPool>>,
        pub(crate) statistics: Mutex<Option<Arc<PipelineStatistics>>>,
    }

    impl VydGstSink {
        fn get_video_info_from_caps(&self) -> Option<gst_video::VideoInfo> {
            let binding = self.obj();
            let element: &super::VydGstSink = binding.as_ref();
            let sink_pad = element.static_pad("sink")?;
            let caps = sink_pad.current_caps()?;
            gst_video::VideoInfo::from_caps(&caps).ok()
        }

        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            let mut timing_data = TimingMeta::extract_timing_data(buffer);
            timing_data.push((TimingMeta::now(), "gst_vydsink_entry".to_string()));

            // Get dimensions from caps instead of hardcoding
            let info = self.get_video_info_from_caps();
            let Some(info) = info else {
                tracing::error!("packed Luma8 caps are required before frame delivery");
                return Err(gst::FlowError::NotNegotiated);
            };
            let (width, height) = (info.width(), info.height());
            let packed = if let Some(meta) = buffer.meta::<gst_video::VideoMeta>() {
                meta.width() == width
                    && meta.height() == height
                    && meta.stride()[0] == width as i32
                    && meta.offset()[0] == 0
            } else {
                info.stride()[0] == width as i32 && info.offset()[0] == 0
            };
            if !packed {
                // TODO(Frame v2): require an explicit GST/Vyd layout converter
                // when its contract can be represented and verified end to end.
                tracing::error!(width, height, "padded Luma8 storage is not yet implemented");
                return Err(gst::FlowError::NotNegotiated);
            }

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

            let managed = self.managed_pool.lock().unwrap().clone();
            let legacy = self.frame_buffer_pool.lock().unwrap().clone();
            if managed.is_none() && legacy.is_none() {
                return self.render_with_copy(
                    buffer,
                    timing_data,
                    "No frame buffer pool configured",
                    width,
                    height,
                );
            }

            let memory_ptr = {
                let Ok(map) = buffer.peek_memory(0).map_readable() else {
                    tracing::error!("Failed to map buffer memory for reading");
                    return Err(gst::FlowError::Error);
                };
                map.as_ptr()
            };

            // Where the ZC magic happens - acquire a Frame from our pool from
            // the slot that corresponds to the memory pointer
            let frame = if let Some(managed) = managed {
                managed.acquire_frame_from_ptr(buffer, memory_ptr, width, height)
            } else {
                legacy
                    .unwrap()
                    .lock()
                    .unwrap()
                    .acquire_frame_from_ptr(memory_ptr, width, height)
            };
            let Ok(frame) = frame else {
                // non-ZC fallback if we couldn't acquire a Frame from our pool using the ptr
                return self.render_with_copy(
                    buffer,
                    timing_data,
                    "Memory pointer not from our pool",
                    width,
                    height,
                );
            };

            if let Some(stats) = self.statistics.lock().unwrap().as_ref() {
                stats
                    .record_memory_operation("gst-source/sink", MemoryOperation::ZeroCopyReference);
            }

            tracing::trace!(
                memory_ptr = ?memory_ptr,
                "Zero-copy: acquired Frame from memory pointer"
            );

            self.write_frame_to_queue(frame, timing_data, buffer)
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

            // TODO(Frame v2): keep this checked row path for an explicit
            // padded-to-packed converter. The render boundary currently
            // rejects padded input before reaching this fallback.
            let pixels = if let Some(info) = self.get_video_info_from_caps() {
                let (offset, stride) = if let Some(meta) = buffer.meta::<gst_video::VideoMeta>() {
                    if meta.width() != width || meta.height() != height {
                        return Err(gst::FlowError::Error);
                    }
                    (meta.offset()[0], meta.stride()[0])
                } else {
                    (info.offset()[0], info.stride()[0])
                };
                let stride = usize::try_from(stride).map_err(|_| gst::FlowError::Error)?;
                let spec = FrameSpec::new(width, height, PixelFormat::Luma8)
                    .map_err(|_| gst::FlowError::Error)?;
                let layout = CpuLayout::new(
                    spec,
                    &[CpuPlaneLayout::new(0, offset, stride)],
                    &[data.len()],
                )
                .map_err(|_| gst::FlowError::Error)?;
                let mut packed = Vec::with_capacity(width as usize * height as usize);
                for row in 0..height as usize {
                    packed.extend_from_slice(
                        layout
                            .row(&[data], 0, row)
                            .map_err(|_| gst::FlowError::Error)?,
                    );
                }
                packed
            } else {
                data.to_vec()
            };
            let Ok(frame) = Frame::from_raw(width, height, pixels) else {
                tracing::error!("Failed to create Frame from buffer");
                return Err(gst::FlowError::Error);
            };

            if let Some(stats) = self.statistics.lock().unwrap().as_ref() {
                let len = frame.pixel_count() as usize;
                stats.record_memory_operation(
                    "gst-source/sink",
                    MemoryOperation::CpuAllocation(len),
                );
                stats.record_memory_operation("gst-source/sink", MemoryOperation::CpuCopy(len));
            }

            self.write_frame_to_queue(frame, timing_data, buffer)
        }

        /// Extract timing metadata and write frame to output queue
        fn write_frame_to_queue(
            &self,
            frame: Frame,
            mut timing_data: Vec<(u64, String)>,
            buffer: &gst::Buffer,
        ) -> Result<gst::FlowSuccess, gst::FlowError> {
            timing_data.push((TimingMeta::now(), "gst_vydsink_exit".to_string()));

            let Some(queue) = self.output_frame_queue.lock().unwrap().clone() else {
                tracing::error!("No frame queue configured");
                return Err(gst::FlowError::Error);
            };

            let media_timing = FrameTiming {
                presentation_timestamp: buffer
                    .pts()
                    .map(|time| Duration::from_nanos(time.nseconds())),
                decode_timestamp: buffer
                    .dts()
                    .map(|time| Duration::from_nanos(time.nseconds())),
                duration: buffer
                    .duration()
                    .map(|time| Duration::from_nanos(time.nseconds())),
            };
            match queue.write_frame_with_metadata(frame, Some(timing_data), media_timing) {
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
        fn set_caps(&self, caps: &gst::Caps) -> Result<(), gst::LoggableError> {
            let info = gst_video::VideoInfo::from_caps(caps).map_err(|_| {
                gst::loggable_error!(gst::CAT_RUST, "unsupported Vyd source video caps")
            })?;
            if info.format() != gst_video::VideoFormat::Gray8
                || info.stride()[0] != info.width() as i32
                || info.offset()[0] != 0
            {
                // TODO(Frame v2): negotiate an explicit stride conversion
                // stage when packed and padded frame transport are implemented.
                return Err(gst::loggable_error!(
                    gst::CAT_RUST,
                    "packed/padded CPU layout negotiation is not yet implemented; insert an explicit converter"
                ));
            }
            self.parent_set_caps(caps)
        }

        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            self.render(buffer)
        }

        fn propose_allocation(
            &self,
            query: &mut gst::query::Allocation,
        ) -> Result<(), gst::LoggableError> {
            tracing::debug!("propose_allocation called");

            // Get our frame buffer pool
            let managed = self.managed_pool.lock().unwrap().clone();
            let legacy = self.frame_buffer_pool.lock().unwrap().clone();
            if managed.is_none() && legacy.is_none() {
                tracing::warn!(
                    "No frame buffer pool configured for allocation proposal. ZC-mode unavailable"
                );
                return Ok(()); // Let parent handle it - will fall back to copy mode
            }

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
            let managed_limit = managed.as_ref().map(|pool| pool.max_buffers());
            if let Some(managed) = managed {
                gst_pool.set_managed_pool(managed);
            } else {
                gst_pool.set_frame_buffer_pool(legacy.unwrap());
            }

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

            let size = u32::try_from(video_info.size()).map_err(|_| {
                gst::loggable_error!(gst::CAT_RUST, "Negotiated video buffer exceeds GST size")
            })?;
            let (min_buffers, max_buffers) = if let Some(limit) = managed_limit {
                let max = u32::try_from(limit).map_err(|_| {
                    gst::loggable_error!(gst::CAT_RUST, "Vyd pool count exceeds GST limit")
                })?;
                (max.min(4), max)
            } else {
                (4, 0) // Legacy proposal retains its historical GST limits.
            };

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

#[cfg(test)]
mod tests {
    use super::*;
    use gst_base::subclass::prelude::BaseSinkImpl;

    #[test]
    fn padded_caps_are_rejected_at_negotiation() {
        gst::init().unwrap();
        let sink = VydGstSink::new();
        let info = gst_video::VideoInfo::builder(gst_video::VideoFormat::Gray8, 5, 2)
            .build()
            .unwrap();
        assert_ne!(info.stride()[0], 5);
        let caps = info.to_caps().unwrap();
        let error = sink.imp().set_caps(&caps).unwrap_err();
        assert!(error.to_string().contains("not yet implemented"));
    }
}
