// wgpu's device/queue auto-traits add depth to RuntimeState's Send proof when
// Vyd's GPU feature is enabled alongside this GST runtime.
#![recursion_limit = "256"]

mod config;
pub mod frame_buffer_pool;
mod gst_buffer_timing_meta;
mod gst_pipeline;
pub mod gst_source;
pub mod managed_pool;
pub mod runtime;
pub mod vyd_frame_queue;
pub mod vyd_gst_allocator;
pub mod vyd_gst_buffer_pool;
pub mod vyd_gst_sink;

use vyd::register_vyd_frame_source;

pub use crate::frame_buffer_pool::{create_shared_pool, FrameBufferPool, SharedFrameBufferPool};
pub use crate::gst_source::GstSource;
pub use crate::vyd_gst_allocator::VydGstAllocator;
pub use crate::vyd_gst_buffer_pool::VydGstBufferPool;

register_vyd_frame_source!(GstSource);

pub mod timing;
