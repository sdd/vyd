// wgpu's device/queue auto-traits add depth to RuntimeState's Send proof when
// Vyd's GPU feature is enabled alongside this GST runtime.
#![recursion_limit = "256"]

mod config;
pub mod frame_buffer_pool;
mod gst_buffer_timing_meta;
mod gst_pipeline;
pub mod gst_source;
pub mod managed_pool;
pub mod pipeline_config;
pub mod runtime;
pub mod source_config;
pub mod vyd_frame_queue;
pub mod vyd_gst_allocator;
pub mod vyd_gst_buffer_pool;
pub mod vyd_gst_sink;

use vyd_core::register_vyd_frame_source;

pub use crate::frame_buffer_pool::{create_shared_pool, FrameBufferPool, SharedFrameBufferPool};
pub use crate::gst_source::GstSource;
pub use crate::vyd_gst_allocator::VydGstAllocator;
pub use crate::vyd_gst_buffer_pool::VydGstBufferPool;

register_vyd_frame_source!(GstSource);

pub mod timing;

/// Install the process-wide GStreamer tracing hooks once, before the first
/// initialization. Both the source and graph runtime use this entrypoint.
pub(crate) fn initialize_gstreamer_tracing() -> Result<(), gst::glib::Error> {
    static EVENTS: std::sync::Once = std::sync::Once::new();
    EVENTS.call_once(|| {
        tracing_gstreamer::integrate_events();
        gst::debug_remove_default_log_function();
    });
    gst::init()?;
    static SPANS: std::sync::Once = std::sync::Once::new();
    SPANS.call_once(tracing_gstreamer::integrate_spans);
    Ok(())
}
