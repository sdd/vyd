mod config;
mod frame_buffer;
mod gst_buffer_timing_meta;
pub mod gst_gpu_source;
mod gst_pipeline;
mod ring_buffer_sink;

use vyd::register_vyd_frame_source;

pub use crate::gst_gpu_source::GstGpuSource;

register_vyd_frame_source!(GstGpuSource);
