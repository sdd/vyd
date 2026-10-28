#![cfg_attr(feature = "gpu-wgpu", recursion_limit = "256")]

use std::any::Any;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

pub mod annotations;
pub mod cancellation;
pub mod caps;
pub mod capture_timing;
pub mod error;
pub mod frame;
pub mod frame_metadata;
#[cfg(feature = "gpu-wgpu")]
pub mod gpu;
pub mod pipeline;
pub mod stages;
pub mod statistics;
pub mod traits;

pub use cancellation::CancellationToken;
pub use error::VydError;
pub use frame::Frame;
pub use traits::{FrameProcessor, FrameSink, FrameSource, StageFactory};

// needed for the exported macros to work without the consuming crate having to import them
pub use inventory;
pub use paste;
pub use toml;

/// Source media time for a frame, independent of stage processing time.
/// Missing fields mean the source did not supply that timestamp. The values
/// travel with the frame's identity through the supported linear graph.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrameTiming {
    pub presentation_timestamp: Option<Duration>,
    pub decode_timestamp: Option<Duration>,
    pub duration: Option<Duration>,
}

/// Opaque relative timing retained by transport adapters.
#[derive(Clone, Copy)]
pub struct FrameContextTiming(u64);

pub struct FrameContext {
    pub metadata: HashMap<String, Box<dyn Any + Send + Sync>>,
    pub typed_metadata: frame_metadata::TypedFrameMetadata,
    pub frame: Frame,
    /// Source-assigned identity in the supported linear stream. Stages must
    /// preserve it when replacing pixels; skips may leave gaps.
    pub frame_index: u64,
    /// Genuine source capture UTC, or a live-source estimate; unknown for untimed files.
    pub frame_capture_ts: Option<SystemTime>,
    pub(crate) elapsed_us: u64,
    pub timing: FrameTiming,
}

impl FrameContext {
    pub fn new(frame: Frame, frame_index: u64) -> Self {
        Self {
            metadata: HashMap::new(),
            typed_metadata: frame_metadata::TypedFrameMetadata::new(),
            frame,
            frame_index,
            frame_capture_ts: None,
            elapsed_us: 0,
            timing: FrameTiming::default(),
        }
    }

    /// Copy the opaque relative timestamp for a transport handoff.
    pub fn capture_timing(&self) -> FrameContextTiming {
        FrameContextTiming(self.elapsed_us)
    }

    /// Restore a timestamp previously copied from the same ingested frame.
    pub fn restore_capture_timing(&mut self, timing: FrameContextTiming) {
        self.elapsed_us = timing.0;
    }

    /// Relative capture time modulo 2^64 microseconds (about 584,000 years).
    pub fn us_since(&self, previous: &Self) -> u64 {
        self.elapsed_us.wrapping_sub(previous.elapsed_us)
    }

    pub fn try_get_as<'a, T: 'static>(&'a self, key: &str) -> Result<&'a T, VydError> {
        let Some(value) = self.metadata.get(key) else {
            tracing::trace!("key \"{}\" not present in frame context metadata", key);
            return Err(VydError::FrameMetadataNotFoundError);
        };

        let Some(downcasted) = value.downcast_ref::<T>() else {
            tracing::error!(
                "Could not downcast metadata value to requested type {}",
                core::any::type_name::<T>()
            );
            return Err(VydError::FrameMetadataTypeError);
        };

        Ok(downcasted)
    }

    pub fn get_as<'a, T: 'static>(&'a self, key: &str) -> &'a T {
        self.try_get_as(key).unwrap()
    }

    pub fn put<T: Any + Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.metadata.insert(key.to_string(), Box::new(value));
    }

    /// Store an immutable annotation. A value must not own this context's
    /// attached pixel buffer when transported through GST.
    pub fn put_typed<T: Any + Send + Sync>(&mut self, value: T) {
        self.typed_metadata.insert(value);
    }

    pub fn get_typed<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.typed_metadata.get::<T>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameProcessorResult {
    Continue,
    Skip,
}
