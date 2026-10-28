use std::collections::VecDeque;
use std::sync::mpsc::Sender;
use std::sync::{Condvar, Mutex};
use std::time::Duration;
use vyd_core::{capture_timing::CaptureClock, Frame, FrameContext, FrameTiming};

#[allow(dead_code)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub timestamp: u64,
    pub frame_size: usize,
}

/// Frame data with associated timing metadata
pub struct FrameWithTiming {
    pub context: FrameContext,
    pub timing_data: Option<Vec<(u64, String)>>,
}

/// `FrameBuffer` buffers frames between stages.
///
/// Fixed capacity ring buffer.
/// Assumes that the pixel format, dimensions, and stride are
/// pre-determined and compatible between sender and receiver.
/// Can be configured in either blocking or frame-dropping modes.
pub struct VydFrameQueue {
    buffer: Mutex<VecDeque<FrameWithTiming>>,
    clock: Mutex<CaptureClock>,
    not_empty: Condvar,
    frame_info: FrameInfo,
    capacity: usize,
    should_spill: bool,
    // Channel sender for buffer occupancy level changes
    state_sender: Mutex<Option<Sender<bool>>>,
    pause_threshold: usize,
    resume_threshold: usize,
}

impl VydFrameQueue {
    pub fn frame_info(&self) -> &FrameInfo {
        &self.frame_info
    }

    pub fn new(frame_info: FrameInfo, capacity: usize, should_spill: bool) -> Self {
        let pause_threshold = (capacity - 3).max(1);

        Self {
            buffer: Mutex::new(VecDeque::with_capacity(capacity)),
            clock: Mutex::new(CaptureClock::new(should_spill, 0, None)),
            not_empty: Condvar::new(),
            frame_info,
            capacity,
            should_spill,
            state_sender: Mutex::new(None),
            pause_threshold,
            resume_threshold: 1,
        }
    }

    pub fn configure_timing(&self, latency_ms: u64, frame_zero: Option<std::time::SystemTime>) {
        *self.clock.lock().unwrap() = CaptureClock::new(self.should_spill, latency_ms, frame_zero);
    }

    pub fn reset_timing(&self) {
        self.clock.lock().unwrap().reset();
    }

    pub fn set_frame_zero(&self, utc: std::time::SystemTime) {
        self.clock.lock().unwrap().set_frame_zero(utc);
    }

    pub fn clear_state_sender(&self) {
        self.state_sender.lock().unwrap().take();
    }

    pub fn set_state_sender(&self, sender: Sender<bool>) {
        let mut state_sender = self.state_sender.lock().unwrap();
        *state_sender = Some(sender);
    }

    #[allow(unused)]
    pub fn write_frame(&self, frame: Frame) -> Result<(), &'static str> {
        self.write_frame_with_timing(frame, None)
    }

    pub fn write_frame_with_timing(
        &self,
        frame: Frame,
        timing_data: Option<Vec<(u64, String)>>,
    ) -> Result<(), &'static str> {
        self.write_frame_with_metadata(frame, timing_data, FrameTiming::default())
    }

    pub fn write_frame_with_metadata(
        &self,
        frame: Frame,
        timing_data: Option<Vec<(u64, String)>>,
        media_timing: FrameTiming,
    ) -> Result<(), &'static str> {
        self.write_captured_frame(frame, timing_data, media_timing, None)
    }

    pub fn write_captured_frame(
        &self,
        frame: Frame,
        timing_data: Option<Vec<(u64, String)>>,
        media_timing: FrameTiming,
        capture_utc: Option<std::time::SystemTime>,
    ) -> Result<(), &'static str> {
        let mut buffer = self.buffer.lock().unwrap();
        if !self.should_spill && buffer.len() == self.capacity {
            return Err("Buffer full");
        }
        let context = self
            .clock
            .lock()
            .unwrap()
            .ingest(frame, media_timing, capture_utc);
        let buffer_len = buffer.len();

        if self.should_spill {
            // drop frame if full.
            if buffer_len == self.capacity {
                buffer.pop_front();
            }
        } else {
            // reject frame if buffer is full
            if buffer_len == self.capacity {
                tracing::warn!("buffer completely full, dropping this frame and pausing");
                return Err("Buffer full");
            }

            // Pause pipeline when we hit the threshold (before full)
            if buffer_len == self.pause_threshold {
                let sender_clone = {
                    let state_sender = self.state_sender.lock().unwrap();
                    state_sender.clone()
                };

                if let Some(sender) = sender_clone {
                    tracing::debug!(%buffer_len, "buffer reached upper capacity threshold");
                    drop(buffer); // Release buffer lock
                    if sender.send(true).is_err() {
                        tracing::warn!("Failed to send message");
                    }
                    buffer = self.buffer.lock().unwrap(); // Re-acquire buffer lock
                }
            }
        }

        let frame_with_timing = FrameWithTiming {
            context,
            timing_data,
        };

        buffer.push_back(frame_with_timing);
        drop(buffer);
        self.not_empty.notify_one();

        Ok(())
    }

    pub fn read_frame(&self, timeout: Duration) -> Option<FrameWithTiming> {
        let mut buffer = self.buffer.lock().unwrap();

        while buffer.is_empty() {
            let (new_buffer, timeout_result) =
                self.not_empty.wait_timeout(buffer, timeout).unwrap();
            buffer = new_buffer;

            if timeout_result.timed_out() {
                return None;
            }
        }

        let buffer_len = buffer.len();
        let frame_with_timing = buffer.pop_front();
        let new_buffer_len = buffer_len - 1;

        drop(buffer);

        if !self.should_spill
            && frame_with_timing.is_some()
            && buffer_len > self.resume_threshold
            && new_buffer_len <= self.resume_threshold
        {
            let sender_clone = {
                let state_sender = self.state_sender.lock().unwrap();
                state_sender.clone()
            };

            if let Some(sender) = sender_clone {
                tracing::debug!(
                    buffer_len = %new_buffer_len,
                    "buffer reached lower occupancy threshold",
                );
                if sender.send(false).is_err() {
                    tracing::warn!("failed to send message");
                }
            }
        }

        frame_with_timing
    }

    pub fn try_read_frame(&self) -> Option<FrameWithTiming> {
        let mut buffer = self.buffer.lock().unwrap();
        let buffer_len = buffer.len();
        let frame_with_timing = buffer.pop_front();

        if frame_with_timing.is_some() {
            let new_buffer_len = buffer_len - 1;

            // Release buffer lock before sending message
            drop(buffer);

            if !self.should_spill
                && buffer_len > self.resume_threshold
                && new_buffer_len <= self.resume_threshold
            {
                let sender_clone = {
                    let state_sender = self.state_sender.lock().unwrap();
                    state_sender.clone()
                };

                if let Some(sender) = sender_clone {
                    tracing::debug!(
                        buffer_len = %new_buffer_len,
                        "buffer reached lower capacity threshold",
                    );
                    if sender.send(false).is_err() {
                        tracing::warn!("failed to send message");
                    }
                }
            }
        } else {
            drop(buffer);
        }

        frame_with_timing
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        let buffer = self.buffer.lock().unwrap();
        buffer.len()
    }

    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        let buffer = self.buffer.lock().unwrap();
        buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spill_preserves_ingestion_indices_and_media_time() {
        let queue = VydFrameQueue::new(
            FrameInfo {
                width: 1,
                height: 1,
                stride: 1,
                timestamp: 0,
                frame_size: 1,
            },
            4,
            true,
        );
        let write = |index: u64| {
            queue
                .write_frame_with_metadata(
                    Frame::from_raw(1, 1, vec![0]).unwrap(),
                    None,
                    FrameTiming {
                        presentation_timestamp: Some(Duration::from_millis(index * 40)),
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        write(0);
        let first = queue.try_read_frame().unwrap().context;
        for index in 1..=8 {
            write(index);
        }
        let next = queue.try_read_frame().unwrap().context;
        assert_eq!(next.frame_index, 5);
        assert_eq!(next.us_since(&first), 200_000);
    }
}
