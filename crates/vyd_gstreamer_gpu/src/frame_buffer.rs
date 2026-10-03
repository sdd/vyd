use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::mpsc::Sender;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

#[allow(dead_code)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub timestamp: u64,
    pub frame_size: usize,
}

/// Frame data with associated timing metadata
#[derive(Debug, Clone)]
pub struct FrameData {
    pub bytes: Bytes,
    pub timing_data: Option<Vec<(u64, String)>>,
}

/// `FrameBuffer` buffers frames between stages.
///
/// Fixed capacity ring buffer.
/// Assumes that the pixel format, dimensions, and stride are
/// pre-determined and compatible between sender and receiver.
/// Can be configured in either blocking or frame-dropping modes.
pub struct FrameBuffer {
    buffer: Mutex<VecDeque<FrameData>>,
    not_empty: Condvar,
    frame_info: FrameInfo,
    capacity: usize,
    should_spill: bool,
    // Channel sender for buffer occupancy level changes
    state_sender: Mutex<Option<Sender<bool>>>,
    pause_threshold: usize,
    resume_threshold: usize,
}

impl FrameBuffer {
    pub fn new(frame_info: FrameInfo, capacity: usize, should_spill: bool) -> Self {
        let pause_threshold = (capacity - 3).max(1);

        Self {
            buffer: Mutex::new(VecDeque::with_capacity(capacity)),
            not_empty: Condvar::new(),
            frame_info,
            capacity,
            should_spill,
            state_sender: Mutex::new(None),
            pause_threshold,
            resume_threshold: 1,
        }
    }

    pub fn set_state_sender(&self, sender: Sender<bool>) {
        let mut state_sender = self.state_sender.lock().unwrap();
        *state_sender = Some(sender);
    }

    pub fn write_frame(&self, frame_data: &[u8]) -> Result<(), &'static str> {
        self.write_frame_with_timing(frame_data, None)
    }

    pub fn write_frame_with_timing(
        &self,
        frame_data: &[u8],
        timing_data: Option<Vec<(u64, String)>>,
    ) -> Result<(), &'static str> {
        if frame_data.len() != self.frame_info.frame_size {
            return Err("Frame size mismatch");
        }

        let mut buffer = self.buffer.lock().unwrap();
        let buffer_len = buffer.len();

        if self.should_spill {
            // drop frame if full.
            // TODO: re-use frame to avoid allocation
            if buffer_len == self.capacity {
                buffer.pop_front();
            }
        } else {
            // reject frame if buffer is completely full
            if buffer_len == self.capacity {
                tracing::warn!("buffer completely full, dropping a frame");
                return Err("Buffer full");
            }

            // Pause pipeline when we hit the threshold (before full)
            if buffer_len == self.pause_threshold {
                // Clone the sender before dropping locks
                let sender_clone = {
                    let state_sender = self.state_sender.lock().unwrap();
                    state_sender.clone()
                };

                if let Some(sender) = sender_clone {
                    tracing::info!(%buffer_len, "buffer reached upper capacity threshold");
                    drop(buffer); // Release buffer lock
                    if sender.send(true).is_err() {
                        tracing::warn!("Failed to send message");
                    }
                    buffer = self.buffer.lock().unwrap(); // Re-acquire buffer lock
                }
            }
        }

        // Add new frame at the back
        let frame_data = FrameData {
            bytes: Bytes::copy_from_slice(frame_data),
            timing_data,
        };
        buffer.push_back(frame_data);

        // Release buffer lock before notify
        drop(buffer);
        self.not_empty.notify_one();

        Ok(())
    }

    pub fn read_frame(&self, timeout: Duration) -> Option<FrameData> {
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
        let frame = buffer.pop_front();
        let new_buffer_len = buffer_len - 1;

        drop(buffer);

        if !self.should_spill
            && frame.is_some()
            && buffer_len > self.resume_threshold
            && new_buffer_len <= self.resume_threshold
        {
            let sender_clone = {
                let state_sender = self.state_sender.lock().unwrap();
                state_sender.clone()
            };

            if let Some(sender) = sender_clone {
                tracing::info!(
                    buffer_len = %new_buffer_len,
                    "buffer reached lower occupancy threshold",
                );
                if sender.send(false).is_err() {
                    tracing::warn!("failed to send message");
                }
            }
        }

        frame
    }

    pub fn try_read_frame(&self) -> Option<FrameData> {
        let mut buffer = self.buffer.lock().unwrap();
        let buffer_len = buffer.len();
        let frame = buffer.pop_front();

        if let Some(_) = frame {
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
                    tracing::info!(
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

        frame
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
