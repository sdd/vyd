//! Source-side timing, sampled before a source queue can drop frames.
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use crate::{Frame, FrameContext, FrameTiming};

/// Per-stream clock. Latency only corrects estimated UTC, never relative time.
pub struct CaptureClock {
    next_index: u64,
    live: bool,
    latency: Duration,
    frame_zero: Option<SystemTime>,
    previous: Option<(Option<SystemTime>, Option<Duration>, Instant)>,
    elapsed: Duration,
    offsets: VecDeque<SystemTime>,
}

impl CaptureClock {
    pub fn new(live: bool, mean_frame_latency_ms: u64, frame_zero: Option<SystemTime>) -> Self {
        Self {
            next_index: 0,
            live,
            latency: Duration::from_millis(mean_frame_latency_ms),
            frame_zero,
            previous: None,
            elapsed: Duration::ZERO,
            offsets: VecDeque::new(),
        }
    }

    /// Set only from an explicit frame-zero capture tag, never a creation date.
    pub fn set_frame_zero(&mut self, utc: SystemTime) {
        if self.frame_zero.is_none() {
            self.frame_zero = Some(utc);
        }
    }

    /// `capture_utc` must be genuine capture UTC, not a container creation date.
    pub fn ingest(
        &mut self,
        frame: Frame,
        timing: FrameTiming,
        capture_utc: Option<SystemTime>,
    ) -> FrameContext {
        self.sample(
            frame,
            timing,
            capture_utc,
            Instant::now(),
            SystemTime::now(),
        )
    }

    fn sample(
        &mut self,
        frame: Frame,
        timing: FrameTiming,
        capture: Option<SystemTime>,
        now: Instant,
        utc: SystemTime,
    ) -> FrameContext {
        let pts = timing.presentation_timestamp;
        if let Some((old_capture, old_pts, old_now)) = self.previous {
            let delta = capture
                .zip(old_capture)
                .and_then(|(a, b)| a.duration_since(b).ok())
                .or_else(|| pts.zip(old_pts).map(|(a, b)| a.saturating_sub(b)))
                .unwrap_or_else(|| now.saturating_duration_since(old_now));
            self.elapsed = self.elapsed.saturating_add(delta);
        }
        let retained_pts = pts.map(|pts| {
            self.previous
                .and_then(|(_, old, _)| old)
                .map_or(pts, |old| old.max(pts))
        });
        self.previous = Some((capture, retained_pts, now));
        let estimated = if self.live {
            if let Some(pts) = pts {
                if let Some(offset) = utc.checked_sub(pts) {
                    if self.offsets.len() == 32 {
                        self.offsets.pop_front();
                    }
                    self.offsets.push_back(offset);
                }
                // Average UTC-minus-PTS samples relative to the first sample,
                // avoiding assumptions about the platform's SystemTime epoch.
                self.offsets.front().copied().and_then(|base| {
                    let total: i128 = self
                        .offsets
                        .iter()
                        .map(|t| match t.duration_since(base) {
                            Ok(d) => d.as_nanos() as i128,
                            Err(e) => -(e.duration().as_nanos() as i128),
                        })
                        .sum();
                    let mean = total / self.offsets.len() as i128;
                    let nanos = mean.unsigned_abs();
                    let delta = Duration::new(
                        u64::try_from(nanos / 1_000_000_000).ok()?,
                        (nanos % 1_000_000_000) as u32,
                    );
                    let offset = if mean >= 0 {
                        base.checked_add(delta)
                    } else {
                        base.checked_sub(delta)
                    };
                    offset
                        .and_then(|t| t.checked_add(pts))
                        .and_then(|t| t.checked_sub(self.latency))
                })
            } else {
                utc.checked_sub(self.latency)
            }
        } else {
            self.frame_zero
                .and_then(|zero| zero.checked_add(self.elapsed))
        };
        let mut context = FrameContext::new(frame, self.next_index);
        self.next_index = self.next_index.wrapping_add(1);
        context.timing = timing;
        context.frame_capture_ts = capture.or(estimated);
        context.elapsed_us = self.elapsed.as_micros() as u32;
        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> Frame {
        Frame::from_raw(1, 1, vec![0]).unwrap()
    }
    fn pts(us: u64) -> FrameTiming {
        FrameTiming {
            presentation_timestamp: Some(Duration::from_micros(us)),
            ..Default::default()
        }
    }
    #[test]
    fn wrapping_and_dropped_frames_keep_source_time() {
        let mut clock = CaptureClock::new(false, 0, None);
        let now = Instant::now();
        let utc = SystemTime::UNIX_EPOCH;
        let first = clock.sample(frame(), pts(0), None, now, utc);
        let _dropped = clock.sample(frame(), pts(u32::MAX as u64), None, now, utc);
        let last = clock.sample(frame(), pts(u32::MAX as u64 + 21), None, now, utc);
        assert_eq!(last.frame_index, 2);
        assert_eq!(last.us_since(&first), 20);
    }
    #[test]
    fn offline_pts_ignores_processing_speed_and_latency() {
        let zero = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let mut clock = CaptureClock::new(false, 999, Some(zero));
        let now = Instant::now();
        let first = clock.sample(frame(), pts(5_000_000), None, now, zero);
        let next = clock.sample(
            frame(),
            pts(5_040_000),
            None,
            now + Duration::from_secs(9),
            zero,
        );
        assert_eq!(next.us_since(&first), 40_000);
        assert_eq!(first.frame_capture_ts, Some(zero));
        assert_eq!(
            next.frame_capture_ts,
            zero.checked_add(Duration::from_millis(40))
        );
    }
    #[test]
    fn capture_deltas_take_priority_over_pts_and_arrival_jitter() {
        let zero = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let mut clock = CaptureClock::new(true, 500, None);
        let now = Instant::now();
        let first = clock.sample(frame(), pts(0), Some(zero), now, zero);
        let capture = zero + Duration::from_millis(40);
        let next = clock.sample(
            frame(),
            pts(90_000),
            Some(capture),
            now + Duration::from_secs(1),
            zero,
        );
        assert_eq!(next.us_since(&first), 40_000);
        assert_eq!(next.frame_capture_ts, Some(capture));
    }
    #[test]
    fn latency_only_adjusts_estimated_utc_and_offsets_average_jitter() {
        let zero = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let now = Instant::now();
        let mut clocks = [
            CaptureClock::new(true, 0, None),
            CaptureClock::new(true, 10, None),
        ];
        for (pts_us, arrival_us) in [(0, 0), (40_000, 60_000), (80_000, 80_000)] {
            let a = clocks[0].sample(
                frame(),
                pts(pts_us),
                None,
                now,
                zero + Duration::from_micros(arrival_us),
            );
            let b = clocks[1].sample(
                frame(),
                pts(pts_us),
                None,
                now,
                zero + Duration::from_micros(arrival_us),
            );
            assert_eq!(a.elapsed_us, b.elapsed_us);
            assert_eq!(
                a.frame_capture_ts
                    .unwrap()
                    .duration_since(b.frame_capture_ts.unwrap())
                    .unwrap(),
                Duration::from_millis(10)
            );
            if pts_us == 80_000 {
                assert_eq!(a.elapsed_us, 80_000);
                assert_eq!(
                    a.frame_capture_ts
                        .unwrap()
                        .duration_since(zero)
                        .unwrap()
                        .as_micros(),
                    86_666
                );
            }
        }
    }
    #[test]
    fn regressing_pts_does_not_double_count_time() {
        let mut clock = CaptureClock::new(false, 0, None);
        let now = Instant::now();
        let utc = SystemTime::UNIX_EPOCH;
        let first = clock.sample(frame(), pts(0), None, now, utc);
        for time in [40_000, 39_000, 80_000] {
            let next = clock.sample(frame(), pts(time), None, now, utc);
            assert_eq!(next.us_since(&first), time.max(40_000) as u32);
        }
    }

    #[test]
    fn no_pts_uses_monotonic_time_and_live_utc_latency() {
        let zero = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let now = Instant::now();
        let mut clock = CaptureClock::new(true, 10, None);
        let first = clock.sample(frame(), FrameTiming::default(), None, now, zero);
        let next = clock.sample(
            frame(),
            FrameTiming::default(),
            None,
            now + Duration::from_millis(40),
            zero,
        );
        assert_eq!(next.us_since(&first), 40_000);
        assert_eq!(
            next.frame_capture_ts,
            zero.checked_sub(Duration::from_millis(10))
        );
        let mut offline = CaptureClock::new(false, 0, None);
        assert_eq!(
            offline
                .sample(frame(), pts(0), None, now, zero)
                .frame_capture_ts,
            None
        );
    }
}
