use vyd_core::statistics::PipelineStatistics;

/// Record element latency events produced by the GStreamer source.
pub fn record_gst_timing_data(stats: &PipelineStatistics, timing_data: &[(u64, String)]) {
    use once_cell::sync::Lazy;
    use std::collections::HashSet;

    // Elements that commonly don't have matching entry/exit pairs due to their nature
    static TIMING_EXEMPT_PATTERNS: Lazy<HashSet<&'static str>> =
        Lazy::new(|| ["src", "sink", "bin", "demux"].into_iter().collect());

    // Group events by stage and calculate durations between entry and exit
    let mut stage_entries = std::collections::HashMap::new();

    for (timestamp_us, event_name) in timing_data {
        // Parse stage name and event type from format "gst_stagename_entry" or "gst_stagename_exit"
        if let Some(stage_and_event) = event_name.strip_prefix("gst_") {
            if let Some((stage_name, event_type)) = stage_and_event.rsplit_once('_') {
                match event_type {
                    "entry" => {
                        // Record entry time for this stage
                        stage_entries.insert(stage_name.to_string(), *timestamp_us);
                    }
                    "exit" => {
                        // Look for matching entry and calculate duration
                        if let Some(entry_time) = stage_entries.remove(stage_name) {
                            let duration_us = timestamp_us.saturating_sub(entry_time);

                            // Create a clean stage name for display
                            let clean_stage_name = format!("external/{}", stage_name);

                            // Record the timing
                            stats.record_stage_timing(&clean_stage_name, duration_us);

                            tracing::trace!(
                                stage = stage_name,
                                duration_us,
                                entry_time,
                                exit_time = timestamp_us,
                                "Recorded GST stage duration"
                            );
                        } else {
                            // Only warn if the element doesn't contain timing-exempt patterns
                            let is_exempt = TIMING_EXEMPT_PATTERNS
                                .iter()
                                .any(|pattern| stage_name.contains(pattern));

                            if !is_exempt {
                                tracing::warn!(
                                    stage = stage_name,
                                    "Found exit event without matching entry event"
                                );
                            }
                        }
                    }
                    _ => {
                        tracing::warn!(
                            event_name,
                            "Unknown GST event type, expected 'entry' or 'exit'"
                        );
                    }
                }
            }
        }
    }

    // Warn about any unmatched entry events (except timing-exempt elements)
    for (stage_name, _entry_time) in stage_entries {
        let is_exempt = TIMING_EXEMPT_PATTERNS
            .iter()
            .any(|pattern| stage_name.contains(pattern));

        if !is_exempt {
            tracing::warn!(
                stage = stage_name,
                "Found entry event without matching exit event"
            );
        }
    }
}

/// Only known UTC reference clock domains are usable as absolute capture time.
pub fn capture_utc(buffer: &gst::BufferRef) -> Option<std::time::SystemTime> {
    buffer
        .iter_meta::<gst::ReferenceTimestampMeta>()
        .find_map(|meta| {
            let reference = meta.reference().structure(0)?.name();
            let nanos = meta.timestamp().nseconds();
            let unix_nanos = match reference.as_str() {
                "timestamp/x-unix" => nanos,
                "timestamp/x-ntp" => nanos.checked_sub(2_208_988_800u64 * 1_000_000_000)?,
                _ => return None,
            };
            std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_nanos(unix_nanos))
        })
}

/// Opt-in container convention: an extended comment in Unix microseconds.
/// Generic date/date-time tags describe creation and are deliberately ignored.
pub fn frame_zero_capture_utc(tags: &gst::TagListRef) -> Option<std::time::SystemTime> {
    tags.iter_tag::<gst::tags::ExtendedComment>()
        .find_map(|value| {
            let us = value
                .get()
                .strip_prefix("frame-zero-capture-unix-us=")?
                .parse::<u64>()
                .ok()?;
            std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_micros(us))
        })
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    #[test]
    fn utc_reference_domains_and_explicit_container_tag() {
        gst::init().unwrap();
        for (domain, seconds, expected) in [
            ("timestamp/x-unix", 100, Some(100)),
            ("timestamp/x-ntp", 2_208_988_900, Some(100)),
            ("timestamp/x-ntp", 10, None),
            ("timestamp/x-monotonic", 100, None),
        ] {
            let mut buffer = gst::Buffer::new();
            gst::ReferenceTimestampMeta::add(
                buffer.get_mut().unwrap(),
                &gst::Caps::builder(domain).build(),
                gst::ClockTime::from_seconds(seconds),
                None,
            );
            assert_eq!(
                capture_utc(&buffer),
                expected.map(|s| std::time::UNIX_EPOCH + std::time::Duration::from_secs(s))
            );
        }
        let mut tags = gst::TagList::new();
        assert_eq!(frame_zero_capture_utc(&tags), None);
        tags.get_mut().unwrap().add::<gst::tags::ExtendedComment>(
            &"frame-zero-capture-unix-us=100000000",
            gst::TagMergeMode::Append,
        );
        assert_eq!(
            frame_zero_capture_utc(&tags),
            Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(100))
        );
    }
}
