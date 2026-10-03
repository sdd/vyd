const TRACING_TARGET: &str = "vyd_timing";

/// Helper function to add vyd stage timing to existing timing data (in microseconds)
#[allow(unused)]
pub(crate) fn add_stage_timing(
    frame_ctx: &mut crate::FrameContext,
    stage_name: &str,
    stage_type: &str,
    duration_us: u64,
) {
    let timing_data = get_or_create_timing_data(frame_ctx);

    // Get current timestamp (we'll use the time since we got the frame for relative timing)
    // This is approximate but gives us relative timing within the vyd pipeline
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    // Add pipeline timing with a prefix to distinguish from source timing
    let event_name = format!("pipeline_{}", stage_name);
    timing_data.push((timestamp_us, event_name));

    tracing::trace!(
        target: TRACING_TARGET,
        stage_name,
        stage_type,
        duration_us,
        timestamp_us,
        total_timing_events = timing_data.len(),
        "pipeline.stage_latency_us"
    );
}

#[allow(unused)]
pub(crate) fn get_or_create_timing_data(
    frame_ctx: &mut crate::FrameContext,
) -> &mut Vec<(u64, String)> {
    if !frame_ctx.metadata.contains_key("timing_data") {
        frame_ctx.put("timing_data", Vec::<(u64, String)>::new());
    }

    // We know this will succeed because we just created it if it didn't exist
    frame_ctx
        .metadata
        .get_mut("timing_data")
        .unwrap()
        .downcast_mut::<Vec<(u64, String)>>()
        .unwrap()
}

/// Helper function to log timing summary for one frame
pub(crate) fn log_frame_timing_summary(
    frame_ctx: &crate::FrameContext,
    frame_count: u64,
    total_frame_duration_us: u64,
) {
    if let Some(timing_data) = frame_ctx
        .metadata
        .get("timing_data")
        .and_then(|data| data.downcast_ref::<Vec<(u64, String)>>())
    {
        if timing_data.is_empty() {
            tracing::info!(
                target: TRACING_TARGET,
                frame_count,
                "No timing data found on frame"
            );
            return;
        }

        tracing::debug!(
            target: TRACING_TARGET,
            frame_count,
            "=== Frame Processing Timeline ==="
        );

        // Sort all events by timestamp (chronological order)
        let mut all_events = timing_data.clone();
        all_events.sort_by_key(|(timestamp, _)| *timestamp);

        // Get the start time from the first event
        let start_time = all_events[0].0;
        let mut prev_time = start_time;

        for (timestamp_us, event_name) in all_events {
            let elapsed_us = timestamp_us.saturating_sub(start_time);
            let delta_us = timestamp_us.saturating_sub(prev_time);
            tracing::debug!(
                target: TRACING_TARGET,
                frame_count,
                "{}: Δ{} μs (+{} μs)",
                event_name,
                delta_us,
                elapsed_us
            );
            prev_time = timestamp_us;
        }

        tracing::debug!(
            target: TRACING_TARGET,
            frame_count,
            "==================================="
        );

        let fps = (total_frame_duration_us as f32 / 1e6).recip();
        tracing::debug!(
            target: TRACING_TARGET,
            fps,
            "Frame rate (instantaneous)"
        )
    }
}
