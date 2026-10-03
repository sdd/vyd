use gst::prelude::*;

pub struct TimingMeta;

impl TimingMeta {
    const META_API_NAME: &'static str = "application/x-vyd-timing";

    /// Create caps for a specific stage and event type
    fn create_timing_caps(stage: &str, event: &str) -> gst::Caps {
        gst::Caps::builder(Self::META_API_NAME)
            .field("stage", stage)
            .field("event", event)
            .build()
    }

    /// Record stage entry time using ReferenceTimestampMeta
    pub fn record_stage_entry(buffer: &mut gst::BufferRef, stage: &str) {
        let now_micros = Self::now();

        // Convert microseconds to nanoseconds for GStreamer's ClockTime
        let now_ns = gst::ClockTime::from_nseconds(now_micros * 1000);

        let caps = Self::create_timing_caps(stage, "entry");

        gst::ReferenceTimestampMeta::add(buffer, &caps, now_ns, gst::ClockTime::NONE);

        tracing::trace!(
            stage = stage,
            timestamp_us = now_micros,
            "Recorded stage entry"
        );
    }

    /// Record stage exit time using ReferenceTimestampMeta
    pub fn record_stage_exit(buffer: &mut gst::BufferRef, stage: &str) {
        let now_micros = Self::now();

        // Convert microseconds to nanoseconds for GStreamer's ClockTime
        let now_ns = gst::ClockTime::from_nseconds(now_micros * 1000);

        let caps = Self::create_timing_caps(stage, "exit");

        gst::ReferenceTimestampMeta::add(buffer, &caps, now_ns, gst::ClockTime::NONE);

        tracing::trace!(
            stage = stage,
            timestamp_us = now_micros,
            "Recorded stage exit"
        );
    }

    pub fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Extract all timing information from a buffer as a chronologically sorted Vec
    /// Returns Vec<(timestamp_us, event_name)> sorted by timestamp (earliest first)
    pub fn extract_timing_data(buffer: &gst::BufferRef) -> Vec<(u64, String)> {
        let mut timing_events = Vec::new();

        // Iterate through all ReferenceTimestampMeta entries
        for meta in buffer.iter_meta::<gst::ReferenceTimestampMeta>() {
            let caps = meta.reference();

            // Check if this is our timing metadata
            if let Some(structure) = caps.structure(0) {
                if structure.name() == Self::META_API_NAME {
                    if let (Ok(stage), Ok(event)) = (
                        structure.get::<&str>("stage"),
                        structure.get::<&str>("event"),
                    ) {
                        let event_name = format!("gst_{}_{}", stage, event);
                        let timestamp_ns = meta.timestamp().nseconds();
                        let timestamp_us = timestamp_ns / 1000; // Convert back to microseconds

                        timing_events.push((timestamp_us, event_name));

                        tracing::trace!(
                            stage = stage,
                            event = event,
                            timestamp_us = timestamp_us,
                            "Extracted timing data"
                        );
                    }
                }
            }
        }

        // Sort by timestamp (chronological order)
        timing_events.sort_by_key(|(timestamp, _)| *timestamp);

        timing_events
    }

    /// Print timing summary for a buffer showing elapsed time from start and delta from previous
    #[allow(dead_code)]
    pub fn print_timing_summary(buffer: &gst::BufferRef) {
        let timing_events = Self::extract_timing_data(buffer);

        if timing_events.is_empty() {
            tracing::info!("No timing data found on buffer");
            return;
        }

        tracing::info!("=== Buffer Processing Timeline ===");

        // Get the start time from the first event
        let start_time = timing_events[0].0;
        let mut prev_time = start_time;

        for (timestamp_us, event_name) in timing_events {
            let elapsed_us = timestamp_us.saturating_sub(start_time);
            let delta_us = timestamp_us.saturating_sub(prev_time);
            tracing::info!("{}: Δ{} μs (+{} μs)", event_name, delta_us, elapsed_us);
            prev_time = timestamp_us;
        }

        tracing::info!("===================================");
    }
}

/// Instrument pipeline with timing metadata attached to buffers
pub fn instrument_pipeline_with_timing_meta(pipeline: &gst::Pipeline) {
    // Cast pipeline to Element for the recursive function
    let pipeline_element = pipeline.upcast_ref::<gst::Element>();
    instrument_element_recursive(pipeline_element);

    // Also set up dynamic instrumentation for bins like decodebin
    setup_dynamic_instrumentation(pipeline_element);

    tracing::debug!("Pipeline instrumentation completed");
}

/// Recursively instrument an element and all its children (if it's a bin)
fn instrument_element_recursive(element: &gst::Element) {
    // First instrument this element itself
    instrument_regular_element(element);

    // If this element is a bin, recursively instrument its children
    if let Ok(bin) = element.clone().downcast::<gst::Bin>() {
        tracing::debug!(
            "Found bin element: {}, instrumenting children",
            element.name()
        );

        // Iterate through all child elements
        let elements: Result<Vec<_>, _> = bin.iterate_elements().into_iter().collect();
        if let Ok(children) = elements {
            for child in children {
                tracing::debug!("Instrumenting child element: {}", child.name());
                // Recursively instrument each child (in case it's also a bin)
                instrument_element_recursive(&child);
            }
        }
    };
}

/// Set up dynamic instrumentation for elements that create children later
fn setup_dynamic_instrumentation(element: &gst::Element) {
    // If this element is a bin, set up pad-added signal
    if let Ok(bin) = element.clone().downcast::<gst::Bin>() {
        tracing::debug!(
            "Setting up dynamic instrumentation for bin: {}",
            element.name()
        );

        bin.connect("element-added", false, |values| {
            if let (Some(_bin), Some(element)) = (
                values[0].get::<gst::Element>().ok(),
                values[1].get::<gst::Element>().ok(),
            ) {
                tracing::debug!("New element added to bin: {}", element.name());
                // Instrument the newly added element
                instrument_element_recursive(&element);
            }
            None
        });

        // Recursively set up for existing children too
        let elements: Result<Vec<_>, _> = bin.iterate_elements().into_iter().collect();
        if let Ok(children) = elements {
            for child in children {
                setup_dynamic_instrumentation(&child);
            }
        }
    };
}

/// Instrument a regular (non-bin) element
fn instrument_regular_element(element: &gst::Element) {
    let elem_name = element.name();
    let stage_name = normalize_stage_name(&elem_name);

    tracing::debug!(
        "Instrumenting element: {} (normalized: {})",
        elem_name,
        stage_name
    );

    // Record stage entry on sink pads
    let sink_pads: Result<Vec<_>, _> = element.iterate_sink_pads().into_iter().collect();
    if let Ok(pads) = sink_pads {
        for pad in pads {
            let stage_entry = stage_name.clone();
            pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
                if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                    let buffer = buffer.make_mut();

                    TimingMeta::record_stage_entry(buffer, &stage_entry);
                }
                gst::PadProbeReturn::Ok
            });
        }
    }

    // Record stage exit on src pads
    let src_pads: Result<Vec<_>, _> = element.iterate_src_pads().into_iter().collect();
    if let Ok(pads) = src_pads {
        for pad in pads {
            let stage_exit = stage_name.clone();
            pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
                if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                    let buffer = buffer.make_mut();

                    TimingMeta::record_stage_exit(buffer, &stage_exit);
                }
                gst::PadProbeReturn::Ok
            });
        }
    }
}

/// Normalize element names to readable stage names
#[allow(dead_code)]
fn normalize_stage_name(element_name: &str) -> String {
    element_name
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}
