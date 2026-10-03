use gst::prelude::*;
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};

/// Check if element uses GPU memory types in its caps
fn is_gpu_element_by_caps(element: &gst::Element) -> bool {
    // Check both src and sink pads
    let src_pads: Vec<_> = element.src_pads().into_iter().collect();
    let sink_pads: Vec<_> = element.sink_pads().into_iter().collect();

    for pad in src_pads.iter().chain(sink_pads.iter()) {
        if let Some(caps) = pad.current_caps().or_else(|| pad.caps()) {
            if caps_contains_gpu_memory(&caps) {
                return true;
            }
        }

        // Also check pad templates for potential GPU support
        if let Some(pad_template) = pad.pad_template() {
            let template_caps = pad_template.caps();
            if caps_contains_gpu_memory(&template_caps) {
                return true;
            }
        }
    }

    false
}

/// Check if caps contain GPU memory features
fn caps_contains_gpu_memory(caps: &gst::Caps) -> bool {
    static GPU_MEMORY_FEATURES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
        [
            "memory:CUDAMemory",
            "memory:GLMemory",
            "memory:DMABuf",
            "memory:VASurface",
            "memory:D3D11Memory",
            "memory:OpenCLMemory",
            "memory:VulkanMemory",
            "memory:KMSMemory",
        ]
        .into_iter()
        .collect()
    });

    for i in 0..caps.size() {
        if let Some(_structure) = caps.structure(i) {
            if let Some(features) = caps.features(i) {
                for feature_idx in 0..features.size() {
                    if let Some(feature) = features.nth(feature_idx) {
                        if GPU_MEMORY_FEATURES.contains(feature.as_str()) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

/// Get processing type for a pipeline by analyzing all its elements
pub fn analyze_pipeline_processing_types(pipeline: &gst::Pipeline) -> HashMap<String, bool> {
    let mut processing_types = HashMap::new();

    // if let Ok(iterator) = pipeline.iterate_recurse() {
    for element in pipeline.iterate_recurse().into_iter().filter_map(|item| {
        item.ok()
            .and_then(|value| value.downcast::<gst::Element>().ok())
    }) {
        if let Some(factory) = element.factory() {
            let element_name = factory.name().to_string();
            let is_gpu = is_gpu_element_by_caps(&element);
            processing_types.insert(element_name.clone(), is_gpu);

            tracing::info!(
                element_name,
                is_gpu,
                "Detected GStreamer element processing type"
            );
        }
    }
    // }

    processing_types
}
