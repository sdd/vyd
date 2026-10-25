//! GStreamer-specific configuration of a Vyd run.
use crate::runtime::{FrameCorrespondence, NativeGeometry, NativeNode};
use serde::Deserialize;
#[cfg(feature = "gpu-wgpu")]
use std::sync::Arc;
use vyd_core::pipeline::PipelineContext;
use vyd_core::VydError;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeConfig {
    boundary: usize,
    pipeline: String,
    bridge_capacity: Option<usize>,
    correspondence: Option<String>,
    input_size: Option<[u32; 2]>,
    output_size: Option<[u32; 2]>,
    annotation_policy: Option<String>,
}

fn invalid(message: impl std::fmt::Display) -> VydError {
    VydError::GeneralPluginError(format!("GStreamer node configuration: {message}"))
}

/// A temporary config convention until stages declare capabilities directly.
/// Any stage with `gpu = true` requests a GPU, regardless of its type name.
pub fn gpu_stage_requested(config: &toml::Value) -> bool {
    config
        .get("stages")
        .and_then(toml::Value::as_array)
        .is_some_and(|stages| {
            stages
                .iter()
                .any(|stage| stage.get("gpu").and_then(toml::Value::as_bool) == Some(true))
        })
}

/// Build native sections from GStreamer pipeline descriptions. Geometry
/// changes must be declared and invalidate all input-grid annotations.
pub fn nodes_from_config(config: &toml::Value) -> Result<Vec<NativeNode>, VydError> {
    let Some(raw_nodes) = config.get("gstreamer_nodes") else {
        return Ok(Vec::new());
    };
    let nodes = raw_nodes
        .as_array()
        .ok_or_else(|| invalid("gstreamer_nodes must be an array of tables"))?;
    nodes
        .iter()
        .map(|value| {
            let node: NodeConfig = value
                .clone()
                .try_into()
                .map_err(|error| invalid(format!("invalid node: {error}")))?;
            if node.pipeline.trim().is_empty() {
                return Err(invalid("pipeline description must not be empty"));
            }
            let mut section = NativeNode::description(node.boundary, node.pipeline);
            match (node.bridge_capacity, node.correspondence.as_deref()) {
                (Some(capacity), Some("one_to_one_ordered")) => {
                    section =
                        section.with_fifo_bridge(capacity, FrameCorrespondence::OneToOneOrdered)?;
                }
                (None, None) => {}
                _ => {
                    return Err(invalid(
                        "bridge_capacity requires correspondence = 'one_to_one_ordered'",
                    ))
                }
            }
            match (node.input_size, node.output_size) {
                (Some(input), Some(output)) => {
                    let geometry =
                        NativeGeometry::new((input[0], input[1]), (output[0], output[1]))?;
                    section = section.with_packed_gray8_geometry(geometry);
                    match node.annotation_policy.as_deref() {
                        Some("drop_all") => {
                            section = section.with_annotation_update(|annotations, _| {
                                annotations.clear();
                                Ok(())
                            });
                        }
                        None if input == output => {}
                        None => {
                            return Err(invalid(
                                "geometry changes require annotation_policy = 'drop_all'",
                            ))
                        }
                        _ => return Err(invalid("unsupported annotation_policy")),
                    }
                }
                (None, None) if node.annotation_policy.is_none() => {}
                _ => {
                    return Err(invalid(
                        "input_size and output_size must be declared together",
                    ))
                }
            }
            Ok(section)
        })
        .collect()
}

/// Provision the GPU for the whole run before the stage session starts.
pub fn configure_gpu(config: &toml::Value, context: &mut PipelineContext) -> Result<(), VydError> {
    if !gpu_stage_requested(config) {
        return Ok(());
    }
    #[cfg(feature = "gpu-wgpu")]
    {
        let gpu =
            vyd_core::gpu::WgpuContext::request_blocking(Default::default()).map_err(invalid)?;
        tracing::info!(adapter = %gpu.adapter_info().name, backend = ?gpu.adapter_info().backend, "GPU adapter selected");
        context.set_wgpu_context(Arc::new(gpu)).map_err(invalid)?;
        Ok(())
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        let _ = context;
        Err(invalid("GPU stages require the gpu-wgpu feature"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_request_is_stage_name_independent() {
        let config: toml::Value =
            toml::from_str("[[stages]]\nstage_type = 'unrelated::Stage'\ngpu = true\n").unwrap();
        assert!(gpu_stage_requested(&config));
    }

    #[test]
    fn geometry_change_requires_invalidation_policy() {
        let config: toml::Value = toml::from_str(
            "[[gstreamer_nodes]]\nboundary = 0\npipeline = 'videoscale'\ninput_size = [4, 2]\noutput_size = [8, 2]\n",
        )
        .unwrap();
        assert!(nodes_from_config(&config).is_err());
    }
}
