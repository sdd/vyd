//! Native single-input/single-output sections in the shared GST graph.
use super::{bridge, error, Shared};
use gst::glib::{self, prelude::*, value::ToValue};
use gst::prelude::*;
use std::sync::Arc;
use vyd_core::VydError;

use super::transport::NativeFrameAnnotations;

/// Runs on the validated native output, before its context reaches a Rust stage.
/// The callback receives the negotiated output size, not the input frame size.
pub type AnnotationUpdate = Arc<
    dyn for<'a> Fn(&mut NativeFrameAnnotations<'a>, NativeGeometry) -> Result<(), VydError>
        + Send
        + Sync,
>;

/// The boundary after the source is 0; the boundary after processor `n` is
/// `n + 1`. Nodes at the same boundary are inserted in declaration order.
pub struct NativeNode {
    pub boundary: usize,
    pub section: NativeSection,
    bridge_capacity: Option<usize>,
    geometry: Option<NativeGeometry>,
    annotation_update: Option<AnnotationUpdate>,
}

/// A fixed packed GRAY8 geometry transition. Both sides must use one
/// SystemMemory plane with zero offset and a stride equal to the width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeGeometry {
    pub input: (u32, u32),
    pub output: (u32, u32),
}

impl NativeGeometry {
    pub fn new(input: (u32, u32), output: (u32, u32)) -> Result<Self, VydError> {
        if [input.0, input.1, output.0, output.1]
            .iter()
            .any(|&n| n == 0 || n > i32::MAX as u32)
        {
            return Err(error(
                "native geometry dimensions must be positive GST integers",
            ));
        }
        Ok(Self { input, output })
    }
}

/// A FIFO bridge is valid only when each admitted input produces exactly one
/// output in the same order. This is a caller-declared contract, not something
/// caps or buffer counts can prove.
pub enum FrameCorrespondence {
    OneToOneOrdered,
    MayDrop,
    MayDuplicate,
    MayReorder,
    MayAggregate,
}

impl NativeNode {
    pub fn factory(
        boundary: usize,
        name: impl Into<String>,
        properties: Vec<NativeProperty>,
    ) -> Self {
        Self {
            boundary,
            section: NativeSection::Factory {
                name: name.into(),
                properties,
            },
            bridge_capacity: None,
            geometry: None,
            annotation_update: None,
        }
    }

    pub fn element(boundary: usize, element: gst::Element) -> Self {
        Self {
            boundary,
            section: NativeSection::Element(element),
            bridge_capacity: None,
            geometry: None,
            annotation_update: None,
        }
    }

    /// Parse a chain as one bin with ghosted unlinked input/output pads.
    pub fn description(boundary: usize, description: impl Into<String>) -> Self {
        Self {
            boundary,
            section: NativeSection::Description(description.into()),
            bridge_capacity: None,
            geometry: None,
            annotation_update: None,
        }
    }

    pub fn with_fifo_bridge(
        mut self,
        capacity: usize,
        contract: FrameCorrespondence,
    ) -> Result<Self, VydError> {
        if !matches!(contract, FrameCorrespondence::OneToOneOrdered) {
            return Err(error(
                "FIFO context bridge requires one-to-one ordered frame correspondence",
            ));
        }
        if capacity == 0 {
            return Err(error("FIFO context bridge capacity must be positive"));
        }
        self.bridge_capacity = Some(capacity);
        Ok(self)
    }

    pub fn with_packed_gray8_geometry(mut self, geometry: NativeGeometry) -> Self {
        self.geometry = Some(geometry);
        self
    }

    pub fn with_annotation_update<F>(mut self, update: F) -> Self
    where
        F: for<'a> Fn(&mut NativeFrameAnnotations<'a>, NativeGeometry) -> Result<(), VydError>
            + Send
            + Sync
            + 'static,
    {
        self.annotation_update = Some(Arc::new(update));
        self
    }

    pub(super) fn geometry(&self) -> Option<NativeGeometry> {
        self.geometry
    }
}

pub enum NativeSection {
    Factory {
        name: String,
        properties: Vec<NativeProperty>,
    },
    Element(gst::Element),
    Description(String),
}

pub struct NativeProperty {
    pub name: String,
    pub value: NativePropertyValue,
}

impl NativeProperty {
    pub fn new(name: impl Into<String>, value: NativePropertyValue) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

pub enum NativePropertyValue {
    Bool(bool),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
}

impl NativePropertyValue {
    fn to_value(&self) -> glib::Value {
        match self {
            Self::Bool(value) => value.to_value(),
            Self::I32(value) => value.to_value(),
            Self::U32(value) => value.to_value(),
            Self::I64(value) => value.to_value(),
            Self::U64(value) => value.to_value(),
            Self::F32(value) => value.to_value(),
            Self::F64(value) => value.to_value(),
            Self::String(value) => value.to_value(),
        }
    }
}

impl NativeSection {
    pub(super) fn build(self) -> Result<gst::Element, VydError> {
        let element = match self {
            Self::Factory { name, properties } => {
                let element = gst::ElementFactory::make(&name).build().map_err(error)?;
                for property in properties {
                    let Some(spec) = element.find_property(&property.name) else {
                        return Err(error(format!(
                            "native factory {name} has no property {}",
                            property.name
                        )));
                    };
                    if !spec.flags().contains(glib::ParamFlags::WRITABLE)
                        || spec.flags().contains(glib::ParamFlags::CONSTRUCT_ONLY)
                    {
                        return Err(error(format!(
                            "native factory {name} property {} is not writable after construction",
                            property.name
                        )));
                    }
                    let value = property.value.to_value();
                    if value.type_() != spec.value_type() {
                        return Err(error(format!(
                            "native factory {name} property {} expects {:?}, got {:?}",
                            property.name,
                            spec.value_type(),
                            value.type_()
                        )));
                    }
                    element.set_property_from_value(&property.name, &value);
                }
                element
            }
            Self::Element(element) => element,
            Self::Description(description) => gst::parse_bin_from_description(&description, true)
                .map_err(error)?
                .upcast(),
        };
        let sinks = element.sink_pads();
        let sources = element.src_pads();
        if sinks.len() != 1 || sources.len() != 1 {
            return Err(error(format!("native section {} needs exactly one existing sink pad and one existing source pad; found {} and {}", element.name(), sinks.len(), sources.len())));
        }
        Ok(element)
    }
}

fn reject_dropping_queue(element: &gst::Element) -> Result<(), VydError> {
    if element
        .factory()
        .is_some_and(|factory| factory.name() == "queue")
    {
        let leaky = element.property_value("leaky");
        let value = leaky.get::<&glib::EnumValue>().map_err(error)?;
        if value.value() != 0 {
            return Err(error(format!(
                "dropping queue {} must be outside a FIFO bridged section",
                element.name()
            )));
        }
    }
    if let Some(bin) = element.downcast_ref::<gst::Bin>() {
        for child in bin.children() {
            reject_dropping_queue(&child)?;
        }
    }
    Ok(())
}

pub(super) fn place(
    nodes: Vec<NativeNode>,
    processor_count: usize,
) -> Result<Vec<Vec<NativeNode>>, VydError> {
    let mut boundaries = (0..=processor_count)
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for node in nodes {
        let Some(sections) = boundaries.get_mut(node.boundary) else {
            return Err(error(format!(
                "native section boundary {} exceeds processor count {processor_count}",
                node.boundary
            )));
        };
        sections.push(node);
    }
    Ok(boundaries)
}

pub(super) fn append(
    graph: &gst::Pipeline,
    previous: &mut gst::Element,
    sections: Vec<NativeNode>,
    shared: &Shared,
    bridges: &mut Vec<bridge::BridgeHandle>,
) -> Result<(), VydError> {
    for node in sections {
        let geometry = node.geometry;
        let annotation_update = node.annotation_update;
        if annotation_update.is_some() && geometry.is_none() {
            return Err(error(
                "native annotation update requires declared packed GRAY8 geometry",
            ));
        }
        let element = node.section.build()?;
        if node.bridge_capacity.is_some() {
            reject_dropping_queue(&element)?;
        }
        graph.add(&element).map_err(error)?;
        if let Some(capacity) = node.bridge_capacity {
            bridges.push(bridge::install(
                &element,
                capacity,
                shared.clone(),
                geometry,
                annotation_update,
            )?);
        } else if let Some(geometry) = geometry {
            bridge::install_geometry_validation(
                &element,
                geometry,
                shared.clone(),
                annotation_update,
            )?;
        }
        previous.link(&element).map_err(error)?;
        *previous = element;
    }
    Ok(())
}
