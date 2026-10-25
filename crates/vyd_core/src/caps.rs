//! Typed stage port declarations. A declaration describes acceptable formats;
//! actual frame storage is still checked at the runtime boundary.
use crate::VydError;
use crate::frame::PixelFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameMemory {
    /// Processor-visible rows are tightly packed.
    CpuPacked,
    /// Processor-visible rows may have a positive padded stride; packed rows
    /// are also valid. GST media caps alone cannot describe the actual pitch.
    CpuRowStrided,
    /// Requires a GPU-specific adapter and resource contract.
    Wgpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramePortSpec {
    pixel_format: PixelFormat,
    memory: FrameMemory,
}

impl FramePortSpec {
    /// Declare a capability, not a conversion or an allocated frame. The
    /// selected runtime must still validate actual storage at each boundary.
    pub const fn new(pixel_format: PixelFormat, memory: FrameMemory) -> Self {
        Self {
            pixel_format,
            memory,
        }
    }

    pub const fn luma8_cpu_packed() -> Self {
        Self::new(PixelFormat::Luma8, FrameMemory::CpuPacked)
    }

    pub const fn pixel_format(self) -> PixelFormat {
        self.pixel_format
    }
    pub const fn memory(self) -> FrameMemory {
        self.memory
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramePortSpecs {
    specs: Vec<FramePortSpec>,
}

impl FramePortSpecs {
    pub fn one(spec: FramePortSpec) -> Self {
        Self { specs: vec![spec] }
    }

    pub fn try_from_vec(specs: Vec<FramePortSpec>) -> Result<Self, VydError> {
        let mut values = Vec::new();
        for spec in specs {
            if !values.contains(&spec) {
                values.push(spec);
            }
        }
        if values.is_empty() {
            return Err(VydError::GeneralPluginError(
                "frame port needs at least one format".into(),
            ));
        }
        Ok(Self { specs: values })
    }

    pub fn with_alternative(mut self, spec: FramePortSpec) -> Self {
        if !self.specs.contains(&spec) {
            self.specs.push(spec);
        }
        self
    }

    pub fn as_slice(&self) -> &[FramePortSpec] {
        &self.specs
    }
}

impl Default for FramePortSpecs {
    fn default() -> Self {
        Self::one(FramePortSpec::luma8_cpu_packed())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameFormatRelation {
    pub input: FramePortSpec,
    pub output: FramePortSpec,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessorPortSpecs {
    relations: Vec<FrameFormatRelation>,
}

impl ProcessorPortSpecs {
    /// Every alternative preserves its format. The output is constrained by
    /// the selected input, rather than chosen from an unrelated output list.
    pub fn passthrough(specs: FramePortSpecs) -> Self {
        Self {
            relations: specs
                .as_slice()
                .iter()
                .copied()
                .map(|spec| FrameFormatRelation {
                    input: spec,
                    output: spec,
                })
                .collect(),
        }
    }

    pub fn transforming(relations: Vec<FrameFormatRelation>) -> Result<Self, VydError> {
        let mut values = Vec::new();
        for relation in relations {
            if !values.contains(&relation) {
                values.push(relation);
            }
        }
        if values.is_empty() {
            return Err(VydError::GeneralPluginError(
                "processor needs at least one input/output relation".into(),
            ));
        }
        Ok(Self { relations: values })
    }

    pub fn relations(&self) -> &[FrameFormatRelation] {
        &self.relations
    }
}

impl Default for ProcessorPortSpecs {
    fn default() -> Self {
        Self::passthrough(FramePortSpecs::default())
    }
}

#[derive(Clone, Debug)]
pub struct PipelinePortDeclarations {
    pub source: FramePortSpecs,
    pub processors: Vec<ProcessorPortSpecs>,
    pub sink: FramePortSpecs,
}
