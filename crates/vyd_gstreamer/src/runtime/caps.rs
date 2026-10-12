//! Translate typed Vyd port alternatives into GST caps. GST performs the
//! actual intersection/acceptance when the graph links and streams.
use super::error;
use gst::prelude::*;
use vyd::caps::{
    FrameMemory, FramePortSpec, FramePortSpecs, PipelinePortDeclarations, ProcessorPortSpecs,
};
use vyd::frame::PixelFormat;
use vyd::VydError;

pub(super) fn negotiated_gray8(pad: &gst::Pad) -> Result<gst_video::VideoInfo, VydError> {
    let caps = pad
        .current_caps()
        .ok_or_else(|| error("pad has no negotiated video caps"))?;
    gray8_info(&caps)
}

fn gray8_info(caps: &gst::CapsRef) -> Result<gst_video::VideoInfo, VydError> {
    if caps.size() != 1 || !caps.is_fixed() {
        return Err(error(
            "native boundary requires one fixed GRAY8 caps structure",
        ));
    }
    let features = caps
        .features(0)
        .ok_or_else(|| error("caps have no memory features"))?;
    if features.is_any()
        || !(features.is_empty()
            || (features.size() == 1 && features.contains("memory:SystemMemory")))
    {
        return Err(error("native boundary requires SystemMemory caps"));
    }
    let info = gst_video::VideoInfo::from_caps(&caps).map_err(error)?;
    if info.format() != gst_video::VideoFormat::Gray8 {
        return Err(error("native boundary requires GRAY8 format"));
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyd::caps::FrameFormatRelation;

    fn gpu(format: PixelFormat) -> FramePortSpec {
        FramePortSpec::new(format, FrameMemory::Wgpu)
    }

    fn plan(
        processors: Vec<ProcessorPortSpecs>,
        native_boundaries: &[bool],
        gpu_context_available: bool,
    ) -> Result<CapsPlan, VydError> {
        CapsPlan::new(
            PipelinePortDeclarations {
                source: FramePortSpecs::default(),
                processors,
                sink: FramePortSpecs::default(),
            },
            native_boundaries,
            gpu_context_available,
        )
    }

    fn gpu_passthrough(format: PixelFormat) -> ProcessorPortSpecs {
        ProcessorPortSpecs::passthrough(FramePortSpecs::one(gpu(format)))
    }

    #[test]
    fn native_caps_require_fixed_gray8_system_memory() {
        gst::init().unwrap();
        for (format, features, valid) in [
            ("GRAY8", None, true),
            ("RGB", None, false),
            ("GRAY8", Some("memory:GLMemory"), false),
            ("GRAY8", Some("memory:SystemMemory"), true),
        ] {
            let mut caps = gst::Caps::builder("video/x-raw")
                .field("format", format)
                .field("width", 8i32)
                .field("height", 2i32)
                .build();
            if let Some(feature) = features {
                caps.get_mut()
                    .unwrap()
                    .set_features(0, Some(gst::CapsFeatures::new([feature])));
            }
            assert_eq!(gray8_info(&caps).is_ok(), valid, "{format} {features:?}");
        }
        let unfixed = gst::Caps::builder("video/x-raw")
            .field("format", "GRAY8")
            .field("width", gst::IntRange::<i32>::new(4, 8))
            .field("height", 2i32)
            .build();
        assert!(gray8_info(&unfixed).is_err());
    }

    #[test]
    fn selects_luma8_and_nv12_gpu_relations_without_claiming_gpu_gst_caps() {
        gst::init().unwrap();
        let conversion = ProcessorPortSpecs::transforming(vec![FrameFormatRelation {
            input: gpu(PixelFormat::Luma8),
            output: gpu(PixelFormat::Nv12),
        }])
        .unwrap();
        let selected = plan(
            vec![gpu_passthrough(PixelFormat::Luma8), conversion],
            &[false, false, false],
            true,
        )
        .unwrap();
        assert_eq!(
            selected.selections,
            vec![
                ProcessorSelection::Gpu {
                    input: PixelFormat::Luma8,
                    output: PixelFormat::Luma8,
                },
                ProcessorSelection::Gpu {
                    input: PixelFormat::Luma8,
                    output: PixelFormat::Nv12,
                },
            ]
        );
        assert_eq!(selected.gpu_segments, vec![GpuSegment { start: 0, end: 2 }]);
        for caps in [&selected.source, &selected.sink].into_iter().chain(
            selected
                .processors
                .iter()
                .flat_map(|pair| [&pair.0, &pair.1]),
        ) {
            assert_eq!(caps.size(), 1);
            assert_eq!(
                caps.structure(0).unwrap().get::<&str>("format").unwrap(),
                "GRAY8"
            );
            let features = caps.features(0).unwrap();
            assert!(
                features.is_empty()
                    || (features.size() == 1 && features.contains("memory:SystemMemory"))
            );
        }
    }

    #[test]
    fn cpu_stages_and_native_boundaries_split_maximal_gpu_segments() {
        gst::init().unwrap();
        let selected = plan(
            vec![
                gpu_passthrough(PixelFormat::Nv12),
                gpu_passthrough(PixelFormat::Luma8),
                ProcessorPortSpecs::default(),
                gpu_passthrough(PixelFormat::Nv12),
                gpu_passthrough(PixelFormat::Luma8),
            ],
            &[true, false, false, false, true, true],
            true,
        )
        .unwrap();
        assert_eq!(
            selected.gpu_segments,
            vec![
                GpuSegment { start: 0, end: 2 },
                GpuSegment { start: 3, end: 4 },
                GpuSegment { start: 4, end: 5 },
            ]
        );
        assert_eq!(selected.selections[2], ProcessorSelection::Cpu);
    }

    #[test]
    fn missing_gpu_context_and_ambiguous_alternatives_fail_before_execution() {
        gst::init().unwrap();
        let missing = plan(
            vec![gpu_passthrough(PixelFormat::Nv12)],
            &[false, false],
            false,
        )
        .err()
        .unwrap();
        assert!(missing.to_string().contains("no GPU context"), "{missing}");
        let alternatives = ProcessorPortSpecs::passthrough(
            FramePortSpecs::one(gpu(PixelFormat::Luma8)).with_alternative(gpu(PixelFormat::Nv12)),
        );
        let ambiguous = plan(vec![alternatives], &[false, false], true)
            .err()
            .unwrap();
        assert!(ambiguous.to_string().contains("ambiguous"), "{ambiguous}");

        let cpu_or_gpu = ProcessorPortSpecs::passthrough(
            FramePortSpecs::default().with_alternative(gpu(PixelFormat::Luma8)),
        );
        assert_eq!(
            plan(vec![cpu_or_gpu.clone()], &[false, false], false)
                .unwrap()
                .selections,
            vec![ProcessorSelection::Cpu]
        );
        assert!(plan(vec![cpu_or_gpu], &[false, false], true)
            .err()
            .unwrap()
            .to_string()
            .contains("ambiguous"));
    }

    #[test]
    fn rejects_unsupported_cpu_layout_mixed_memory_and_external_ports() {
        gst::init().unwrap();
        let strided = FramePortSpec::new(PixelFormat::Luma8, FrameMemory::CpuRowStrided);
        assert!(matches!(
            plan(
                vec![ProcessorPortSpecs::passthrough(FramePortSpecs::one(
                    strided
                ))],
                &[false, false],
                true,
            ),
            Err(VydError::NotYetImplemented { .. })
        ));
        let mixed = ProcessorPortSpecs::transforming(vec![FrameFormatRelation {
            input: FramePortSpec::luma8_cpu_packed(),
            output: gpu(PixelFormat::Luma8),
        }])
        .unwrap();
        assert!(plan(vec![mixed], &[false, false], true)
            .err()
            .unwrap()
            .to_string()
            .contains("mixed memory"));
        for invalid in [
            FramePortSpec::new(PixelFormat::Nv12, FrameMemory::CpuPacked),
            gpu(PixelFormat::Luma8),
        ] {
            let result = CapsPlan::new(
                PipelinePortDeclarations {
                    source: FramePortSpecs::one(invalid),
                    processors: Vec::new(),
                    sink: FramePortSpecs::default(),
                },
                &[false],
                true,
            );
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("packed CPU Luma8"));
        }
        assert!(plan(Vec::new(), &[], false)
            .err()
            .unwrap()
            .to_string()
            .contains("boundary count"));
    }
}

pub(super) struct CapsPlan {
    pub source: gst::Caps,
    pub processors: Vec<(gst::Caps, gst::Caps)>,
    pub sink: gst::Caps,
    /// The selected relation for each Rust processor. A GPU relation is an
    /// internal frame contract; it never becomes a GST memory feature.
    pub selections: Vec<ProcessorSelection>,
    /// Half-open stage ranges. A native node at boundary `n` separates stages
    /// `n - 1` and `n`, even when both select GPU relations.
    pub gpu_segments: Vec<GpuSegment>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProcessorSelection {
    Cpu,
    Gpu {
        input: PixelFormat,
        output: PixelFormat,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GpuSegment {
    pub start: usize,
    pub end: usize,
}

impl CapsPlan {
    pub fn new(
        declarations: PipelinePortDeclarations,
        native_boundaries: &[bool],
        gpu_context_available: bool,
    ) -> Result<Self, VydError> {
        if native_boundaries.len() != declarations.processors.len() + 1 {
            return Err(error(
                "native boundary count does not match processor count",
            ));
        }
        // TODO(Frame v2): Select packed/strided alternatives only after the
        // graph has an explicit layout converter and guarded frame transport.
        // GST media caps do not encode the concrete row stride.
        let unsupported = declarations
            .source
            .as_slice()
            .iter()
            .copied()
            .chain(declarations.processors.iter().flat_map(|stage| {
                stage
                    .relations()
                    .iter()
                    .flat_map(|relation| [relation.input, relation.output])
            }))
            .chain(declarations.sink.as_slice().iter().copied())
            .any(|spec| spec.memory() == FrameMemory::CpuRowStrided);
        if unsupported {
            return Err(VydError::NotYetImplemented {
                feature: "packed/padded CPU layout negotiation",
                detail: "a CpuRowStrided port needs an explicit converter and frame adapter".into(),
            });
        }
        let source = external_gray8(&declarations.source, "source")?;
        let sink = external_gray8(&declarations.sink, "sink")?;
        let selections = declarations
            .processors
            .iter()
            .enumerate()
            .map(|(index, ports)| select_processor(index, ports, gpu_context_available))
            .collect::<Result<Vec<_>, _>>()?;
        let processors = selections
            .iter()
            .map(|_| {
                let caps = gray8_caps();
                (caps.clone(), caps)
            })
            .collect();
        let gpu_segments = gpu_segments(&selections, native_boundaries);
        Ok(Self {
            source,
            processors,
            sink,
            selections,
            gpu_segments,
        })
    }
}

fn gray8_caps() -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .build()
}

fn external_gray8(specs: &FramePortSpecs, role: &str) -> Result<gst::Caps, VydError> {
    if specs
        .as_slice()
        .contains(&FramePortSpec::luma8_cpu_packed())
    {
        Ok(gray8_caps())
    } else {
        Err(error(format!(
            "{role} requires packed CPU Luma8 for the GST SystemMemory GRAY8 boundary; a GPU adapter cannot expose GPU memory on a GST pad"
        )))
    }
}

fn select_processor(
    index: usize,
    ports: &ProcessorPortSpecs,
    gpu_context_available: bool,
) -> Result<ProcessorSelection, VydError> {
    let mut cpu = false;
    let mut gpu = Vec::new();
    for relation in ports.relations() {
        match (relation.input.memory(), relation.output.memory()) {
            (FrameMemory::Wgpu, FrameMemory::Wgpu) => {
                gpu.push((
                    relation.input.pixel_format(),
                    relation.output.pixel_format(),
                ));
            }
            (FrameMemory::CpuPacked, FrameMemory::CpuPacked)
                if relation.input == FramePortSpec::luma8_cpu_packed()
                    && relation.output == relation.input =>
            {
                cpu = true;
            }
            (FrameMemory::CpuPacked, FrameMemory::CpuPacked) => {}
            _ => {
                return Err(error(format!(
                    "processor {index} has an unsupported mixed memory or CPU layout relation"
                )));
            }
        }
    }
    if !gpu_context_available {
        if cpu {
            return Ok(ProcessorSelection::Cpu);
        }
        if !gpu.is_empty() {
            return Err(error(format!(
                "processor {index} selects Wgpu but no GPU context is configured"
            )));
        }
    }
    match (cpu, gpu.as_slice()) {
        (true, []) => Ok(ProcessorSelection::Cpu),
        (false, [(input, output)]) => Ok(ProcessorSelection::Gpu {
            input: *input,
            output: *output,
        }),
        (true, _) | (false, [_, ..]) => Err(error(format!(
            "processor {index} has ambiguous CPU/GPU or GPU format alternatives"
        ))),
        (false, []) => Err(error(format!(
            "processor {index} has no supported relation; format-changing CPU processors require an explicit conversion adapter and GST boundary is packed GRAY8"
        ))),
    }
}

fn gpu_segments(selections: &[ProcessorSelection], native_boundaries: &[bool]) -> Vec<GpuSegment> {
    let mut segments = Vec::new();
    let mut start = None;
    for (index, selection) in selections.iter().enumerate() {
        if matches!(selection, ProcessorSelection::Gpu { .. }) {
            if start.is_some() && native_boundaries[index] {
                segments.push(GpuSegment {
                    start: start.take().unwrap(),
                    end: index,
                });
            }
            start.get_or_insert(index);
        } else if let Some(start) = start.take() {
            segments.push(GpuSegment { start, end: index });
        }
    }
    if let Some(start) = start {
        segments.push(GpuSegment {
            start,
            end: selections.len(),
        });
    }
    segments
}

pub(super) fn filter(name: &str, caps: &gst::Caps) -> Result<gst::Element, VydError> {
    gst::ElementFactory::make("capsfilter")
        .name(name)
        .property("caps", caps)
        .build()
        .map_err(error)
}
