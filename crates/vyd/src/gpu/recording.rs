//! Managed compute recording over retained frame plane leases.
//!
//! A recording owns its read leases and exclusive output reservations. Plane
//! IDs and bind groups are meaningful only within that recording. Dropping it
//! without submission discards the candidates and leaves every frame intact.
//!
//! Safe callers cannot extract a pooled frame texture or the encoder:
//! ```compile_fail
//! use vyd::frame::GpuFrameRead;
//! fn raw_plane(read: &GpuFrameRead) { let _ = read.plane(0); }
//! ```
//! ```compile_fail
//! use vyd::gpu::recording::GpuRecording;
//! fn raw_encoder(recording: &GpuRecording<'_>) { let _ = &recording.encoder; }
//! ```

use super::pool::GpuResourceKey;
use super::{DeviceKey, WgpuContext};
use crate::frame::layout::FrameSpec;
use crate::frame::{GpuFrameRead, GpuFrameWrite, PixelFormat};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

static NEXT_RECORDING_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_PROGRAM_ID: AtomicU64 = AtomicU64::new(1);
// wgpu error scopes use a device-local stack. Serialize Vyd program builds so
// two callers cannot pop each other's creation errors on a shared device.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaneAccess {
    Read,
    Write,
}

/// One WGSL group-0 plane binding. A read is `texture_2d<u32>` and a write is
/// `texture_storage_2d<rgba8uint, write>`; all physical texels are RGBA8Uint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBindingSpec {
    pub binding: u32,
    pub access: PlaneAccess,
    pub format: PixelFormat,
    pub plane: usize,
}

impl GpuBindingSpec {
    pub const fn read(binding: u32, format: PixelFormat, plane: usize) -> Self {
        Self {
            binding,
            access: PlaneAccess::Read,
            format,
            plane,
        }
    }
    pub const fn write(binding: u32, format: PixelFormat, plane: usize) -> Self {
        Self {
            binding,
            access: PlaneAccess::Write,
            format,
            plane,
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum GpuRecordingError {
    #[error("GPU recording device {actual:?} differs from {expected:?}")]
    WrongDevice {
        expected: DeviceKey,
        actual: DeviceKey,
    },
    #[error("frame specification differs from the recording's expected specification")]
    SpecMismatch,
    #[error("GPU plane {0} does not exist")]
    MissingPlane(usize),
    #[error("GPU plane resource has the wrong format, extent, device or usage")]
    InvalidPlaneResource,
    #[error("GPU plane ID belongs to another recording")]
    WrongRecording,
    #[error("GPU plane binding direction does not match the shader declaration")]
    WrongDirection,
    #[error("GPU plane logical format or plane index does not match the shader declaration")]
    WrongPlaneSpec,
    #[error("duplicate GPU binding number {0}")]
    DuplicateBinding(u32),
    #[error("GPU binding {0} is missing or undeclared")]
    BindingMismatch(u32),
    #[error("GPU bind group belongs to another program")]
    WrongProgram,
    #[error("GPU bind group ID belongs to another recording")]
    WrongBindGroup,
    #[error("GPU dispatch dimensions must all be nonzero")]
    EmptyDispatch,
    #[error("GPU compute program {stage} failed validation: {message}")]
    ProgramCreation {
        stage: &'static str,
        message: String,
    },
}

/// A reusable, device-specific compute pipeline with one managed plane group.
/// Invalid WGSL and shader/layout mismatch return a structured error at
/// construction. The schema is checked again for each recording-local group.
pub struct GpuComputeProgram {
    id: u64,
    device: DeviceKey,
    bindings: Vec<GpuBindingSpec>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl GpuComputeProgram {
    pub fn new(
        context: &WgpuContext,
        wgsl: &str,
        entry_point: &str,
        bindings: &[GpuBindingSpec],
    ) -> Result<Self, GpuRecordingError> {
        let mut seen = HashSet::new();
        for spec in bindings {
            if !seen.insert(spec.binding) {
                return Err(GpuRecordingError::DuplicateBinding(spec.binding));
            }
            if spec.plane >= spec.format.plane_count() {
                return Err(GpuRecordingError::MissingPlane(spec.plane));
            }
        }
        let entries: Vec<_> = bindings
            .iter()
            .map(|spec| wgpu::BindGroupLayoutEntry {
                binding: spec.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match spec.access {
                    PlaneAccess::Read => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    PlaneAccess::Write => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                count: None,
            })
            .collect();
        let _creation_guard = super::DEVICE_ERROR_SCOPE_LOCK.lock().map_err(|_| {
            GpuRecordingError::ProgramCreation {
                stage: "validation lock",
                message: "program creation validation lock is poisoned".into(),
            }
        })?;
        let layout = capture_creation(context, "bind group layout", || {
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("vyd managed plane group"),
                    entries: &entries,
                })
        })?;
        let pipeline_layout = capture_creation(context, "pipeline layout", || {
            context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("vyd managed compute layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                })
        })?;
        let shader = capture_creation(context, "shader module", || {
            context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("vyd managed compute shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                })
        })?;
        let pipeline = capture_creation(context, "compute pipeline", || {
            context
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("vyd managed compute pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })?;
        Ok(Self {
            id: NEXT_PROGRAM_ID.fetch_add(1, Ordering::Relaxed),
            device: context.device_key(),
            bindings: bindings.to_vec(),
            layout,
            pipeline,
        })
    }
}

/// wgpu 24 reports creation failures through device error scopes, not the
/// handle-returning creation methods. Pop every filter before returning so an
/// error cannot leak into another recording's validation scope.
fn capture_creation<T>(
    context: &WgpuContext,
    stage: &'static str,
    create: impl FnOnce() -> T,
) -> Result<T, GpuRecordingError> {
    let device = context.device();
    device.push_error_scope(wgpu::ErrorFilter::Internal);
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = create();
    let validation = pollster::block_on(device.pop_error_scope());
    let out_of_memory = pollster::block_on(device.pop_error_scope());
    let internal = pollster::block_on(device.pop_error_scope());
    match validation.or(out_of_memory).or(internal) {
        Some(error) => Err(GpuRecordingError::ProgramCreation {
            stage,
            message: error.to_string(),
        }),
        None => Ok(value),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuReadPlaneId {
    recording: u64,
    frame: usize,
    plane: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuWritePlaneId {
    recording: u64,
    frame: usize,
    plane: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPlaneBinding {
    Read(GpuReadPlaneId),
    Write(GpuWritePlaneId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBindGroupId {
    recording: u64,
    index: usize,
    program: u64,
}

struct RecordedGroup {
    program: u64,
    group: wgpu::BindGroup,
    writes: Vec<(usize, usize)>,
}

/// A mutable output borrow stays here until this recording is dropped or
/// handed to the submission layer. The encoder cannot be accessed publicly.
pub struct GpuRecording<'a> {
    id: u64,
    context: Arc<WgpuContext>,
    reads: Vec<GpuFrameRead>,
    writes: Vec<GpuFrameWrite<'a>>,
    groups: Vec<RecordedGroup>,
    written: Vec<Vec<bool>>,
    encoder: Option<wgpu::CommandEncoder>,
}

impl<'a> GpuRecording<'a> {
    pub fn new(context: Arc<WgpuContext>) -> Self {
        let encoder = context
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vyd managed compute recording"),
            });
        Self {
            id: NEXT_RECORDING_ID.fetch_add(1, Ordering::Relaxed),
            context,
            reads: Vec::new(),
            writes: Vec::new(),
            groups: Vec::new(),
            written: Vec::new(),
            encoder: Some(encoder),
        }
    }

    pub fn add_read(
        &mut self,
        read: GpuFrameRead,
        expected: FrameSpec,
    ) -> Result<Vec<GpuReadPlaneId>, GpuRecordingError> {
        self.validate_guard(
            read.device_key(),
            read.spec(),
            expected,
            read.plane_count(),
            |index| read.plane(index).map(|lease| lease.key()),
            PlaneAccess::Read,
        )?;
        let frame = self.reads.len();
        let ids = (0..read.plane_count())
            .map(|plane| GpuReadPlaneId {
                recording: self.id,
                frame,
                plane,
            })
            .collect();
        self.reads.push(read);
        Ok(ids)
    }

    pub fn add_write(
        &mut self,
        write: GpuFrameWrite<'a>,
        expected: FrameSpec,
    ) -> Result<Vec<GpuWritePlaneId>, GpuRecordingError> {
        self.validate_guard(
            write.device_key(),
            write.spec(),
            expected,
            write.plane_count(),
            |index| write.plane(index).map(|lease| lease.key()),
            PlaneAccess::Write,
        )?;
        let frame = self.writes.len();
        let ids = (0..write.plane_count())
            .map(|plane| GpuWritePlaneId {
                recording: self.id,
                frame,
                plane,
            })
            .collect();
        self.writes.push(write);
        self.written.push(vec![false; expected.plane_count()]);
        Ok(ids)
    }

    fn validate_guard(
        &self,
        device: DeviceKey,
        spec: FrameSpec,
        expected: FrameSpec,
        plane_count: usize,
        key: impl Fn(usize) -> Option<GpuResourceKey>,
        access: PlaneAccess,
    ) -> Result<(), GpuRecordingError> {
        if device != self.context.device_key() {
            return Err(GpuRecordingError::WrongDevice {
                expected: self.context.device_key(),
                actual: device,
            });
        }
        if spec != expected {
            return Err(GpuRecordingError::SpecMismatch);
        }
        if plane_count != spec.plane_count() {
            return Err(GpuRecordingError::MissingPlane(plane_count));
        }
        for index in 0..plane_count {
            let resource = key(index).ok_or(GpuRecordingError::MissingPlane(index))?;
            validate_resource(resource, device, spec, index, access)?;
        }
        Ok(())
    }

    /// Build a recording-local group from typed plane IDs. The schema must be
    /// satisfied exactly, including logical format, plane number and access.
    pub fn bind_group(
        &mut self,
        program: &GpuComputeProgram,
        bindings: &[(u32, GpuPlaneBinding)],
    ) -> Result<GpuBindGroupId, GpuRecordingError> {
        if program.device != self.context.device_key() {
            return Err(GpuRecordingError::WrongDevice {
                expected: self.context.device_key(),
                actual: program.device,
            });
        }
        let mut seen = HashSet::new();
        for (number, _) in bindings {
            if !seen.insert(*number) {
                return Err(GpuRecordingError::DuplicateBinding(*number));
            }
        }
        if seen.len() != program.bindings.len() {
            return Err(GpuRecordingError::BindingMismatch(
                program
                    .bindings
                    .iter()
                    .find(|spec| !seen.contains(&spec.binding))
                    .map_or(u32::MAX, |spec| spec.binding),
            ));
        }
        // Hold the view objects locally until wgpu has created the group.
        let mut views = Vec::with_capacity(bindings.len());
        let mut writes = Vec::new();
        for (number, id) in bindings {
            let spec = program
                .bindings
                .iter()
                .find(|spec| spec.binding == *number)
                .ok_or(GpuRecordingError::BindingMismatch(*number))?;
            let (actual_format, actual_plane, key, texture) = match id {
                GpuPlaneBinding::Read(id) => {
                    if spec.access != PlaneAccess::Read {
                        return Err(GpuRecordingError::WrongDirection);
                    }
                    if id.recording != self.id {
                        return Err(GpuRecordingError::WrongRecording);
                    }
                    let read = self
                        .reads
                        .get(id.frame)
                        .ok_or(GpuRecordingError::WrongRecording)?;
                    let lease = read
                        .plane(id.plane)
                        .ok_or(GpuRecordingError::MissingPlane(id.plane))?;
                    (
                        read.spec().pixel_format(),
                        id.plane,
                        lease.key(),
                        lease.texture(),
                    )
                }
                GpuPlaneBinding::Write(id) => {
                    if spec.access != PlaneAccess::Write {
                        return Err(GpuRecordingError::WrongDirection);
                    }
                    if id.recording != self.id {
                        return Err(GpuRecordingError::WrongRecording);
                    }
                    let write = self
                        .writes
                        .get(id.frame)
                        .ok_or(GpuRecordingError::WrongRecording)?;
                    let lease = write
                        .plane(id.plane)
                        .ok_or(GpuRecordingError::MissingPlane(id.plane))?;
                    writes.push((id.frame, id.plane));
                    (
                        write.spec().pixel_format(),
                        id.plane,
                        lease.key(),
                        lease.texture(),
                    )
                }
            };
            if actual_format != spec.format || actual_plane != spec.plane {
                return Err(GpuRecordingError::WrongPlaneSpec);
            }
            let GpuResourceKey::Texture { usage, .. } = key else {
                return Err(GpuRecordingError::InvalidPlaneResource);
            };
            let required = match spec.access {
                PlaneAccess::Read => wgpu::TextureUsages::TEXTURE_BINDING,
                PlaneAccess::Write => wgpu::TextureUsages::STORAGE_BINDING,
            };
            if !usage.contains(required) {
                return Err(GpuRecordingError::InvalidPlaneResource);
            }
            views.push((
                *number,
                texture.create_view(&wgpu::TextureViewDescriptor::default()),
            ));
        }
        let entries: Vec<_> = views
            .iter()
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        let group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vyd managed plane bindings"),
                layout: &program.layout,
                entries: &entries,
            });
        let index = self.groups.len();
        self.groups.push(RecordedGroup {
            program: program.id,
            group,
            writes,
        });
        Ok(GpuBindGroupId {
            recording: self.id,
            index,
            program: program.id,
        })
    }

    /// Record one compute pass. Frame textures, views and encoder remain
    /// private; a group from another recording or program is rejected.
    pub fn dispatch(
        &mut self,
        program: &GpuComputeProgram,
        group: GpuBindGroupId,
        workgroups: [u32; 3],
    ) -> Result<(), GpuRecordingError> {
        if program.device != self.context.device_key() {
            return Err(GpuRecordingError::WrongDevice {
                expected: self.context.device_key(),
                actual: program.device,
            });
        }
        if group.recording != self.id {
            return Err(GpuRecordingError::WrongBindGroup);
        }
        if group.program != program.id {
            return Err(GpuRecordingError::WrongProgram);
        }
        let bound = self
            .groups
            .get(group.index)
            .ok_or(GpuRecordingError::WrongBindGroup)?;
        if bound.program != program.id {
            return Err(GpuRecordingError::WrongProgram);
        }
        if workgroups.contains(&0) {
            return Err(GpuRecordingError::EmptyDispatch);
        }
        let mut pass = self
            .encoder
            .as_mut()
            .expect("unfinished recording")
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vyd managed compute pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&program.pipeline);
        pass.set_bind_group(0, &bound.group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        drop(pass);
        for &(frame, plane) in &bound.writes {
            self.written[frame][plane] = true;
        }
        Ok(())
    }

    /// Only the submission layer can turn this into queue work. It must retain
    /// `reads` and `writes` through GPU completion before publishing outputs.
    #[allow(dead_code)] // G4.10 consumes the prepared recording.
    pub(crate) fn finish(mut self) -> PreparedRecording<'a> {
        PreparedRecording {
            context: self.context,
            command_buffer: self.encoder.take().expect("unfinished recording").finish(),
            reads: self.reads,
            writes: self.writes,
            written: self.written,
        }
    }
}

#[allow(dead_code)] // G4.10 owns the queue admission and publication protocol.
pub(crate) struct PreparedRecording<'a> {
    pub(crate) context: Arc<WgpuContext>,
    pub(crate) command_buffer: wgpu::CommandBuffer,
    pub(crate) reads: Vec<GpuFrameRead>,
    pub(crate) writes: Vec<GpuFrameWrite<'a>>,
    pub(crate) written: Vec<Vec<bool>>,
}

fn validate_resource(
    key: GpuResourceKey,
    device: DeviceKey,
    spec: FrameSpec,
    index: usize,
    access: PlaneAccess,
) -> Result<(), GpuRecordingError> {
    let (row, rows) = spec
        .plane_shape(index)
        .map_err(|_| GpuRecordingError::MissingPlane(index))?;
    let width = if index == 1 { row / 2 } else { row };
    let GpuResourceKey::Texture {
        device: resource_device,
        width: actual_width,
        height: actual_height,
        format,
        usage,
    } = key
    else {
        return Err(GpuRecordingError::InvalidPlaneResource);
    };
    let required = match access {
        PlaneAccess::Read => wgpu::TextureUsages::TEXTURE_BINDING,
        PlaneAccess::Write => wgpu::TextureUsages::STORAGE_BINDING,
    };
    if resource_device != device
        || actual_width as usize != width
        || actual_height as usize != rows
        || format != wgpu::TextureFormat::Rgba8Uint
        || !usage.contains(required)
    {
        return Err(GpuRecordingError::InvalidPlaneResource);
    }
    Ok(())
}

trait PlaneCount {
    fn plane_count(self) -> usize;
}
impl PlaneCount for PixelFormat {
    fn plane_count(self) -> usize {
        if self == PixelFormat::Nv12 { 2 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::frame::factory::FrameFactory;
    use crate::frame::pool::WaitPolicy;
    use crate::gpu::pool::{GpuPoolLimits, GpuPoolManager};
    use crate::gpu::transfer::GpuTransfer;
    use crate::gpu::{GpuContextError, WgpuContextOptions};
    use crate::statistics::{PipelineStatistics, StatsContext};
    use std::time::Duration;

    const COPY_SHADER: &str = r#"
        @group(0) @binding(0) var source: texture_2d<u32>;
        @group(0) @binding(1) var output: texture_storage_2d<rgba8uint, write>;
        @compute @workgroup_size(1, 1, 1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let xy = vec2<i32>(id.xy);
            textureStore(output, xy, textureLoad(source, xy, 0));
        }
    "#;

    fn fixture() -> Option<(Arc<WgpuContext>, GpuPoolManager, GpuTransfer)> {
        let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Arc::new(context),
            Err(GpuContextError::Unavailable) => {
                eprintln!("recording GPU tests unverified: no compatible adapter");
                return None;
            }
            Err(error) => panic!("recording GPU hardware gate failed: {error}"),
        };
        eprintln!(
            "recording adapter={} backend={:?}",
            context.adapter_info().name,
            context.adapter_info().backend
        );
        let pool =
            GpuPoolManager::new(context.clone(), GpuPoolLimits::new(8, 1024 * 1024).unwrap());
        let stats = StatsContext::new(Arc::new(PipelineStatistics::new())).for_stage("recording");
        let transfer = GpuTransfer::new(context.clone(), pool.clone(), stats).unwrap();
        Some((context, pool, transfer))
    }

    fn wait() -> WaitPolicy {
        WaitPolicy::Timeout(Duration::from_secs(3))
    }
    fn spec() -> FrameSpec {
        FrameSpec::new(2, 2, PixelFormat::Luma8).unwrap()
    }
    fn program(context: &WgpuContext) -> GpuComputeProgram {
        GpuComputeProgram::new(
            context,
            COPY_SHADER,
            "main",
            &[
                GpuBindingSpec::read(0, PixelFormat::Luma8, 0),
                GpuBindingSpec::write(1, PixelFormat::Luma8, 0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn invalid_wgsl_and_shader_layout_return_structured_errors() {
        let Some((context, _pool, _transfer)) = fixture() else {
            return;
        };
        let bindings = [
            GpuBindingSpec::read(0, PixelFormat::Luma8, 0),
            GpuBindingSpec::write(1, PixelFormat::Luma8, 0),
        ];
        let Err(GpuRecordingError::ProgramCreation { stage, message }) =
            GpuComputeProgram::new(&context, "@compute fn main(", "main", &bindings)
        else {
            panic!("malformed WGSL must return a structured error");
        };
        assert_eq!(stage, "shader module");
        assert!(!message.is_empty());
        let Err(GpuRecordingError::ProgramCreation { stage, message }) =
            GpuComputeProgram::new(&context, COPY_SHADER, "main", &bindings[..1])
        else {
            panic!("shader/layout mismatch must return a structured error");
        };
        assert_eq!(stage, "compute pipeline");
        assert!(!message.is_empty());
        // An earlier validation failure must not poison a later valid build.
        program(&context);
    }

    #[test]
    fn records_a_real_dispatch_without_publishing_before_admission() {
        let Some((context, pool, transfer)) = fixture() else {
            return;
        };
        let factory = FrameFactory::default();
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut destination = Frame::from_raw(2, 2, vec![9, 9, 9, 9]).unwrap();
        let read = source
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let before = pool.snapshot().unwrap().active.len();
        let output = destination
            .try_write_gpu_as(&transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let shader = program(&context);
        let mut recording = GpuRecording::new(context);
        let src = recording.add_read(read, spec()).unwrap()[0];
        let dst = recording.add_write(output, spec()).unwrap()[0];
        let group = recording
            .bind_group(
                &shader,
                &[
                    (0, GpuPlaneBinding::Read(src)),
                    (1, GpuPlaneBinding::Write(dst)),
                ],
            )
            .unwrap();
        recording.dispatch(&shader, group, [2, 2, 1]).unwrap();
        assert_eq!(pool.snapshot().unwrap().active.len(), before + 1);
        let prepared = recording.finish();
        assert_eq!(prepared.reads.len(), 1);
        assert_eq!(prepared.writes.len(), 1);
        drop(prepared); // G4.10 will own admission and completion.
        assert_eq!(
            destination
                .try_read_cpu_luma8(WaitPolicy::Immediate)
                .unwrap(),
            &[9, 9, 9, 9]
        );
        assert_eq!(pool.snapshot().unwrap().active.len(), before);
    }

    #[test]
    fn rejected_binding_and_drop_preserve_output_and_cache() {
        let Some((context, pool, transfer)) = fixture() else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let read = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let active = pool.snapshot().unwrap().active.len();
        let shader = program(&context);
        {
            let output = frame
                .try_write_gpu_as(&transfer, PixelFormat::Luma8, wait())
                .unwrap();
            let mut recording = GpuRecording::new(context.clone());
            let src = recording.add_read(read.clone(), spec()).unwrap()[0];
            let dst = recording.add_write(output, spec()).unwrap()[0];
            assert_eq!(
                recording
                    .bind_group(
                        &shader,
                        &[
                            (0, GpuPlaneBinding::Write(dst)),
                            (1, GpuPlaneBinding::Read(src)),
                        ]
                    )
                    .unwrap_err(),
                GpuRecordingError::WrongDirection
            );
            assert_eq!(
                recording
                    .bind_group(
                        &shader,
                        &[
                            (0, GpuPlaneBinding::Read(src)),
                            (0, GpuPlaneBinding::Read(src)),
                        ]
                    )
                    .unwrap_err(),
                GpuRecordingError::DuplicateBinding(0)
            );
            assert_eq!(pool.snapshot().unwrap().active.len(), active + 1);
        }
        assert_eq!(pool.snapshot().unwrap().active.len(), active);
        assert_eq!(
            frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[1, 2, 3, 4]
        );
        let again = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert_eq!(again.device_key(), read.device_key());
    }

    #[test]
    fn rejects_foreign_recording_device_spec_and_usage() {
        let Some((context, _pool, transfer)) = fixture() else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let read = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let mut first = GpuRecording::new(context.clone());
        assert_eq!(
            first
                .add_read(
                    read.clone(),
                    FrameSpec::new(3, 2, PixelFormat::Luma8).unwrap()
                )
                .unwrap_err(),
            GpuRecordingError::SpecMismatch
        );
        let id = first.add_read(read, spec()).unwrap()[0];
        let shader = program(&context);
        let mut second = GpuRecording::new(context.clone());
        assert_eq!(
            second
                .bind_group(&shader, &[(0, GpuPlaneBinding::Read(id))])
                .unwrap_err(),
            GpuRecordingError::BindingMismatch(1)
        );
        let mut destination = Frame::from_raw(2, 2, vec![0; 4]).unwrap();
        let output = destination
            .try_write_gpu_as(&transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let dst = second.add_write(output, spec()).unwrap()[0];
        assert_eq!(
            second
                .bind_group(
                    &shader,
                    &[
                        (0, GpuPlaneBinding::Read(id)),
                        (1, GpuPlaneBinding::Write(dst)),
                    ]
                )
                .unwrap_err(),
            GpuRecordingError::WrongRecording
        );
        assert_eq!(
            second
                .bind_group(
                    &shader,
                    &[
                        (0, GpuPlaneBinding::Write(dst)),
                        (1, GpuPlaneBinding::Write(dst)),
                    ]
                )
                .unwrap_err(),
            GpuRecordingError::WrongDirection
        );
        let foreign =
            Arc::new(WgpuContext::request_blocking(WgpuContextOptions::default()).unwrap());
        let mut wrong_device = GpuRecording::new(foreign);
        let read = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert!(matches!(
            wrong_device.add_read(read, spec()),
            Err(GpuRecordingError::WrongDevice { .. })
        ));
        assert_eq!(
            validate_resource(
                GpuResourceKey::Texture {
                    device: context.device_key(),
                    width: 2,
                    height: 2,
                    format: wgpu::TextureFormat::Rgba8Uint,
                    usage: wgpu::TextureUsages::COPY_SRC
                },
                context.device_key(),
                spec(),
                0,
                PlaneAccess::Read,
            ),
            Err(GpuRecordingError::InvalidPlaneResource)
        );
    }

    #[test]
    fn logical_format_and_program_ids_are_checked_before_dispatch() {
        let Some((context, _pool, transfer)) = fixture() else {
            return;
        };
        let factory = FrameFactory::default();
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut destination = Frame::from_raw(2, 2, vec![0; 4]).unwrap();
        let read = source
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let write = destination
            .try_write_gpu_as(&transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let mut recording = GpuRecording::new(context.clone());
        let src = recording.add_read(read, spec()).unwrap()[0];
        let dst = recording.add_write(write, spec()).unwrap()[0];
        let wrong_format = GpuComputeProgram::new(
            &context,
            COPY_SHADER,
            "main",
            &[
                GpuBindingSpec::read(0, PixelFormat::Nv12, 0),
                GpuBindingSpec::write(1, PixelFormat::Luma8, 0),
            ],
        )
        .unwrap();
        assert_eq!(
            recording
                .bind_group(
                    &wrong_format,
                    &[
                        (0, GpuPlaneBinding::Read(src)),
                        (1, GpuPlaneBinding::Write(dst)),
                    ],
                )
                .unwrap_err(),
            GpuRecordingError::WrongPlaneSpec
        );
        let first = program(&context);
        let second = program(&context);
        let group = recording
            .bind_group(
                &first,
                &[
                    (0, GpuPlaneBinding::Read(src)),
                    (1, GpuPlaneBinding::Write(dst)),
                ],
            )
            .unwrap();
        assert_eq!(
            recording.dispatch(&second, group, [1, 1, 1]),
            Err(GpuRecordingError::WrongProgram)
        );
        assert_eq!(
            recording.dispatch(&first, group, [0, 1, 1]),
            Err(GpuRecordingError::EmptyDispatch)
        );
    }
}
