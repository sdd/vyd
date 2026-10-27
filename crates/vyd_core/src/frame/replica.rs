//! Requested-format reads. A mutable frame borrow makes replica publication
//! explicit and lets us notice replacement of the public legacy `img` field.

use super::factory::FrameFactory;
use super::pool::WaitPolicy;
use super::{
    CpuFrame, CpuLuma8Write, CpuNv12Frame, CpuNv12Write, Frame, FrameError, Nv12Encoding,
    PixelFormat,
};
use std::time::Instant;

fn remaining(wait: WaitPolicy, started: Instant) -> WaitPolicy {
    match wait {
        WaitPolicy::Immediate => WaitPolicy::Immediate,
        WaitPolicy::Timeout(limit) => WaitPolicy::Timeout(limit.saturating_sub(started.elapsed())),
    }
}

/// A borrowed CPU replica in the requested logical format. The source frame
/// owns any converted allocation until its next mutation or replacement.
pub struct CpuFrameRead<'a> {
    frame: &'a Frame,
}

/// Scoped mutable CPU access in the requested logical format. Each successful
/// mutable row or byte access invalidates replicas. Writes are published as
/// they occur, including if the guard is dropped before every row is filled.
pub enum CpuFrameWrite<'a> {
    Luma8(CpuLuma8Write<'a>),
    Nv12(CpuNv12Write<'a>),
}

impl CpuFrameWrite<'_> {
    pub fn pixel_format(&self) -> PixelFormat {
        match self {
            Self::Luma8(_) => PixelFormat::Luma8,
            Self::Nv12(_) => PixelFormat::Nv12,
        }
    }

    pub fn luma8_bytes_mut(&mut self) -> Result<&mut [u8], FrameError> {
        match self {
            Self::Luma8(write) => write.try_bytes_mut(),
            Self::Nv12(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Luma8,
                actual: PixelFormat::Nv12,
            }),
        }
    }

    pub fn nv12_row_mut(&mut self, plane: usize, row: usize) -> Result<&mut [u8], FrameError> {
        match self {
            Self::Nv12(write) => write.row_mut(plane, row),
            Self::Luma8(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Nv12,
                actual: PixelFormat::Luma8,
            }),
        }
    }
}

impl CpuFrameRead<'_> {
    pub fn pixel_format(&self) -> PixelFormat {
        self.frame.pixel_format()
    }

    pub fn luma8_bytes(&self) -> Result<&[u8], FrameError> {
        self.frame.try_read_cpu_luma8(WaitPolicy::Immediate)
    }

    pub fn nv12(&self) -> Result<super::CpuNv12Read<'_>, FrameError> {
        self.frame.try_read_cpu_nv12(WaitPolicy::Immediate)
    }
}

impl CpuFrame {
    fn refresh_replicas(&mut self) {
        self.cache.refresh(vec![self.img.as_raw().stamp()]);
    }
}

impl CpuNv12Frame {
    fn refresh_replicas(&mut self) {
        self.cache
            .refresh(self.regions.iter().map(|region| region.stamp()).collect());
    }
}

impl Frame {
    /// Request a writable CPU version. A format change or GPU readback is
    /// prepared separately; the logical frame is replaced only after a
    /// writable destination has been acquired. Existing GPU readers and frame
    /// clones retain the preceding bytes through their own leases.
    pub fn try_write_cpu_as(
        &mut self,
        factory: &FrameFactory,
        format: PixelFormat,
        wait: WaitPolicy,
    ) -> Result<CpuFrameWrite<'_>, FrameError> {
        self.try_write_cpu_as_impl(
            factory,
            format,
            wait,
            #[cfg(feature = "gpu-wgpu")]
            None,
        )
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn try_write_cpu_as_with_transfer(
        &mut self,
        factory: &FrameFactory,
        transfer: &crate::gpu::transfer::GpuTransfer,
        format: PixelFormat,
        wait: WaitPolicy,
    ) -> Result<CpuFrameWrite<'_>, FrameError> {
        self.try_write_cpu_as_impl(factory, format, wait, Some(transfer))
    }

    /// Materialize the requested CPU format as the current frame version for
    /// an external CPU boundary. Existing GPU readers retain their leases.
    #[cfg(feature = "gpu-wgpu")]
    pub fn ensure_cpu_as_with_transfer(
        &mut self,
        factory: &FrameFactory,
        transfer: &crate::gpu::transfer::GpuTransfer,
        format: PixelFormat,
        wait: WaitPolicy,
    ) -> Result<(), FrameError> {
        if self.is_cpu() && self.pixel_format() == format {
            return Ok(());
        }
        drop(self.try_write_cpu_as_with_transfer(factory, transfer, format, wait)?);
        Ok(())
    }

    fn try_write_cpu_as_impl(
        &mut self,
        factory: &FrameFactory,
        format: PixelFormat,
        wait: WaitPolicy,
        #[cfg(feature = "gpu-wgpu")] transfer: Option<&crate::gpu::transfer::GpuTransfer>,
    ) -> Result<CpuFrameWrite<'_>, FrameError> {
        if self.is_cpu() && self.pixel_format() == format {
            return self.write_cpu_current(format, wait);
        }
        let started = Instant::now();
        let mut candidate = match self {
            Frame::Cpu(_) | Frame::CpuNv12(_) => match format {
                PixelFormat::Luma8 => factory.convert_nv12_to_luma8(self, wait)?,
                PixelFormat::Nv12 => {
                    factory.convert_luma8_to_nv12(self, Nv12Encoding::FullRange, wait)?
                }
            },
            Frame::Gpu(_) => return Err(FrameError::UnsupportedLegacyGpuAccess),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(gpu) => {
                let transfer = transfer.ok_or(FrameError::GpuConversionRequired)?;
                // Keep readback and conversion off the live frame so an
                // acquisition failure cannot replace its source version.
                let source_format = gpu.spec().pixel_format();
                let mut source = Frame::ManagedGpu(gpu.clone());
                source.try_read_cpu_as_with_transfer(
                    factory,
                    transfer,
                    source_format,
                    remaining(wait, started),
                )?;
                let Frame::ManagedGpu(mut downloaded) = source else {
                    unreachable!()
                };
                let mut cpu = *downloaded.cpu.take().unwrap();
                if cpu.pixel_format() != format {
                    cpu = match format {
                        PixelFormat::Luma8 => {
                            factory.convert_nv12_to_luma8(&cpu, remaining(wait, started))?
                        }
                        PixelFormat::Nv12 => factory.convert_luma8_to_nv12(
                            &cpu,
                            Nv12Encoding::FullRange,
                            remaining(wait, started),
                        )?,
                    };
                }
                cpu
            }
        };
        // Probe the writable backing before replacing the current version.
        // The candidate is exclusive after this guard drops, so reacquisition
        // below cannot need another COW allocation.
        drop(candidate.write_cpu_current(format, remaining(wait, started))?);
        *self = candidate;
        self.write_cpu_current(format, remaining(wait, started))
    }

    fn write_cpu_current(
        &mut self,
        format: PixelFormat,
        wait: WaitPolicy,
    ) -> Result<CpuFrameWrite<'_>, FrameError> {
        match (self, format) {
            (Frame::Cpu(cpu), PixelFormat::Luma8) => {
                Ok(CpuFrameWrite::Luma8(cpu.try_write_luma8(wait)?))
            }
            (Frame::CpuNv12(cpu), PixelFormat::Nv12) => {
                Ok(CpuFrameWrite::Nv12(cpu.try_write(wait)?))
            }
            _ => unreachable!("candidate format was checked"),
        }
    }

    /// Request a coherent CPU replica. Conversion is performed once per source
    /// version and recorded by `FrameFactory`. The `&mut` receiver is needed
    /// because a read may populate the cache; it also protects public `img`
    /// replacement and mutable storage access from racing with publication.
    pub fn try_read_cpu_as(
        &mut self,
        factory: &FrameFactory,
        format: PixelFormat,
        wait: WaitPolicy,
    ) -> Result<CpuFrameRead<'_>, FrameError> {
        self.try_read_cpu_as_impl(
            factory,
            format,
            wait,
            #[cfg(feature = "gpu-wgpu")]
            None,
        )
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn try_read_cpu_as_with_transfer(
        &mut self,
        factory: &FrameFactory,
        transfer: &crate::gpu::transfer::GpuTransfer,
        format: PixelFormat,
        wait: WaitPolicy,
    ) -> Result<CpuFrameRead<'_>, FrameError> {
        self.try_read_cpu_as_impl(factory, format, wait, Some(transfer))
    }

    fn try_read_cpu_as_impl(
        &mut self,
        factory: &FrameFactory,
        format: PixelFormat,
        wait: WaitPolicy,
        #[cfg(feature = "gpu-wgpu")] transfer: Option<&crate::gpu::transfer::GpuTransfer>,
    ) -> Result<CpuFrameRead<'_>, FrameError> {
        match self {
            Frame::Cpu(cpu) if format == PixelFormat::Luma8 => {
                cpu.refresh_replicas();
                Ok(CpuFrameRead { frame: self })
            }
            Frame::Cpu(cpu) => {
                cpu.refresh_replicas();
                if cpu.cache.converted.is_none() {
                    let source = Frame::Cpu(cpu.clone());
                    cpu.cache.converted = Some(Box::new(factory.convert_luma8_to_nv12(
                        &source,
                        Nv12Encoding::FullRange,
                        wait,
                    )?));
                }
                Ok(CpuFrameRead {
                    frame: cpu.cache.converted.as_deref().unwrap(),
                })
            }
            Frame::CpuNv12(cpu) if format == PixelFormat::Nv12 => {
                cpu.refresh_replicas();
                Ok(CpuFrameRead { frame: self })
            }
            Frame::CpuNv12(cpu) => {
                cpu.refresh_replicas();
                if cpu.cache.converted.is_none() {
                    let source = Frame::CpuNv12(cpu.clone());
                    cpu.cache.converted =
                        Some(Box::new(factory.convert_nv12_to_luma8(&source, wait)?));
                }
                Ok(CpuFrameRead {
                    frame: cpu.cache.converted.as_deref().unwrap(),
                })
            }
            Frame::Gpu(_) => Err(FrameError::UnsupportedLegacyGpuAccess),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(gpu) => {
                let transfer = transfer.ok_or(FrameError::GpuConversionRequired)?;
                let started = Instant::now();
                gpu.ensure_cpu(factory, transfer, wait)?;
                gpu.cpu.as_mut().unwrap().try_read_cpu_as_impl(
                    factory,
                    format,
                    remaining(wait, started),
                    Some(transfer),
                )
            }
        }
    }
}

#[cfg(feature = "gpu-wgpu")]
mod gpu {
    use super::*;
    use crate::frame::layout::{CpuLayout, FrameSpec};
    use crate::gpu::DeviceKey;
    use crate::gpu::submission::SubmissionState;
    use crate::gpu::transfer::{GpuPlaneSet, GpuTransfer};
    use std::sync::Arc;

    /// A GPU-resident source version. The CPU cache is filled on demand, and
    /// its clones retain the same plane leases independently.
    pub struct ManagedGpuFrame {
        source: GpuPlaneSet,
        pub(super) cpu: Option<Box<Frame>>,
        converted_gpu: Option<Box<ManagedGpuFrame>>,
        pending: Option<Arc<SubmissionState>>,
    }

    impl Clone for ManagedGpuFrame {
        fn clone(&self) -> Self {
            Self {
                source: self.source.clone(),
                cpu: None,
                converted_gpu: None,
                pending: self.pending.clone(),
            }
        }
    }

    impl ManagedGpuFrame {
        pub fn spec(&self) -> FrameSpec {
            self.source.spec()
        }
        pub fn device_key(&self) -> DeviceKey {
            self.source.device_key()
        }

        pub(super) fn wait_pending(&self, wait: WaitPolicy) -> Result<(), FrameError> {
            if let Some(pending) = &self.pending {
                pending.try_wait(wait)?;
            }
            Ok(())
        }

        pub(super) fn ensure_cpu(
            &mut self,
            factory: &FrameFactory,
            transfer: &GpuTransfer,
            wait: WaitPolicy,
        ) -> Result<(), FrameError> {
            self.wait_pending(wait)?;
            if self.cpu.is_some() {
                return Ok(());
            }
            let started = Instant::now();
            let readback = transfer.readback(&self.source, wait)?;
            let spec = self.source.spec();
            let (width, height) = spec.dimensions();
            let mut destination = match spec.pixel_format() {
                PixelFormat::Luma8 => {
                    factory.create_luma8(width, height, remaining(wait, started))?
                }
                PixelFormat::Nv12 => factory.create_nv12(
                    width,
                    height,
                    Nv12Encoding::FullRange,
                    remaining(wait, started),
                )?,
            };
            let layout = CpuLayout::packed(spec)?;
            match &mut destination {
                Frame::Cpu(cpu) => {
                    let mut write = cpu.try_write_luma8(wait)?;
                    readback.copy_into(&layout, &mut [write.try_bytes_mut()?])?;
                }
                Frame::CpuNv12(cpu) => {
                    let mut regions: Vec<&mut [u8]> = cpu
                        .regions
                        .iter_mut()
                        .map(|region| region.try_write_luma8(wait))
                        .collect::<Result<_, _>>()?;
                    readback.copy_into(&layout, &mut regions)?;
                }
                _ => unreachable!(),
            }
            self.cpu = Some(Box::new(destination));
            Ok(())
        }
    }

    /// Owns retained plane leases without exposing their raw textures. Future
    /// managed recordings can use crate-private `plane` bindings.
    #[derive(Clone)]
    pub struct GpuFrameRead {
        planes: GpuPlaneSet,
    }

    impl GpuFrameRead {
        pub(crate) fn quarantine(&self) {
            self.planes.quarantine();
        }
        pub fn spec(&self) -> FrameSpec {
            self.planes.spec()
        }
        pub fn device_key(&self) -> DeviceKey {
            self.planes.device_key()
        }
        pub fn plane_count(&self) -> usize {
            self.planes.plane_count()
        }
        #[allow(dead_code)] // G4.9 recording-local bindings will consume these leases.
        pub(crate) fn plane(&self, index: usize) -> Option<&crate::gpu::pool::TextureLease> {
            self.planes.plane(index)
        }
        /// Promote the retained GPU replica to a managed frame source.
        pub fn into_frame(self) -> Frame {
            self.into_frame_with_cpu(None)
        }

        fn into_frame_with_cpu(self, cpu: Option<Box<Frame>>) -> Frame {
            Frame::ManagedGpu(ManagedGpuFrame {
                source: self.planes,
                cpu,
                converted_gpu: None,
                pending: None,
            })
        }
    }

    /// An uninitialized, bounded plane reservation for a future managed GPU
    /// recording. Its mutable frame borrow prevents another version from
    /// being installed concurrently. Dropping it before submission releases
    /// the reservation without changing the frame or its replica caches.
    pub struct GpuFrameWrite<'a> {
        target: &'a mut Frame,
        candidate: GpuPlaneSet,
    }

    impl<'a> GpuFrameWrite<'a> {
        pub fn spec(&self) -> FrameSpec {
            self.candidate.spec()
        }
        pub fn device_key(&self) -> DeviceKey {
            self.candidate.device_key()
        }
        pub fn plane_count(&self) -> usize {
            self.candidate.plane_count()
        }
        /// Recording code may bind these private leases, but cannot publish
        /// them through this safe public guard before submission completion.
        #[allow(dead_code)] // G4.9 will consume recording-local bindings.
        pub(crate) fn plane(&self, index: usize) -> Option<&crate::gpu::pool::TextureLease> {
            self.candidate.plane(index)
        }
        /// G4.9 consumes the guard into a recording. The exclusive target
        /// borrow stays with that recording until G4.10 admits or discards it.
        pub(crate) fn retained_candidate(&self) -> GpuPlaneSet {
            self.candidate.clone()
        }

        /// Called only after queue admission, while the submission callback
        /// already retains an independent candidate lease.
        pub(crate) fn publish(self, pending: Arc<SubmissionState>) {
            *self.target = Frame::ManagedGpu(ManagedGpuFrame {
                source: self.candidate,
                cpu: None,
                converted_gpu: None,
                pending: Some(pending),
            });
        }
    }

    impl Frame {
        /// Reserve a fresh GPU destination for a complete overwrite in the
        /// requested format. No conversion/upload or logical mutation occurs
        /// at acquisition. A managed recording must initialize and submit
        /// the candidate before it becomes a readable frame version.
        pub fn try_write_gpu_as(
            &mut self,
            transfer: &GpuTransfer,
            format: PixelFormat,
            wait: WaitPolicy,
        ) -> Result<GpuFrameWrite<'_>, FrameError> {
            if matches!(self, Frame::Gpu(_)) {
                return Err(FrameError::UnsupportedLegacyGpuAccess);
            }
            if let Frame::ManagedGpu(gpu) = &self
                && gpu.device_key() != transfer.device_key()
            {
                return Err(crate::gpu::transfer::GpuTransferError::Context(
                    crate::gpu::GpuContextError::IncompatibleDevice {
                        expected: transfer.device_key(),
                        actual: gpu.device_key(),
                    },
                )
                .into());
            }
            let (width, height) = self.dimensions();
            let spec = FrameSpec::new(width, height, format)?;
            let candidate = transfer.reserve_write(spec, wait)?;
            Ok(GpuFrameWrite {
                target: self,
                candidate,
            })
        }

        /// Request a GPU replica on the transfer service's device. A cache hit
        /// neither transfers nor converts bytes. A GPU-resident source is
        /// converted on the same device without CPU readback or upload.
        pub fn try_read_gpu_as(
            &mut self,
            factory: &FrameFactory,
            transfer: &GpuTransfer,
            format: PixelFormat,
            wait: WaitPolicy,
        ) -> Result<GpuFrameRead, FrameError> {
            let started = Instant::now();
            if let Frame::ManagedGpu(gpu) = &*self {
                gpu.wait_pending(remaining(wait, started))?;
            }
            transfer.ensure_healthy()?;
            match self {
                Frame::Cpu(cpu) => {
                    cpu.refresh_replicas();
                    if let Some(planes) = cpu.cache.gpu.iter().find(|planes| {
                        planes.device_key() == transfer.device_key()
                            && planes.spec().pixel_format() == format
                    }) {
                        return Ok(GpuFrameRead {
                            planes: planes.clone(),
                        });
                    }
                }
                Frame::CpuNv12(cpu) => {
                    cpu.refresh_replicas();
                    if let Some(planes) = cpu.cache.gpu.iter().find(|planes| {
                        planes.device_key() == transfer.device_key()
                            && planes.spec().pixel_format() == format
                    }) {
                        return Ok(GpuFrameRead {
                            planes: planes.clone(),
                        });
                    }
                }
                Frame::ManagedGpu(gpu) => {
                    if gpu.source.device_key() != transfer.device_key() {
                        return Err(crate::gpu::transfer::GpuTransferError::Context(
                            crate::gpu::GpuContextError::IncompatibleDevice {
                                expected: transfer.device_key(),
                                actual: gpu.source.device_key(),
                            },
                        )
                        .into());
                    }
                    if gpu.source.spec().pixel_format() == format {
                        return Ok(GpuFrameRead {
                            planes: gpu.source.clone(),
                        });
                    }
                    if let Some(converted) = &gpu.converted_gpu {
                        converted.wait_pending(remaining(wait, started))?;
                        return Ok(GpuFrameRead {
                            planes: converted.source.clone(),
                        });
                    }

                    let read = GpuFrameRead {
                        planes: gpu.source.clone(),
                    };
                    let (destination, _receipt) = crate::gpu::conversion::convert(
                        read,
                        transfer,
                        format,
                        remaining(wait, started),
                    )?;
                    let Frame::ManagedGpu(converted) = destination else {
                        unreachable!("managed recording publishes a GPU output")
                    };
                    gpu.converted_gpu = Some(Box::new(converted));
                    let converted = gpu.converted_gpu.as_ref().unwrap();
                    converted.wait_pending(remaining(wait, started))?;
                    return Ok(GpuFrameRead {
                        planes: converted.source.clone(),
                    });
                }
                Frame::Gpu(_) => return Err(FrameError::UnsupportedLegacyGpuAccess),
            }
            let source = self.try_read_cpu_as_with_transfer(
                factory,
                transfer,
                format,
                remaining(wait, started),
            )?;
            let planes = upload_cpu(&source, transfer, remaining(wait, started))?;
            match self {
                Frame::Cpu(cpu) => cpu.cache.gpu.push(planes.clone()),
                Frame::CpuNv12(cpu) => cpu.cache.gpu.push(planes.clone()),
                Frame::ManagedGpu(_) => unreachable!("GPU sources use resident conversion"),
                Frame::Gpu(_) => unreachable!(),
            }
            Ok(GpuFrameRead { planes })
        }

        /// Compatibility promotion for callers that previously used
        /// `ensure_gpu`. The old parameterless method still returns its
        /// original unsupported error until services are supplied here.
        pub fn ensure_gpu_with(
            &mut self,
            factory: &FrameFactory,
            transfer: &GpuTransfer,
            wait: WaitPolicy,
        ) -> Result<(), FrameError> {
            let format = self.pixel_format();
            if matches!(self, Frame::ManagedGpu(_)) {
                self.try_read_gpu_as(factory, transfer, format, wait)?;
                return Ok(());
            }
            let cpu = if self.is_cpu() {
                Some(Box::new(self.clone_shared()))
            } else {
                None
            };
            let read = self.try_read_gpu_as(factory, transfer, format, wait)?;
            *self = read.into_frame_with_cpu(cpu);
            Ok(())
        }
    }

    fn upload_cpu(
        read: &CpuFrameRead<'_>,
        transfer: &GpuTransfer,
        wait: WaitPolicy,
    ) -> Result<GpuPlaneSet, FrameError> {
        match read.frame {
            Frame::Cpu(cpu) => {
                let spec = FrameSpec::new(cpu.width(), cpu.height(), PixelFormat::Luma8)?;
                let layout = CpuLayout::packed(spec)?;
                Ok(transfer.upload(&layout, &[cpu.img.as_raw()], None, wait)?)
            }
            Frame::CpuNv12(cpu) => {
                let regions: Vec<&[u8]> = cpu.regions.iter().map(|region| &region[..]).collect();
                Ok(transfer.upload(&cpu.layout, &regions, Some(cpu.encoding), wait)?)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "gpu-wgpu")]
pub use gpu::{GpuFrameRead, GpuFrameWrite, ManagedGpuFrame};
