//! Checked CPU-plane to RGBA texture transfers. This layer submits its own
//! copies. A timed-out upload or readback retains leases in its queue or map
//! callback until the GPU has finished with them. Context device loss closes
//! the associated pool so no failed resource can be reused.

use super::pool::{
    GpuPoolError, GpuPoolManager, STAGING_READBACK, STAGING_UPLOAD, StagingLease, StagingMapHolder,
    TextureLease,
};
use super::{DEVICE_ERROR_SCOPE_LOCK, DeviceKey, GpuContextError, WgpuContext};
use crate::frame::layout::{CpuLayout, FrameSpec, LayoutError};
use crate::frame::pool::WaitPolicy;
use crate::frame::{Nv12Encoding, PixelFormat};
use crate::statistics::{
    FrameOperationAttempt, FrameOperationBytes, FrameOperationKind, PackingDirection,
    ResourceOwnerId, StatsContext,
};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};
use thiserror::Error;

const PLANE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::COPY_SRC
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::STORAGE_BINDING);

#[derive(Debug, Error)]
pub enum GpuTransferError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Pool(#[from] GpuPoolError),
    #[error(transparent)]
    Context(#[from] GpuContextError),
    #[error("CPU layout does not match GPU frame specification")]
    SpecMismatch,
    #[error("NV12 upload requires an explicit full-range encoding declaration")]
    UnsupportedNv12Encoding,
    #[error("Luma8 upload must not declare NV12 encoding")]
    UnexpectedNv12Encoding,
    #[error("GPU transfer arithmetic overflow")]
    Overflow,
    #[error("GPU transfer would block")]
    WouldBlock,
    #[error("GPU transfer timed out after {0:?}")]
    Timeout(Duration),
    #[error("GPU staging map failed: {0}")]
    Map(String),
    #[error("GPU transfer completion callback was lost")]
    CompletionLost,
    #[error("GPU transfer submission failed validation: {0}")]
    Submission(String),
}

#[derive(Clone)]
pub struct GpuPlaneSet {
    spec: FrameSpec,
    device: DeviceKey,
    planes: Vec<TextureLease>,
}

impl GpuPlaneSet {
    pub(crate) fn quarantine(&self) {
        for plane in &self.planes {
            plane.quarantine_pool();
        }
    }
    pub fn spec(&self) -> FrameSpec {
        self.spec
    }
    pub fn device_key(&self) -> DeviceKey {
        self.device
    }
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }
    #[allow(dead_code)] // G4.7/G4.9 consume recording-local plane bindings.
    pub(crate) fn plane(&self, index: usize) -> Option<&TextureLease> {
        self.planes.get(index)
    }
}

#[derive(Clone, Copy)]
struct PlaneCopy {
    width: u32,
    height: u32,
    pitch: u32,
    offset: u64,
}

struct CopyPlan {
    planes: Vec<PlaneCopy>,
    // Entire allocated/mapped footprint, including aligned row tails.
    staging_bytes: u64,
    // RGBA texel bytes copied by wgpu; row tails are not image content.
    copied_bytes: u64,
    visible_bytes: u64,
}

impl CopyPlan {
    fn new(spec: FrameSpec) -> Result<Self, GpuTransferError> {
        let mut planes = Vec::with_capacity(spec.plane_count());
        let mut offset = 0_u64;
        let mut visible_bytes = 0_u64;
        let mut copied_bytes = 0_u64;
        for index in 0..spec.plane_count() {
            let (logical_row, rows) = spec.plane_shape(index)?;
            let width = if index == 1 {
                logical_row / 2
            } else {
                logical_row
            };
            let width = u32::try_from(width).map_err(|_| GpuTransferError::Overflow)?;
            let height = u32::try_from(rows).map_err(|_| GpuTransferError::Overflow)?;
            let physical_row = u64::from(width)
                .checked_mul(4)
                .ok_or(GpuTransferError::Overflow)?;
            let align = u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let pitch = physical_row
                .checked_add(align - 1)
                .ok_or(GpuTransferError::Overflow)?
                / align
                * align;
            let pitch = u32::try_from(pitch).map_err(|_| GpuTransferError::Overflow)?;
            let bytes = u64::from(pitch)
                .checked_mul(u64::from(height))
                .ok_or(GpuTransferError::Overflow)?;
            copied_bytes = copied_bytes
                .checked_add(
                    physical_row
                        .checked_mul(u64::from(height))
                        .ok_or(GpuTransferError::Overflow)?,
                )
                .ok_or(GpuTransferError::Overflow)?;
            planes.push(PlaneCopy {
                width,
                height,
                pitch,
                offset,
            });
            offset = offset
                .checked_add(bytes)
                .ok_or(GpuTransferError::Overflow)?;
            visible_bytes = visible_bytes
                .checked_add(
                    u64::try_from(logical_row)
                        .map_err(|_| GpuTransferError::Overflow)?
                        .checked_mul(u64::from(height))
                        .ok_or(GpuTransferError::Overflow)?,
                )
                .ok_or(GpuTransferError::Overflow)?;
        }
        Ok(Self {
            planes,
            staging_bytes: offset,
            copied_bytes,
            visible_bytes,
        })
    }
}

struct Deadline {
    wait: WaitPolicy,
    start: Instant,
}

impl Deadline {
    fn new(wait: WaitPolicy) -> Self {
        Self {
            wait,
            start: Instant::now(),
        }
    }
    fn remaining(&self) -> WaitPolicy {
        match self.wait {
            WaitPolicy::Immediate => WaitPolicy::Immediate,
            WaitPolicy::Timeout(duration) => {
                WaitPolicy::Timeout(duration.saturating_sub(self.start.elapsed()))
            }
        }
    }
    fn expired_error(&self) -> GpuTransferError {
        match self.wait {
            WaitPolicy::Immediate => GpuTransferError::WouldBlock,
            WaitPolicy::Timeout(duration) => GpuTransferError::Timeout(duration),
        }
    }
    fn wait_for<T>(
        &self,
        context: &WgpuContext,
        receiver: mpsc::Receiver<T>,
        attempt: &mut FrameOperationAttempt,
    ) -> Result<T, GpuTransferError> {
        let started = Instant::now();
        loop {
            context.ensure_healthy()?;
            context.device().poll(wgpu::Maintain::Poll);
            context.ensure_healthy()?;
            match receiver.try_recv() {
                Ok(value) => {
                    attempt.record_wait(started.elapsed());
                    return Ok(value);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    attempt.record_wait(started.elapsed());
                    return Err(GpuTransferError::CompletionLost);
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            let remaining = match self.remaining() {
                WaitPolicy::Immediate => Duration::ZERO,
                WaitPolicy::Timeout(duration) => duration,
            };
            if remaining.is_zero() {
                attempt.record_wait(started.elapsed());
                return Err(self.expired_error());
            }
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
    }
}

/// The transfer service uses the pipeline's shared context and bounded pool.
pub struct GpuTransfer {
    context: Arc<WgpuContext>,
    pool: GpuPoolManager,
    stats: StatsContext,
}

impl GpuTransfer {
    fn operation_stats(&self) -> StatsContext {
        // The service is shared by every stage in one pipeline. Capture the
        // current synchronous stage scope when an operation begins; completion
        // keeps its ticket even if it runs after that scope has exited.
        self.stats.with_current_stage()
    }

    pub(crate) fn context(&self) -> Arc<WgpuContext> {
        self.context.clone()
    }

    pub(crate) fn conversion_attempt(
        &self,
        from: PixelFormat,
        to: PixelFormat,
    ) -> FrameOperationAttempt {
        self.operation_stats().begin_frame_operation(
            FrameOperationKind::Conversion {
                from,
                to,
                location: crate::statistics::ConversionLocation::Gpu,
            },
            Some(ResourceOwnerId(self.pool.id())),
        )
    }

    pub fn device_key(&self) -> DeviceKey {
        self.context.device_key()
    }

    pub(crate) fn ensure_healthy(&self) -> Result<(), GpuTransferError> {
        self.context.ensure_healthy()?;
        Ok(())
    }

    pub fn new(
        context: Arc<WgpuContext>,
        pool: GpuPoolManager,
        stats: StatsContext,
    ) -> Result<Self, GpuTransferError> {
        pool.ensure_device(&context)?;
        Ok(Self {
            context,
            pool,
            stats,
        })
    }

    fn submit_checked(&self, submit: impl FnOnce()) -> Result<(), GpuTransferError> {
        let _guard = DEVICE_ERROR_SCOPE_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let device = self.context.device();
        for filter in [
            wgpu::ErrorFilter::Internal,
            wgpu::ErrorFilter::OutOfMemory,
            wgpu::ErrorFilter::Validation,
        ] {
            device.push_error_scope(filter);
        }
        submit();
        let validation = device.pop_error_scope();
        let out_of_memory = device.pop_error_scope();
        let internal = device.pop_error_scope();
        let error = pollster::block_on(async {
            let validation = validation.await;
            let out_of_memory = out_of_memory.await;
            let internal = internal.await;
            validation.or(out_of_memory).or(internal)
        });
        match error {
            Some(error) => Err(GpuTransferError::Submission(error.to_string())),
            None => Ok(()),
        }
    }

    /// Reserve uninitialized plane textures for a future managed compute
    /// write. The caller must not publish these as readable frame content
    /// until a submitted recording has initialized every visible texel.
    pub(crate) fn reserve_write(
        &self,
        spec: FrameSpec,
        wait: WaitPolicy,
    ) -> Result<GpuPlaneSet, GpuTransferError> {
        let deadline = Deadline::new(wait);
        let plan = CopyPlan::new(spec)?;
        let mut planes = Vec::with_capacity(plan.planes.len());
        for plane in &plan.planes {
            planes.push(self.pool.try_acquire_texture(
                &self.context,
                plane.width,
                plane.height,
                wgpu::TextureFormat::Rgba8Uint,
                PLANE_USAGE,
                deadline.remaining(),
            )?);
        }
        Ok(GpuPlaneSet {
            spec,
            device: self.context.device_key(),
            planes,
        })
    }

    /// Upload initialized visible rows. NV12 requires an explicit full-range
    /// declaration; Luma8 passes `None`. The physical RGBA channels are R=Y,
    /// G/B=0, A=255 for Y and R=U, G=V, B=0, A=255 for UV. All staging
    /// padding is zeroed on every upload, including when a pooled buffer is
    /// reused.
    pub fn upload(
        &self,
        layout: &CpuLayout,
        regions: &[&[u8]],
        encoding: Option<Nv12Encoding>,
        wait: WaitPolicy,
    ) -> Result<GpuPlaneSet, GpuTransferError> {
        let spec = layout.spec();
        let owner = Some(ResourceOwnerId(self.pool.id()));
        let stats = self.operation_stats();
        let mut upload =
            stats.begin_frame_operation(FrameOperationKind::Upload(spec.pixel_format()), owner);
        let pack = stats.begin_frame_operation(
            FrameOperationKind::PlanePacking {
                format: spec.pixel_format(),
                direction: PackingDirection::Pack,
            },
            owner,
        );
        let deadline = Deadline::new(wait);
        match (spec.pixel_format(), encoding) {
            (PixelFormat::Luma8, None) | (PixelFormat::Nv12, Some(Nv12Encoding::FullRange)) => {}
            (PixelFormat::Luma8, Some(_)) => return Err(GpuTransferError::UnexpectedNv12Encoding),
            (PixelFormat::Nv12, _) => return Err(GpuTransferError::UnsupportedNv12Encoding),
        }
        let plan = CopyPlan::new(spec)?;
        let mut planes = Vec::with_capacity(spec.plane_count());
        for plane in &plan.planes {
            planes.push(self.pool.try_acquire_texture(
                &self.context,
                plane.width,
                plane.height,
                wgpu::TextureFormat::Rgba8Uint,
                PLANE_USAGE,
                deadline.remaining(),
            )?);
        }
        let staging = self.pool.try_acquire_staging(
            &self.context,
            plan.staging_bytes,
            STAGING_UPLOAD,
            deadline.remaining(),
        )?;
        self.map(
            &staging,
            wgpu::MapMode::Write,
            &deadline,
            &mut upload,
            Vec::new(),
        )?;
        let packed = (|| {
            let mut mapped = staging.buffer().slice(..).get_mapped_range_mut();
            pack_rows(layout, regions, &plan, &mut mapped)?;
            drop(mapped);
            Ok::<_, GpuTransferError>(())
        })();
        staging.buffer().unmap();
        packed?;
        pack.complete(FrameOperationBytes {
            visible: plan.visible_bytes,
            staging: plan.staging_bytes,
            transferred: 0,
        });
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vyd plane upload"),
                });
        for (plane, texture) in plan.planes.iter().zip(&planes) {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: staging.buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: plane.offset,
                        bytes_per_row: Some(plane.pitch),
                        rows_per_image: Some(plane.height),
                    },
                },
                texture_copy(texture),
                extent(plane),
            );
        }
        self.context.progress.begin_transfer()?;
        let (sender, receiver) = mpsc::channel();
        let held_staging = staging.clone();
        let held_planes = planes.clone();
        let progress = Arc::downgrade(&self.context.progress);
        self.submit_checked(|| {
            self.context.queue().submit([encoder.finish()]);
            self.context.queue().on_submitted_work_done(move || {
                drop(held_staging);
                drop(held_planes);
                let _ = sender.send(());
                if let Some(progress) = progress.upgrade() {
                    progress.end_transfer();
                }
            });
        })?;
        deadline.wait_for(&self.context, receiver, &mut upload)?;
        upload.complete(FrameOperationBytes {
            visible: plan.visible_bytes,
            staging: plan.staging_bytes,
            transferred: plan.copied_bytes,
        });
        Ok(GpuPlaneSet {
            spec,
            device: self.context.device_key(),
            planes,
        })
    }

    /// Copy all planes into a mapped staging lease. The returned guard owns
    /// the mapping and source textures until it is dropped.
    pub fn readback(
        &self,
        source: &GpuPlaneSet,
        wait: WaitPolicy,
    ) -> Result<GpuReadbackGuard, GpuTransferError> {
        let owner = Some(ResourceOwnerId(self.pool.id()));
        let stats = self.operation_stats();
        let mut download = stats.begin_frame_operation(
            FrameOperationKind::Download(source.spec.pixel_format()),
            owner,
        );
        if source.device != self.context.device_key() {
            return Err(GpuContextError::IncompatibleDevice {
                expected: self.context.device_key(),
                actual: source.device,
            }
            .into());
        }
        let deadline = Deadline::new(wait);
        let plan = CopyPlan::new(source.spec)?;
        let staging = self.pool.try_acquire_staging(
            &self.context,
            plan.staging_bytes,
            STAGING_READBACK,
            deadline.remaining(),
        )?;
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vyd plane readback"),
                });
        for (plane, texture) in plan.planes.iter().zip(&source.planes) {
            encoder.copy_texture_to_buffer(
                texture_copy(texture),
                wgpu::TexelCopyBufferInfo {
                    buffer: staging.buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: plane.offset,
                        bytes_per_row: Some(plane.pitch),
                        rows_per_image: Some(plane.height),
                    },
                },
                extent(plane),
            );
        }
        self.context.progress.begin_transfer()?;
        let held_staging = staging.clone();
        let held_planes = source.planes.clone();
        let progress = Arc::downgrade(&self.context.progress);
        self.submit_checked(|| {
            self.context.queue().submit([encoder.finish()]);
            self.context.queue().on_submitted_work_done(move || {
                drop(held_staging);
                drop(held_planes);
                if let Some(progress) = progress.upgrade() {
                    progress.end_transfer();
                }
            });
        })?;
        self.map(
            &staging,
            wgpu::MapMode::Read,
            &deadline,
            &mut download,
            source.planes.clone(),
        )?;
        download.complete(FrameOperationBytes {
            visible: plan.visible_bytes,
            staging: plan.staging_bytes,
            transferred: plan.copied_bytes,
        });
        Ok(GpuReadbackGuard {
            source: source.clone(),
            plan,
            mapped: staging.mapping_holder(),
            stats,
            owner,
        })
    }

    fn map(
        &self,
        staging: &StagingLease,
        mode: wgpu::MapMode,
        deadline: &Deadline,
        attempt: &mut FrameOperationAttempt,
        retained_planes: Vec<TextureLease>,
    ) -> Result<(), GpuTransferError> {
        self.context.progress.begin_transfer()?;
        let (sender, receiver) = mpsc::channel();
        let held = staging.clone();
        let progress = Arc::downgrade(&self.context.progress);
        staging.buffer().slice(..).map_async(mode, move |result| {
            drop(retained_planes);
            if let Err(mpsc::SendError(result)) = sender.send(result) {
                if result.is_ok() {
                    held.buffer().unmap();
                }
            }
            if let Some(progress) = progress.upgrade() {
                progress.end_transfer();
            }
        });
        deadline
            .wait_for(&self.context, receiver, attempt)?
            .map_err(|error| GpuTransferError::Map(error.to_string()))
    }
}

/// A mapped physical readback. `copy_into` writes only visible logical rows;
/// its destination can have independent regions, offsets and padded strides.
pub struct GpuReadbackGuard {
    source: GpuPlaneSet,
    plan: CopyPlan,
    mapped: StagingMapHolder,
    stats: StatsContext,
    owner: Option<ResourceOwnerId>,
}

impl GpuReadbackGuard {
    pub fn spec(&self) -> FrameSpec {
        self.source.spec
    }

    pub fn copy_into(
        &self,
        layout: &CpuLayout,
        regions: &mut [&mut [u8]],
    ) -> Result<(), GpuTransferError> {
        let unpack = self.stats.begin_frame_operation(
            FrameOperationKind::PlanePacking {
                format: self.source.spec.pixel_format(),
                direction: PackingDirection::Unpack,
            },
            self.owner,
        );
        if layout.spec() != self.source.spec {
            return Err(GpuTransferError::SpecMismatch);
        }
        let mapped = self.mapped.buffer().slice(..).get_mapped_range();
        unpack_rows(layout, regions, &self.plan, &mapped)?;
        drop(mapped);
        unpack.complete(FrameOperationBytes {
            visible: self.plan.visible_bytes,
            staging: self.plan.staging_bytes,
            transferred: 0,
        });
        Ok(())
    }
}

fn row_start(plane: &PlaneCopy, row: usize) -> Result<usize, GpuTransferError> {
    usize::try_from(plane.offset)
        .map_err(|_| GpuTransferError::Overflow)?
        .checked_add(
            row.checked_mul(plane.pitch as usize)
                .ok_or(GpuTransferError::Overflow)?,
        )
        .ok_or(GpuTransferError::Overflow)
}

fn pack_rows(
    layout: &CpuLayout,
    regions: &[&[u8]],
    plan: &CopyPlan,
    staging: &mut [u8],
) -> Result<(), GpuTransferError> {
    staging.fill(0);
    for (index, plane) in plan.planes.iter().enumerate() {
        for row in 0..plane.height as usize {
            let source = layout.row(regions, index, row)?;
            let start = row_start(plane, row)?;
            let physical = &mut staging[start..start + plane.width as usize * 4];
            if index == 0 {
                for (texel, &y) in physical.chunks_exact_mut(4).zip(source) {
                    texel.copy_from_slice(&[y, 0, 0, 255]);
                }
            } else {
                for (texel, uv) in physical.chunks_exact_mut(4).zip(source.chunks_exact(2)) {
                    texel.copy_from_slice(&[uv[0], uv[1], 0, 255]);
                }
            }
        }
    }
    Ok(())
}

fn unpack_rows(
    layout: &CpuLayout,
    regions: &mut [&mut [u8]],
    plan: &CopyPlan,
    staging: &[u8],
) -> Result<(), GpuTransferError> {
    for (index, plane) in plan.planes.iter().enumerate() {
        for row in 0..plane.height as usize {
            let destination = layout.row_mut(regions, index, row)?;
            let start = row_start(plane, row)?;
            let physical = &staging[start..start + plane.width as usize * 4];
            if index == 0 {
                for (y, texel) in destination.iter_mut().zip(physical.chunks_exact(4)) {
                    *y = texel[0];
                }
            } else {
                for (uv, texel) in destination
                    .chunks_exact_mut(2)
                    .zip(physical.chunks_exact(4))
                {
                    uv.copy_from_slice(&texel[..2]);
                }
            }
        }
    }
    Ok(())
}

fn texture_copy(texture: &TextureLease) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture: texture.texture(),
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    }
}

fn extent(plane: &PlaneCopy) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: plane.width,
        height: plane.height,
        depth_or_array_layers: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::PixelFormat;
    use crate::frame::layout::CpuPlaneLayout;
    use crate::gpu::WgpuContextOptions;
    use crate::gpu::pool::GpuPoolLimits;
    use crate::statistics::PipelineStatistics;

    fn fixture() -> Option<(GpuTransfer, Arc<PipelineStatistics>)> {
        let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Arc::new(context),
            Err(GpuContextError::Unavailable) => {
                eprintln!("GPU transfer tests unverified: no compatible adapter");
                return None;
            }
            Err(error) => panic!("GPU transfer hardware gate failed: {error}"),
        };
        eprintln!(
            "GPU transfer adapter={} backend={:?}",
            context.adapter_info().name,
            context.adapter_info().backend
        );
        let pool = GpuPoolManager::new(context.clone(), GpuPoolLimits::new(8, 128 * 1024).unwrap());
        let stats = Arc::new(PipelineStatistics::new());
        let transfer = GpuTransfer::new(
            context,
            pool,
            StatsContext::new(stats.clone()).for_stage("transfer"),
        )
        .unwrap();
        Some((transfer, stats))
    }

    fn wait() -> WaitPolicy {
        WaitPolicy::Timeout(Duration::from_secs(3))
    }

    #[test]
    fn timed_out_readback_retires_without_a_caller_poll_loop() {
        let Some((transfer, _stats)) = fixture() else {
            return;
        };
        let spec = FrameSpec::new(2, 2, PixelFormat::Luma8).unwrap();
        let layout = CpuLayout::packed(spec).unwrap();
        let pixels = [1_u8, 2, 3, 4];
        let planes = transfer.upload(&layout, &[&pixels], None, wait()).unwrap();
        let result = transfer.readback(&planes, WaitPolicy::Timeout(Duration::ZERO));
        assert!(matches!(&result, Ok(_) | Err(GpuTransferError::Timeout(_))));
        drop(result);
        drop(planes);
        // A timeout leaves callback-owned leases in flight. The context's
        // autonomous driver must settle them before bounded shutdown.
        transfer.context.shutdown(wait()).unwrap();
        assert!(transfer.pool.snapshot().unwrap().active.is_empty());
    }

    fn layout(spec: FrameSpec, distinct: bool) -> CpuLayout {
        let (y_row, y_rows) = spec.plane_shape(0).unwrap();
        if spec.plane_count() == 1 {
            return CpuLayout::new(
                spec,
                &[CpuPlaneLayout::new(0, 2, y_row + 3)],
                &[2 + (y_row + 3) * y_rows],
            )
            .unwrap();
        }
        let (uv_row, uv_rows) = spec.plane_shape(1).unwrap();
        let y_len = 2 + (y_row + 3) * y_rows;
        let uv_len = 3 + (uv_row + 4) * uv_rows;
        if distinct {
            CpuLayout::new(
                spec,
                &[
                    CpuPlaneLayout::new(0, 2, y_row + 3),
                    CpuPlaneLayout::new(1, 3, uv_row + 4),
                ],
                &[y_len, uv_len],
            )
            .unwrap()
        } else {
            CpuLayout::new(
                spec,
                &[
                    CpuPlaneLayout::new(0, 2, y_row + 3),
                    CpuPlaneLayout::new(0, y_len + 3, uv_row + 4),
                ],
                &[y_len + uv_len + 3],
            )
            .unwrap()
        }
    }

    fn test_roundtrip(format: PixelFormat, width: u32, height: u32, distinct: bool) {
        let Some((transfer, stats)) = fixture() else {
            return;
        };
        let spec = FrameSpec::new(width, height, format).unwrap();
        let input_layout = layout(spec, distinct);
        let mut source: Vec<Vec<u8>> = input_layout
            .region_lengths()
            .iter()
            .map(|&len| vec![0xE7; len])
            .collect();
        for plane in 0..spec.plane_count() {
            let (_, rows) = spec.plane_shape(plane).unwrap();
            for row in 0..rows {
                let mut regions: Vec<&mut [u8]> =
                    source.iter_mut().map(Vec::as_mut_slice).collect();
                for (col, byte) in input_layout
                    .row_mut(&mut regions, plane, row)
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    *byte = (plane * 71 + row * 13 + col * 7) as u8;
                }
            }
        }
        let input: Vec<&[u8]> = source.iter().map(Vec::as_slice).collect();
        let encoding = (format == PixelFormat::Nv12).then_some(Nv12Encoding::FullRange);
        let planes = transfer
            .upload(&input_layout, &input, encoding, wait())
            .unwrap();
        assert_eq!(planes.spec(), spec);
        assert_eq!(planes.plane_count(), spec.plane_count());
        let guard = transfer.readback(&planes, wait()).unwrap();
        let output_layout = if distinct {
            CpuLayout::packed(spec).unwrap()
        } else {
            layout(spec, true)
        };
        let mut output: Vec<Vec<u8>> = output_layout
            .region_lengths()
            .iter()
            .map(|&len| vec![0xBB; len])
            .collect();
        let mut output_regions: Vec<&mut [u8]> = output.iter_mut().map(Vec::as_mut_slice).collect();
        guard
            .copy_into(&output_layout, &mut output_regions)
            .unwrap();
        drop(output_regions);
        for plane in 0..spec.plane_count() {
            let (_, rows) = spec.plane_shape(plane).unwrap();
            for row in 0..rows {
                let output_refs: Vec<&[u8]> = output.iter().map(Vec::as_slice).collect();
                assert_eq!(
                    input_layout.row(&input, plane, row).unwrap(),
                    output_layout.row(&output_refs, plane, row).unwrap()
                );
            }
        }
        let mut visible = output
            .iter()
            .map(|region| vec![false; region.len()])
            .collect::<Vec<_>>();
        for plane in 0..spec.plane_count() {
            let region = output_layout.plane(plane).unwrap().region();
            let (_, rows) = spec.plane_shape(plane).unwrap();
            for row in 0..rows {
                for index in output_layout.row_range(plane, row).unwrap() {
                    visible[region][index] = true;
                }
            }
        }
        for (region, mask) in output.iter().zip(visible) {
            for (&byte, visible) in region.iter().zip(mask) {
                if !visible {
                    assert_eq!(byte, 0xBB, "output padding was modified");
                }
            }
        }
        let operations = &stats.snapshot().frame_operations["transfer"];
        let prefix = match format {
            PixelFormat::Luma8 => "luma8",
            PixelFormat::Nv12 => "nv12",
        };
        let upload = operations
            .iter()
            .find(|(name, _)| name.starts_with(&format!("upload_{prefix}@")))
            .unwrap()
            .1;
        let download = operations
            .iter()
            .find(|(name, _)| name.starts_with(&format!("download_{prefix}@")))
            .unwrap()
            .1;
        let plan = CopyPlan::new(spec).unwrap();
        for op in [upload, download] {
            assert_eq!((op.attempted, op.completed, op.failed), (1, 1, 0));
            assert_eq!(op.visible_bytes, plan.visible_bytes);
            assert_eq!(op.staging_bytes, plan.staging_bytes);
            assert_eq!(op.transfer_bytes, plan.copied_bytes);
        }
        assert!(plan.staging_bytes > plan.copied_bytes);
    }

    #[test]
    fn luma_odd_padded_roundtrip() {
        test_roundtrip(PixelFormat::Luma8, 3, 3, false);
    }

    #[test]
    fn nv12_odd_distinct_regions_roundtrip() {
        test_roundtrip(PixelFormat::Nv12, 3, 3, true);
    }

    #[test]
    fn nv12_even_shared_region_roundtrip() {
        test_roundtrip(PixelFormat::Nv12, 4, 2, false);
    }

    #[test]
    fn packing_initializes_every_padding_byte_and_unused_channel() {
        let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
        let layout = CpuLayout::packed(spec).unwrap();
        let plan = CopyPlan::new(spec).unwrap();
        let source = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
        let mut physical = vec![0xC9; plan.staging_bytes as usize];
        pack_rows(&layout, &[&source], &plan, &mut physical).unwrap();
        for (index, plane) in plan.planes.iter().enumerate() {
            for row in 0..plane.height as usize {
                let start = row_start(plane, row).unwrap();
                let copied = plane.width as usize * 4;
                let visible = &physical[start..start + copied];
                if index == 0 {
                    assert!(
                        visible
                            .chunks_exact(4)
                            .all(|texel| texel[1..] == [0, 0, 255])
                    );
                } else {
                    assert!(visible.chunks_exact(4).all(|texel| texel[2..] == [0, 255]));
                }
                assert!(
                    physical[start + copied..start + plane.pitch as usize]
                        .iter()
                        .all(|&byte| byte == 0)
                );
            }
        }
        let mut recovered = vec![0_u8; source.len()];
        unpack_rows(&layout, &mut [&mut recovered], &plan, &physical).unwrap();
        assert_eq!(recovered, source);
    }

    #[test]
    fn mapped_guard_retains_source_and_staging_until_release() {
        let Some((transfer, _)) = fixture() else {
            return;
        };
        let spec = FrameSpec::new(3, 3, PixelFormat::Luma8).unwrap();
        let layout = CpuLayout::packed(spec).unwrap();
        let source = [9_u8; 9];
        let planes = transfer.upload(&layout, &[&source], None, wait()).unwrap();
        let guard = transfer.readback(&planes, wait()).unwrap();
        drop(planes);
        let active = transfer.pool.snapshot().unwrap().active;
        assert_eq!(
            active.len(),
            2,
            "source texture and mapped staging stay active"
        );
        drop(guard);
        assert!(transfer.pool.snapshot().unwrap().active.is_empty());
    }

    #[test]
    fn invalid_readback_destination_fails_unpack_without_completed_bytes() {
        let Some((transfer, stats)) = fixture() else {
            return;
        };
        let spec = FrameSpec::new(3, 3, PixelFormat::Luma8).unwrap();
        let layout = CpuLayout::packed(spec).unwrap();
        let source = [9_u8; 9];
        let planes = transfer.upload(&layout, &[&source], None, wait()).unwrap();
        let guard = transfer.readback(&planes, wait()).unwrap();
        let mut short = [0_u8; 1];
        assert!(matches!(
            guard.copy_into(&layout, &mut [&mut short]),
            Err(GpuTransferError::Layout(LayoutError::ShortRegion))
        ));
        let operations = &stats.snapshot().frame_operations["transfer"];
        let unpack = operations
            .iter()
            .find(|(name, _)| name.starts_with("unpack_luma8@"))
            .unwrap()
            .1;
        assert_eq!(
            (unpack.attempted, unpack.failed, unpack.completed),
            (1, 1, 0)
        );
        assert_eq!(unpack.visible_bytes, 0);
        assert_eq!(unpack.staging_bytes, 0);
    }

    #[test]
    fn short_source_fails_without_completed_bytes() {
        let Some((transfer, stats)) = fixture() else {
            return;
        };
        let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
        let layout = layout(spec, true);
        let source = vec![0_u8; layout.region_lengths()[0]];
        let short_uv = [0_u8; 1];
        assert!(matches!(
            transfer.upload(
                &layout,
                &[&source, &short_uv],
                Some(Nv12Encoding::FullRange),
                wait()
            ),
            Err(GpuTransferError::Layout(LayoutError::ShortRegion))
        ));
        let operations = &stats.snapshot().frame_operations["transfer"];
        for (name, operation) in operations {
            if name.starts_with("upload_nv12@") || name.starts_with("pack_nv12@") {
                assert_eq!(
                    (operation.attempted, operation.failed, operation.completed),
                    (1, 1, 0)
                );
                assert_eq!(operation.visible_bytes, 0);
                assert_eq!(operation.staging_bytes, 0);
                assert_eq!(operation.transfer_bytes, 0);
            }
        }
    }

    #[test]
    fn nv12_upload_rejects_unstated_or_limited_range() {
        let Some((transfer, stats)) = fixture() else {
            return;
        };
        let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
        let layout = CpuLayout::packed(spec).unwrap();
        let source = [0_u8; 17];
        for encoding in [
            None,
            Some(Nv12Encoding::Unknown),
            Some(Nv12Encoding::LimitedRange),
        ] {
            assert!(matches!(
                transfer.upload(&layout, &[&source], encoding, wait()),
                Err(GpuTransferError::UnsupportedNv12Encoding)
            ));
        }
        let operations = &stats.snapshot().frame_operations["transfer"];
        for (name, operation) in operations {
            if name.starts_with("upload_nv12@") || name.starts_with("pack_nv12@") {
                assert_eq!(
                    (operation.attempted, operation.failed, operation.completed),
                    (3, 3, 0)
                );
                assert_eq!(operation.visible_bytes, 0);
                assert_eq!(operation.staging_bytes, 0);
            }
        }
    }
}
