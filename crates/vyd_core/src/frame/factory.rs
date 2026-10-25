//! Initial pooled CPU frame creation. The legacy image mutation API still
//! detaches to a counted heap copy; use Frame::try_update_pooled_luma8 for
//! bounded writes through the issuing pool.
use super::layout::{CpuLayout, FrameSpec};
use super::pool::{BufferPoolManager, PoolLimits, WaitPolicy};
use super::{CpuNv12Frame, CpuStorage, Frame, FrameError, Nv12Encoding, PixelFormat};
use std::time::Instant;

const DEFAULT_MAX_BUFFERS: usize = 32;
const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone)]
pub struct FrameFactory {
    pool: BufferPoolManager,
}

impl Default for FrameFactory {
    fn default() -> Self {
        let limits = PoolLimits::new(Some(DEFAULT_MAX_BUFFERS), Some(DEFAULT_MAX_BYTES))
            .expect("positive default frame pool limits");
        Self::new(BufferPoolManager::new(limits))
    }
}

impl FrameFactory {
    pub fn new(pool: BufferPoolManager) -> Self {
        Self { pool }
    }

    pub fn cpu_pool(&self) -> &BufferPoolManager {
        &self.pool
    }

    /// Allocate a zero-initialized, packed CPU NV12 frame. The caller must
    /// declare the source's full-range encoding before allocation.
    #[track_caller]
    pub fn create_nv12(
        &self,
        width: u32,
        height: u32,
        encoding: Nv12Encoding,
        wait: WaitPolicy,
    ) -> Result<Frame, FrameError> {
        let spec = FrameSpec::new(width, height, PixelFormat::Nv12)?;
        self.create_nv12_with_layout(CpuLayout::packed(spec)?, encoding, wait)
    }

    /// Allocate checked NV12 regions, including independently backed planes.
    /// A single timeout covers every region acquisition.
    #[track_caller]
    pub fn create_nv12_with_layout(
        &self,
        layout: CpuLayout,
        encoding: Nv12Encoding,
        wait: WaitPolicy,
    ) -> Result<Frame, FrameError> {
        if encoding != Nv12Encoding::FullRange {
            return Err(FrameError::UnsupportedNv12Encoding);
        }
        if layout.spec().pixel_format() != PixelFormat::Nv12 {
            return Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Nv12,
                actual: layout.spec().pixel_format(),
            });
        }
        let started = Instant::now();
        let mut regions = Vec::with_capacity(layout.region_lengths().len());
        for &length in layout.region_lengths() {
            let remaining = match wait {
                WaitPolicy::Immediate => WaitPolicy::Immediate,
                WaitPolicy::Timeout(limit) => {
                    WaitPolicy::Timeout(limit.saturating_sub(started.elapsed()))
                }
            };
            regions.push(CpuStorage::from_shared(
                self.pool.try_acquire_zeroed(length, remaining)?,
            ));
        }
        Ok(Frame::from_cpu_nv12(CpuNv12Frame::from_regions(
            layout, encoding, regions,
        )?))
    }

    /// Create a zero-initialized packed Luma8 frame backed by this factory's
    /// pool. The returned value is an ordinary Frame.
    #[track_caller]
    pub fn create_luma8(
        &self,
        width: u32,
        height: u32,
        wait: WaitPolicy,
    ) -> Result<Frame, FrameError> {
        let spec = FrameSpec::new(width, height, PixelFormat::Luma8)?;
        let (row_bytes, rows) = spec.plane_shape(0)?;
        let bytes = row_bytes
            .checked_mul(rows)
            .ok_or(super::layout::LayoutError::Overflow)?;
        let lease = self.pool.try_acquire_zeroed(bytes, wait)?;
        Ok(Frame::from_shared(lease, width, height)?)
    }

    /// Fill the uniquely leased backing before publishing it as a Frame.
    /// Reused contents are initialized but are not cleared. A failed fill
    /// returns the lease to its issuing pool.
    #[track_caller]
    pub fn create_luma8_with(
        &self,
        width: u32,
        height: u32,
        wait: WaitPolicy,
        fill: impl FnOnce(&mut [u8]) -> Result<(), FrameError>,
    ) -> Result<Frame, FrameError> {
        let spec = FrameSpec::new(width, height, PixelFormat::Luma8)?;
        let (row_bytes, rows) = spec.plane_shape(0)?;
        let bytes = row_bytes
            .checked_mul(rows)
            .ok_or(super::layout::LayoutError::Overflow)?;
        let mut lease = self.pool.try_acquire(bytes, wait)?;
        fill(lease.try_write(WaitPolicy::Immediate)?)?;
        Ok(Frame::from_shared(lease, width, height)?)
    }
}
