use image::{GrayImage, ImageBuffer, Luma};
use std::any::Any;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::statistics::{MemoryOperation, StatsContext};
use pool::{PoolError, WaitPolicy};

pub mod conversion;
pub mod factory;
pub mod layout;
pub mod pool;
mod replica;
pub use replica::{CpuFrameRead, CpuFrameWrite};
#[cfg(feature = "gpu-wgpu")]
pub use replica::{GpuFrameRead, GpuFrameWrite, ManagedGpuFrame};

static NEXT_CPU_STORAGE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct ReplicaCache {
    stamp: Vec<(u64, u64)>,
    converted: Option<Box<Frame>>,
    #[cfg(feature = "gpu-wgpu")]
    gpu: Vec<crate::gpu::transfer::GpuPlaneSet>,
}

impl ReplicaCache {
    fn refresh(&mut self, stamp: Vec<(u64, u64)>) {
        if self.stamp != stamp {
            self.stamp = stamp;
            self.converted = None;
            #[cfg(feature = "gpu-wgpu")]
            self.gpu.clear();
        }
    }
}

/// Errors that can occur when working with frames
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("Invalid CPU frame layout: {0}")]
    Layout(#[from] layout::LayoutError),
    #[error("CPU frame pool error: {0}")]
    Pool(#[from] pool::PoolError),
    #[error("GPU download timeout after {duration:?}")]
    GpuDownloadTimeout { duration: Duration },

    #[error("GPU sync failed: {message}")]
    GpuSyncFailed { message: String },

    #[error("GPU download failed: {message}")]
    GpuDownloadFailed { message: String },

    #[error("GPU upload not yet implemented")]
    GpuUploadNotImplemented,

    #[error("Frame buffer is unavailable")]
    BufferUnavailable,

    #[error("CPU frame is not backed by a writable Vyd pool lease")]
    NotPooledCpuFrame,

    #[error("Invalid frame dimensions: {width}x{height} with {data_len} bytes")]
    InvalidDimensions {
        width: u32,
        height: u32,
        data_len: usize,
    },

    #[error("GPU conversion required")]
    GpuConversionRequired,

    #[error("requested {requested:?} access to a {actual:?} frame requires format conversion")]
    FormatConversionRequired {
        requested: PixelFormat,
        actual: PixelFormat,
    },

    #[error("NV12 requires an explicit full-range grayscale encoding declaration")]
    UnsupportedNv12Encoding,

    #[cfg(feature = "gpu-wgpu")]
    #[error(transparent)]
    GpuTransfer(#[from] crate::gpu::transfer::GpuTransferError),

    #[cfg(feature = "gpu-wgpu")]
    #[error(transparent)]
    GpuSubmission(#[from] crate::gpu::submission::GpuSubmissionError),

    #[cfg(feature = "gpu-wgpu")]
    #[error(transparent)]
    GpuRecording(#[from] crate::gpu::recording::GpuRecordingError),

    #[error("legacy GPU handle does not support managed requested-format access")]
    UnsupportedLegacyGpuAccess,
}

/// Pixel formats supported by frame backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    Nv12,
    Luma8,
}

pub type GpuPixelFormat = PixelFormat;

/// The unified Frame type.
#[derive(Clone)]
pub enum Frame {
    Cpu(CpuFrame),
    CpuNv12(CpuNv12Frame),
    Gpu(Arc<dyn GpuFrame>),
    #[cfg(feature = "gpu-wgpu")]
    ManagedGpu(ManagedGpuFrame),
}

pub type CpuImgBuf = ImageBuffer<Luma<u8>, CpuStorage>;

/// Where the current CPU allocation came from. This does not say whether the
/// frame is the only reference or can be mutated in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuAllocationOrigin {
    Heap,
    Pool,
    External,
}

/// A simple CPU-backed frame type for grayscale images (GRAY8).
pub struct CpuFrame {
    /// Legacy packed-Luma8 image access. Mutating this image can detach a pool
    /// lease to a counted heap copy. Use `try_write_luma8` for bounded writes.
    pub img: CpuImgBuf,
    cache: ReplicaCache,
}

impl Clone for CpuFrame {
    fn clone(&self) -> Self {
        Self {
            img: self.img.clone(),
            cache: ReplicaCache::default(),
        }
    }
}

/// The only NV12 sample interpretation supported by this milestone. Importers
/// must map their source metadata to this value explicitly; unknown and
/// limited-range inputs are rejected rather than relabelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nv12Encoding {
    FullRange,
    LimitedRange,
    Unknown,
}

/// Managed CPU NV12 storage. Regions are private so callers can only access
/// visible Y and UV row bytes through the checked layout.
pub struct CpuNv12Frame {
    layout: layout::CpuLayout,
    regions: Vec<CpuStorage>,
    encoding: Nv12Encoding,
    cache: ReplicaCache,
}

impl Clone for CpuNv12Frame {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            regions: self.regions.clone(),
            encoding: self.encoding,
            cache: ReplicaCache::default(),
        }
    }
}

pub struct CpuNv12Read<'a> {
    frame: &'a CpuNv12Frame,
}

impl CpuNv12Read<'_> {
    pub fn layout(&self) -> &layout::CpuLayout {
        &self.frame.layout
    }
    pub fn encoding(&self) -> Nv12Encoding {
        self.frame.encoding
    }
    pub fn row(&self, plane: usize, row: usize) -> Result<&[u8], FrameError> {
        let plane_layout = self
            .frame
            .layout
            .plane(plane)
            .ok_or(layout::LayoutError::WrongPlaneCount)?;
        let range = self
            .frame
            .layout
            .row_range(plane, row)
            .ok_or(layout::LayoutError::InvalidRow)?;
        let region = self
            .frame
            .regions
            .get(plane_layout.region())
            .ok_or(layout::LayoutError::MissingRegion)?;
        if region.len() < self.frame.layout.region_lengths()[plane_layout.region()] {
            return Err(layout::LayoutError::ShortRegion.into());
        }
        Ok(&region[range])
    }
}

pub struct CpuNv12Write<'a> {
    frame: &'a mut CpuNv12Frame,
    wait: WaitPolicy,
    started: Instant,
}

impl CpuNv12Write<'_> {
    pub fn layout(&self) -> &layout::CpuLayout {
        &self.frame.layout
    }
    pub fn encoding(&self) -> Nv12Encoding {
        self.frame.encoding
    }
    pub fn row_mut(&mut self, plane: usize, row: usize) -> Result<&mut [u8], FrameError> {
        let plane_layout = self
            .frame
            .layout
            .plane(plane)
            .ok_or(layout::LayoutError::WrongPlaneCount)?;
        let range = self
            .frame
            .layout
            .row_range(plane, row)
            .ok_or(layout::LayoutError::InvalidRow)?;
        let region = self
            .frame
            .regions
            .get_mut(plane_layout.region())
            .ok_or(layout::LayoutError::MissingRegion)?;
        if region.len() < self.frame.layout.region_lengths()[plane_layout.region()] {
            return Err(layout::LayoutError::ShortRegion.into());
        }
        let wait = match self.wait {
            WaitPolicy::Immediate => WaitPolicy::Immediate,
            WaitPolicy::Timeout(limit) => {
                WaitPolicy::Timeout(limit.saturating_sub(self.started.elapsed()))
            }
        };
        let bytes = region.try_write_luma8(wait)?;
        self.frame.cache = ReplicaCache::default();
        Ok(&mut bytes[range])
    }
}

impl CpuNv12Frame {
    /// Wrap initialized regions described by a checked NV12 layout. Physical
    /// padding remains inaccessible through the row guards.
    pub fn from_regions(
        layout: layout::CpuLayout,
        encoding: Nv12Encoding,
        regions: Vec<CpuStorage>,
    ) -> Result<Self, FrameError> {
        if encoding != Nv12Encoding::FullRange {
            return Err(FrameError::UnsupportedNv12Encoding);
        }
        if layout.spec().pixel_format() != PixelFormat::Nv12 {
            return Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Nv12,
                actual: layout.spec().pixel_format(),
            });
        }
        if regions.len() != layout.region_lengths().len() {
            return Err(layout::LayoutError::MissingRegion.into());
        }
        for (region, length) in regions.iter().zip(layout.region_lengths()) {
            if region.len() < *length {
                return Err(layout::LayoutError::ShortRegion.into());
            }
        }
        Ok(Self {
            layout,
            regions,
            encoding,
            cache: ReplicaCache::default(),
        })
    }

    pub fn from_packed(
        width: u32,
        height: u32,
        encoding: Nv12Encoding,
        bytes: Vec<u8>,
    ) -> Result<Self, FrameError> {
        let spec = layout::FrameSpec::new(width, height, PixelFormat::Nv12)?;
        let layout = layout::CpuLayout::packed(spec)?;
        Self::from_regions(layout, encoding, vec![CpuStorage::from_vec(bytes)])
    }

    pub fn layout(&self) -> &layout::CpuLayout {
        &self.layout
    }
    pub fn encoding(&self) -> Nv12Encoding {
        self.encoding
    }
    pub fn dimensions(&self) -> (u32, u32) {
        self.layout.spec().dimensions()
    }
    pub fn try_read(&self, _wait: WaitPolicy) -> Result<CpuNv12Read<'_>, FrameError> {
        Ok(CpuNv12Read { frame: self })
    }
    pub fn try_write(&mut self, wait: WaitPolicy) -> Result<CpuNv12Write<'_>, FrameError> {
        Ok(CpuNv12Write {
            frame: self,
            wait,
            started: Instant::now(),
        })
    }
}

/// A trait for shared CPU storage backends that can be efficiently cloned and potentially
/// converted back to owned storage. This enables zero-copy optimizations when possible.
pub trait CpuStorageShared: Deref<Target = [u8]> + Send + Sync + 'static {
    /// Clone this shared storage (object-safe version of Clone)
    fn clone_shared(&self) -> Box<dyn CpuStorageShared>;

    /// Transfer a Vec<u8> without copying if this is the only reference.
    /// Backends that must copy return `Err(self)` so the caller can count it.
    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>>;

    /// Allocation provenance, independent of current reference count.
    fn allocation_origin(&self) -> CpuAllocationOrigin {
        CpuAllocationOrigin::External
    }

    /// Return a bounded writable view for managed pool storage. Other shared
    /// backends keep their existing immutable contract.
    fn try_write_pooled(&mut self, _wait: WaitPolicy) -> Result<Option<&mut [u8]>, PoolError> {
        Ok(None)
    }
}

/// Internal storage variants - implementation detail hidden from users
enum StorageInner {
    /// Heap storage shared until a writer requests mutable access.
    Owned(Arc<Vec<u8>>),
    /// Immutable shared storage (e.g., from a frame buffer pool)
    Shared(Box<dyn CpuStorageShared>),
}

impl Clone for StorageInner {
    fn clone(&self) -> Self {
        match self {
            StorageInner::Owned(v) => StorageInner::Owned(Arc::clone(v)),
            StorageInner::Shared(shared) => StorageInner::Shared(shared.clone_shared()),
        }
    }
}

/// CPU-based image storage that can be either owned or shared with copy-on-write semantics.
pub struct CpuStorage {
    inner: StorageInner,
    version: Arc<(u64, AtomicU64)>,
}

impl Clone for CpuStorage {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            version: self.version.clone(),
        }
    }
}

fn new_storage_version() -> Arc<(u64, AtomicU64)> {
    Arc::new((
        NEXT_CPU_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
        AtomicU64::new(0),
    ))
}

impl CpuStorage {
    /// Create new owned storage from a Vec<u8>
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            inner: StorageInner::Owned(Arc::new(vec)),
            version: new_storage_version(),
        }
    }

    /// Create new shared storage from any type implementing CpuStorageShared
    pub fn from_shared<T: CpuStorageShared>(shared: T) -> Self {
        Self {
            inner: StorageInner::Shared(Box::new(shared)),
            version: new_storage_version(),
        }
    }

    /// Convert to owned storage, consuming self.
    ///
    /// This transfers a uniquely owned Vec without copying when the backend
    /// supports it; otherwise it makes one counted heap copy. A pooled backing
    /// is returned to its issuing pool when its final reference is released.
    pub fn to_owned(self) -> Self {
        match self.inner {
            StorageInner::Owned(data) => {
                if Arc::strong_count(&data) == 1 {
                    Self {
                        inner: StorageInner::Owned(data),
                        version: new_storage_version(),
                    }
                } else {
                    StatsContext::record_memory_operation(MemoryOperation::CpuCopy(data.len()));
                    StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(
                        data.len(),
                    ));
                    Self::from_vec((*data).clone())
                }
            }
            StorageInner::Shared(shared) => Self::convert_shared_to_owned(shared),
        }
    }

    /// True only for a unique heap `Vec`, not for pooled provenance. Use
    /// `allocation_origin` when the question is where storage came from.
    pub fn is_owned(&self) -> bool {
        matches!(&self.inner, StorageInner::Owned(data) if Arc::strong_count(data) == 1)
    }

    /// Allocation provenance. A pool-origin buffer may be shared by clones;
    /// an exclusive heap buffer is still heap-origin.
    pub fn allocation_origin(&self) -> CpuAllocationOrigin {
        match &self.inner {
            StorageInner::Owned(_) => CpuAllocationOrigin::Heap,
            StorageInner::Shared(shared) => shared.allocation_origin(),
        }
    }

    /// Internal helper to convert shared storage to owned
    fn convert_shared_to_owned(shared: Box<dyn CpuStorageShared>) -> Self {
        let data_len = shared.len();

        // Try to steal the buffer first (zero-copy)
        match shared.try_into_owned() {
            Ok(vec) => {
                tracing::trace!("Transferred shared Vec without copying");
                StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
                Self {
                    inner: StorageInner::Owned(Arc::new(vec)),
                    version: new_storage_version(),
                }
            }
            Err(shared) => {
                tracing::trace!("Shared storage requires one heap copy for owned access");
                StatsContext::record_memory_operation(MemoryOperation::CpuCopy(data_len));
                StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(data_len));
                let vec = shared.to_vec();
                Self {
                    inner: StorageInner::Owned(Arc::new(vec)),
                    version: new_storage_version(),
                }
            }
        }
    }

    /// Create a shared clone of this storage.
    /// If already shared, increments reference count.
    /// If owned, converts to shared.
    pub fn clone_shared(&self) -> CpuStorage {
        match &self.inner {
            StorageInner::Shared(shared) => {
                // Use the CpuStorageShared::clone_shared method
                let new_shared = shared.clone_shared();
                CpuStorage {
                    inner: StorageInner::Shared(new_shared),
                    version: self.version.clone(),
                }
            }
            StorageInner::Owned(vec) => {
                // Convert to shared storage
                let shared_vec = SharedVec {
                    inner: Arc::clone(vec),
                };
                CpuStorage {
                    inner: StorageInner::Shared(Box::new(shared_vec)),
                    version: self.version.clone(),
                }
            }
        }
    }

    fn try_write_pooled(&mut self, wait: WaitPolicy) -> Result<Option<&mut [u8]>, PoolError> {
        let result = match &mut self.inner {
            StorageInner::Shared(shared) => shared.try_write_pooled(wait),
            StorageInner::Owned(_) => Ok(None),
        }?;
        if result.is_some() {
            self.version.1.fetch_add(1, Ordering::Relaxed);
        }
        Ok(result)
    }

    fn stamp(&self) -> (u64, u64) {
        (self.version.0, self.version.1.load(Ordering::Relaxed))
    }

    fn try_write_luma8(&mut self, wait: WaitPolicy) -> Result<&mut [u8], FrameError> {
        if self.allocation_origin() == CpuAllocationOrigin::Pool {
            return self
                .try_write_pooled(wait)?
                .ok_or(FrameError::NotPooledCpuFrame);
        }
        // Legacy or external storage has no issuing pool. Its COW path makes
        // a counted heap copy when the bytes cannot be transferred.
        Ok(&mut *self)
    }
}

impl Deref for CpuStorage {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.inner {
            StorageInner::Owned(v) => v.as_slice(),
            StorageInner::Shared(shared) => shared,
        }
    }
}

impl DerefMut for CpuStorage {
    /// Get mutable access to the underlying data.
    ///
    /// Shared storage transfers a unique Vec when possible; other backings
    /// detach through one counted heap copy.
    fn deref_mut(&mut self) -> &mut Self::Target {
        if matches!(self.inner, StorageInner::Shared(_)) {
            *self = std::mem::take(self).to_owned();
        }

        self.version.1.fetch_add(1, Ordering::Relaxed);

        match &mut self.inner {
            StorageInner::Owned(v) => {
                if Arc::strong_count(v) > 1 {
                    let bytes = v.len();
                    StatsContext::record_memory_operation(MemoryOperation::CpuCopy(bytes));
                    StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(bytes));
                }
                Arc::make_mut(v).as_mut_slice()
            }
            StorageInner::Shared(_) => unreachable!("to_owned should always return owned"),
        }
    }
}

impl Default for CpuStorage {
    fn default() -> Self {
        Self::from_vec(Vec::new())
    }
}

/// Wrapper for Arc<Vec<u8>> that can be unwrapped efficiently
#[derive(Clone)]
pub struct SharedVec {
    inner: Arc<Vec<u8>>,
}

impl SharedVec {
    pub fn new(vec: Vec<u8>) -> Self {
        Self {
            inner: Arc::new(vec),
        }
    }
}

impl Deref for SharedVec {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl CpuStorageShared for SharedVec {
    fn clone_shared(&self) -> Box<dyn CpuStorageShared> {
        Box::new(self.clone())
    }

    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>> {
        match Arc::try_unwrap(self.inner) {
            Ok(vec) => Ok(vec),
            Err(arc) => Err(Box::new(SharedVec { inner: arc }) as Box<dyn CpuStorageShared>),
        }
    }

    fn allocation_origin(&self) -> CpuAllocationOrigin {
        CpuAllocationOrigin::Heap
    }
}

// Also support Arc<[u8]> but it can't be unwrapped efficiently
impl CpuStorageShared for Arc<[u8]> {
    fn clone_shared(&self) -> Box<dyn CpuStorageShared> {
        Box::new(self.clone())
    }

    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>> {
        // Arc<[u8]> can't be unwrapped to avoid copying, so always fail
        Err(self as Box<dyn CpuStorageShared>)
    }
}

/// image 0.24 exposes only shared access to its container. Move it out for a
/// scoped fallible write and restore it even when the callback unwinds.
struct PooledImageWrite<'a> {
    target: &'a mut CpuImgBuf,
    storage: CpuStorage,
    width: u32,
    height: u32,
}

impl<'a> PooledImageWrite<'a> {
    fn new(target: &'a mut CpuImgBuf) -> Self {
        let placeholder = ImageBuffer::from_raw(0, 0, CpuStorage::default())
            .expect("empty image placeholder is valid");
        let image = std::mem::replace(target, placeholder);
        let (width, height) = image.dimensions();
        Self {
            target,
            storage: image.into_raw(),
            width,
            height,
        }
    }
}

/// Scoped mutable access to packed Luma8 bytes. Dropping the guard restores
/// the legacy image wrapper, including during unwinding.
pub struct CpuLuma8Write<'a> {
    image: PooledImageWrite<'a>,
}

impl Deref for CpuLuma8Write<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.image.storage
    }
}

impl CpuLuma8Write<'_> {
    /// Obtain mutable bytes. Access remains fallible even for a backend whose
    /// writable mapping can disappear after the guard was acquired.
    pub fn try_bytes_mut(&mut self) -> Result<&mut [u8], FrameError> {
        self.image.storage.try_write_luma8(WaitPolicy::Immediate)
    }
}

impl Drop for PooledImageWrite<'_> {
    fn drop(&mut self) {
        let storage = std::mem::take(&mut self.storage);
        *self.target = ImageBuffer::from_raw(self.width, self.height, storage)
            .expect("original image dimensions remain valid");
    }
}

impl From<Vec<u8>> for CpuStorage {
    fn from(v: Vec<u8>) -> Self {
        CpuStorage::from_vec(v)
    }
}

impl From<Arc<[u8]>> for CpuStorage {
    fn from(a: Arc<[u8]>) -> Self {
        CpuStorage::from_shared(a)
    }
}

impl From<SharedVec> for CpuStorage {
    fn from(sv: SharedVec) -> Self {
        CpuStorage::from_shared(sv)
    }
}

/// Trait representing a GPU-backed frame handle.
/// Implement this for specific backends (GLMemory, Metal texture, Vulkan image, etc).
pub trait GpuFrame: Any + Send + Sync + fmt::Debug {
    /// Image dimensions
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn pixel_format(&self) -> GpuPixelFormat;

    /// Blocking download of GPU memory to a CPU frame.
    /// If possible, this should be implemented efficiently (zero-copy fallback where possible).
    fn download_to_cpu(&self, timeout: Option<Duration>) -> Result<CpuFrame, FrameError>;

    /// Optional: make sure GPU work producing this texture has completed.
    /// Should be cheap if a fence is already available.
    fn sync_gpu(&self) -> Result<(), FrameError> {
        Ok(())
    }

    /// Synchronize GPU work and download to CPU in one operation.
    /// This is the recommended way to download GPU frames as it combines sync and download efficiently.
    fn sync_and_download(&self, timeout: Option<Duration>) -> Result<CpuFrame, FrameError> {
        self.sync_gpu()?;
        self.download_to_cpu(timeout)
    }

    /// Get platform-specific texture handle for OpenGL
    #[cfg(target_os = "macos")]
    fn as_gl_texture(&self) -> Option<u32> {
        None // Default implementation - override in concrete types
    }

    /// Get platform-specific texture handle for Metal
    #[cfg(target_os = "macos")]
    fn as_metal_texture(&self) -> Option<*mut std::ffi::c_void> {
        None // Default implementation - override in concrete types
    }

    /// Legacy backend-specific access. `Frame::try_gpu_downcast` uses the actual
    /// handle's `Any` identity rather than trusting this method's return value.
    fn as_any(&self) -> &dyn Any;
}

impl CpuFrame {
    /// Borrow initialized packed Luma8 bytes. CPU storage is already resident,
    /// so neither wait policy blocks; the parameter aligns this access with
    /// future staging-backed Frame access.
    pub fn try_read_luma8(&self, _wait: WaitPolicy) -> Result<&[u8], FrameError> {
        Ok(self.img.as_raw())
    }

    /// Borrow writable packed Luma8 bytes. Call `try_bytes_mut` on the guard.
    /// Shared pool storage performs COW
    /// within its issuing pool, waiting up to `wait`; failed acquisition leaves
    /// this frame and retained readers unchanged. Heap-origin storage uses
    /// counted heap COW. The legacy `img` field can detach a pool lease to heap.
    pub fn try_write_luma8(&mut self, wait: WaitPolicy) -> Result<CpuLuma8Write<'_>, FrameError> {
        let Self { img, cache } = self;
        let mut image = PooledImageWrite::new(img);
        image.storage.try_write_luma8(wait)?;
        *cache = ReplicaCache::default();
        Ok(CpuLuma8Write { image })
    }

    /// Mutate packed Luma8 bytes through a bounded Vyd pool lease. Shared
    /// storage detaches within its issuing pool; acquisition failure preserves
    /// this frame. A callback error after mutation cannot roll back its writes.
    pub fn try_update_pooled<R>(
        &mut self,
        wait: WaitPolicy,
        update: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<R, FrameError> {
        let Self { img, cache } = self;
        let mut guard = PooledImageWrite::new(img);
        let bytes = guard
            .storage
            .try_write_pooled(wait)?
            .ok_or(FrameError::NotPooledCpuFrame)?;
        *cache = ReplicaCache::default();
        Ok(update(bytes))
    }

    pub fn from_vec(width: u32, height: u32, v: Vec<u8>) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::InvalidDimensions {
                width,
                height,
                data_len: v.len(),
            });
        }
        let expected_len = width
            .checked_mul(height)
            .map(|pixels| pixels as usize)
            .ok_or(FrameError::InvalidDimensions {
                width,
                height,
                data_len: v.len(),
            })?;
        if v.len() != expected_len {
            return Err(FrameError::InvalidDimensions {
                width,
                height,
                data_len: v.len(),
            });
        }

        let storage = CpuStorage::from_vec(v);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage)
            .expect("dimensions validated above");
        Ok(CpuFrame {
            img,
            cache: ReplicaCache::default(),
        })
    }

    pub fn from_shared(
        width: u32,
        height: u32,
        v: impl CpuStorageShared,
    ) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::InvalidDimensions {
                width,
                height,
                data_len: v.len(),
            });
        }
        let expected_len = width
            .checked_mul(height)
            .map(|pixels| pixels as usize)
            .ok_or(FrameError::InvalidDimensions {
                width,
                height,
                data_len: v.len(),
            })?;
        if v.len() != expected_len {
            return Err(FrameError::InvalidDimensions {
                width,
                height,
                data_len: v.len(),
            });
        }

        let storage = CpuStorage::from_shared(v);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage)
            .expect("dimensions validated above");
        Ok(CpuFrame {
            img,
            cache: ReplicaCache::default(),
        })
    }

    pub fn from_img(img: GrayImage) -> Self {
        let width = img.width();
        let height = img.height();

        // GrayImage is ImageBuffer<Luma<u8>, Vec<u8>>, so we can take ownership
        let mut vec = img.into_raw();
        // ImageBuffer permits trailing backing bytes. Keep CpuFrame's packed
        // Luma8 invariant without reallocating the image's backing storage.
        let pixel_len = usize::try_from(u64::from(width) * u64::from(height))
            .expect("a valid GrayImage's pixels must fit its backing buffer");
        vec.truncate(pixel_len);
        // This should never fail for a valid GrayImage
        Self::from_vec(width, height, vec).expect("GrayImage should have valid dimensions")
    }

    pub fn width(&self) -> u32 {
        self.img.width()
    }

    pub fn height(&self) -> u32 {
        self.img.height()
    }

    /// Get frame dimensions as a tuple
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    /// Get the total number of pixels
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width()) * u64::from(self.height())
    }

    pub fn to_shared(self) -> CpuFrame {
        let width = self.img.width();
        let height = self.img.height();
        let storage = match self.img.into_raw().inner {
            StorageInner::Shared(shared) => CpuStorage {
                inner: StorageInner::Shared(shared),
                version: new_storage_version(),
            },
            StorageInner::Owned(v) => CpuStorage::from_shared(SharedVec { inner: v }),
        };
        CpuFrame {
            img: ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage).unwrap(),
            cache: ReplicaCache::default(),
        }
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Frame::Cpu(c) => f
                .debug_struct("Frame::Cpu")
                .field("w", &c.width())
                .field("h", &c.height())
                .finish(),
            Frame::CpuNv12(c) => f
                .debug_struct("Frame::CpuNv12")
                .field("dimensions", &c.dimensions())
                .field("encoding", &c.encoding())
                .finish(),
            Frame::Gpu(g) => f
                .debug_struct("Frame::Gpu")
                .field("w", &g.width())
                .field("h", &g.height())
                .finish(),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(g) => f
                .debug_struct("Frame::ManagedGpu")
                .field("spec", &g.spec())
                .finish(),
        }
    }
}

impl Frame {
    /// Borrow initialized packed Luma8 bytes from a CPU frame. CPU access is
    /// immediate under either wait policy; GPU frames require a separate
    /// download and report `GpuConversionRequired` here.
    pub fn try_read_cpu_luma8(&self, wait: WaitPolicy) -> Result<&[u8], FrameError> {
        match self {
            Frame::Cpu(cpu) => cpu.try_read_luma8(wait),
            Frame::CpuNv12(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Luma8,
                actual: PixelFormat::Nv12,
            }),
            Frame::Gpu(_) => Err(FrameError::GpuConversionRequired),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    /// Borrow writable packed Luma8 bytes. Pooled COW remains bounded by the
    /// issuing pool and `wait`; on failure the old frame remains intact.
    pub fn try_write_cpu_luma8(
        &mut self,
        wait: WaitPolicy,
    ) -> Result<CpuLuma8Write<'_>, FrameError> {
        match self {
            Frame::Cpu(cpu) => cpu.try_write_luma8(wait),
            Frame::CpuNv12(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Luma8,
                actual: PixelFormat::Nv12,
            }),
            Frame::Gpu(_) => Err(FrameError::GpuConversionRequired),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    /// Mutate packed Luma8 bytes without detaching to a heap-backed image.
    /// Only managed pooled CPU frames support this initial access path.
    pub fn try_update_pooled_luma8<R>(
        &mut self,
        wait: WaitPolicy,
        update: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<R, FrameError> {
        match self {
            Frame::Cpu(cpu) => cpu.try_update_pooled(wait, update),
            Frame::CpuNv12(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Luma8,
                actual: PixelFormat::Nv12,
            }),
            Frame::Gpu(_) => Err(FrameError::GpuConversionRequired),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    pub fn from_cpu_nv12(frame: CpuNv12Frame) -> Self {
        Self::CpuNv12(frame)
    }

    pub fn as_cpu_nv12(&self) -> Option<&CpuNv12Frame> {
        match self {
            Self::CpuNv12(frame) => Some(frame),
            _ => None,
        }
    }

    pub fn to_cpu_nv12(self) -> Option<CpuNv12Frame> {
        match self {
            Self::CpuNv12(frame) => Some(frame),
            _ => None,
        }
    }

    pub fn try_read_cpu_nv12(&self, wait: WaitPolicy) -> Result<CpuNv12Read<'_>, FrameError> {
        match self {
            Self::CpuNv12(frame) => frame.try_read(wait),
            Self::Cpu(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Nv12,
                actual: PixelFormat::Luma8,
            }),
            Self::Gpu(_) => Err(FrameError::GpuConversionRequired),
            #[cfg(feature = "gpu-wgpu")]
            Self::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    pub fn try_write_cpu_nv12(&mut self, wait: WaitPolicy) -> Result<CpuNv12Write<'_>, FrameError> {
        match self {
            Self::CpuNv12(frame) => frame.try_write(wait),
            Self::Cpu(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Nv12,
                actual: PixelFormat::Luma8,
            }),
            Self::Gpu(_) => Err(FrameError::GpuConversionRequired),
            #[cfg(feature = "gpu-wgpu")]
            Self::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    /// Create from a CPU-owned buffer
    pub fn from_cpu_frame(cpu: CpuFrame) -> Self {
        Frame::Cpu(cpu)
    }

    pub fn from_img(img: GrayImage) -> Self {
        Self::from_cpu_frame(CpuFrame::from_img(img))
    }

    /// Create from Arc-backed CPU storage
    pub fn from_shared(
        buf: impl CpuStorageShared,
        width: u32,
        height: u32,
    ) -> Result<Self, FrameError> {
        let cpu_frame = CpuFrame::from_shared(width, height, buf)?;
        Ok(Frame::Cpu(cpu_frame))
    }

    /// Create from a GPU handle
    pub fn from_gpu(handle: Arc<dyn GpuFrame>) -> Self {
        Frame::Gpu(handle)
    }

    /// Get frame dimensions as a tuple
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Frame::Cpu(c) => c.dimensions(),
            Frame::CpuNv12(c) => c.dimensions(),
            Frame::Gpu(g) => (g.width(), g.height()),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(g) => g.spec().dimensions(),
        }
    }

    pub fn pixel_format(&self) -> PixelFormat {
        match self {
            Frame::Cpu(_) => PixelFormat::Luma8,
            Frame::CpuNv12(_) => PixelFormat::Nv12,
            Frame::Gpu(g) => g.pixel_format(),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(g) => g.spec().pixel_format(),
        }
    }

    /// Check if this frame is CPU-backed
    pub fn is_cpu(&self) -> bool {
        matches!(self, Frame::Cpu(_) | Frame::CpuNv12(_))
    }

    /// Check if this frame is GPU-backed
    pub fn is_gpu(&self) -> bool {
        match self {
            Frame::Gpu(_) => true,
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(_) => true,
            _ => false,
        }
    }

    /// Get the total number of pixels, if available
    pub fn pixel_count(&self) -> u64 {
        match self {
            Frame::Cpu(c) => u64::from(c.width()) * u64::from(c.height()),
            Frame::CpuNv12(c) => {
                let (w, h) = c.dimensions();
                u64::from(w) * u64::from(h)
            }
            Frame::Gpu(g) => u64::from(g.width()) * u64::from(g.height()),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(g) => {
                let (w, h) = g.spec().dimensions();
                u64::from(w) * u64::from(h)
            }
        }
    }

    /// Return a legacy packed-Luma8 `CpuFrame`, if present.
    pub fn as_cpu_frame(&self) -> Option<&CpuFrame> {
        match self {
            Frame::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// Consume a legacy packed-Luma8 `CpuFrame`, if present.
    pub fn to_cpu_frame(self) -> Option<CpuFrame> {
        match self {
            Frame::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// Return a legacy packed-Luma8 image, if present.
    pub fn as_cpu_image(&self) -> Option<&CpuImgBuf> {
        match self {
            Frame::Cpu(c) => Some(&c.img),
            _ => None,
        }
    }

    /// If the Frame is CPU, return a reference; otherwise return None.
    pub fn as_cpu_ref(&self) -> Option<&[u8]> {
        self.as_cpu_image().map(|i| i.as_ref())
    }

    // Clean, easy-to-use methods that automatically track operations
    pub fn from_raw(width: u32, height: u32, buf: Vec<u8>) -> Result<Self, FrameError> {
        let bytes = buf.len();
        StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(bytes));

        let cpu_frame = CpuFrame::from_vec(width, height, buf)?;
        Ok(Frame::Cpu(cpu_frame))
    }

    pub fn to_cpu(&self, timeout: Option<Duration>) -> Result<CpuFrame, FrameError> {
        match self {
            Frame::Cpu(c) => {
                // CpuStorage clones share their backing allocation and copy only on mutation.
                StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
                Ok(c.clone())
            }
            Frame::CpuNv12(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Luma8,
                actual: PixelFormat::Nv12,
            }),
            Frame::Gpu(g) => {
                let bytes = (g.width() * g.height()) as usize;
                StatsContext::record_memory_operation(MemoryOperation::GpuDownload(bytes));
                g.sync_and_download(timeout)
            }
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    pub fn ensure_cpu(&mut self, timeout: Option<Duration>) -> Result<(), FrameError> {
        #[cfg(feature = "gpu-wgpu")]
        if matches!(self, Frame::ManagedGpu(_)) {
            return Err(FrameError::GpuConversionRequired);
        }
        if let Frame::Gpu(g) = self {
            let bytes = (g.width() * g.height()) as usize;
            StatsContext::record_memory_operation(MemoryOperation::GpuDownload(bytes));
            let cpu = g.sync_and_download(timeout)?;
            *self = Frame::Cpu(cpu);
        } else {
            StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
        }
        Ok(())
    }

    // For future GPU operations
    pub fn ensure_gpu(&mut self, _timeout: Option<Duration>) -> Result<(), FrameError> {
        if let Frame::Cpu(c) = self {
            let bytes = (c.width() * c.height()) as usize;
            StatsContext::record_memory_operation(MemoryOperation::GpuUpload(bytes));
            // TODO: Implement GPU upload
            Err(FrameError::GpuUploadNotImplemented)
        } else if matches!(self, Frame::CpuNv12(_)) {
            Err(FrameError::GpuUploadNotImplemented)
        } else {
            StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
            Ok(())
        }
    }

    /// Consume the frame and return its CPU handle; may trigger a download.
    /// Pixel storage can still be shared and detaches when mutated.
    pub fn into_cpu(self, timeout: Option<Duration>) -> Result<CpuFrame, FrameError> {
        match self {
            Frame::Cpu(c) => Ok(c),
            Frame::CpuNv12(_) => Err(FrameError::FormatConversionRequired {
                requested: PixelFormat::Luma8,
                actual: PixelFormat::Nv12,
            }),
            Frame::Gpu(g) => g.sync_and_download(timeout),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(_) => Err(FrameError::GpuConversionRequired),
        }
    }

    /// If this is a GPU frame, try to downcast to a concrete backend
    /// (e.g., if you need GL-specific access).
    pub fn try_gpu_downcast<T: 'static + GpuFrame>(&self) -> Option<Arc<T>> {
        match self {
            Frame::Gpu(a) => {
                let handle: Arc<dyn Any + Send + Sync> = a.clone();
                handle.downcast::<T>().ok()
            }
            _ => None,
        }
    }

    /// Create a shared reference to this frame's data.
    pub fn clone_shared(&self) -> Frame {
        match self {
            Frame::Cpu(cpu_frame) => {
                let (width, height) = cpu_frame.dimensions();
                let shared_storage = cpu_frame.img.as_raw().clone_shared();

                let img = ImageBuffer::from_raw(width, height, shared_storage)
                    .expect("Failed to create ImageBuffer from shared storage");
                Frame::Cpu(CpuFrame {
                    img,
                    cache: ReplicaCache::default(),
                })
            }
            Frame::CpuNv12(frame) => Frame::CpuNv12(frame.clone()),
            Frame::Gpu(gpu) => Frame::Gpu(Arc::clone(gpu)),
            #[cfg(feature = "gpu-wgpu")]
            Frame::ManagedGpu(gpu) => Frame::ManagedGpu(gpu.clone()),
        }
    }
}

impl From<ImageBuffer<Luma<u8>, Vec<u8>>> for Frame {
    fn from(img: ImageBuffer<Luma<u8>, Vec<u8>>) -> Self {
        Self::from_img(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cpu_frame_roundtrip() {
        let f = CpuFrame::from_vec(16, 8, vec![0; 16 * 8]).unwrap();
        let mut frame = Frame::from_cpu_frame(f.clone());
        // ensure_cpu is a no-op for CPU frame
        frame.ensure_cpu(Some(Duration::from_secs(1))).unwrap();
        let dumped = frame.to_cpu(None).unwrap();
        assert_eq!(dumped.img.width(), 16);
    }

    #[test]
    fn convenience_methods() {
        let frame = Frame::from_raw(10, 5, vec![42; 50]).unwrap();
        assert_eq!(frame.dimensions(), (10, 5));
        assert_eq!(frame.pixel_count(), 50);
        assert_eq!(frame.pixel_format(), PixelFormat::Luma8);
        assert!(frame.is_cpu());
        assert!(!frame.is_gpu());
    }

    #[test]
    fn invalid_dimensions() {
        let result = CpuFrame::from_vec(10, 10, vec![0; 50]);
        assert!(matches!(result, Err(FrameError::InvalidDimensions { .. })));
        let overflow = CpuFrame::from_vec(u32::MAX, u32::MAX, vec![]);
        assert!(matches!(
            overflow,
            Err(FrameError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn image_conversions_accept_trailing_backing_bytes() {
        let img = GrayImage::from_raw(2, 1, vec![1, 2, 99]).unwrap();
        let backing = img.as_raw().as_ptr();
        let frame = Frame::from_img(img);
        assert_eq!(frame.as_cpu_ref().unwrap(), &[1, 2]);
        assert_eq!(frame.as_cpu_ref().unwrap().as_ptr(), backing);

        let img = GrayImage::from_raw(2, 1, vec![3, 4, 99]).unwrap();
        let frame: Frame = img.into();
        assert_eq!(frame.as_cpu_ref().unwrap(), &[3, 4]);
    }

    #[derive(Debug)]
    struct TestGpu;

    impl GpuFrame for TestGpu {
        fn width(&self) -> u32 {
            1
        }
        fn height(&self) -> u32 {
            1
        }
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::Luma8
        }
        fn download_to_cpu(&self, _: Option<Duration>) -> Result<CpuFrame, FrameError> {
            CpuFrame::from_vec(1, 1, vec![0])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug)]
    struct DelegatingGpu {
        inner: TestGpu,
    }

    impl GpuFrame for DelegatingGpu {
        fn width(&self) -> u32 {
            self.inner.width()
        }
        fn height(&self) -> u32 {
            self.inner.height()
        }
        fn pixel_format(&self) -> PixelFormat {
            self.inner.pixel_format()
        }
        fn download_to_cpu(&self, timeout: Option<Duration>) -> Result<CpuFrame, FrameError> {
            self.inner.download_to_cpu(timeout)
        }
        fn as_any(&self) -> &dyn Any {
            &self.inner
        }
    }

    #[test]
    fn gpu_downcasts_use_the_handle_identity() {
        let handle = Arc::new(DelegatingGpu { inner: TestGpu });
        let frame = Frame::from_gpu(handle.clone());
        // A backend can safely return a member from as_any; it cannot justify
        // reinterpreting the allocation containing the backend as that member.
        assert!(frame.try_gpu_downcast::<TestGpu>().is_none());
        let recovered = frame.try_gpu_downcast::<DelegatingGpu>().unwrap();
        assert!(Arc::ptr_eq(&handle, &recovered));
    }

    #[test]
    fn cpu_clones_share_until_mutated() {
        let frame = Frame::from_raw(2, 1, vec![1, 2]).unwrap();
        let mut copy = frame.to_cpu(None).unwrap();
        let original = frame.as_cpu_frame().unwrap();
        assert_eq!(copy.img.as_raw().as_ptr(), original.img.as_raw().as_ptr());
        copy.img.as_mut()[0] = 9;
        assert_eq!(&original.img.as_raw()[..], &[1, 2]);
        assert_eq!(&copy.img.as_raw()[..], &[9, 2]);
        assert_ne!(copy.img.as_raw().as_ptr(), original.img.as_raw().as_ptr());
        let detached = copy.img.as_raw().as_ptr();
        copy.img.as_mut()[1] = 8;
        assert_eq!(copy.img.as_raw().as_ptr(), detached);
    }
}
