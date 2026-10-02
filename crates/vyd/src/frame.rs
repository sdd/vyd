use image::{GrayImage, ImageBuffer, Luma};
use std::any::Any;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

use crate::statistics::{MemoryOperation, StatsContext};

/// Errors that can occur when working with frames
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("GPU download timeout after {duration:?}")]
    GpuDownloadTimeout { duration: Duration },

    #[error("GPU sync failed: {message}")]
    GpuSyncFailed { message: String },

    #[error("GPU download failed: {message}")]
    GpuDownloadFailed { message: String },

    #[error("GPU upload not yet implemented")]
    GpuUploadNotImplemented,

    #[error("Cannot operate on Frame::None")]
    FrameIsNone,

    #[error("Invalid frame dimensions: {width}x{height} with {data_len} bytes")]
    InvalidDimensions {
        width: u32,
        height: u32,
        data_len: usize,
    },

    #[error("GPU conversion required")]
    GpuConversionRequired,
}

/// GPU pixel formats supported by the pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPixelFormat {
    Nv12,
    Luma8,
}

/// The unified Frame type.
#[derive(Clone)]
pub enum Frame {
    Cpu(CpuFrame),
    Gpu(Arc<dyn GpuFrame>),
    None,
}

pub type CpuImgBuf = ImageBuffer<Luma<u8>, CpuStorage>;

/// A simple CPU-backed frame type for grayscale images (GRAY8).
#[derive(Clone)]
pub struct CpuFrame {
    pub img: CpuImgBuf,
}

/// A trait for shared CPU storage backends that can be efficiently cloned and potentially
/// converted back to owned storage. This enables zero-copy optimizations when possible.
pub trait CpuStorageShared: Deref<Target = [u8]> + Send + Sync + 'static {
    /// Clone this shared storage (object-safe version of Clone)
    fn clone_shared(&self) -> Box<dyn CpuStorageShared>;

    /// Try to convert this shared storage to an owned Vec<u8> if this is the only reference.
    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>>;

    /// Lease a new buffer from the same pool and copy this storage's data into it.
    ///
    /// Returns `Some(new_storage)` if a pool buffer was available and data was copied,
    /// `None` if the pool is exhausted or this storage type doesn't support pooling.
    ///
    /// The returned storage may still be pool-backed (and thus shareable), allowing
    /// it to be returned to the pool on drop.
    fn lease_and_copy(&self) -> Option<CpuStorage> {
        None // Default: no pooling support
    }
}

/// Internal storage variants - implementation detail hidden from users
enum StorageInner {
    /// Owned mutable storage backed by a Vec<u8>
    Owned(Vec<u8>),
    /// Immutable shared storage (e.g., from a frame buffer pool)
    Shared(Box<dyn CpuStorageShared>),
}

impl Clone for StorageInner {
    fn clone(&self) -> Self {
        match self {
            StorageInner::Owned(v) => StorageInner::Owned(v.clone()),
            StorageInner::Shared(shared) => StorageInner::Shared(shared.clone_shared()),
        }
    }
}

/// CPU-based image storage that can be either owned or shared with copy-on-write semantics.
#[derive(Clone)]
pub struct CpuStorage {
    inner: StorageInner,
}

impl CpuStorage {
    /// Create new owned storage from a Vec<u8>
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            inner: StorageInner::Owned(vec),
        }
    }

    /// Create new shared storage from any type implementing CpuStorageShared
    pub fn from_shared<T: CpuStorageShared>(shared: T) -> Self {
        Self {
            inner: StorageInner::Shared(Box::new(shared)),
        }
    }

    /// Convert to owned storage, consuming self.
    ///
    /// This will attempt to steal the buffer from shared storage if possible,
    /// otherwise it will try to lease a new buffer from the pool, falling back
    /// to heap allocation if necessary.
    pub fn to_owned(self) -> Self {
        match self.inner {
            StorageInner::Owned(_) => self, // Already owned
            StorageInner::Shared(shared) => Self::convert_shared_to_owned(shared),
        }
    }

    /// Check if this storage is currently in owned form
    pub fn is_owned(&self) -> bool {
        matches!(self.inner, StorageInner::Owned(_))
    }

    /// Internal helper to convert shared storage to owned
    fn convert_shared_to_owned(shared: Box<dyn CpuStorageShared>) -> Self {
        let data_len = shared.len();

        // Try to steal the buffer first (zero-copy)
        match shared.try_into_owned() {
            Ok(vec) => {
                tracing::info!("Successfully stole buffer from pool, zero-copy conversion");
                StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
                Self {
                    inner: StorageInner::Owned(vec),
                }
            }
            Err(shared) => {
                // Try to lease and copy from the same pool
                if let Some(new_storage) = shared.lease_and_copy() {
                    tracing::info!(
                        "Leased new buffer from pool and copied data, avoiding heap fragmentation"
                    );
                    StatsContext::record_memory_operation(MemoryOperation::CpuCopy(data_len));
                    return new_storage.to_owned(); // Ensure the result is owned
                }

                tracing::warn!(
                    "Buffer still has other references and no pool available, performing heap allocation and copy"
                );
                StatsContext::record_memory_operation(MemoryOperation::CpuCopy(data_len));
                StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(data_len));
                let vec = (&**shared).to_vec();
                Self {
                    inner: StorageInner::Owned(vec),
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
                }
            }
            StorageInner::Owned(vec) => {
                // Convert to shared storage
                let shared_vec = SharedVec::new(vec.clone());
                CpuStorage::from_shared(shared_vec)
            }
        }
    }
}

impl Deref for CpuStorage {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.inner {
            StorageInner::Owned(v) => v,
            StorageInner::Shared(shared) => &**shared,
        }
    }
}

impl DerefMut for CpuStorage {
    /// Get mutable access to the underlying data.
    ///
    /// This will convert shared storage to owned storage using the most efficient
    /// method available (steal > pool lease > heap allocation).
    fn deref_mut(&mut self) -> &mut Self::Target {
        *self = std::mem::take(self).to_owned();

        match &mut self.inner {
            StorageInner::Owned(v) => v,
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
        &*self.inner
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
pub trait GpuFrame: Send + Sync + fmt::Debug {
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

    /// For downcasts if caller needs backend-specific access.
    fn as_any(&self) -> &dyn Any;
}

impl CpuFrame {
    pub fn from_vec(width: u32, height: u32, v: Vec<u8>) -> Result<Self, FrameError> {
        let expected_len = (width * height) as usize;
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
        Ok(CpuFrame { img })
    }

    pub fn from_shared(
        width: u32,
        height: u32,
        v: impl CpuStorageShared,
    ) -> Result<Self, FrameError> {
        let expected_len = (width * height) as usize;
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
        Ok(CpuFrame { img })
    }

    pub fn from_img(img: GrayImage) -> Self {
        let width = img.width();
        let height = img.height();

        // GrayImage is ImageBuffer<Luma<u8>, Vec<u8>>, so we can take ownership
        let vec = img.into_raw();
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
    pub fn pixel_count(&self) -> u32 {
        self.width() * self.height()
    }

    pub fn to_shared(self) -> CpuFrame {
        let width = self.img.width();
        let height = self.img.height();
        let storage = match self.img.into_raw().inner {
            StorageInner::Shared(shared) => CpuStorage {
                inner: StorageInner::Shared(shared),
            },
            StorageInner::Owned(v) => CpuStorage::from_shared(SharedVec::new(v)),
        };
        CpuFrame {
            img: ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage).unwrap(),
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
            Frame::Gpu(g) => f
                .debug_struct("Frame::Gpu")
                .field("w", &g.width())
                .field("h", &g.height())
                .finish(),
            Frame::None => f.debug_struct("Frame::None").finish(),
        }
    }
}

impl Frame {
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
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        match self {
            Frame::Cpu(c) => Some(c.dimensions()),
            Frame::Gpu(g) => Some((g.width(), g.height())),
            Frame::None => None,
        }
    }

    /// Check if this frame is CPU-backed
    pub fn is_cpu(&self) -> bool {
        matches!(self, Frame::Cpu(_))
    }

    /// Check if this frame is GPU-backed
    pub fn is_gpu(&self) -> bool {
        matches!(self, Frame::Gpu(_))
    }

    /// Check if this frame is None
    pub fn is_none(&self) -> bool {
        matches!(self, Frame::None)
    }

    /// Get the total number of pixels, if available
    pub fn pixel_count(&self) -> Option<u32> {
        match self {
            Frame::Cpu(c) => Some(c.pixel_count()),
            Frame::Gpu(g) => Some(g.width() * g.height()),
            Frame::None => None,
        }
    }

    /// If the Frame is CPU, return a CpuFrame ref; otherwise return None.
    pub fn as_cpu_frame(&self) -> Option<&CpuFrame> {
        match self {
            Frame::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// If the Frame is CPU, return a CpuFrame ref; otherwise return None.
    pub fn to_cpu_frame(self) -> Option<CpuFrame> {
        match self {
            Frame::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// If the Frame is CPU, return an image reference; otherwise return None.
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
                StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
                Ok(c.clone())
            }
            Frame::Gpu(g) => {
                let bytes = (g.width() * g.height()) as usize;
                StatsContext::record_memory_operation(MemoryOperation::GpuDownload(bytes));
                g.sync_and_download(timeout)
            }
            Frame::None => Err(FrameError::FrameIsNone),
        }
    }

    pub fn ensure_cpu(&mut self, timeout: Option<Duration>) -> Result<(), FrameError> {
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
        } else {
            StatsContext::record_memory_operation(MemoryOperation::ZeroCopyReference);
            Ok(())
        }
    }

    /// Consume self and ensure CPU ownership; may trigger a download.
    pub fn into_cpu(self, timeout: Option<Duration>) -> Result<CpuFrame, FrameError> {
        match self {
            Frame::Cpu(c) => Ok(c),
            Frame::Gpu(g) => g.sync_and_download(timeout),
            Frame::None => Err(FrameError::FrameIsNone),
        }
    }

    /// If this is a GPU frame, try to downcast to a concrete backend
    /// (e.g., if you need GL-specific access).
    pub fn try_gpu_downcast<T: 'static + GpuFrame>(&self) -> Option<Arc<T>> {
        match self {
            Frame::Gpu(a) => a
                .as_any()
                .downcast_ref::<T>()
                .map(|_| Arc::clone(&(a.clone().downcast_arc::<T>().ok().unwrap()))),
            _ => None,
        }
    }

    pub fn get_image(&mut self, timeout: Option<Duration>) -> Result<&CpuImgBuf, FrameError> {
        self.ensure_cpu(timeout)?;

        match self {
            &mut Frame::Cpu(ref cpu_frame) => Ok(&cpu_frame.img),
            _ => {
                unreachable!()
            }
        }
    }

    pub fn get_pixels(&mut self, timeout: Option<Duration>) -> Result<&[u8], FrameError> {
        Ok(self.get_image(timeout)?.as_raw())
    }

    /// Create a shared reference to this frame's data.
    pub fn clone_shared(&self) -> Frame {
        match self {
            Frame::Cpu(cpu_frame) => {
                let (width, height) = cpu_frame.dimensions();
                // Access the storage by cloning the ImageBuffer and extracting the storage
                let cloned_img = cpu_frame.img.clone();
                let storage = cloned_img.into_raw(); // This gives us CpuStorage
                let shared_storage = storage.clone_shared();

                let img = ImageBuffer::from_raw(width, height, shared_storage)
                    .expect("Failed to create ImageBuffer from shared storage");
                Frame::Cpu(CpuFrame { img })
            }
            Frame::Gpu(gpu) => Frame::Gpu(Arc::clone(gpu)),
            Frame::None => Frame::None,
        }
    }
}

impl From<ImageBuffer<Luma<u8>, Vec<u8>>> for Frame {
    fn from(img: ImageBuffer<Luma<u8>, Vec<u8>>) -> Self {
        let width = img.width();
        let height = img.height();
        let cpu_storage = CpuStorage::from_vec(img.into_raw());
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, cpu_storage).unwrap();

        Frame::Cpu(CpuFrame { img })
    }
}

/// Convenience helper trait to make downcastable Arc<dyn GpuFrame>
/// (requires this helper impl in your runtime)
trait ArcDowncastExt {
    fn downcast_arc<T: 'static>(self: Arc<Self>) -> Result<Arc<T>, Arc<Self>>;
}
impl ArcDowncastExt for dyn GpuFrame {
    fn downcast_arc<T: 'static>(self: Arc<Self>) -> Result<Arc<T>, Arc<Self>> {
        // Attempt to downcast via Any (best-effort)
        if self.as_any().is::<T>() {
            let raw = Arc::into_raw(self) as *const T;
            // SAFETY: we checked the type above
            Ok(unsafe { Arc::from_raw(raw) })
        } else {
            Err(self)
        }
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
        assert_eq!(frame.dimensions(), Some((10, 5)));
        assert_eq!(frame.pixel_count(), Some(50));
        assert!(frame.is_cpu());
        assert!(!frame.is_gpu());
        assert!(!frame.is_none());
    }

    #[test]
    fn invalid_dimensions() {
        let result = CpuFrame::from_vec(10, 10, vec![0; 50]);
        assert!(matches!(result, Err(FrameError::InvalidDimensions { .. })));
    }
}
