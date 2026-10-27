//! Pipeline-owned wgpu device, capability gate and managed work progress.

pub mod conversion;
pub mod pool;
pub mod recording;
pub mod submission;
pub mod transfer;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;

// wgpu error scopes are a device stack. Program construction and queue
// admission must not pop one another's scopes.
pub(crate) static DEVICE_ERROR_SCOPE_LOCK: Mutex<()> = Mutex::new(());

static NEXT_DEVICE_KEY: AtomicU64 = AtomicU64::new(1);

/// Identity of one Vyd-owned wgpu device. Clones of a context keep this key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceKey(u64);

impl DeviceKey {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GpuContextError {
    #[error("no compatible wgpu adapter is available")]
    Unavailable,
    #[error("no GPU context has been configured for this pipeline")]
    NotConfigured,
    #[error("GPU context cannot be changed after pipeline startup")]
    ConfigurationLocked,
    #[error(
        "adapter lacks required Rgba8Uint texture usages: required {required:?}, available {available:?}"
    )]
    UnsupportedTextureUsages {
        required: wgpu::TextureUsages,
        available: wgpu::TextureUsages,
    },
    #[error("adapter lacks Rgba8Uint storage write support")]
    UnsupportedStorageWrite,
    #[error("GPU requires at least two storage and two sampled texture bindings for NV12 planes")]
    UnsupportedPlaneBindingLimits,
    #[error(
        "invalid or unsupported GPU plane extent {width}x{height}; device limit is {max_dimension}"
    )]
    UnsupportedExtent {
        width: u32,
        height: u32,
        max_dimension: u32,
    },
    #[error("wgpu device request failed: {message}")]
    DeviceRequest { message: String },
    #[error("GPU device {actual:?} is incompatible with pipeline device {expected:?}")]
    IncompatibleDevice {
        expected: DeviceKey,
        actual: DeviceKey,
    },
    #[error("GPU device lost: {message}")]
    DeviceLost { message: String },
    #[error("uncaptured GPU error: {message}")]
    UncapturedError { message: String },
    #[error("GPU context is closed")]
    Closed,
}

/// Adapter selection for a headless compute pipeline.
#[derive(Clone, Copy, Debug)]
pub struct WgpuContextOptions {
    pub backends: wgpu::Backends,
    pub power_preference: wgpu::PowerPreference,
    pub force_fallback_adapter: bool,
}

impl Default for WgpuContextOptions {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::PRIMARY,
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }
    }
}

/// The one device and queue shared by a pipeline's GPU frames and stages.
pub struct WgpuContext {
    key: DeviceKey,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    pub(crate) progress: Arc<submission::Progress>,
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuContext")
            .field("key", &self.key)
            .field("adapter_info", &self.adapter_info)
            .finish_non_exhaustive()
    }
}

impl WgpuContext {
    pub async fn request(options: WgpuContextOptions) -> Result<Self, GpuContextError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: options.backends,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuContextError::Unavailable)?;
        let format = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Uint);
        validate_capabilities(format, &adapter.limits())?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|error| GpuContextError::DeviceRequest {
                message: error.to_string(),
            })?;
        validate_capabilities(format, &device.limits())?;
        let progress = submission::Progress::new(device.clone());
        let lost_progress = Arc::downgrade(&progress);
        device.set_device_lost_callback(move |reason, message| {
            if let Some(progress) = lost_progress.upgrade() {
                progress.device_lost(format!("{reason:?}: {message}"));
            }
        });
        let error_progress = Arc::downgrade(&progress);
        device.on_uncaptured_error(Box::new(move |error| {
            if let Some(progress) = error_progress.upgrade() {
                progress.fail_pending(format!("uncaptured GPU error: {error}"));
            }
        }));
        Ok(Self {
            key: DeviceKey(NEXT_DEVICE_KEY.fetch_add(1, Ordering::Relaxed)),
            device,
            queue,
            adapter_info,
            progress,
        })
    }

    /// Synchronous entry point for current serial pipeline construction.
    pub fn request_blocking(options: WgpuContextOptions) -> Result<Self, GpuContextError> {
        pollster::block_on(Self::request(options))
    }

    pub fn device_key(&self) -> DeviceKey {
        self.key
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Shader and ordinary non-frame buffer creation can use this device.
    /// Frame planes remain accessible only through managed bindings.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn validate_plane_extent(&self, width: u32, height: u32) -> Result<(), GpuContextError> {
        validate_plane_extent(width, height, self.device.limits().max_texture_dimension_2d)
    }

    pub fn ensure_same_device(&self, other: &Self) -> Result<(), GpuContextError> {
        ensure_device_key(self.key, other.key)
    }

    pub fn ensure_healthy(&self) -> Result<(), GpuContextError> {
        if let Some(error) = self.progress.device_failure() {
            return Err(error);
        }
        Ok(())
    }

    /// Stop accepting managed compute work and wait up to the caller's
    /// deadline for accepted submissions to retire. A timeout leaves the
    /// progress driver running so already accepted work remains protected.
    pub fn shutdown(
        &self,
        wait: crate::frame::pool::WaitPolicy,
    ) -> Result<(), submission::GpuSubmissionError> {
        let result = self.progress.shutdown(wait);
        if result.is_ok() {
            self.progress.stop_driver(Duration::from_millis(100));
        }
        result
    }
}

impl Drop for WgpuContext {
    fn drop(&mut self) {
        self.progress.stop_driver(Duration::from_millis(100));
    }
}

const REQUIRED_PLANE_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::COPY_SRC
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::STORAGE_BINDING);

fn validate_capabilities(
    format: wgpu::TextureFormatFeatures,
    limits: &wgpu::Limits,
) -> Result<(), GpuContextError> {
    if !format.allowed_usages.contains(REQUIRED_PLANE_USAGES) {
        return Err(GpuContextError::UnsupportedTextureUsages {
            required: REQUIRED_PLANE_USAGES,
            available: format.allowed_usages,
        });
    }
    if !format
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
    {
        return Err(GpuContextError::UnsupportedStorageWrite);
    }
    if limits.max_storage_textures_per_shader_stage < 2
        || limits.max_sampled_textures_per_shader_stage < 2
    {
        return Err(GpuContextError::UnsupportedPlaneBindingLimits);
    }
    Ok(())
}

fn validate_plane_extent(
    width: u32,
    height: u32,
    max_dimension: u32,
) -> Result<(), GpuContextError> {
    if width == 0 || height == 0 || width > max_dimension || height > max_dimension {
        Err(GpuContextError::UnsupportedExtent {
            width,
            height,
            max_dimension,
        })
    } else {
        Ok(())
    }
}

fn ensure_device_key(expected: DeviceKey, actual: DeviceKey) -> Result<(), GpuContextError> {
    if expected == actual {
        Ok(())
    } else {
        Err(GpuContextError::IncompatibleDevice { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported() -> wgpu::TextureFormatFeatures {
        wgpu::TextureFormatFeatures {
            allowed_usages: REQUIRED_PLANE_USAGES,
            flags: wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY,
        }
    }

    #[test]
    fn rejects_missing_format_usage_and_storage_write() {
        let limits = wgpu::Limits::default();
        let mut features = supported();
        features
            .allowed_usages
            .remove(wgpu::TextureUsages::STORAGE_BINDING);
        assert!(matches!(
            validate_capabilities(features, &limits),
            Err(GpuContextError::UnsupportedTextureUsages { .. })
        ));
        features = supported();
        features.flags = wgpu::TextureFormatFeatureFlags::empty();
        assert_eq!(
            validate_capabilities(features, &limits),
            Err(GpuContextError::UnsupportedStorageWrite)
        );
    }

    #[test]
    fn rejects_bad_limits_and_extents() {
        let mut limits = wgpu::Limits {
            max_storage_textures_per_shader_stage: 1,
            ..Default::default()
        };
        assert_eq!(
            validate_capabilities(supported(), &limits),
            Err(GpuContextError::UnsupportedPlaneBindingLimits)
        );
        limits.max_storage_textures_per_shader_stage = 2;
        limits.max_sampled_textures_per_shader_stage = 1;
        assert_eq!(
            validate_capabilities(supported(), &limits),
            Err(GpuContextError::UnsupportedPlaneBindingLimits)
        );
        assert!(validate_plane_extent(0, 1, 4096).is_err());
        assert!(validate_plane_extent(4097, 1, 4096).is_err());
        assert!(validate_plane_extent(4096, 4096, 4096).is_ok());
    }

    #[test]
    fn rejects_incompatible_device_key() {
        let first = DeviceKey(1);
        let second = DeviceKey(2);
        assert!(ensure_device_key(first, first).is_ok());
        assert_eq!(
            ensure_device_key(first, second),
            Err(GpuContextError::IncompatibleDevice {
                expected: first,
                actual: second,
            })
        );
    }

    #[test]
    #[ignore = "explicit hardware smoke; reports the local adapter or unavailable status"]
    fn hardware_smoke_and_pipeline_ownership() {
        let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(gpu) => std::sync::Arc::new(gpu),
            Err(GpuContextError::Unavailable) => {
                eprintln!("GPU hardware smoke: no compatible adapter available");
                return;
            }
            Err(error) => panic!("GPU hardware smoke failed: {error}"),
        };
        eprintln!(
            "GPU hardware smoke: adapter={} backend={:?} type={:?}",
            gpu.adapter_info().name,
            gpu.adapter_info().backend,
            gpu.adapter_info().device_type
        );
        gpu.validate_plane_extent(17, 9).unwrap();
        let mut pipeline = crate::pipeline::PipelineContext::new();
        assert_eq!(
            pipeline.wgpu_context().unwrap_err(),
            GpuContextError::NotConfigured
        );
        pipeline.set_wgpu_context(gpu.clone()).unwrap();
        pipeline.set_wgpu_context(gpu.clone()).unwrap();
        assert_eq!(
            pipeline.wgpu_context().unwrap().device_key(),
            gpu.device_key()
        );
        let other = WgpuContext::request_blocking(WgpuContextOptions::default()).unwrap();
        assert!(matches!(
            pipeline.set_wgpu_context(std::sync::Arc::new(other)),
            Err(GpuContextError::IncompatibleDevice { .. })
        ));
        assert_eq!(
            pipeline.wgpu_context().unwrap().device_key(),
            gpu.device_key()
        );
        pipeline.freeze_wgpu_context();
        assert_eq!(
            pipeline.set_wgpu_context(gpu.clone()),
            Err(GpuContextError::ConfigurationLocked)
        );
    }
}
