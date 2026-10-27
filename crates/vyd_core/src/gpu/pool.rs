//! Bounded, device-specific wgpu texture and staging storage.
//!
//! A lease (and every clone or mapping holder) keeps its backing active. Idle
//! backings are reused only for an identical descriptor. Allocation happens
//! outside the state lock after reserving capacity. New staging bytes are
//! explicitly zeroed; a plane texture is not published as frame content until
//! the transfer or compute path has initialized its visible texels.

use super::{DeviceKey, GpuContextError, WgpuContext};
use crate::frame::pool::WaitPolicy;
use std::collections::{HashMap, VecDeque};
use std::panic::Location;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPoolLimits {
    max_resources: usize,
    max_bytes: u64,
}

impl GpuPoolLimits {
    pub fn new(max_resources: usize, max_bytes: u64) -> Result<Self, GpuPoolError> {
        if max_resources == 0 || max_bytes == 0 {
            return Err(GpuPoolError::InvalidLimits);
        }
        Ok(Self {
            max_resources,
            max_bytes,
        })
    }

    pub fn max_resources(self) -> usize {
        self.max_resources
    }

    pub fn max_bytes(self) -> u64 {
        self.max_bytes
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum GpuPoolError {
    #[error("GPU pool limits must be positive")]
    InvalidLimits,
    #[error("invalid GPU resource size")]
    InvalidSize,
    #[error("GPU resource exceeds pool capacity")]
    Oversized,
    #[error("GPU pool is exhausted")]
    Exhausted,
    #[error("GPU pool acquisition timed out after {0:?}")]
    Timeout(Duration),
    #[error("GPU pool acquisition was cancelled")]
    Cancelled,
    #[error("GPU pool is closed")]
    Closed,
    #[error("GPU pool state is poisoned")]
    Poisoned,
    #[error("GPU pool accounting overflow")]
    Overflow,
    #[error("unsupported GPU texture format or usages")]
    InvalidTextureDescriptor,
    #[error("staging usage must be MAP_WRITE|COPY_SRC or MAP_READ|COPY_DST")]
    InvalidStagingUsage,
    #[error("staging buffer exceeds device buffer limit")]
    UnsupportedBufferSize,
    #[error(transparent)]
    Context(#[from] GpuContextError),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuResourceKey {
    Texture {
        device: DeviceKey,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    Staging {
        device: DeviceKey,
        size: u64,
        usage: wgpu::BufferUsages,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuLeaseOwner {
    pub pool_id: u64,
    pub lease_id: u64,
    pub key: GpuResourceKey,
    pub bytes: u64,
    pub handles: usize,
    pub origin: &'static str,
    pub line: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuPoolSnapshot {
    pub pool_id: u64,
    pub device: DeviceKey,
    pub limits: GpuPoolLimits,
    pub allocated_resources: usize,
    /// Payload capacity: RGBA texel bytes plus staging bytes; excludes driver overhead.
    pub allocated_bytes: u64,
    pub pending_resources: usize,
    pub pending_bytes: u64,
    pub idle_resources: usize,
    pub idle_bytes: u64,
    pub active: Vec<GpuLeaseOwner>,
    pub allocations: u64,
    pub reuses: u64,
    pub waiters: usize,
    pub closed: bool,
}

#[allow(dead_code)] // G4.6 consumes the texture handle.
enum Resource {
    Texture(wgpu::Texture),
    Staging(wgpu::Buffer),
}

struct Idle {
    key: GpuResourceKey,
    bytes: u64,
    resource: Resource,
}

struct Active {
    key: GpuResourceKey,
    bytes: u64,
    handles: Arc<AtomicUsize>,
    origin: &'static str,
    line: u32,
}

struct State {
    idle: VecDeque<Idle>,
    active: HashMap<u64, Active>,
    allocated_resources: usize,
    allocated_bytes: u64,
    pending_resources: usize,
    pending_bytes: u64,
    allocations: u64,
    reuses: u64,
    waiters: usize,
    cancel_epoch: u64,
    next_lease_id: u64,
    closed: bool,
}

struct Inner {
    id: u64,
    context: Arc<WgpuContext>,
    limits: GpuPoolLimits,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Clone)]
pub struct GpuPoolManager(Arc<Inner>);

impl GpuPoolManager {
    pub fn new(context: Arc<WgpuContext>, limits: GpuPoolLimits) -> Self {
        let progress = context.progress.clone();
        let pool = Self(Arc::new(Inner {
            id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            context,
            limits,
            state: Mutex::new(State {
                idle: VecDeque::new(),
                active: HashMap::new(),
                allocated_resources: 0,
                allocated_bytes: 0,
                pending_resources: 0,
                pending_bytes: 0,
                allocations: 0,
                reuses: 0,
                waiters: 0,
                cancel_epoch: 0,
                next_lease_id: 1,
                closed: false,
            }),
            changed: Condvar::new(),
        }));
        let weak = Arc::downgrade(&pool.0);
        progress.register_pool(Arc::new(move || {
            if let Some(inner) = weak.upgrade() {
                let _ = GpuPoolManager(inner).close();
            }
        }));
        pool
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }
    pub fn device_key(&self) -> DeviceKey {
        self.0.context.device_key()
    }

    pub fn ensure_device(&self, context: &WgpuContext) -> Result<(), GpuPoolError> {
        self.0.context.ensure_same_device(context)?;
        context.ensure_healthy()?;
        Ok(())
    }

    /// Acquire one physical RGBA plane. The descriptor is matched exactly on
    /// reuse, including all usage bits. The format is intentionally fixed for
    /// the milestone-4 portable plane representation.
    #[track_caller]
    pub fn try_acquire_texture(
        &self,
        context: &WgpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        wait: WaitPolicy,
    ) -> Result<TextureLease, GpuPoolError> {
        self.ensure_device(context)?;
        context.validate_plane_extent(width, height)?;
        if format != wgpu::TextureFormat::Rgba8Uint
            || usage.is_empty()
            || !PLANE_USAGES.contains(usage)
        {
            return Err(GpuPoolError::InvalidTextureDescriptor);
        }
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(4))
            .ok_or(GpuPoolError::Overflow)?;
        let key = GpuResourceKey::Texture {
            device: context.device_key(),
            width,
            height,
            format,
            usage,
        };
        self.acquire(key, bytes, wait).map(TextureLease)
    }

    /// Acquire upload or readback staging. New buffers are initialized at
    /// creation. A returned idle buffer preserves its previous contents; the
    /// transfer path must initialize copy padding before submitting an upload.
    #[track_caller]
    pub fn try_acquire_staging(
        &self,
        context: &WgpuContext,
        size: u64,
        usage: wgpu::BufferUsages,
        wait: WaitPolicy,
    ) -> Result<StagingLease, GpuPoolError> {
        self.ensure_device(context)?;
        if size == 0 || !size.is_multiple_of(wgpu::MAP_ALIGNMENT) {
            return Err(GpuPoolError::InvalidSize);
        }
        if size > context.device().limits().max_buffer_size {
            return Err(GpuPoolError::UnsupportedBufferSize);
        }
        if usage != STAGING_UPLOAD && usage != STAGING_READBACK {
            return Err(GpuPoolError::InvalidStagingUsage);
        }
        let key = GpuResourceKey::Staging {
            device: context.device_key(),
            size,
            usage,
        };
        self.acquire(key, size, wait).map(StagingLease)
    }

    #[track_caller]
    fn acquire(
        &self,
        key: GpuResourceKey,
        bytes: u64,
        wait: WaitPolicy,
    ) -> Result<Arc<LeaseInner>, GpuPoolError> {
        if bytes > self.0.limits.max_bytes {
            return Err(GpuPoolError::Oversized);
        }
        let deadline = match wait {
            WaitPolicy::Immediate => None,
            WaitPolicy::Timeout(duration) => Some(
                Instant::now()
                    .checked_add(duration)
                    .ok_or(GpuPoolError::Overflow)?,
            ),
        };
        let caller = Location::caller();
        let mut state = self.0.state.lock().map_err(|_| GpuPoolError::Poisoned)?;
        let cancel_epoch = state.cancel_epoch;
        loop {
            if state.closed {
                return Err(GpuPoolError::Closed);
            }
            if state.cancel_epoch != cancel_epoch {
                return Err(GpuPoolError::Cancelled);
            }
            if state.next_lease_id == u64::MAX {
                return Err(GpuPoolError::Overflow);
            }
            if let Some(position) = state.idle.iter().position(|idle| idle.key == key) {
                let idle = state.idle.remove(position).expect("matched idle entry");
                state.reuses = state.reuses.checked_add(1).ok_or(GpuPoolError::Overflow)?;
                return Ok(self.issue(&mut state, idle.resource, key, bytes, caller));
            }
            // An unmatched idle descriptor cannot strand bounded capacity.
            while !fits(&state, self.0.limits, bytes) {
                let Some(idle) = state.idle.pop_front() else {
                    break;
                };
                state.allocated_resources -= 1;
                state.allocated_bytes -= idle.bytes;
                drop(idle);
            }
            if fits(&state, self.0.limits, bytes) {
                state.pending_resources += 1;
                state.pending_bytes += bytes;
                drop(state);
                let result = self.allocate(key);
                state = self.0.state.lock().unwrap_or_else(|p| p.into_inner());
                state.pending_resources -= 1;
                state.pending_bytes -= bytes;
                self.0.changed.notify_all();
                if state.closed {
                    return Err(GpuPoolError::Closed);
                }
                if state.cancel_epoch != cancel_epoch {
                    return Err(GpuPoolError::Cancelled);
                }
                let resource = result?;
                if state.next_lease_id == u64::MAX {
                    return Err(GpuPoolError::Overflow);
                }
                state.allocated_resources += 1;
                state.allocated_bytes += bytes;
                state.allocations = state
                    .allocations
                    .checked_add(1)
                    .ok_or(GpuPoolError::Overflow)?;
                return Ok(self.issue(&mut state, resource, key, bytes, caller));
            }
            let Some(deadline) = deadline else {
                return Err(GpuPoolError::Exhausted);
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(GpuPoolError::Timeout(match wait {
                    WaitPolicy::Timeout(duration) => duration,
                    WaitPolicy::Immediate => unreachable!(),
                }));
            }
            state.waiters += 1;
            let result = self.0.changed.wait_timeout(state, remaining);
            state = match result {
                Ok((guard, _)) => guard,
                Err(_) => return Err(GpuPoolError::Poisoned),
            };
            state.waiters -= 1;
        }
    }

    fn allocate(&self, key: GpuResourceKey) -> Result<Resource, GpuPoolError> {
        let device = self.0.context.device();
        match key {
            GpuResourceKey::Texture {
                width,
                height,
                format,
                usage,
                ..
            } => Ok(Resource::Texture(device.create_texture(
                &wgpu::TextureDescriptor {
                    label: Some("vyd GPU plane pool"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                },
            ))),
            GpuResourceKey::Staging { size, usage, .. } => {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vyd GPU staging pool"),
                    size,
                    usage,
                    mapped_at_creation: true,
                });
                {
                    let mut view = buffer.slice(..).get_mapped_range_mut();
                    view.fill(0);
                }
                buffer.unmap();
                Ok(Resource::Staging(buffer))
            }
        }
    }

    fn issue(
        &self,
        state: &mut State,
        resource: Resource,
        key: GpuResourceKey,
        bytes: u64,
        caller: &'static Location<'static>,
    ) -> Arc<LeaseInner> {
        let id = state.next_lease_id;
        state.next_lease_id += 1;
        let handles = Arc::new(AtomicUsize::new(1));
        let lease = Arc::new(LeaseInner {
            issuer: self.0.clone(),
            id,
            key,
            bytes,
            resource: Some(resource),
            handles: handles.clone(),
        });
        state.active.insert(
            id,
            Active {
                key,
                bytes,
                handles,
                origin: caller.file(),
                line: caller.line(),
            },
        );
        lease
    }

    pub fn cancel_waiters(&self) -> Result<(), GpuPoolError> {
        let mut state = self.0.state.lock().map_err(|_| GpuPoolError::Poisoned)?;
        state.cancel_epoch = state.cancel_epoch.wrapping_add(1);
        self.0.changed.notify_all();
        Ok(())
    }

    pub fn close(&self) -> Result<(), GpuPoolError> {
        let mut state = self.0.state.lock().map_err(|_| GpuPoolError::Poisoned)?;
        state.closed = true;
        let idle: Vec<_> = state.idle.drain(..).collect();
        for entry in &idle {
            state.allocated_resources -= 1;
            state.allocated_bytes -= entry.bytes;
        }
        self.0.changed.notify_all();
        drop(state);
        // wgpu handle destruction may invoke a callback; never do it while
        // holding the pool mutex (including in a device-loss callback).
        drop(idle);
        Ok(())
    }

    pub fn snapshot(&self) -> Result<GpuPoolSnapshot, GpuPoolError> {
        let state = self.0.state.lock().map_err(|_| GpuPoolError::Poisoned)?;
        let mut active = state
            .active
            .iter()
            .map(|(&lease_id, entry)| GpuLeaseOwner {
                pool_id: self.0.id,
                lease_id,
                key: entry.key,
                bytes: entry.bytes,
                handles: entry.handles.load(Ordering::Relaxed),
                origin: entry.origin,
                line: entry.line,
            })
            .collect::<Vec<_>>();
        active.sort_by_key(|owner| owner.lease_id);
        Ok(GpuPoolSnapshot {
            pool_id: self.0.id,
            device: self.device_key(),
            limits: self.0.limits,
            allocated_resources: state.allocated_resources,
            allocated_bytes: state.allocated_bytes,
            pending_resources: state.pending_resources,
            pending_bytes: state.pending_bytes,
            idle_resources: state.idle.len(),
            idle_bytes: state.idle.iter().map(|entry| entry.bytes).sum(),
            active,
            allocations: state.allocations,
            reuses: state.reuses,
            waiters: state.waiters,
            closed: state.closed,
        })
    }
}

const PLANE_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::COPY_SRC
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::STORAGE_BINDING);
pub const STAGING_UPLOAD: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_WRITE.union(wgpu::BufferUsages::COPY_SRC);
pub const STAGING_READBACK: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

fn fits(state: &State, limits: GpuPoolLimits, bytes: u64) -> bool {
    state
        .allocated_resources
        .checked_add(state.pending_resources)
        .is_some_and(|n| n < limits.max_resources)
        && state
            .allocated_bytes
            .checked_add(state.pending_bytes)
            .and_then(|n| n.checked_add(bytes))
            .is_some_and(|n| n <= limits.max_bytes)
}

struct LeaseInner {
    issuer: Arc<Inner>,
    id: u64,
    key: GpuResourceKey,
    bytes: u64,
    resource: Option<Resource>,
    handles: Arc<AtomicUsize>,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let mut state = self.issuer.state.lock().unwrap_or_else(|p| p.into_inner());
        state.active.remove(&self.id);
        let mut discarded = None;
        if let Some(resource) = self.resource.take() {
            if state.closed {
                state.allocated_resources -= 1;
                state.allocated_bytes -= self.bytes;
                discarded = Some(resource);
            } else {
                state.idle.push_back(Idle {
                    key: self.key,
                    bytes: self.bytes,
                    resource,
                });
            }
        }
        self.issuer.changed.notify_all();
        drop(state);
        drop(discarded);
    }
}

pub struct TextureLease(Arc<LeaseInner>);

impl Clone for TextureLease {
    fn clone(&self) -> Self {
        self.0.handles.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}
impl Drop for TextureLease {
    fn drop(&mut self) {
        self.0.handles.fetch_sub(1, Ordering::Relaxed);
    }
}
impl TextureLease {
    /// A lost device must not put a submitted plane back into an idle pool.
    pub(crate) fn quarantine_pool(&self) {
        let _ = GpuPoolManager(self.0.issuer.clone()).close();
    }
    pub fn pool_id(&self) -> u64 {
        self.0.issuer.id
    }
    pub fn lease_id(&self) -> u64 {
        self.0.id
    }
    pub fn key(&self) -> GpuResourceKey {
        self.0.key
    }
    #[allow(dead_code)] // G4.6 uses it for copy commands and views.
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        let Resource::Texture(texture) = self.0.resource.as_ref().expect("live texture lease")
        else {
            unreachable!()
        };
        texture
    }
}

pub struct StagingLease(Arc<LeaseInner>);

impl Clone for StagingLease {
    fn clone(&self) -> Self {
        self.0.handles.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}
impl Drop for StagingLease {
    fn drop(&mut self) {
        self.0.handles.fetch_sub(1, Ordering::Relaxed);
    }
}
impl StagingLease {
    pub fn pool_id(&self) -> u64 {
        self.0.issuer.id
    }
    pub fn lease_id(&self) -> u64 {
        self.0.id
    }
    pub fn key(&self) -> GpuResourceKey {
        self.0.key
    }
    #[allow(dead_code)] // G4.6 uses it for staging copies and maps.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        let Resource::Staging(buffer) = self.0.resource.as_ref().expect("live staging lease")
        else {
            unreachable!()
        };
        buffer
    }

    /// Call after a successful map. Retains the lease across a mapped view;
    /// drop the view before dropping this holder so it can unmap safely.
    /// A pending map callback must separately retain a lease until it runs.
    #[allow(dead_code)] // G4.6 retains mapped readback guards.
    pub(crate) fn mapping_holder(&self) -> StagingMapHolder {
        StagingMapHolder(self.clone())
    }
}

#[allow(dead_code)] // G4.6 retains mapped readback guards.
pub(crate) struct StagingMapHolder(StagingLease);
impl StagingMapHolder {
    #[allow(dead_code)] // G4.6 uses it for mapped row access.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        self.0.buffer()
    }
}
impl Drop for StagingMapHolder {
    fn drop(&mut self) {
        self.0.buffer().unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::WgpuContextOptions;
    use std::thread;

    fn context() -> Option<Arc<WgpuContext>> {
        match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Some(Arc::new(context)),
            Err(GpuContextError::Unavailable) => {
                eprintln!("GPU pool tests unverified: no compatible adapter");
                None
            }
            Err(error) => panic!("GPU pool hardware gate failed: {error}"),
        }
    }

    fn one(context: Arc<WgpuContext>, bytes: u64) -> GpuPoolManager {
        GpuPoolManager::new(context, GpuPoolLimits::new(1, bytes).unwrap())
    }

    fn plane(
        pool: &GpuPoolManager,
        context: &WgpuContext,
        width: u32,
        height: u32,
        usage: wgpu::TextureUsages,
        wait: WaitPolicy,
    ) -> Result<TextureLease, GpuPoolError> {
        pool.try_acquire_texture(
            context,
            width,
            height,
            wgpu::TextureFormat::Rgba8Uint,
            usage,
            wait,
        )
    }

    fn wait_for_waiter(pool: &GpuPoolManager) {
        let start = Instant::now();
        while pool.snapshot().unwrap().waiters == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "waiter did not park"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn texture_lease_is_unique_until_last_handle_and_reuses_exact_key() {
        let Some(context) = context() else {
            return;
        };
        let pool = one(context.clone(), 16);
        let first = plane(&pool, &context, 2, 2, PLANE_USAGES, WaitPolicy::Immediate).unwrap();
        let clone = first.clone();
        let first_id = first.lease_id();
        let snapshot = pool.snapshot().unwrap();
        assert_eq!(snapshot.pool_id, first.pool_id());
        assert_eq!(snapshot.allocated_resources, 1);
        assert_eq!(snapshot.allocated_bytes, 16);
        assert_eq!(snapshot.active[0].handles, 2);
        assert_eq!(snapshot.active[0].lease_id, first_id);
        assert_eq!(snapshot.allocations, 1);
        drop(first);
        assert!(matches!(
            plane(&pool, &context, 2, 2, PLANE_USAGES, WaitPolicy::Immediate),
            Err(GpuPoolError::Exhausted)
        ));
        drop(clone);
        assert_eq!(pool.snapshot().unwrap().idle_resources, 1);
        let second = plane(&pool, &context, 2, 2, PLANE_USAGES, WaitPolicy::Immediate).unwrap();
        assert_ne!(second.lease_id(), first_id);
        assert_eq!(pool.snapshot().unwrap().reuses, 1);
        assert_eq!(pool.snapshot().unwrap().allocations, 1);
        assert_eq!(pool.snapshot().unwrap().idle_resources, 0);
    }

    #[test]
    fn descriptors_are_exact_and_incompatible_devices_are_rejected() {
        let Some(context) = context() else {
            return;
        };
        let pool = one(context.clone(), 24);
        let first = plane(&pool, &context, 2, 2, PLANE_USAGES, WaitPolicy::Immediate).unwrap();
        drop(first);
        let smaller_usage = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let second = plane(&pool, &context, 2, 2, smaller_usage, WaitPolicy::Immediate).unwrap();
        assert_eq!(pool.snapshot().unwrap().allocations, 2);
        assert_eq!(pool.snapshot().unwrap().reuses, 0);
        drop(second);
        let third = plane(&pool, &context, 3, 2, smaller_usage, WaitPolicy::Immediate).unwrap();
        assert_eq!(pool.snapshot().unwrap().allocated_bytes, 24);
        assert_eq!(pool.snapshot().unwrap().idle_resources, 0);
        drop(third);
        assert!(matches!(
            pool.try_acquire_texture(
                &context,
                1,
                1,
                wgpu::TextureFormat::R8Unorm,
                smaller_usage,
                WaitPolicy::Immediate
            ),
            Err(GpuPoolError::InvalidTextureDescriptor)
        ));
        assert!(matches!(
            plane(
                &pool,
                &context,
                1,
                1,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                WaitPolicy::Immediate
            ),
            Err(GpuPoolError::InvalidTextureDescriptor)
        ));
        let other = WgpuContext::request_blocking(WgpuContextOptions::default()).unwrap();
        assert!(matches!(
            plane(&pool, &other, 1, 1, PLANE_USAGES, WaitPolicy::Immediate),
            Err(GpuPoolError::Context(
                GpuContextError::IncompatibleDevice { .. }
            ))
        ));
    }

    #[test]
    fn staging_holder_keeps_backing_active_and_usage_is_exact() {
        let Some(context) = context() else {
            return;
        };
        let pool = one(context.clone(), 256);
        let staging = pool
            .try_acquire_staging(&context, 256, STAGING_UPLOAD, WaitPolicy::Immediate)
            .unwrap();
        let pending = staging.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .buffer()
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |result| {
                sender.send(result).unwrap();
            });
        context.device().poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
        drop(pending);
        let holder = staging.mapping_holder();
        assert_eq!(holder.buffer().size(), 256);
        {
            let view = holder.buffer().slice(..).get_mapped_range();
            assert!(view.iter().all(|byte| *byte == 0));
        }
        drop(staging);
        assert_eq!(pool.snapshot().unwrap().active[0].handles, 1);
        assert!(matches!(
            pool.try_acquire_staging(&context, 256, STAGING_UPLOAD, WaitPolicy::Immediate),
            Err(GpuPoolError::Exhausted)
        ));
        drop(holder);
        let readback = pool
            .try_acquire_staging(&context, 256, STAGING_READBACK, WaitPolicy::Immediate)
            .unwrap();
        assert_eq!(pool.snapshot().unwrap().allocations, 2);
        drop(readback);
        assert!(matches!(
            pool.try_acquire_staging(&context, 255, STAGING_UPLOAD, WaitPolicy::Immediate),
            Err(GpuPoolError::InvalidSize)
        ));
        assert!(matches!(
            pool.try_acquire_staging(
                &context,
                256,
                wgpu::BufferUsages::COPY_DST,
                WaitPolicy::Immediate
            ),
            Err(GpuPoolError::InvalidStagingUsage)
        ));
    }

    #[test]
    fn wait_has_one_deadline_and_cancel_does_not_close_pool() {
        let Some(context) = context() else {
            return;
        };
        let pool = one(context.clone(), 256);
        let held = pool
            .try_acquire_staging(&context, 256, STAGING_UPLOAD, WaitPolicy::Immediate)
            .unwrap();
        assert!(matches!(
            pool.try_acquire_staging(
                &context,
                256,
                STAGING_UPLOAD,
                WaitPolicy::Timeout(Duration::from_millis(20))
            ),
            Err(GpuPoolError::Timeout(_))
        ));
        let worker_pool = pool.clone();
        let worker_context = context.clone();
        let worker = thread::spawn(move || {
            worker_pool
                .try_acquire_staging(
                    &worker_context,
                    256,
                    STAGING_UPLOAD,
                    WaitPolicy::Timeout(Duration::from_secs(2)),
                )
                .map(|_| ())
        });
        wait_for_waiter(&pool);
        pool.cancel_waiters().unwrap();
        assert!(matches!(
            worker.join().unwrap(),
            Err(GpuPoolError::Cancelled)
        ));
        drop(held);
        assert!(
            pool.try_acquire_staging(&context, 256, STAGING_UPLOAD, WaitPolicy::Immediate)
                .is_ok()
        );
        assert!(!pool.snapshot().unwrap().closed);
    }

    #[test]
    fn close_wakes_waiters_and_retires_active_capacity() {
        let Some(context) = context() else {
            return;
        };
        let pool = one(context.clone(), 256);
        let held = pool
            .try_acquire_staging(&context, 256, STAGING_UPLOAD, WaitPolicy::Immediate)
            .unwrap();
        let worker_pool = pool.clone();
        let worker_context = context.clone();
        let worker = thread::spawn(move || {
            worker_pool
                .try_acquire_staging(
                    &worker_context,
                    256,
                    STAGING_UPLOAD,
                    WaitPolicy::Timeout(Duration::from_secs(2)),
                )
                .map(|_| ())
        });
        wait_for_waiter(&pool);
        pool.close().unwrap();
        assert!(matches!(worker.join().unwrap(), Err(GpuPoolError::Closed)));
        assert_eq!(pool.snapshot().unwrap().allocated_resources, 1);
        drop(held);
        let snapshot = pool.snapshot().unwrap();
        assert_eq!(snapshot.allocated_resources, 0);
        assert_eq!(snapshot.allocated_bytes, 0);
        assert!(matches!(
            pool.try_acquire_staging(&context, 256, STAGING_UPLOAD, WaitPolicy::Immediate),
            Err(GpuPoolError::Closed)
        ));
    }
}
