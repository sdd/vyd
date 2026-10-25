//! Adapter between Vyd's bounded CPU pool and GStreamer buffer ownership.
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use gst::glib::translate::from_glib_full;
use vyd_core::frame::pool::{BufferPoolManager, CpuBufferLease, PoolError, PoolLimits, WaitPolicy};
use vyd_core::frame::{CpuStorageShared, FrameError};
use vyd_core::statistics::{MemoryOperation, PipelineStatistics};
use vyd_core::Frame;

struct CurrentPool {
    bytes: usize,
    manager: BufferPoolManager,
}

/// Shares the same backing leases between upstream GST memory and Vyd frames.
///
/// The pointer table identifies only memory issued by this adapter. GST owns
/// the writable memory until it reaches the sink; Vyd obtains a read-only
/// shared lease there. An old manager remains alive through its leases after
/// negotiated buffer size changes.
pub struct ManagedFramePool {
    max_buffers: usize,
    current: Mutex<Option<CurrentPool>>,
    active: Mutex<HashMap<usize, CpuBufferLease>>,
    statistics: Mutex<Option<Arc<PipelineStatistics>>>,
}

impl ManagedFramePool {
    pub fn new(
        max_buffers: Option<usize>,
        max_bytes: Option<usize>,
    ) -> Result<Arc<Self>, PoolError> {
        let limits = PoolLimits::new(max_buffers, max_bytes)?;
        u32::try_from(limits.max_buffers()).map_err(|_| PoolError::Overflow)?;
        Ok(Arc::new(Self {
            max_buffers: limits.max_buffers(),
            current: Mutex::new(None),
            active: Mutex::new(HashMap::new()),
            statistics: Mutex::new(None),
        }))
    }

    pub fn set_statistics(&self, stats: Arc<PipelineStatistics>) {
        let configured_bytes = self
            .current
            .lock()
            .ok()
            .and_then(|current| current.as_ref().map(|pool| pool.bytes));
        let mut statistics = self.statistics.lock().unwrap();
        if statistics.is_some() {
            return;
        }
        *statistics = Some(stats.clone());
        if let Some(bytes) = configured_bytes {
            for _ in 0..self.max_buffers {
                stats.record_memory_operation(
                    "gst-source/pool",
                    MemoryOperation::CpuAllocation(bytes),
                );
            }
        }
    }

    pub fn configure(&self, bytes: usize) -> Result<(), PoolError> {
        let total = bytes
            .checked_mul(self.max_buffers)
            .ok_or(PoolError::Overflow)?;
        let limits = PoolLimits::new(Some(self.max_buffers), Some(total))?;
        let mut current = self.current.lock().map_err(|_| PoolError::Poisoned)?;
        if current.as_ref().is_some_and(|pool| pool.bytes == bytes) {
            return Ok(());
        }
        let manager = BufferPoolManager::new(limits);
        manager.preallocate(bytes, self.max_buffers)?;
        if let Some(stats) = self
            .statistics
            .lock()
            .map_err(|_| PoolError::Poisoned)?
            .as_ref()
        {
            for _ in 0..self.max_buffers {
                stats.record_memory_operation(
                    "gst-source/pool",
                    MemoryOperation::CpuAllocation(bytes),
                );
            }
        }
        if let Some(old) = current.replace(CurrentPool { bytes, manager }) {
            old.manager.close()?;
        }
        Ok(())
    }

    pub fn buffer_size(&self) -> Result<usize, PoolError> {
        self.current
            .lock()
            .map_err(|_| PoolError::Poisoned)?
            .as_ref()
            .map(|pool| pool.bytes)
            .ok_or(PoolError::InvalidSize)
    }

    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }

    pub fn manager(&self) -> Result<BufferPoolManager, PoolError> {
        self.current
            .lock()
            .map_err(|_| PoolError::Poisoned)?
            .as_ref()
            .map(|pool| pool.manager.clone())
            .ok_or(PoolError::InvalidSize)
    }

    pub fn acquire_buffer(self: &Arc<Self>) -> Result<gst::Buffer, PoolError> {
        let (manager, bytes) = {
            let current = self.current.lock().map_err(|_| PoolError::Poisoned)?;
            let pool = current.as_ref().ok_or(PoolError::InvalidSize)?;
            (pool.manager.clone(), pool.bytes)
        };
        let lease = manager.try_acquire(bytes, WaitPolicy::Immediate)?;
        let ptr = lease.as_ptr() as usize;
        self.active
            .lock()
            .map_err(|_| PoolError::Poisoned)?
            .insert(ptr, lease);
        let data = Box::new(MemoryOwner {
            pool: self.clone(),
            ptr,
        });
        let data = Box::into_raw(data);
        let memory = unsafe {
            gst::ffi::gst_memory_new_wrapped(
                0,
                ptr as *mut std::ffi::c_void,
                bytes,
                0,
                bytes,
                data as *mut std::ffi::c_void,
                Some(release_memory),
            )
        };
        if memory.is_null() {
            unsafe { release_memory(data as *mut std::ffi::c_void) };
            return Err(PoolError::AllocationFailed);
        }
        let memory = unsafe { from_glib_full(memory) };
        let mut buffer = gst::Buffer::new();
        buffer.get_mut().unwrap().append_memory(memory);
        tracing::debug!(bytes, "acquired GST buffer from Vyd manager");
        Ok(buffer)
    }

    pub fn acquire_frame_from_ptr(
        &self,
        buffer: &gst::BufferRef,
        ptr: *const u8,
        width: u32,
        height: u32,
    ) -> Result<Frame, FrameError> {
        let len = usize::try_from(u64::from(width) * u64::from(height)).map_err(|_| {
            FrameError::InvalidDimensions {
                width,
                height,
                data_len: 0,
            }
        })?;
        let lease = self
            .active
            .lock()
            .map_err(|_| FrameError::BufferUnavailable)?
            .get(&(ptr as usize))
            .cloned()
            .ok_or(FrameError::BufferUnavailable)?;
        if lease.len() < len {
            return Err(FrameError::InvalidDimensions {
                width,
                height,
                data_len: lease.len(),
            });
        }
        // Keep a readable GST mapping for as long as Vyd exposes this slice.
        // The map holds the exact GST buffer and its read lock; a native writer
        // must produce a writable copy. Checking the pointer prevents pairing
        // a pool lease with an unrelated GST allocation.
        if buffer.n_memory() != 1
            || buffer
                .peek_memory(0)
                .map_readable()
                .map_err(|_| FrameError::BufferUnavailable)?
                .as_ptr()
                != ptr
        {
            return Err(FrameError::BufferUnavailable);
        }
        // BufferRef::to_owned() copies the miniobject. Retain this exact
        // miniobject so GST sees the outstanding reader.
        let retained = unsafe { gst::Buffer::from_glib_none(buffer.as_ptr()) };
        let mapped = retained
            .into_mapped_buffer_readable()
            .map_err(|_| FrameError::BufferUnavailable)?;
        if mapped.as_ptr() != ptr || mapped.len() < len {
            return Err(FrameError::BufferUnavailable);
        }
        Frame::from_shared(
            PooledFrameSlice {
                lease,
                len,
                mapped: Arc::new(mapped),
            },
            width,
            height,
        )
    }
}

struct MemoryOwner {
    pool: Arc<ManagedFramePool>,
    ptr: usize,
}

unsafe extern "C" fn release_memory(data: *mut std::ffi::c_void) {
    if data.is_null() {
        return;
    }
    let owner = Box::from_raw(data as *mut MemoryOwner);
    let removed = owner
        .pool
        .active
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&owner.ptr);
    drop(removed);
}

struct PooledFrameSlice {
    lease: CpuBufferLease,
    len: usize,
    mapped: Arc<gst::MappedBuffer<gst::buffer::Readable>>,
}

impl Deref for PooledFrameSlice {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        debug_assert_eq!(self.lease.as_ptr(), self.mapped.as_ptr());
        &self.mapped[..self.len]
    }
}

impl CpuStorageShared for PooledFrameSlice {
    fn clone_shared(&self) -> Box<dyn CpuStorageShared> {
        Box::new(Self {
            lease: self.lease.clone(),
            len: self.len,
            mapped: self.mapped.clone(),
        })
    }

    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>> {
        // A pooled backing cannot transfer its Vec without leaving the pool.
        Err(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vyd_gst_buffer_pool::VydGstBufferPool;
    use gst::prelude::{BufferPoolExt, BufferPoolExtManual};

    #[test]
    fn managed_preallocation_is_counted_even_if_statistics_arrive_after_configuration() {
        let pool = ManagedFramePool::new(Some(2), None).unwrap();
        pool.configure(4).unwrap();
        let stats = Arc::new(PipelineStatistics::new());
        pool.set_statistics(stats.clone());
        pool.set_statistics(stats.clone());
        pool.configure(4).unwrap();
        pool.configure(8).unwrap();
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["gst-source/pool_cpu_alloc_count"], 4);
        assert_eq!(memory["gst-source/pool_cpu_alloc_bytes"], 24);
    }

    #[test]
    fn gst_memory_and_vyd_frame_share_bounded_lease_until_both_release_it() {
        gst::init().unwrap();
        let pool = ManagedFramePool::new(Some(1), None).unwrap();
        pool.configure(4).unwrap();
        let mut buffer = pool.acquire_buffer().unwrap();
        {
            let mut map = buffer.get_mut().unwrap().map_writable().unwrap();
            map.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
        }
        let ptr = buffer.peek_memory(0).map_readable().unwrap().as_ptr();
        let frame = pool.acquire_frame_from_ptr(&buffer, ptr, 2, 2).unwrap();
        assert!(buffer.get_mut().is_none());
        let Frame::Cpu(cpu) = &frame else {
            panic!("expected CPU frame")
        };
        assert_eq!(&cpu.img.as_raw()[..], &[1, 2, 3, 4]);
        let mut detached = frame.clone();
        let Frame::Cpu(detached_cpu) = &mut detached else {
            panic!("expected CPU frame")
        };
        detached_cpu.img.put_pixel(0, 0, image::Luma([9]));
        assert_eq!(frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(detached.as_cpu_ref().unwrap(), &[9, 2, 3, 4]);
        drop(detached);
        assert_eq!(pool.manager().unwrap().snapshot().unwrap().active.len(), 1);
        assert!(pool.acquire_buffer().is_err());

        drop(buffer);
        assert_eq!(pool.manager().unwrap().snapshot().unwrap().active.len(), 1);
        assert!(pool.acquire_buffer().is_err());
        drop(frame);
        assert_eq!(pool.manager().unwrap().snapshot().unwrap().idle_buffers, 1);
        let reused = pool.acquire_buffer().unwrap();
        assert_eq!(reused.map_readable().unwrap().as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn custom_gst_buffer_pool_uses_vyd_manager() {
        gst::init().unwrap();
        let managed = ManagedFramePool::new(Some(2), None).unwrap();
        let gst_pool = VydGstBufferPool::new();
        gst_pool.set_managed_pool(managed.clone());
        let mut config = gst_pool.config();
        config.set_params(None, 4, 1, 2);
        gst_pool.set_config(config).unwrap();
        gst_pool.set_active(true).unwrap();

        let first = gst_pool.acquire_buffer(None).unwrap();
        let second = gst_pool.acquire_buffer(None).unwrap();
        let snapshot = managed.manager().unwrap().snapshot().unwrap();
        assert_eq!((snapshot.allocated_buffers, snapshot.active.len()), (2, 2));
        drop(first);
        drop(second);
        gst_pool.set_active(false).unwrap();
        assert_eq!(
            managed.manager().unwrap().snapshot().unwrap().active.len(),
            0
        );
    }

    #[test]
    fn reconfiguration_keeps_old_frame_backing_alive_until_release() {
        gst::init().unwrap();
        let pool = ManagedFramePool::new(None, None).unwrap();
        pool.configure(4).unwrap();
        let old_manager = pool.manager().unwrap();
        let mut buffer = pool.acquire_buffer().unwrap();
        buffer
            .get_mut()
            .unwrap()
            .map_writable()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[4, 3, 2, 1]);
        let ptr = buffer.peek_memory(0).map_readable().unwrap().as_ptr();
        let frame = pool.acquire_frame_from_ptr(&buffer, ptr, 2, 2).unwrap();

        pool.configure(8).unwrap();
        assert!(old_manager.snapshot().unwrap().closed);
        drop(buffer);
        let Frame::Cpu(cpu) = &frame else {
            panic!("expected CPU frame")
        };
        assert_eq!(&cpu.img.as_raw()[..], &[4, 3, 2, 1]);
        assert_eq!(old_manager.snapshot().unwrap().active.len(), 1);
        let newer = pool.acquire_buffer().unwrap();
        assert_eq!(newer.size(), 8);
        drop(frame);
        assert_eq!(old_manager.snapshot().unwrap().allocated_buffers, 0);
    }
}
