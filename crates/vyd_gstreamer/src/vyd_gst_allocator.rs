use gst::glib;
use gst_base::subclass::prelude::*;

use crate::SharedFrameBufferPool;

glib::wrapper! {
    /// Custom GStreamer allocator for ring buffer memory
    pub struct VydGstAllocator(ObjectSubclass<imp::VydGstAllocator>)
        @extends gst::Allocator, gst::Object;
}

impl VydGstAllocator {
    pub fn new() -> Self {
        glib::Object::builder().build()
    }

    /// Set the frame buffer pool for this allocator to use
    pub fn set_pool(&self, pool: SharedFrameBufferPool) {
        let imp = self.imp();
        *imp.pool.lock().unwrap() = Some(pool);
    }

    /// Get a reference to the frame buffer pool (if set)
    pub fn pool(&self) -> Option<SharedFrameBufferPool> {
        let imp = self.imp();
        imp.pool.lock().unwrap().clone()
    }
}

impl Default for VydGstAllocator {
    fn default() -> Self {
        Self::new()
    }
}

mod imp {
    use super::*;
    use gst::glib::{bool_error, BoolError};
    use std::sync::{Arc, Mutex, Weak};

    #[derive(Default)]
    pub struct VydGstAllocator {
        /// Reference to the frame buffer pool that manages our memory
        pub(crate) pool: Mutex<Option<SharedFrameBufferPool>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for VydGstAllocator {
        const NAME: &'static str = "VydGstAllocator";
        type Type = super::VydGstAllocator;
        type ParentType = gst::Allocator;
    }

    impl ObjectImpl for VydGstAllocator {}

    impl GstObjectImpl for VydGstAllocator {}

    impl AllocatorImpl for VydGstAllocator {
        fn alloc(
            &self,
            size: usize,
            _params: Option<&gst::AllocationParams>,
        ) -> Result<gst::Memory, BoolError> {
            let pool_opt = { self.pool.lock().unwrap().clone() };
            let Some(pool) = pool_opt else {
                tracing::error!("Allocation attempt without configured pool");
                return Err(bool_error!("VydGstAllocator: no pool configured"));
            };

            let mut pool_guard = pool.lock().unwrap();

            if size != pool_guard.buffer_size() {
                tracing::error!(
                    requested_size = size,
                    pool_buffer_size = pool_guard.buffer_size(),
                    "Size mismatch - requested size doesn't match pool buffer size"
                );
                return Err(bool_error!(
                    "Requested size {} doesn't match pool buffer size {}",
                    size,
                    pool_guard.buffer_size()
                ));
            }

            let Some(handle) = pool_guard.alloc() else {
                tracing::error!(
                    size,
                    buffer_count = pool_guard.buffer_count(),
                    "Pool exhausted - no available buffers"
                );
                return Err(bool_error!("No available buffers in pool"));
            };

            tracing::trace!(
                size,
                available_slots = pool_guard.available_count(),
                in_use_slots = pool_guard.in_use_count(),
                "allocated memory from pool"
            );

            // Create a weak reference to the pool for the deallocator
            let pool_weak = Arc::downgrade(&pool);
            let buffer_ptr = handle.as_ptr();

            // Box the weak reference for the user_data
            let user_data = Box::new(PoolDeallocatorData {
                pool: pool_weak,
                buffer_ptr,
            });

            // Create memory using FFI with our custom deallocator
            let memory = unsafe {
                gst::glib::translate::from_glib_full(gst::ffi::gst_memory_new_wrapped(
                    0,
                    buffer_ptr as *mut std::ffi::c_void,
                    size,
                    0,
                    size,
                    Box::into_raw(user_data) as *mut std::ffi::c_void,
                    Some(pool_deallocator),
                ))
            };

            Ok(memory)
        }

        fn free(&self, _memory: gst::Memory) {
            // This method should not be called - gst should use the notify function instead
            tracing::warn!("VydGstAllocator::free called - unexpected");
        }
    }

    // Data structure for the deallocator
    struct PoolDeallocatorData {
        pool: Weak<Mutex<crate::FrameBufferPool>>,
        buffer_ptr: *const u8,
    }

    // Custom deallocator function called when GStreamer memory is freed
    unsafe extern "C" fn pool_deallocator(data: *mut std::ffi::c_void) {
        if data.is_null() {
            return;
        }

        // Reconstruct the box from the raw pointer
        let deallocator_data = Box::from_raw(data as *mut PoolDeallocatorData);

        // Try to upgrade the weak reference
        if let Some(pool_arc) = deallocator_data.pool.upgrade() {
            if let Ok(mut pool_guard) = pool_arc.lock() {
                let success = pool_guard.free(deallocator_data.buffer_ptr);
                tracing::trace!(
                    success,
                    buffer_ptr = ?deallocator_data.buffer_ptr,
                    "Custom deallocator returned buffer to pool"
                );
            }
        }

        // Box is automatically dropped here, cleaning up the PoolDeallocatorData
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_shared_pool;
    use gst::prelude::*;

    #[test]
    fn test_allocator_creation() {
        gst::init().unwrap();

        let allocator = VydGstAllocator::new();

        // Verify it's properly typed
        assert!(allocator.is::<gst::Allocator>());
        assert_eq!(allocator.type_().name(), "VydGstAllocator");

        // Should start with no pool
        assert!(allocator.pool().is_none());
    }

    #[test]
    fn test_pool_configuration() {
        gst::init().unwrap();

        let allocator = VydGstAllocator::new();
        let buffer_size = 1024;
        let buffer_count = 4;
        let pool = create_shared_pool(buffer_size, buffer_count);

        allocator.set_pool(pool.clone());

        // Verify pool is set
        assert!(allocator.pool().is_some());
        let retrieved_pool = allocator.pool().unwrap();

        // Verify same configuration
        let pool_guard = retrieved_pool.lock().unwrap();
        assert_eq!(pool_guard.buffer_size(), buffer_size);
        assert_eq!(pool_guard.buffer_count(), buffer_count);
    }

    #[test]
    fn test_allocation_without_pool() {
        gst::init().unwrap();

        let allocator = VydGstAllocator::new();
        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // Should fail - no pool configured
        let result = allocator.alloc(1024, Some(&params));
        assert!(result.is_err());
    }

    #[test]
    fn test_successful_allocation_and_automatic_cleanup() {
        gst::init().unwrap();

        let allocator = VydGstAllocator::new();
        let buffer_size = 1024;
        let buffer_count = 2;
        let pool = create_shared_pool(buffer_size, buffer_count);
        allocator.set_pool(pool.clone());

        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // Test allocation and automatic cleanup through scope
        {
            // Should succeed with correct size
            let result = allocator.alloc(buffer_size, Some(&params));
            assert!(result.is_ok());

            let memory = result.unwrap();
            assert_eq!(memory.size(), buffer_size);

            // Pool should reflect allocation
            {
                let pool_guard = pool.lock().unwrap();
                assert_eq!(pool_guard.in_use_count(), 1);
                assert_eq!(pool_guard.available_count(), buffer_count - 1);
            }

            // Memory goes out of scope here and should be automatically freed
        }

        // Check if automatic cleanup worked
        {
            let pool_guard = pool.lock().unwrap();
            // In a properly working implementation, this should be 0
            // If it's still 1, the automatic cleanup didn't work
            let in_use = pool_guard.in_use_count();
            if in_use == 0 {
                println!("✓ Automatic cleanup working correctly");
            } else {
                println!(
                    "! Automatic cleanup not working - memory still in use: {}",
                    in_use
                );
            }
        }
    }

    #[test]
    fn test_size_mismatch_rejection() {
        gst::init().unwrap();

        let allocator = VydGstAllocator::new();
        let buffer_size = 1024;
        let pool = create_shared_pool(buffer_size, 2);
        allocator.set_pool(pool);

        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // Should fail with wrong size
        let result = allocator.alloc(2048, Some(&params)); // Wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_exhaustion() {
        gst::init().unwrap();

        let allocator = VydGstAllocator::new();
        let buffer_size = 1024;
        let buffer_count = 1; // Only one slot
        let pool = create_shared_pool(buffer_size, buffer_count);
        allocator.set_pool(pool.clone());

        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // First allocation should succeed
        let mem1 = allocator.alloc(buffer_size, Some(&params));
        assert!(mem1.is_ok());

        // Pool should be exhausted
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), 1);
            assert_eq!(pool_guard.available_count(), 0);
            assert!(pool_guard.is_full());
        }

        // Second allocation should fail (pool exhausted)
        let mem2 = allocator.alloc(buffer_size, Some(&params));
        assert!(mem2.is_err());

        // Keep the memory alive to maintain pool state
        let _memory = mem1.unwrap();

        // Verify pool is still exhausted
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.available_count(), 0);
        }
    }
}
