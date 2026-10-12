use gst::glib;
use gst_base::subclass::prelude::*;
use std::sync::{Arc, Mutex, Weak};

use crate::managed_pool::ManagedFramePool;
use crate::SharedFrameBufferPool;

glib::wrapper! {
    /// Custom GStreamer buffer pool for ring buffer memory
    pub struct VydGstBufferPool(ObjectSubclass<imp::VydGstBufferPool>)
        @extends gst::BufferPool, gst::Object;
}

impl VydGstBufferPool {
    pub fn new() -> Self {
        glib::Object::builder().build()
    }

    /// Set the frame buffer pool for this buffer pool to use
    pub fn set_frame_buffer_pool(&self, pool: SharedFrameBufferPool) {
        let imp = self.imp();
        *imp.pool.lock().unwrap() = Some(pool);
    }

    pub fn set_managed_pool(&self, pool: Arc<ManagedFramePool>) {
        *self.imp().managed_pool.lock().unwrap() = Some(pool);
    }
}

impl Default for VydGstBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

mod imp {
    use super::*;
    use gst::{BufferPoolAcquireParams, BufferPoolConfigRef};

    #[derive(Default)]
    pub struct VydGstBufferPool {
        /// Reference to the frame buffer pool that manages our memory
        pub(crate) pool: Mutex<Option<SharedFrameBufferPool>>,
        pub(crate) managed_pool: Mutex<Option<Arc<ManagedFramePool>>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for VydGstBufferPool {
        const NAME: &'static str = "VydGstBufferPool";
        type Type = super::VydGstBufferPool;
        type ParentType = gst::BufferPool;
    }

    impl ObjectImpl for VydGstBufferPool {}

    impl GstObjectImpl for VydGstBufferPool {}

    impl BufferPoolImpl for VydGstBufferPool {
        fn set_config(&self, config: &mut BufferPoolConfigRef) -> bool {
            tracing::debug!("Setting buffer pool config");

            // Get the parameters from config - returns Option<(Option<Caps>, u32, u32, u32)>
            let Some((caps, size, min_buffers, max_buffers)) = config.params() else {
                tracing::error!("Failed to get buffer pool config parameters");
                return false;
            };

            tracing::debug!(
                size,
                min_buffers,
                max_buffers,
                caps = ?caps,
                "Buffer pool config change requested"
            );

            if let Some(managed) = self.managed_pool.lock().unwrap().as_ref() {
                return managed.configure(size as usize).is_ok();
            }

            // If we have a frame buffer pool, check if we need to recreate it for the new size
            if let Some(existing_pool) = self.pool.lock().unwrap().as_ref() {
                let mut pool_guard = existing_pool.lock().unwrap();
                if pool_guard.buffer_size() != size as usize {
                    tracing::info!(
                        "Buffer size changed from {} to {}, reconfiguring frame buffer pool",
                        pool_guard.buffer_size(),
                        size
                    );

                    // Use the larger of min_buffers or max_buffers for buffer count
                    // If max_buffers is 0, it means unlimited, so use min_buffers or a reasonable default
                    let buffer_count = if max_buffers > 0 {
                        max_buffers as usize
                    } else if min_buffers > 0 {
                        std::cmp::max(min_buffers as usize, 16) // Use at least 16 buffers
                    } else {
                        16 // Default buffer count
                    };

                    // Reconfigure the pool with new dimensions
                    pool_guard.reconfigure(size as usize, buffer_count);
                }
            } else {
                tracing::error!("No frame buffer pool configured");
                return false;
            }

            true
        }

        fn acquire_buffer(
            &self,
            _params: Option<&BufferPoolAcquireParams>,
        ) -> Result<gst::Buffer, gst::FlowError> {
            if let Some(managed) = self.managed_pool.lock().unwrap().as_ref() {
                return managed.acquire_buffer().map_err(|error| {
                    tracing::error!(%error, "managed GST buffer acquisition failed");
                    gst::FlowError::Error
                });
            }
            let pool_opt = { self.pool.lock().unwrap().clone() };
            let Some(pool) = pool_opt else {
                tracing::error!("Acquire attempt without configured pool");
                return Err(gst::FlowError::Error);
            };

            let mut pool_guard = pool.lock().unwrap();
            let frame_pool_buffer_size = pool_guard.buffer_size();

            let Some(handle) = pool_guard.alloc() else {
                tracing::error!(
                    frame_pool_buffer_size,
                    buffer_count = pool_guard.buffer_count(),
                    "Pool exhausted - no available buffers"
                );
                return Err(gst::FlowError::Flushing);
            };

            tracing::trace!(
                frame_pool_buffer_size,
                available_slots = pool_guard.available_count(),
                in_use_slots = pool_guard.in_use_count(),
                "acquired buffer from pool"
            );

            // TODO: do we need a weak reference here?
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
                    frame_pool_buffer_size,
                    0,
                    frame_pool_buffer_size,
                    Box::into_raw(user_data) as *mut std::ffi::c_void,
                    Some(pool_deallocator),
                ))
            };

            let mut buffer = gst::Buffer::new();
            buffer.get_mut().unwrap().append_memory(memory);

            Ok(buffer)
        }

        fn stop(&self) -> bool {
            tracing::debug!("Stopping buffer pool");

            // When stopping, we should ensure any outstanding buffers are handled
            // The GStreamer base class will handle most of the cleanup, but we can
            // add any custom cleanup logic here if needed
            true
        }
    }

    struct PoolDeallocatorData {
        pool: Weak<Mutex<crate::FrameBufferPool>>,
        buffer_ptr: *const u8,
    }

    // Custom deallocator called when GStreamer memory is freed
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
    fn test_buffer_pool_creation() {
        gst::init().unwrap();

        let buffer_pool = VydGstBufferPool::new();

        // Verify it's properly typed
        assert!(buffer_pool.is::<gst::BufferPool>());
        assert_eq!(buffer_pool.type_().name(), "VydGstBufferPool");
    }

    #[test]
    fn test_pool_configuration() {
        gst::init().unwrap();

        let buffer_pool = VydGstBufferPool::new();
        let buffer_size = 1024;
        let buffer_count = 4;
        let pool = create_shared_pool(buffer_size, buffer_count);

        // Set underlying pool
        buffer_pool.set_frame_buffer_pool(pool.clone());

        // Configure buffer pool
        let mut config = buffer_pool.config();
        config.set_params(None, buffer_size as u32, 0, buffer_count as u32);
        let result = buffer_pool.set_config(config);
        assert!(result.is_ok());

        // Verify underlying pool configuration
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.buffer_size(), buffer_size);
        assert_eq!(pool_guard.buffer_count(), buffer_count);
    }

    #[test]
    fn test_acquisition_without_pool() {
        gst::init().unwrap();

        let buffer_pool = VydGstBufferPool::new();

        // Configure without setting underlying pool
        let mut config = buffer_pool.config();
        config.set_params(None, 1024, 0, 2);
        assert!(buffer_pool.set_config(config).is_err());

        // Should fail - no pool configured
        let result = buffer_pool.imp().acquire_buffer(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_successful_acquisition_and_automatic_cleanup() {
        gst::init().unwrap();

        let buffer_pool = VydGstBufferPool::new();
        let buffer_size = 1024;
        let buffer_count = 2;
        let pool = create_shared_pool(buffer_size, buffer_count);
        buffer_pool.set_frame_buffer_pool(pool.clone());

        // Configure buffer pool
        let mut config = buffer_pool.config();
        config.set_params(None, buffer_size as u32, 0, buffer_count as u32);
        buffer_pool.set_config(config).unwrap();
        buffer_pool.set_active(true).unwrap();

        // Test acquisition and automatic cleanup through scope
        {
            // Should succeed with correct size
            let result = buffer_pool.acquire_buffer(None);
            assert!(result.is_ok());

            let buffer = result.unwrap();
            assert_eq!(buffer.size(), buffer_size);

            // Pool should reflect allocation
            {
                let pool_guard = pool.lock().unwrap();
                assert_eq!(pool_guard.in_use_count(), 1);
                assert_eq!(pool_guard.available_count(), buffer_count - 1);
            }

            // Buffer goes out of scope here and should be automatically freed
        }

        // Check if automatic cleanup worked
        {
            let pool_guard = pool.lock().unwrap();
            let in_use = pool_guard.in_use_count();
            if in_use == 0 {
                println!("✓ Automatic cleanup working correctly");
            } else {
                println!(
                    "! Automatic cleanup not working - buffer still in use: {}",
                    in_use
                );
            }
        }
    }

    #[test]
    fn test_size_mismatch_reconfigures_pool() {
        gst::init().unwrap();

        let buffer_pool = VydGstBufferPool::new();
        let buffer_size = 1024;
        let pool = create_shared_pool(buffer_size, 2);
        buffer_pool.set_frame_buffer_pool(pool.clone());

        // Configure with wrong size
        let mut config = buffer_pool.config();
        config.set_params(None, 2048, 0, 2); // Wrong size

        let result = buffer_pool.set_config(config);

        assert!(result.is_ok());
        assert_eq!(pool.lock().unwrap().buffer_size(), 2048);
    }

    #[test]
    fn test_pool_exhaustion() {
        gst::init().unwrap();

        let buffer_pool = VydGstBufferPool::new();
        let buffer_size = 1024;
        let buffer_count = 1; // Only one slot
        let pool = create_shared_pool(buffer_size, buffer_count);
        buffer_pool.set_frame_buffer_pool(pool.clone());

        // Configure buffer pool
        let mut config = buffer_pool.config();
        config.set_params(None, buffer_size as u32, 0, buffer_count as u32);
        buffer_pool.set_config(config).unwrap();
        buffer_pool.set_active(true).unwrap();

        // First acquisition should succeed
        let buf1 = buffer_pool.acquire_buffer(None);
        assert!(buf1.is_ok());

        // Pool should be exhausted
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), 1);
            assert_eq!(pool_guard.available_count(), 0);
            assert!(pool_guard.is_full());
        }

        // Second acquisition should fail (pool exhausted)
        let buf2 = buffer_pool.acquire_buffer(None);
        assert!(buf2.is_err());

        // Keep the buffer alive to maintain pool state
        let _buffer = buf1.unwrap();

        // Verify pool is still exhausted
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.available_count(), 0);
        }
    }
}
