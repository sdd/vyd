use bytes::{Bytes, BytesMut};
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use tracing;
use vyd_core::frame::{CpuStorageShared, FrameError};
use vyd_core::Frame;

/// A frame buffer that owns memory from the pool and implements CpuStorageShared
pub struct FrameBuffer {
    pool_state: Arc<Mutex<FrameBufferPoolState>>,
    slot_index: usize,
    data: Bytes,
}

impl Deref for FrameBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl CpuStorageShared for FrameBuffer {
    fn clone_shared(&self) -> Box<dyn CpuStorageShared> {
        {
            self.pool_state
                .lock()
                .unwrap()
                .inc_slot_ref_count(self.slot_index);
        }

        Box::new(FrameBuffer {
            pool_state: Arc::clone(&self.pool_state),
            slot_index: self.slot_index,
            data: self.data.clone(),
        })
    }

    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>> {
        // Bytes::to_vec copies; never report it as an ownership transfer.
        Err(self)
    }
}

impl Drop for FrameBuffer {
    fn drop(&mut self) {
        if let Ok(mut state) = self.pool_state.lock() {
            state.dec_slot_ref_count(self.slot_index);
        }
    }
}

#[derive(Debug)]
enum SlotUsage {
    Free,
    AllocRaw,
    AllocArc,
}

impl SlotUsage {
    fn is_free(&self) -> bool {
        matches!(self, SlotUsage::Free)
    }
}

/// A memory slot in the frame buffer pool
#[derive(Debug)]
struct MemorySlot {
    /// Bytes slice for this slot (view into the arena)
    buffer: Bytes,
    /// Whether this slot is currently in use
    usage: SlotUsage,
    /// Number of FrameBuffer instances referencing this slot
    ref_count: usize,
}

impl MemorySlot {
    fn is_free(&self) -> bool {
        self.usage.is_free()
    }
}

/// Handle that represents ownership of a buffer from the pool
pub struct BufferHandle {
    /// The buffer data as a mutable slice
    buffer: *mut [u8],
    /// Pointer to identify this buffer when returning it
    buffer_ptr: *const u8,
    /// Store the length to avoid dereferencing
    len: usize,
}

impl BufferHandle {
    /// Get a mutable slice to the buffer data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { &mut *self.buffer }
    }

    /// Get the buffer pointer for returning to the pool
    pub fn as_ptr(&self) -> *const u8 {
        self.buffer_ptr
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Safety: BufferHandle contains a valid pointer that we control
unsafe impl Send for BufferHandle {}

/// Frame buffer pool that manages pre-allocated memory buffers for zero-copy operations
#[derive(Debug, Clone)]
pub struct FrameBufferPool {
    state: Arc<Mutex<FrameBufferPoolState>>,

    arena_base: usize,
    arena_top: usize,
    buffer_size: usize,
    buffer_count: usize,
}

/// Internal state of the frame buffer pool
#[derive(Debug)]
struct FrameBufferPoolState {
    /// All memory slots
    slots: Vec<MemorySlot>,
    /// Size of each buffer slot
    buffer_size: usize,
    /// Total number of buffer slots
    buffer_count: usize,
    /// The underlying memory arena
    memory_arena: Bytes,
}

impl FrameBufferPool {
    /// Creates a new frame buffer pool with the specified buffer size and count
    pub fn new(buffer_size: usize, buffer_count: usize) -> Self {
        let total_size = buffer_size * buffer_count;

        // Create zeroed arena
        let arena = BytesMut::zeroed(total_size);
        let memory_arena = arena.freeze();

        // Create slots as views into the arena
        let mut slots = Vec::with_capacity(buffer_count);
        for i in 0..buffer_count {
            let start = i * buffer_size;
            let end = start + buffer_size;
            let slot_buffer = memory_arena.slice(start..end);

            slots.push(MemorySlot {
                buffer: slot_buffer,
                usage: SlotUsage::Free,
                ref_count: 0,
            });
        }

        let total_allocated_bytes = total_size;

        tracing::debug!(
            buffer_count,
            buffer_size,
            total_allocated_bytes,
            "FrameBufferPool created successfully"
        );

        let state = FrameBufferPoolState {
            slots,
            buffer_size,
            buffer_count,
            memory_arena,
        };

        let arena_base = state.memory_arena.as_ptr() as usize;
        let arena_top = arena_base + state.memory_arena.len();

        let state = Arc::new(Mutex::new(state));

        Self {
            state,
            arena_base,
            arena_top,
            buffer_size,
            buffer_count,
        }
    }

    /// Recreates the pool with new buffer size and count
    /// This will invalidate all existing buffers and reset the pool
    pub fn reconfigure(&mut self, new_buffer_size: usize, new_buffer_count: usize) {
        tracing::info!(
            old_size = self.buffer_size,
            new_size = new_buffer_size,
            old_count = self.buffer_count,
            new_count = new_buffer_count,
            "Reconfiguring frame buffer pool with new dimensions"
        );

        // Create new memory arena
        let total_size = new_buffer_size * new_buffer_count;

        // Create zeroed arena
        let arena = BytesMut::zeroed(total_size);
        let memory_arena = arena.freeze();

        let arena_base = memory_arena.as_ptr() as usize;
        let arena_top = arena_base + total_size;

        // Create new slots
        let mut new_slots = Vec::with_capacity(new_buffer_count);
        for i in 0..new_buffer_count {
            let start = i * new_buffer_size;
            let end = start + new_buffer_size;
            let buffer = memory_arena.slice(start..end);

            new_slots.push(MemorySlot {
                buffer,
                usage: SlotUsage::Free,
                ref_count: 0,
            });
        }

        // Update the state
        let mut state = self.state.lock().unwrap();
        *state = FrameBufferPoolState {
            slots: new_slots,
            buffer_size: new_buffer_size,
            buffer_count: new_buffer_count,
            memory_arena,
        };

        // Update the pool's cached values
        self.arena_base = arena_base;
        self.arena_top = arena_top;
        self.buffer_size = new_buffer_size;
        self.buffer_count = new_buffer_count;

        tracing::info!("Frame buffer pool reconfiguration complete");
    }

    /// Allocate a buffer handle from the pool
    pub fn alloc(&mut self) -> Option<BufferHandle> {
        self.state.lock().unwrap().alloc()
    }

    /// Free a buffer handle back to the pool by pointer
    pub fn free(&mut self, buffer_ptr: *const u8) -> bool {
        self.state.lock().unwrap().free(buffer_ptr)
    }

    /// Try to create a Frame from a pointer that might be from this pool
    /// Returns Ok(Frame) if the pointer is from this pool and the slot is in a valid state,
    /// Err(FrameError::BufferUnavailable) if not or if the slot has been freed or is already in use,
    ///
    pub fn acquire_frame_from_ptr(
        &self,
        ptr: *const u8,
        width: u32,
        height: u32,
    ) -> Result<Frame, FrameError> {
        // Check if this pointer belongs to our pool
        let slot_index = match self.get_slot_index(ptr) {
            Some(slot_index) => slot_index,
            None => return Err(FrameError::BufferUnavailable),
        };

        // Check the slot state and update it atomically
        let data = {
            let mut state = self.state.lock().unwrap();
            match state.slots[slot_index].usage {
                SlotUsage::AllocRaw => {
                    // Valid transition: AllocRaw -> AllocArc
                    state.slots[slot_index].usage = SlotUsage::AllocArc;
                    debug_assert_eq!(
                        state.slots[slot_index].ref_count, 0,
                        "ref_count should be zero when frame is in AllocRaw state"
                    );
                    state.inc_slot_ref_count(slot_index);

                    // Get the Bytes buffer for zero-copy access
                    let slot_buffer = &state.slots[slot_index].buffer;

                    // Create a slice of the appropriate size for the frame
                    let frame_size = (width * height) as usize;
                    if slot_buffer.len() >= frame_size {
                        slot_buffer.slice(0..frame_size)
                    } else {
                        slot_buffer.clone()
                    }
                }
                SlotUsage::Free => {
                    // Invalid: trying to acquire frame from freed slot
                    return Err(FrameError::BufferUnavailable);
                }
                SlotUsage::AllocArc => {
                    // Invalid: slot is already being used as Arc
                    return Err(FrameError::BufferUnavailable);
                }
            }
        };

        let frame_buffer = FrameBuffer {
            pool_state: Arc::clone(&self.state),
            slot_index,
            data,
        };

        Frame::from_shared(frame_buffer, width, height)
    }

    /// Check if the pool is full (all slots in use)
    pub fn is_full(&self) -> bool {
        self.available_count() == 0
    }

    /// Returns the number of available (unused) buffer slots
    pub fn available_count(&self) -> usize {
        self.state
            .lock()
            .unwrap()
            .slots
            .iter()
            .filter(|slot| slot.is_free())
            .count()
    }

    /// Returns the number of buffer slots currently in use
    pub fn in_use_count(&self) -> usize {
        self.state
            .lock()
            .unwrap()
            .slots
            .iter()
            .filter(|slot| !slot.is_free())
            .count()
    }

    /// Returns the size of each buffer slot
    pub fn buffer_size(&self) -> usize {
        self.state.lock().unwrap().buffer_size
    }

    /// Returns the total number of buffer slots
    pub fn buffer_count(&self) -> usize {
        self.state.lock().unwrap().buffer_count
    }

    /// Check if a pointer points to memory within this pool's arena
    pub fn is_from_pool(&self, ptr: *const u8) -> bool {
        ptr as usize >= self.arena_base && (ptr as usize) < self.arena_top
    }

    /// Find which slot index corresponds to this pointer
    /// Returns None if pointer is not from this pool or not properly aligned
    pub fn get_slot_index(&self, ptr: *const u8) -> Option<usize> {
        get_slot_index(
            ptr,
            self.arena_base,
            self.arena_top,
            self.buffer_size,
            self.buffer_count,
        )
    }
}

fn get_slot_index(
    ptr: *const u8,
    arena_base: usize,
    arena_top: usize,
    buffer_size: usize,
    buffer_count: usize,
) -> Option<usize> {
    // Quick bounds check
    if (ptr as usize) < arena_base || (ptr as usize) >= arena_top {
        return None;
    }

    // Calculate offset from base
    let offset = (ptr as usize) - arena_base;

    // Check alignment and calculate slot index
    let (slot_index, remainder) = (offset / buffer_size, offset % buffer_size);

    // Verify alignment - pointer should be at start of a slot
    if remainder != 0 {
        return None;
    }

    // Verify slot index is valid
    if slot_index >= buffer_count {
        return None;
    }

    Some(slot_index)
}

impl FrameBufferPoolState {
    /// Allocate a buffer handle from the pool
    pub fn alloc(&mut self) -> Option<BufferHandle> {
        for slot in &mut self.slots {
            if slot.is_free() {
                slot.usage = SlotUsage::AllocRaw;
                // Convert Bytes to mutable slice
                // Safety: We control the lifecycle and ensure exclusive access
                let ptr = slot.buffer.as_ptr() as *mut u8;
                let len = slot.buffer.len();
                let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, len);

                return Some(BufferHandle {
                    buffer: slice_ptr,
                    buffer_ptr: ptr,
                    len,
                });
            }
        }
        None
    }

    /// Free a buffer handle back to the pool by pointer
    pub fn free(&mut self, buffer_ptr: *const u8) -> bool {
        let Some(slot_idx) = self.get_slot_index(buffer_ptr) else {
            tracing::warn!("Freeing buffer not from pool");
            return false;
        };
        let slot = &mut self.slots[slot_idx];

        match slot.usage {
            SlotUsage::AllocRaw => {
                slot.usage = SlotUsage::Free;

                // Zero out the buffer for next use - technically we
                // don't need to do this. remove once we have tested
                // it all and it works
                // let ptr = slot.buffer.as_ptr() as *mut u8;
                // let slice = unsafe { std::slice::from_raw_parts_mut(ptr, slot.buffer.len()) };
                // slice.fill(0);

                true
            }
            SlotUsage::AllocArc => {
                // slot has been returned and reassigned as a Frame
                // but then freed by gstreamer
                true
            }
            SlotUsage::Free => {
                // Double free!
                tracing::warn!("Double free of buffer");
                false
            }
        }
    }

    /// Find which slot index corresponds to this pointer
    /// Returns None if pointer is not from this pool or not properly aligned
    fn get_slot_index(&self, ptr: *const u8) -> Option<usize> {
        let arena_base = self.memory_arena.as_ptr() as usize;
        let arena_top = arena_base + self.memory_arena.len();
        get_slot_index(
            ptr,
            arena_base,
            arena_top,
            self.buffer_size,
            self.buffer_count,
        )
    }

    /// Increment the reference count for a slot when creating a FrameBuffer
    fn inc_slot_ref_count(&mut self, slot_index: usize) {
        if let Some(slot) = self.slots.get_mut(slot_index) {
            slot.ref_count += 1;
        } else {
            tracing::warn!(
                slot_index,
                "inc_slot_ref_count called on invalid slot index"
            );
        }
    }

    /// Decrement the reference count for a slot when a FrameBuffer is dropped
    fn dec_slot_ref_count(&mut self, slot_index: usize) {
        if let Some(slot) = self.slots.get_mut(slot_index) {
            if slot.ref_count > 0 {
                slot.ref_count -= 1;
                if slot.ref_count == 0 {
                    slot.usage = SlotUsage::Free;
                    tracing::trace!(
                        slot_index,
                        "All FrameBuffer references dropped, slot marked as available"
                    );
                }
            } else {
                tracing::warn!(
                    slot_index,
                    "dec_slot_ref_count called on slot with zero references"
                );
            }
        } else {
            tracing::warn!(
                slot_index,
                "dec_slot_ref_count called on invalid slot index"
            );
        }
    }
}

// Safety: FrameBufferPool is safe to send between threads
// The Bytes type is already Send + Sync, and we manage slot state appropriately
unsafe impl Send for FrameBufferPool {}
unsafe impl Sync for FrameBufferPool {}

/// Thread-safe wrapper around the frame buffer pool
pub type SharedFrameBufferPool = Arc<Mutex<FrameBufferPool>>;

/// Creates a new shared frame buffer pool
pub fn create_shared_pool(buffer_size: usize, buffer_count: usize) -> SharedFrameBufferPool {
    Arc::new(Mutex::new(FrameBufferPool::new(buffer_size, buffer_count)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = FrameBufferPool::new(1024, 4);
        assert_eq!(pool.buffer_size(), 1024);
        assert_eq!(pool.buffer_count(), 4);
        assert_eq!(pool.available_count(), 4);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_buffer_allocation_and_return() {
        let mut pool = FrameBufferPool::new(1024, 2);

        // Store pointers first to avoid borrowing issues
        let (buffer1_ptr, buffer2_ptr) = {
            // Get first buffer
            let buffer1 = pool.alloc().unwrap();
            assert_eq!(buffer1.len(), 1024);
            let ptr1 = buffer1.as_ptr();

            // Get second buffer
            let buffer2 = pool.alloc().unwrap();
            assert_eq!(buffer2.len(), 1024);
            let ptr2 = buffer2.as_ptr();

            // No more buffers available
            assert!(pool.alloc().is_none());

            (ptr1, ptr2)
        };

        // Now we can check counts without borrowing conflicts
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.in_use_count(), 2);

        // Return first buffer
        assert!(pool.free(buffer1_ptr));
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Return second buffer
        assert!(pool.free(buffer2_ptr));
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_buffer_handle_approach() {
        let mut pool = FrameBufferPool::new(1024, 2);

        // Get first buffer handle
        let mut handle1 = pool.alloc().unwrap();
        assert_eq!(handle1.len(), 1024);

        // We can check pool status while handle is alive!
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Get second buffer handle
        let mut handle2 = pool.alloc().unwrap();
        assert_eq!(handle2.len(), 1024);
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.in_use_count(), 2);

        // No more buffers available
        assert!(pool.alloc().is_none());

        // We can modify buffer contents
        let slice1 = handle1.as_mut_slice();
        slice1[0] = 42;

        let slice2 = handle2.as_mut_slice();
        slice2[0] = 24;

        // Return handles
        let ptr1 = handle1.as_ptr();
        let ptr2 = handle2.as_ptr();

        assert!(pool.free(ptr1));
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        assert!(pool.free(ptr2));
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_shared_pool() {
        let shared_pool = create_shared_pool(256, 3);

        {
            let pool = shared_pool.lock().unwrap();
            assert_eq!(pool.buffer_size(), 256);
            assert_eq!(pool.buffer_count(), 3);
            assert_eq!(pool.available_count(), 3);
        }

        // Test that we can access from multiple scopes using handles
        let buffer_ptr = {
            let mut pool = shared_pool.lock().unwrap();
            let handle = pool.alloc().unwrap();
            let ptr = handle.as_ptr();
            assert_eq!(pool.available_count(), 2);
            ptr
        };

        {
            let pool = shared_pool.lock().unwrap();
            assert_eq!(pool.in_use_count(), 1);
        }

        // Return the buffer
        {
            let mut pool = shared_pool.lock().unwrap();
            assert!(pool.free(buffer_ptr));
            assert_eq!(pool.in_use_count(), 0);
        }
    }

    #[test]
    fn test_allocate_deallocate_interface() {
        let shared_pool = create_shared_pool(1024, 2);
        let mut pool = shared_pool.lock().unwrap();

        // Test allocation
        let handle = pool.alloc();
        assert!(handle.is_some());

        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Test deallocation
        let handle = handle.unwrap();
        let deallocated = pool.free(handle.as_ptr());
        assert!(deallocated);
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_is_full() {
        let mut pool = FrameBufferPool::new(256, 1);

        assert!(!pool.is_full());
        assert_eq!(pool.available_count(), 1);

        // Allocate the only slot using handle approach
        let _handle = pool.alloc().unwrap();
        assert!(pool.is_full());
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_buffer_handle_len_and_empty() {
        let mut pool = FrameBufferPool::new(512, 1);

        let handle = pool.alloc().unwrap();
        assert_eq!(handle.len(), 512);
        assert!(!handle.is_empty());

        // Test with zero-size pool (edge case)
        let mut zero_pool = FrameBufferPool::new(0, 1);
        let zero_handle = zero_pool.alloc().unwrap();
        assert_eq!(zero_handle.len(), 0);
        assert!(zero_handle.is_empty());
    }

    #[test]
    fn test_get_slot_index() {
        let mut pool = FrameBufferPool::new(1024, 4);

        // Allocate a few buffers and verify we can find their slot indices
        let handle1 = pool.alloc().unwrap();
        let handle2 = pool.alloc().unwrap();
        let handle3 = pool.alloc().unwrap();

        let ptr1 = handle1.as_ptr();
        let ptr2 = handle2.as_ptr();
        let ptr3 = handle3.as_ptr();

        // Should allocate slots in order
        assert_eq!(pool.get_slot_index(ptr1).unwrap(), 0);
        assert_eq!(pool.get_slot_index(ptr2).unwrap(), 1);
        assert_eq!(pool.get_slot_index(ptr3).unwrap(), 2);
    }

    #[test]
    fn test_get_slot_index_invalid_pointers() {
        let pool = FrameBufferPool::new(1024, 4);

        // Null pointer
        assert!(pool.get_slot_index(std::ptr::null()).is_none());

        // Random pointer not from our pool
        let random_vec = [0u8; 100];
        assert!(pool.get_slot_index(random_vec.as_ptr()).is_none());

        // Pointer before our arena (should be impossible but let's test)
        let before_arena = pool.arena_base - 1;
        assert!(pool.get_slot_index(before_arena as *const u8).is_none());

        // Pointer after our arena
        assert!(pool.get_slot_index(pool.arena_top as *const u8).is_none());
    }

    #[test]
    fn test_get_slot_index_misaligned_pointers() {
        let mut pool = FrameBufferPool::new(1024, 4);
        let handle = pool.alloc().unwrap();
        let aligned_ptr = handle.as_ptr();

        // Should work for aligned pointer
        assert!(pool.get_slot_index(aligned_ptr).is_some());

        // Should fail for misaligned pointers within our arena
        let misaligned1 = unsafe { aligned_ptr.add(1) };
        let misaligned2 = unsafe { aligned_ptr.add(512) };
        let misaligned3 = unsafe { aligned_ptr.add(1023) };

        assert!(pool.get_slot_index(misaligned1).is_none());
        assert!(pool.get_slot_index(misaligned2).is_none());
        assert!(pool.get_slot_index(misaligned3).is_none());
    }

    #[test]
    fn test_frame_buffer_deref() {
        let mut pool = FrameBufferPool::new(100, 1);

        let ptr = pool.alloc().unwrap().as_ptr();

        let frame = pool.acquire_frame_from_ptr(ptr, 10, 10).unwrap();

        if let Some(cpu_ref) = frame.as_cpu_ref() {
            // Test Deref implementation through the frame's CPU reference
            assert_eq!(cpu_ref.len(), 100);
            assert_eq!(&cpu_ref[0..10], &[0u8; 10]);
        } else {
            panic!("Expected CPU frame");
        }
    }

    #[test]
    fn test_frame_buffer_drop_single_reference() {
        let mut pool = FrameBufferPool::new(256, 2);
        assert_eq!(pool.available_count(), 2);

        let ptr = pool.alloc().unwrap().as_ptr();

        {
            let _frame = pool.acquire_frame_from_ptr(ptr, 16, 16).unwrap();
            // Check count after frame creation but before it drops
            assert_eq!(pool.available_count(), 1);
        } // frame drops here

        // Slot should be marked available after drop
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_frame_buffer_drop_multiple_references() {
        let mut pool = FrameBufferPool::new(256, 3);

        // Initial state: all slots available
        assert_eq!(pool.available_count(), 3);
        assert_eq!(pool.in_use_count(), 0);

        // Step 1: Allocate a buffer from the pool (normal workflow)
        let ptr = pool.alloc().unwrap().as_ptr();

        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 1);

        // Step 2: Create a frame from the allocated pointer
        let frame = pool.acquire_frame_from_ptr(ptr, 16, 16).unwrap();

        // After acquire_frame_from_ptr, the slot should be marked as AllocArc
        assert_eq!(pool.available_count(), 2); // Still 2 available
        assert_eq!(pool.in_use_count(), 1); // Still 1 in use, but now as AllocArc

        // Create additional references using clone_shared()
        let ref2 = frame.clone_shared();

        // Step 4: GStreamer calls free() while frame is still held by Arc
        // This should succeed but do nothing since slot is AllocArc
        let freed = pool.free(ptr);
        assert!(freed); // Should return true (successful handling)

        // Pool state should remain unchanged - slot still in use by Arc
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 1);

        // Step 5: Drop one reference - slot should still be in use
        drop(ref2);
        assert_eq!(pool.available_count(), 2); // Still in use due to cloned reference

        // Step 6: Drop the last reference - now slot should be freed by Drop impl
        drop(frame);

        // Now the slot should be available since all Arc references are gone
        assert_eq!(pool.available_count(), 3);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_gstreamer_free_while_frame_active() {
        let mut pool = FrameBufferPool::new(512, 2);

        // Step 1: Normal allocation workflow
        let buffer_ptr = {
            let handle = pool.alloc().unwrap();
            assert_eq!(pool.available_count(), 1);
            assert_eq!(pool.in_use_count(), 1);
            handle.as_ptr()
        };

        // Step 2: Convert to frame (slot becomes AllocArc)
        let _frame = pool.acquire_frame_from_ptr(buffer_ptr, 32, 16).unwrap();
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Step 3: GStreamer tries to free the buffer while frame is active
        // This should succeed but have no effect on the slot state
        assert!(pool.free(buffer_ptr)); // Returns true (handled successfully)

        // Slot should still be in use because frame is alive
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Step 4: Try to free again - should still return true (idempotent)
        assert!(pool.free(buffer_ptr));
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Step 5: Frame goes out of scope here
        // Drop impl should free the slot
    } // _frame drops here

    // After test function, we can't easily check the pool state, but the Drop
    // implementation should have freed the slot

    #[test]
    fn test_frame_buffer_clone_shared() {
        let pool = FrameBufferPool::new(100, 1);

        // Create a FrameBuffer directly to test clone_shared
        let frame_buffer = FrameBuffer {
            pool_state: Arc::clone(&pool.state),
            slot_index: 0,
            data: Bytes::from(vec![0u8; 100]),
        };

        let cloned = frame_buffer.clone_shared();

        // Both should access the same data
        assert_eq!(frame_buffer.len(), cloned.len());
        assert_eq!(&frame_buffer[..], &cloned[..]);
    }

    #[test]
    fn test_frame_buffer_refuses_copy_disguised_as_ownership_transfer() {
        let pool = FrameBufferPool::new(16, 2);

        // Bytes remains a pool-owned backing even with one FrameBuffer handle.
        let data = Bytes::from(vec![42u8; 16]);

        // Create FrameBuffer and immediately box it to avoid extra references
        let frame_buffer = FrameBuffer {
            pool_state: Arc::clone(&pool.state),
            slot_index: 0,
            data,
        };

        let boxed_buffer = Box::new(frame_buffer);

        let result = boxed_buffer.try_into_owned();
        match result {
            Err(buffer) => {
                assert_eq!(buffer.len(), 16);
                assert_eq!(buffer[0], 42);
                drop(buffer);
            }
            Ok(_) => panic!("copy was misreported as a transfer"),
        }
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn pooled_frame_cow_counts_one_copy_and_returns_backing_after_last_clone() {
        use vyd_core::statistics::{PipelineStatistics, StatsContext};
        let mut pool = FrameBufferPool::new(4, 1);
        let pointer = pool.alloc().unwrap().as_ptr();
        let mut frame = pool.acquire_frame_from_ptr(pointer, 2, 2).unwrap();
        let retained = frame.clone_shared();
        assert_eq!(pool.available_count(), 0);

        let stats = Arc::new(PipelineStatistics::new());
        let _scope = StatsContext::new(stats.clone()).for_stage("cow").enter();
        let Frame::Cpu(cpu) = &mut frame else {
            panic!("expected CPU frame")
        };
        cpu.img.as_mut()[0] = 9;
        assert_eq!(frame.as_cpu_ref().unwrap()[0], 9);
        assert_eq!(retained.as_cpu_ref().unwrap()[0], 0);
        assert_eq!(pool.available_count(), 0);
        drop(retained);
        assert_eq!(pool.available_count(), 1);
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["cow_cpu_copy_bytes"], 4);
        assert_eq!(memory["cow_cpu_copy_count"], 1);
        assert_eq!(memory["cow_cpu_alloc_bytes"], 4);
        assert!(!memory.contains_key("cow_zero_copy"));
    }

    #[test]
    fn test_acquire_frame_after_gstreamer_free() {
        let mut pool = FrameBufferPool::new(256, 2);

        // Step 1: Normal allocation workflow
        let buffer_ptr = {
            let handle = pool.alloc().unwrap();
            assert_eq!(pool.available_count(), 1);
            assert_eq!(pool.in_use_count(), 1);
            handle.as_ptr()
        };

        // Step 2: GStreamer calls free (slot becomes Free and zeroed)
        assert!(pool.free(buffer_ptr));
        assert_eq!(pool.available_count(), 2); // Slot is now Free
        assert_eq!(pool.in_use_count(), 0);

        // Step 3: Someone tries to acquire a frame from the freed pointer
        // This should now be rejected safely
        let result = pool.acquire_frame_from_ptr(buffer_ptr, 16, 16);

        match result {
            Ok(_frame) => {
                panic!("acquire_frame_from_ptr should reject freed pointers!");
            }
            Err(_) => {
                // This is the expected safe behavior
                assert_eq!(pool.available_count(), 2); // Slot remains Free
                assert_eq!(pool.in_use_count(), 0);
            }
        }
    }

    #[test]
    fn test_acquire_frame_after_gstreamer_free_then_realloc() {
        let mut pool = FrameBufferPool::new(256, 2);

        // Step 1: Allocate and get pointer
        let old_buffer_ptr = {
            let handle = pool.alloc().unwrap();
            handle.as_ptr()
        };

        // Step 2: GStreamer frees the buffer
        assert!(pool.free(old_buffer_ptr));
        assert_eq!(pool.available_count(), 2);

        // Step 3: Pool reallocates the same slot to someone else
        let new_handle = pool.alloc().unwrap();
        let new_buffer_ptr = new_handle.as_ptr();

        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Step 4: Someone tries to acquire a frame using the old pointer
        // Even if old_buffer_ptr == new_buffer_ptr, this should be safe now
        let result = pool.acquire_frame_from_ptr(old_buffer_ptr, 16, 16);
        match result {
            Ok(_frame) => {
                // If this succeeds, it means the slot was AllocRaw and we properly
                // transitioned it to AllocArc. The new_handle holder should be aware
                // their buffer might be converted to a frame.
                assert_eq!(pool.available_count(), 1);
                // This is actually the expected behavior for the AllocRaw -> AllocArc case
            }
            Err(_) => {
                // This would happen if the slot was not AllocRaw
                assert_eq!(pool.available_count(), 1);
            }
        }

        // Clean up
        pool.free(new_buffer_ptr);
    }

    #[test]
    fn test_double_acquire_frame_rejected() {
        let mut pool = FrameBufferPool::new(256, 2);

        // Step 1: Normal allocation workflow
        let buffer_ptr = {
            let handle = pool.alloc().unwrap();
            handle.as_ptr()
        };

        // Step 2: First acquire_frame_from_ptr succeeds (AllocRaw -> AllocArc)
        let frame1 = pool.acquire_frame_from_ptr(buffer_ptr, 16, 16).unwrap();
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Step 3: Second acquire_frame_from_ptr should fail (AllocArc -> AllocArc is invalid)
        let result = pool.acquire_frame_from_ptr(buffer_ptr, 16, 16);
        match result {
            Ok(_frame) => {
                panic!("Second acquire_frame_from_ptr should be rejected!");
            }
            Err(_) => {
                // Expected behavior - slot is already AllocArc
                assert_eq!(pool.available_count(), 1);
                assert_eq!(pool.in_use_count(), 1);
            }
        }

        // Clean up
        drop(frame1);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_valid_state_transitions() {
        let mut pool = FrameBufferPool::new(256, 2);

        // Test the only valid transition: Free -> AllocRaw -> AllocArc -> Free

        // Initially Free
        assert_eq!(pool.available_count(), 2);

        // Free -> AllocRaw via alloc()
        let buffer_ptr = {
            let handle = pool.alloc().unwrap();
            assert_eq!(pool.available_count(), 1);
            handle.as_ptr()
        };

        // AllocRaw -> AllocArc via acquire_frame_from_ptr()
        let frame = pool.acquire_frame_from_ptr(buffer_ptr, 16, 16).unwrap();
        assert_eq!(pool.available_count(), 1);

        // AllocArc -> Free via Drop
        drop(frame);
        assert_eq!(pool.available_count(), 2);
    }
}
