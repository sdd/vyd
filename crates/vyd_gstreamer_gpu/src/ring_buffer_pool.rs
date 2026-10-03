use gst::prelude::*;
use gst_allocators::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use std::sync::{mpsc, Arc, Condvar, Mutex};

use crate::frame_buffer::FrameInfo;
use crate::gst_source::GstSourceError;

// Custom allocator that provides memory from pre-allocated ring buffer slots
pub struct RingBufferAllocator {
    frame_buffer: Arc<RingBufferPool>,
}

// Ring buffer pool that manages pre-allocated memory slots
pub struct RingBufferPool {
    // Pre-allocated memory slots for zero-copy operation
    memory_slots: Mutex<Vec<Option<*mut u8>>>,
    available_slots: Mutex<Vec<usize>>, // Stack of available slot indices
    slot_available: Condvar,
    frame_info: FrameInfo,
    capacity: usize,
    // Occupancy tracking for flow control
    occupied_count: Mutex<usize>,
    state_sender: Mutex<Option<mpsc::Sender<bool>>>,
    pause_threshold: usize,
    resume_threshold: usize,
}

impl RingBufferPool {
    pub fn new(frame_info: FrameInfo, capacity: usize) -> Result<Arc<Self>, GstSourceError> {
        let pause_threshold = (capacity - 3).max(1);

        // Pre-allocate all memory slots
        let mut memory_slots = Vec::with_capacity(capacity);
        let mut available_slots = Vec::with_capacity(capacity);

        for i in 0..capacity {
            // Allocate aligned memory for each slot
            let layout = std::alloc::Layout::from_size_align(
                frame_info.frame_size,
                64, // 64-byte alignment for SIMD/DMA
            )
            .unwrap();

            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GstSourceError::GeneralInitError(
                    "Failed to allocate memory for ring buffer slot".to_string(),
                ));
            }

            memory_slots.push(Some(ptr));
            available_slots.push(i);
        }

        Ok(Arc::new(Self {
            memory_slots: Mutex::new(memory_slots),
            available_slots: Mutex::new(available_slots),
            slot_available: Condvar::new(),
            frame_info,
            capacity,
            occupied_count: Mutex::new(0),
            state_sender: Mutex::new(None),
            pause_threshold,
            resume_threshold: 1,
        }))
    }

    pub fn set_state_sender(&self, sender: mpsc::Sender<bool>) {
        let mut state_sender = self
            .state_sender
            .lock()
            .expect("Failed to lock state sender mutex");
        *state_sender = Some(sender);
    }

    // Called by allocator to get a free memory slot
    pub fn acquire_slot(&self) -> Option<(usize, *mut u8)> {
        let mut available = self
            .available_slots
            .lock()
            .expect("Failed to lock available slots mutex");
        let mut occupied = self
            .occupied_count
            .lock()
            .expect("Failed to lock occupied count mutex");

        if let Some(slot_idx) = available.pop() {
            *occupied += 1;
            let occupied_count = *occupied;

            // Check if we need to pause pipeline
            if occupied_count == self.pause_threshold {
                if let Some(sender) = self.state_sender.lock().unwrap().as_ref() {
                    tracing::info!(occupied_count, "buffer reached upper threshold");
                    drop(available);
                    drop(occupied);
                    let _ = sender.send(true);
                } else {
                    drop(available);
                    drop(occupied);
                }
            } else {
                drop(available);
                drop(occupied);
            }

            let memory_slots = self.memory_slots.lock().unwrap();
            if let Some(Some(ptr)) = memory_slots.get(slot_idx) {
                return Some((slot_idx, *ptr));
            }
        }

        None
    }

    // Called when buffer is released back to pool
    pub fn release_slot(&self, slot_idx: usize) {
        let mut available = self.available_slots.lock().unwrap();
        let mut occupied = self.occupied_count.lock().unwrap();

        available.push(slot_idx);
        let old_occupied = *occupied;
        *occupied -= 1;
        let new_occupied = *occupied;

        drop(available);
        drop(occupied);

        // Notify waiting allocations
        self.slot_available.notify_one();

        // Check if we need to resume pipeline
        if old_occupied > self.resume_threshold && new_occupied <= self.resume_threshold {
            if let Some(sender) = self.state_sender.lock().unwrap().as_ref() {
                tracing::info!(
                    occupied_count = new_occupied,
                    "buffer reached lower threshold"
                );
                let _ = sender.send(false);
            }
        }
    }

    // For reading frames - blocks until data available
    pub fn read_frame_from_slot(&self, slot_idx: usize) -> Option<&[u8]> {
        let memory_slots = self.memory_slots.lock().unwrap();
        if let Some(Some(ptr)) = memory_slots.get(slot_idx) {
            unsafe { Some(std::slice::from_raw_parts(*ptr, self.frame_info.frame_size)) }
        } else {
            None
        }
    }
}

impl Drop for RingBufferPool {
    fn drop(&mut self) {
        // Clean up allocated memory
        let memory_slots = self.memory_slots.lock().unwrap();
        let layout = std::alloc::Layout::from_size_align(self.frame_info.frame_size, 64).unwrap();

        for slot in memory_slots.iter() {
            if let Some(ptr) = slot {
                unsafe {
                    std::alloc::dealloc(*ptr, layout);
                }
            }
        }
    }
}

// Custom memory type that knows which slot it belongs to
pub struct RingBufferMemory {
    slot_idx: usize,
    pool: Arc<RingBufferPool>,
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for RingBufferMemory {}
unsafe impl Sync for RingBufferMemory {}

impl Drop for RingBufferMemory {
    fn drop(&mut self) {
        // Return slot to pool when memory is released
        self.pool.release_slot(self.slot_idx);
    }
}

// GStreamer allocator implementation
gst::glib::wrapper! {
    pub struct RingBufferGstAllocator(ObjectSubclass<imp::RingBufferGstAllocator>)
        @extends gst_allocators::Allocator, gst::Object;
}

impl RingBufferGstAllocator {
    pub fn new(pool: Arc<RingBufferPool>) -> Self {
        gst::glib::Object::builder().property("pool", &pool).build()
    }
}

mod imp {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    pub struct RingBufferGstAllocator {
        pool: Mutex<Option<Arc<RingBufferPool>>>,
    }

    #[gst::glib::object_subclass]
    impl ObjectSubclass for RingBufferGstAllocator {
        const NAME: &'static str = "RingBufferGstAllocator";
        type Type = super::RingBufferGstAllocator;
        type ParentType = gst_allocators::Allocator;
    }

    impl ObjectImpl for RingBufferGstAllocator {
        fn properties() -> &'static [gst::glib::ParamSpec] {
            static PROPERTIES: std::sync::OnceLock<Vec<gst::glib::ParamSpec>> =
                std::sync::OnceLock::new();
            PROPERTIES.get_or_init(|| {
                vec![
                    gst::glib::ParamSpecBoxed::builder::<Arc<RingBufferPool>>("pool")
                        .nick("Pool")
                        .blurb("Ring buffer pool")
                        .build(),
                ]
            })
        }

        fn set_property(&self, _id: usize, value: &gst::glib::Value, pspec: &gst::glib::ParamSpec) {
            match pspec.name() {
                "pool" => {
                    let pool = value.get::<Arc<RingBufferPool>>().unwrap();
                    *self.pool.lock().unwrap() = Some(pool);
                }
                _ => unimplemented!(),
            }
        }
    }

    impl GstObjectImpl for RingBufferGstAllocator {}

    impl AllocatorImpl for RingBufferGstAllocator {
        fn alloc(
            &self,
            size: usize,
            _params: Option<&gst::AllocationParams>,
        ) -> Option<gst::Memory> {
            let pool_guard = self.pool.lock().unwrap();
            let pool = pool_guard.as_ref()?;

            if let Some((slot_idx, ptr)) = pool.acquire_slot() {
                // Create custom memory that will release slot when dropped
                let ring_memory = RingBufferMemory {
                    slot_idx,
                    pool: pool.clone(),
                    ptr,
                    size,
                };

                // Wrap in GStreamer memory
                // Note: This is simplified - you'd need to implement proper GstMemory wrapping
                Some(gst::Memory::from_mut_slice(unsafe {
                    std::slice::from_raw_parts_mut(ptr, size)
                }))
            } else {
                None
            }
        }
    }
}
