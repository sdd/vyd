use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};
use vyd::frame::pool::{BufferPoolManager, CpuPoolProvider, PoolError, PoolLimits, WaitPolicy};
use vyd::statistics::{PipelineStatistics, StatsContext};

fn make_pool(buffers: usize, bytes: usize) -> BufferPoolManager {
    BufferPoolManager::new(PoolLimits::new(buffers, bytes).unwrap())
}

#[test]
fn initialized_backing_keeps_recycled_contents_and_counts_explicit_clears() {
    let pool = make_pool(1, 4);
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone()).for_stage("pool").enter();
    pool.preallocate(4, 1).unwrap();
    assert!(
        !stats
            .snapshot()
            .memory_operations
            .contains_key("pool_cpu_zero_bytes")
    );
    let mut first = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    first
        .try_write(WaitPolicy::Immediate)
        .unwrap()
        .copy_from_slice(&[1, 2, 3, 4]);
    drop(first);
    let reused = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    assert_eq!(&*reused, &[1, 2, 3, 4]);
    assert!(
        !stats
            .snapshot()
            .memory_operations
            .contains_key("pool_cpu_zero_count")
    );
    drop(reused);
    let cleared = pool.try_acquire_zeroed(4, WaitPolicy::Immediate).unwrap();
    assert_eq!(&*cleared, &[0, 0, 0, 0]);
    let memory = stats.snapshot().memory_operations;
    assert_eq!(memory["pool_cpu_zero_bytes"], 4);
    assert_eq!(memory["pool_cpu_zero_count"], 1);
}

#[test]
fn final_handle_returns_initialized_backing_and_reports_owners() {
    let pool = make_pool(2, 8);
    pool.preallocate(4, 2).unwrap();
    let mut first = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    first.try_write(WaitPolicy::Immediate).unwrap()[0] = 9;
    let retained = first.clone();
    let second = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    assert_eq!(
        pool.try_acquire(4, WaitPolicy::Immediate).err(),
        Some(PoolError::Exhausted)
    );
    let snapshot = pool.snapshot().unwrap();
    assert_eq!(
        (
            snapshot.allocated_buffers,
            snapshot.allocated_bytes,
            snapshot.idle_buffers
        ),
        (2, 8, 0)
    );
    assert_eq!(snapshot.active.len(), 2);
    assert_eq!(
        snapshot
            .active
            .iter()
            .find(|owner| owner.id == first.id())
            .unwrap()
            .handles,
        2
    );
    assert!(
        snapshot
            .active
            .iter()
            .all(|owner| owner.origin.ends_with("frame_pool.rs") && owner.line > 0)
    );
    drop(first);
    assert_eq!(pool.snapshot().unwrap().active.len(), 2);
    drop(retained);
    drop(second);
    let reused = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    assert_eq!(&*reused, &[0, 0, 0, 0]);
    assert_eq!(pool.snapshot().unwrap().idle_buffers, 1);
}

#[test]
fn shared_write_detaches_within_pool_and_exhaustion_keeps_old_value() {
    let pool = make_pool(2, 8);
    pool.preallocate(4, 2).unwrap();
    let mut writable = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    writable.try_write(WaitPolicy::Immediate).unwrap()[0] = 1;
    let retained = writable.clone();
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone()).for_stage("cow").enter();
    writable.try_write(WaitPolicy::Immediate).unwrap()[0] = 9;
    assert_eq!((writable[0], retained[0]), (9, 1));
    assert_ne!(writable.id(), retained.id());
    assert_eq!(stats.snapshot().memory_operations["cow_cpu_copy_bytes"], 4);
    assert_eq!(pool.snapshot().unwrap().active.len(), 2);

    let one = make_pool(1, 4);
    let mut only = one.try_acquire(4, WaitPolicy::Immediate).unwrap();
    only.try_write(WaitPolicy::Immediate).unwrap()[0] = 3;
    let copy = only.clone();
    assert_eq!(
        only.try_write(WaitPolicy::Immediate).err(),
        Some(PoolError::Exhausted)
    );
    assert_eq!((only[0], copy[0]), (3, 3));
    drop(copy);
    let old_id = only.id();
    only.try_write(WaitPolicy::Immediate).unwrap()[0] = 7;
    assert_eq!(only.id(), old_id);
}

#[test]
fn byte_limit_idle_eviction_timeout_and_close_are_bounded() {
    assert_eq!(PoolLimits::new(0, 1), Err(PoolError::InvalidLimits));
    let pool = make_pool(1, 8);
    pool.preallocate(4, 1).unwrap();
    assert_eq!(pool.preallocate(4, 1), Err(PoolError::Exhausted));
    assert_eq!(pool.snapshot().unwrap().allocated_bytes, 4);
    let held = pool.try_acquire(6, WaitPolicy::Immediate).unwrap();
    assert_eq!(
        (
            pool.snapshot().unwrap().allocated_buffers,
            pool.snapshot().unwrap().allocated_bytes
        ),
        (1, 6)
    );
    assert_eq!(
        pool.try_acquire(3, WaitPolicy::Immediate).err(),
        Some(PoolError::Exhausted)
    );
    let started = Instant::now();
    assert!(matches!(
        pool.try_acquire(3, WaitPolicy::Timeout(Duration::from_millis(25))),
        Err(PoolError::Timeout(_))
    ));
    assert!(started.elapsed() < Duration::from_secs(1));

    let waiting_pool = pool.clone();
    let worker = std::thread::spawn(move || {
        waiting_pool.try_acquire(3, WaitPolicy::Timeout(Duration::from_secs(2)))
    });
    let deadline = Instant::now() + Duration::from_secs(1);
    while pool.snapshot().unwrap().waiters == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    let was_waiting = pool.snapshot().unwrap().waiters == 1;
    pool.close().unwrap();
    assert!(was_waiting, "worker never entered bounded pool wait");
    assert_eq!(worker.join().unwrap().err(), Some(PoolError::Closed));
    assert_eq!(held.len(), 6);
    drop(held);
    let final_state = pool.snapshot().unwrap();
    assert_eq!(
        (
            final_state.allocated_buffers,
            final_state.allocated_bytes,
            final_state.idle_buffers
        ),
        (0, 0, 0)
    );
    assert_eq!(
        pool.try_acquire(1, WaitPolicy::Immediate).err(),
        Some(PoolError::Closed)
    );
}

struct TestProvider {
    calls: AtomicUsize,
    fail_on: usize,
    manager: Mutex<Option<BufferPoolManager>>,
}

impl CpuPoolProvider for TestProvider {
    fn allocate_initialized(&self, bytes: usize) -> Result<Vec<u8>, PoolError> {
        let snapshot = self.manager.lock().unwrap().as_ref().unwrap().snapshot()?;
        assert_eq!(snapshot.pending_buffers, 2);
        assert_eq!(snapshot.pending_bytes, 8);
        if self.calls.fetch_add(1, Ordering::SeqCst) + 1 == self.fail_on {
            return Err(PoolError::AllocationFailed);
        }
        Ok(vec![7; bytes])
    }
}

#[test]
fn provider_failure_rolls_back_reservation_and_allows_retry() {
    let provider = Arc::new(TestProvider {
        calls: AtomicUsize::new(0),
        fail_on: 2,
        manager: Mutex::new(None),
    });
    let pool = BufferPoolManager::with_provider(PoolLimits::new(2, 8).unwrap(), provider.clone());
    *provider.manager.lock().unwrap() = Some(pool.clone());
    assert_eq!(pool.preallocate(4, 2), Err(PoolError::AllocationFailed));
    let snapshot = pool.snapshot().unwrap();
    assert_eq!(snapshot.allocated_buffers, 0);
    assert_eq!(snapshot.pending_buffers, 0);
    assert_eq!(snapshot.pending_bytes, 0);
    pool.preallocate(4, 2).unwrap();
    assert_eq!(pool.snapshot().unwrap().allocated_bytes, 8);
    assert_eq!(
        &*pool.try_acquire(4, WaitPolicy::Immediate).unwrap(),
        &[7; 4]
    );
    *provider.manager.lock().unwrap() = None;
}

struct BlockingProvider {
    entered: Barrier,
    release: Barrier,
}

impl CpuPoolProvider for BlockingProvider {
    fn allocate_initialized(&self, bytes: usize) -> Result<Vec<u8>, PoolError> {
        self.entered.wait();
        self.release.wait();
        Ok(vec![0; bytes])
    }
}

#[test]
fn pending_growth_respects_limits_and_close_rejects_in_flight_allocation() {
    let provider = Arc::new(BlockingProvider {
        entered: Barrier::new(2),
        release: Barrier::new(2),
    });
    let pool = BufferPoolManager::with_provider(PoolLimits::new(1, 4).unwrap(), provider.clone());
    let worker_pool = pool.clone();
    let worker = std::thread::spawn(move || worker_pool.try_acquire(4, WaitPolicy::Immediate));
    provider.entered.wait();
    let pending = pool.snapshot().unwrap();
    assert_eq!((pending.pending_buffers, pending.pending_bytes), (1, 4));
    assert_eq!(
        pool.try_acquire(4, WaitPolicy::Immediate).err(),
        Some(PoolError::Exhausted)
    );
    pool.close().unwrap();
    provider.release.wait();
    assert_eq!(worker.join().unwrap().err(), Some(PoolError::Closed));
    let closed = pool.snapshot().unwrap();
    assert_eq!((closed.pending_buffers, closed.allocated_buffers), (0, 0));
}

#[test]
fn cancellation_wakes_waiters_but_leaves_pool_usable() {
    let pool = make_pool(1, 4);
    let held = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    let waiting_pool = pool.clone();
    let worker = std::thread::spawn(move || {
        waiting_pool.try_acquire(4, WaitPolicy::Timeout(Duration::from_secs(2)))
    });
    let deadline = Instant::now() + Duration::from_secs(1);
    while pool.snapshot().unwrap().waiters == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(pool.snapshot().unwrap().waiters, 1);
    pool.cancel_waiters().unwrap();
    assert_eq!(worker.join().unwrap().err(), Some(PoolError::Cancelled));
    assert!(!pool.snapshot().unwrap().closed);
    drop(held);
    pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
}

#[test]
fn timed_acquisition_succeeds_after_final_read_holder_releases() {
    let pool = make_pool(1, 4);
    let mut first = pool.try_acquire(4, WaitPolicy::Immediate).unwrap();
    first.try_write(WaitPolicy::Immediate).unwrap()[0] = 42;
    let reader = first.clone();
    let worker_pool = pool.clone();
    let worker = std::thread::spawn(move || {
        worker_pool.try_acquire(4, WaitPolicy::Timeout(Duration::from_secs(1)))
    });
    let deadline = Instant::now() + Duration::from_secs(1);
    while pool.snapshot().unwrap().waiters == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(pool.snapshot().unwrap().waiters, 1);
    drop(first);
    {
        // A borrowed read cannot outlive its lease. This holder keeps the
        // backing active after the original lease is dropped.
        let mapped_read: &[u8] = &reader;
        assert_eq!(mapped_read[0], 42);
        assert_eq!(pool.snapshot().unwrap().active[0].handles, 1);
        assert_eq!(pool.snapshot().unwrap().idle_buffers, 0);
    }
    drop(reader);
    let acquired = worker.join().unwrap().unwrap();
    assert_eq!(acquired[0], 42);
}
