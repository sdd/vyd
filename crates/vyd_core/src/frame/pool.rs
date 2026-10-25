//! Bounded CPU backing pool. Leases retain their issuing manager and return
//! initialized storage only after the last handle has been dropped.
use super::CpuStorageShared;
use crate::statistics::{MemoryOperation, StatsContext};
use std::collections::{BTreeMap, HashMap};
use std::ops::Deref;
use std::panic::Location;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

const DEFAULT_MAX_BUFFERS: usize = 32;
const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitPolicy {
    Immediate,
    Timeout(Duration),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolLimits {
    max_buffers: usize,
    max_bytes: usize,
}

impl PoolLimits {
    pub fn new(max_buffers: Option<usize>, max_bytes: Option<usize>) -> Result<Self, PoolError> {
        let max_buffers = max_buffers.unwrap_or(DEFAULT_MAX_BUFFERS);
        let max_bytes = max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

        if max_buffers == 0 || max_bytes == 0 {
            return Err(PoolError::InvalidLimits);
        }
        Ok(Self {
            max_buffers,
            max_bytes,
        })
    }

    pub fn max_buffers(self) -> usize {
        self.max_buffers
    }

    pub fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PoolError {
    #[error("pool limits must be positive")]
    InvalidLimits,
    #[error("buffer size must be positive")]
    InvalidSize,
    #[error("buffer size exceeds pool byte limit")]
    Oversized,
    #[error("CPU buffer pool exhausted")]
    Exhausted,
    #[error("CPU buffer acquisition timed out after {0:?}")]
    Timeout(Duration),
    #[error("CPU buffer pool is closed")]
    Closed,
    #[error("CPU buffer pool wait was cancelled")]
    Cancelled,
    #[error("CPU buffer allocation failed")]
    AllocationFailed,
    #[error("CPU buffer provider returned a backing with the wrong size")]
    InvalidBacking,
    #[error("CPU buffer pool state is poisoned")]
    Poisoned,
    #[error("CPU buffer pool accounting overflow")]
    Overflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseOwner {
    pub id: u64,
    pub bytes: usize,
    pub origin: &'static str,
    pub line: u32,
    pub handles: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolSnapshot {
    pub limits: PoolLimits,
    pub allocated_buffers: usize,
    pub allocated_bytes: usize,
    /// Capacity reserved for allocations currently in the provider callback.
    pub pending_buffers: usize,
    pub pending_bytes: usize,
    pub idle_buffers: usize,
    pub active: Vec<LeaseOwner>,
    pub waiters: usize,
    pub closed: bool,
}

struct Active {
    bytes: usize,
    origin: &'static str,
    line: u32,
    handles: Arc<AtomicUsize>,
}

struct State {
    idle: BTreeMap<usize, Vec<Vec<u8>>>,
    active: HashMap<u64, Active>,
    allocated_buffers: usize,
    allocated_bytes: usize,
    pending_buffers: usize,
    pending_bytes: usize,
    waiters: usize,
    cancel_epoch: u64,
    next_id: u64,
    closed: bool,
}

struct Inner {
    limits: PoolLimits,
    provider: Arc<dyn CpuPoolProvider>,
    state: Mutex<State>,
    changed: Condvar,
}

/// Supplies initialized CPU backing. The manager owns capacity, waiting and
/// lease return; providers only allocate a safe, exact-size byte vector.
/// Callbacks run outside the manager lock and may inspect its diagnostics.
pub trait CpuPoolProvider: Send + Sync + 'static {
    fn allocate_initialized(&self, bytes: usize) -> Result<Vec<u8>, PoolError>;
}

#[derive(Default)]
pub struct HeapCpuProvider;

impl CpuPoolProvider for HeapCpuProvider {
    fn allocate_initialized(&self, bytes: usize) -> Result<Vec<u8>, PoolError> {
        allocate_zeroed(bytes)
    }
}

#[derive(Clone)]
pub struct BufferPoolManager(Arc<Inner>);

impl BufferPoolManager {
    pub fn new(limits: PoolLimits) -> Self {
        Self::with_provider(limits, Arc::new(HeapCpuProvider))
    }

    pub fn with_provider(limits: PoolLimits, provider: Arc<dyn CpuPoolProvider>) -> Self {
        Self(Arc::new(Inner {
            limits,
            provider,
            state: Mutex::new(State {
                idle: BTreeMap::new(),
                active: HashMap::new(),
                allocated_buffers: 0,
                allocated_bytes: 0,
                pending_buffers: 0,
                pending_bytes: 0,
                waiters: 0,
                cancel_epoch: 0,
                next_id: 1,
                closed: false,
            }),
            changed: Condvar::new(),
        }))
    }

    /// Reserve initialized idle backings. Failure leaves
    /// the accounting unchanged, including when allocation itself fails.
    pub fn preallocate(&self, bytes: usize, count: usize) -> Result<(), PoolError> {
        if bytes == 0 {
            return Err(PoolError::InvalidSize);
        }
        if bytes > self.0.limits.max_bytes {
            return Err(PoolError::Oversized);
        }
        if count == 0 {
            let state = self.0.state.lock().map_err(|_| PoolError::Poisoned)?;
            return if state.closed {
                Err(PoolError::Closed)
            } else {
                Ok(())
            };
        }
        let total_bytes = bytes.checked_mul(count).ok_or(PoolError::Overflow)?;
        {
            let mut state = self.0.state.lock().map_err(|_| PoolError::Poisoned)?;
            if state.closed {
                return Err(PoolError::Closed);
            }
            reserve(&mut state, self.0.limits, count, total_bytes)?;
        }
        let mut prepared = Vec::new();
        let result = (|| {
            prepared
                .try_reserve_exact(count)
                .map_err(|_| PoolError::AllocationFailed)?;
            for _ in 0..count {
                prepared.push(self.allocate(bytes)?);
            }
            Ok(())
        })();
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.pending_buffers -= count;
        state.pending_bytes -= total_bytes;
        if let Err(error) = result {
            self.0.changed.notify_all();
            return Err(error);
        }
        if state.closed {
            self.0.changed.notify_all();
            return Err(PoolError::Closed);
        }
        state.idle.entry(bytes).or_default().extend(prepared);
        state.allocated_buffers += count;
        state.allocated_bytes += total_bytes;
        self.0.changed.notify_all();
        Ok(())
    }

    fn allocate(&self, bytes: usize) -> Result<Vec<u8>, PoolError> {
        let data = self.0.provider.allocate_initialized(bytes)?;
        if data.len() != bytes {
            return Err(PoolError::InvalidBacking);
        }
        StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(bytes));
        Ok(data)
    }

    /// Acquire an initialized exact-size backing. Reused bytes keep their
    /// previous values; new backing is initialized once.
    /// Timeout starts at method entry and covers all pool waiting.
    #[track_caller]
    pub fn try_acquire(&self, bytes: usize, wait: WaitPolicy) -> Result<CpuBufferLease, PoolError> {
        self.acquire(bytes, wait, false).map(CpuBufferLease)
    }

    /// Request a cleared buffer explicitly. This is distinct from the default
    /// initialized acquisition, which preserves recycled contents.
    #[track_caller]
    pub fn try_acquire_zeroed(
        &self,
        bytes: usize,
        wait: WaitPolicy,
    ) -> Result<CpuBufferLease, PoolError> {
        self.acquire(bytes, wait, true).map(CpuBufferLease)
    }

    #[track_caller]
    fn acquire(
        &self,
        bytes: usize,
        wait: WaitPolicy,
        zeroed: bool,
    ) -> Result<Arc<LeaseInner>, PoolError> {
        if bytes == 0 {
            return Err(PoolError::InvalidSize);
        }
        if bytes > self.0.limits.max_bytes {
            return Err(PoolError::Oversized);
        }
        let deadline = match wait {
            WaitPolicy::Immediate => None,
            WaitPolicy::Timeout(duration) => Some(
                Instant::now()
                    .checked_add(duration)
                    .ok_or(PoolError::Overflow)?,
            ),
        };
        let caller = Location::caller();
        let mut state = self.0.state.lock().map_err(|_| PoolError::Poisoned)?;
        let cancel_epoch = state.cancel_epoch;
        loop {
            if state.closed {
                return Err(PoolError::Closed);
            }
            if state.cancel_epoch != cancel_epoch {
                return Err(PoolError::Cancelled);
            }
            if state.next_id == u64::MAX {
                return Err(PoolError::Overflow);
            }
            let idle = state.idle.get_mut(&bytes).and_then(Vec::pop);
            if state.idle.get(&bytes).is_some_and(Vec::is_empty) {
                state.idle.remove(&bytes);
            }
            if idle.is_none() {
                // Idle backings of a different size must not strand capacity.
                while !fits(&state, self.0.limits, bytes) {
                    let Some(size) = state.idle.keys().next_back().copied() else {
                        break;
                    };
                    let bucket = state.idle.get_mut(&size).unwrap();
                    let _old = bucket.pop().unwrap();
                    if bucket.is_empty() {
                        state.idle.remove(&size);
                    }
                    state.allocated_buffers -= 1;
                    state.allocated_bytes -= size;
                }
            }
            let reused = idle.is_some();
            let backing = if let Some(idle) = idle {
                Some(idle)
            } else if fits(&state, self.0.limits, bytes) {
                reserve(&mut state, self.0.limits, 1, bytes)?;
                drop(state);
                let allocation = self.allocate(bytes);
                state = self
                    .0
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state.pending_buffers -= 1;
                state.pending_bytes -= bytes;
                self.0.changed.notify_all();
                if state.closed {
                    return Err(PoolError::Closed);
                }
                if state.cancel_epoch != cancel_epoch {
                    return Err(PoolError::Cancelled);
                }
                let backing = allocation?;
                if state.next_id == u64::MAX {
                    return Err(PoolError::Overflow);
                }
                state.allocated_buffers += 1;
                state.allocated_bytes += bytes;
                Some(backing)
            } else {
                None
            };
            if let Some(mut backing) = backing {
                if zeroed && reused {
                    backing.fill(0);
                    StatsContext::record_memory_operation(MemoryOperation::CpuZero(bytes));
                }
                let id = state.next_id;
                state.next_id += 1;
                let handles = Arc::new(AtomicUsize::new(1));
                let lease = Arc::new(LeaseInner {
                    issuer: self.0.clone(),
                    id,
                    bytes,
                    data: Some(backing),
                    handles: handles.clone(),
                });
                state.active.insert(
                    id,
                    Active {
                        bytes,
                        origin: caller.file(),
                        line: caller.line(),
                        handles,
                    },
                );
                return Ok(lease);
            }
            let Some(deadline) = deadline else {
                return Err(PoolError::Exhausted);
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(PoolError::Timeout(match wait {
                    WaitPolicy::Timeout(duration) => duration,
                    _ => unreachable!(),
                }));
            }
            state.waiters += 1;
            let result = self.0.changed.wait_timeout(state, remaining);
            state = match result {
                Ok((guard, _)) => guard,
                Err(_) => return Err(PoolError::Poisoned),
            };
            state.waiters -= 1;
        }
    }

    /// Cancel acquisitions already waiting or in a provider callback without
    /// closing the pool. New acquisitions may start immediately afterward.
    pub fn cancel_waiters(&self) -> Result<(), PoolError> {
        let mut state = self.0.state.lock().map_err(|_| PoolError::Poisoned)?;
        state.cancel_epoch = state.cancel_epoch.wrapping_add(1);
        self.0.changed.notify_all();
        Ok(())
    }

    /// Stop acquisition and wake all waiters. Existing leases remain usable
    /// and retain the issuer until their final handle drops.
    pub fn close(&self) -> Result<(), PoolError> {
        let mut state = self.0.state.lock().map_err(|_| PoolError::Poisoned)?;
        state.closed = true;
        let idle_count = state.idle.values().map(Vec::len).sum::<usize>();
        let idle_bytes = state
            .idle
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum::<usize>();
        state.idle.clear();
        state.allocated_buffers -= idle_count;
        state.allocated_bytes -= idle_bytes;
        self.0.changed.notify_all();
        Ok(())
    }

    pub fn snapshot(&self) -> Result<PoolSnapshot, PoolError> {
        let state = self.0.state.lock().map_err(|_| PoolError::Poisoned)?;
        let mut active = state
            .active
            .iter()
            .map(|(&id, entry)| LeaseOwner {
                id,
                bytes: entry.bytes,
                origin: entry.origin,
                line: entry.line,
                handles: entry.handles.load(Ordering::Relaxed),
            })
            .collect::<Vec<_>>();
        active.sort_by_key(|owner| owner.id);
        Ok(PoolSnapshot {
            limits: self.0.limits,
            allocated_buffers: state.allocated_buffers,
            allocated_bytes: state.allocated_bytes,
            pending_buffers: state.pending_buffers,
            pending_bytes: state.pending_bytes,
            idle_buffers: state.idle.values().map(Vec::len).sum(),
            active,
            waiters: state.waiters,
            closed: state.closed,
        })
    }
}

fn fits(state: &State, limits: PoolLimits, bytes: usize) -> bool {
    state.allocated_buffers + state.pending_buffers < limits.max_buffers
        && state
            .allocated_bytes
            .checked_add(state.pending_bytes)
            .and_then(|total| total.checked_add(bytes))
            .is_some_and(|total| total <= limits.max_bytes)
}

fn reserve(
    state: &mut State,
    limits: PoolLimits,
    count: usize,
    bytes: usize,
) -> Result<(), PoolError> {
    if state
        .allocated_buffers
        .checked_add(state.pending_buffers)
        .and_then(|total| total.checked_add(count))
        .ok_or(PoolError::Overflow)?
        > limits.max_buffers
        || state
            .allocated_bytes
            .checked_add(state.pending_bytes)
            .and_then(|total| total.checked_add(bytes))
            .ok_or(PoolError::Overflow)?
            > limits.max_bytes
    {
        return Err(PoolError::Exhausted);
    }
    state.pending_buffers += count;
    state.pending_bytes += bytes;
    Ok(())
}

fn allocate_zeroed(bytes: usize) -> Result<Vec<u8>, PoolError> {
    let mut data = Vec::new();
    data.try_reserve_exact(bytes)
        .map_err(|_| PoolError::AllocationFailed)?;
    data.resize(bytes, 0);
    Ok(data)
}

struct LeaseInner {
    issuer: Arc<Inner>,
    id: u64,
    bytes: usize,
    data: Option<Vec<u8>>,
    handles: Arc<AtomicUsize>,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let mut state = self
            .issuer
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.active.remove(&self.id);
        if let Some(data) = self.data.take() {
            if state.closed {
                state.allocated_buffers -= 1;
                state.allocated_bytes -= self.bytes;
            } else {
                state.idle.entry(self.bytes).or_default().push(data);
            }
        }
        self.issuer.changed.notify_all();
    }
}

/// Cheap clones share one initialized backing. A clone or read holder keeps it leased.
pub struct CpuBufferLease(Arc<LeaseInner>);

impl Clone for CpuBufferLease {
    fn clone(&self) -> Self {
        self.0.handles.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}

impl Drop for CpuBufferLease {
    fn drop(&mut self) {
        self.0.handles.fetch_sub(1, Ordering::Relaxed);
    }
}

impl CpuBufferLease {
    pub fn id(&self) -> u64 {
        self.0.id
    }
    pub fn handles(&self) -> usize {
        self.0.handles.load(Ordering::Relaxed)
    }

    /// A shared lease detaches into another backing from its issuing pool.
    /// If the pool is exhausted, the old value remains unchanged.
    #[track_caller]
    pub fn try_write(&mut self, wait: WaitPolicy) -> Result<&mut [u8], PoolError> {
        if Arc::get_mut(&mut self.0).is_none() {
            let issuer = BufferPoolManager(self.0.issuer.clone());
            let mut replacement = issuer.try_acquire(self.len(), wait)?;
            replacement
                .try_write(WaitPolicy::Immediate)?
                .copy_from_slice(self);
            StatsContext::record_memory_operation(MemoryOperation::CpuCopy(self.len()));
            *self = replacement;
        }
        Ok(Arc::get_mut(&mut self.0)
            .unwrap()
            .data
            .as_mut()
            .unwrap()
            .as_mut_slice())
    }
}

impl Deref for CpuBufferLease {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.0.data.as_ref().unwrap().as_slice()
    }
}

impl CpuStorageShared for CpuBufferLease {
    fn allocation_origin(&self) -> super::CpuAllocationOrigin {
        super::CpuAllocationOrigin::Pool
    }

    fn clone_shared(&self) -> Box<dyn CpuStorageShared> {
        Box::new(self.clone())
    }

    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>> {
        // Returning this Vec would detach its allocation from the issuing pool.
        Err(self)
    }

    fn try_write_pooled(&mut self, wait: WaitPolicy) -> Result<Option<&mut [u8]>, PoolError> {
        self.try_write(wait).map(Some)
    }
}
