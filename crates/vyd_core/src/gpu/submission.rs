//! Admission and completion ownership for managed compute recordings.
//!
//! Admission is synchronous: validate every reserved output, submit the
//! command buffer, publish pending frame versions, and register completion
//! ownership for all input/output leases. No cancellation point separates
//! these steps. Dropping a receipt or frame cannot return a lease early.

use super::recording::{GpuRecording, PreparedRecording};
use super::transfer::GpuPlaneSet;
use super::{DEVICE_ERROR_SCOPE_LOCK, GpuContextError};
use crate::frame::GpuFrameRead;
use crate::frame::pool::WaitPolicy;
use crate::statistics::{FrameOperationAttempt, FrameOperationBytes};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GpuSubmissionError {
    #[error("GPU recording has no output reservation")]
    NoOutputs,
    #[error("GPU output {frame} plane {plane} was not bound in a dispatched pass")]
    UnwrittenOutput { frame: usize, plane: usize },
    #[error("GPU submission would block")]
    WouldBlock,
    #[error("GPU submission timed out after {0:?}")]
    Timeout(Duration),
    #[error("GPU submission failed: {0}")]
    Failed(String),
    #[error("GPU context has stopped accepting submissions")]
    Closed,
}

#[derive(Clone)]
enum Completion {
    Pending,
    Complete,
    Failed(String),
}

#[derive(Clone)]
struct Retained {
    reads: Vec<GpuFrameRead>,
    outputs: Vec<GpuPlaneSet>,
}

impl Retained {
    fn quarantine(&self) {
        for read in &self.reads {
            read.quarantine();
        }
        for output in &self.outputs {
            output.quarantine();
        }
    }
}

struct StateData {
    completion: Completion,
    queue_done: bool,
    validation_done: bool,
    lost: bool,
    retained: Option<Retained>,
    operation: Option<(FrameOperationAttempt, FrameOperationBytes)>,
}

/// Shared by a receipt and every published output clone. The progress
/// registry also owns it until queue completion and validation both resolve.
pub(crate) struct SubmissionState {
    device: wgpu::Device,
    data: Mutex<StateData>,
    changed: Condvar,
}

// Erase the registry edge so auto-trait evaluation does not recurse through
// Progress -> State -> pooled lease -> WgpuContext -> Progress.
trait PendingSubmission: Send + Sync {
    fn fail(&self, message: String, device_lost: bool);
}

impl PendingSubmission for SubmissionState {
    fn fail(&self, message: String, device_lost: bool) {
        SubmissionState::fail(self, message, device_lost);
    }
}

impl SubmissionState {
    fn new(
        device: wgpu::Device,
        retained: Retained,
        operation: Option<(FrameOperationAttempt, FrameOperationBytes)>,
    ) -> Self {
        Self {
            device,
            data: Mutex::new(StateData {
                completion: Completion::Pending,
                queue_done: false,
                validation_done: false,
                lost: false,
                retained: Some(retained),
                operation,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn try_wait(&self, wait: WaitPolicy) -> Result<(), GpuSubmissionError> {
        let start = Instant::now();
        loop {
            let data = self.data.lock().unwrap_or_else(|p| p.into_inner());
            match &data.completion {
                Completion::Complete => return Ok(()),
                Completion::Failed(message) => {
                    return Err(GpuSubmissionError::Failed(message.clone()));
                }
                Completion::Pending => {}
            }
            drop(data);
            // This assists the autonomous driver but never holds the state
            // mutex while polling: wgpu may call us back on this thread.
            self.device.poll(wgpu::Maintain::Poll);
            let data = self.data.lock().unwrap_or_else(|p| p.into_inner());
            match &data.completion {
                Completion::Complete => return Ok(()),
                Completion::Failed(message) => {
                    return Err(GpuSubmissionError::Failed(message.clone()));
                }
                Completion::Pending => {}
            }
            let remaining = match wait {
                WaitPolicy::Immediate => return Err(GpuSubmissionError::WouldBlock),
                WaitPolicy::Timeout(limit) => limit.saturating_sub(start.elapsed()),
            };
            if remaining.is_zero() {
                return Err(GpuSubmissionError::Timeout(match wait {
                    WaitPolicy::Timeout(limit) => limit,
                    WaitPolicy::Immediate => unreachable!(),
                }));
            }
            let _ = self
                .changed
                .wait_timeout(data, remaining.min(Duration::from_millis(1)));
        }
    }

    fn queue_done(&self) -> bool {
        let retired = {
            let mut data = self.data.lock().unwrap_or_else(|p| p.into_inner());
            data.queue_done = true;
            self.finish_if_ready(&mut data)
        };
        drop(retired.1);
        retired.0
    }

    fn validation_done(&self, error: Option<String>) -> bool {
        let retired = {
            let mut data = self.data.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(error) = error {
                data.completion = Completion::Failed(error);
            }
            data.validation_done = true;
            self.finish_if_ready(&mut data)
        };
        drop(retired.1);
        retired.0
    }

    fn finish_if_ready(&self, data: &mut StateData) -> (bool, Option<Retained>) {
        if !data.queue_done || !data.validation_done || data.lost {
            self.changed.notify_all();
            return (false, None);
        }
        if matches!(data.completion, Completion::Pending) {
            data.completion = Completion::Complete;
        }
        if matches!(data.completion, Completion::Complete) {
            if let Some((attempt, bytes)) = data.operation.take() {
                attempt.complete(bytes);
            }
        } else {
            // An accepted submission that failed contributes an attempted and
            // failed operation, but never contributes completed bytes.
            data.operation.take();
        }
        self.changed.notify_all();
        (true, data.retained.take())
    }

    fn fail(&self, message: String, device_lost: bool) {
        let mut data = self.data.lock().unwrap_or_else(|p| p.into_inner());
        if !matches!(data.completion, Completion::Complete) {
            data.completion = Completion::Failed(message);
            data.operation.take();
        }
        let quarantine = if device_lost {
            data.lost = true;
            data.retained.clone()
        } else {
            None
        };
        self.changed.notify_all();
        drop(data);
        if let Some(retained) = quarantine {
            retained.quarantine();
        }
    }
}

/// Context-owned polling and lifetime registry. The driver holds no pool or
/// completion lock while calling wgpu, because callbacks can run in poll.
pub(crate) struct Progress {
    device: wgpu::Device,
    pending: Mutex<HashMap<u64, Arc<dyn PendingSubmission>>>,
    changed: Condvar,
    next_id: AtomicU64,
    transfers: AtomicUsize,
    closed: AtomicBool,
    stopped: AtomicBool,
    #[cfg(test)]
    pause_driver: AtomicBool,
    loss: Mutex<Option<String>>,
    uncaptured: Mutex<Option<String>>,
    pools: Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
    driver: Mutex<Option<JoinHandle<()>>>,
}

impl Progress {
    pub(crate) fn new(device: wgpu::Device) -> Arc<Self> {
        let progress = Arc::new(Self {
            device,
            pending: Mutex::new(HashMap::new()),
            changed: Condvar::new(),
            next_id: AtomicU64::new(1),
            transfers: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            #[cfg(test)]
            pause_driver: AtomicBool::new(false),
            loss: Mutex::new(None),
            uncaptured: Mutex::new(None),
            pools: Mutex::new(Vec::new()),
            driver: Mutex::new(None),
        });
        let weak = Arc::downgrade(&progress);
        let driver = thread::Builder::new()
            .name("vyd-gpu-progress".into())
            .spawn(move || {
                while let Some(progress) = weak.upgrade() {
                    if progress.stopped.load(Ordering::Acquire) {
                        break;
                    }
                    let pending = progress.pending.lock().unwrap_or_else(|p| p.into_inner());
                    if pending.is_empty() && progress.transfers.load(Ordering::Acquire) == 0 {
                        let _ = progress
                            .changed
                            .wait_timeout(pending, Duration::from_millis(5));
                        continue;
                    }
                    drop(pending);
                    #[cfg(test)]
                    if progress.pause_driver.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    progress.device.poll(wgpu::Maintain::Poll);
                    thread::sleep(Duration::from_millis(1));
                }
            })
            .expect("GPU progress thread");
        *progress.driver.lock().unwrap_or_else(|p| p.into_inner()) = Some(driver);
        progress
    }

    fn register(&self, state: Arc<SubmissionState>) -> Result<u64, GpuSubmissionError> {
        let loss = self.loss.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(message) = &*loss {
            return Err(GpuSubmissionError::Failed(message.clone()));
        }
        if let Some(message) = self
            .uncaptured
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            return Err(GpuSubmissionError::Failed(message.clone()));
        }
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if self.closed.load(Ordering::Acquire) {
            return Err(GpuSubmissionError::Closed);
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        pending.insert(id, state);
        drop(pending);
        drop(loss);
        self.changed.notify_all();
        Ok(id)
    }

    fn retire(&self, id: u64) {
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id);
        self.changed.notify_all();
    }

    pub(crate) fn begin_transfer(&self) -> Result<(), GpuContextError> {
        let loss = self.loss.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(message) = &*loss {
            return Err(GpuContextError::DeviceLost {
                message: message.clone(),
            });
        }
        if let Some(message) = self
            .uncaptured
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            return Err(GpuContextError::UncapturedError {
                message: message.clone(),
            });
        }
        let _pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if self.closed.load(Ordering::Acquire) {
            return Err(GpuContextError::Closed);
        }
        self.transfers.fetch_add(1, Ordering::AcqRel);
        drop(loss);
        self.changed.notify_all();
        Ok(())
    }

    pub(crate) fn end_transfer(&self) {
        self.transfers.fetch_sub(1, Ordering::AcqRel);
        self.changed.notify_all();
    }

    pub(crate) fn fail_pending(&self, message: String) {
        self.closed.store(true, Ordering::Release);
        *self.uncaptured.lock().unwrap_or_else(|p| p.into_inner()) = Some(message.clone());
        self.close_pools();
        let states: Vec<_> = self
            .pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .cloned()
            .collect();
        for state in states {
            state.fail(message.clone(), false);
        }
        self.changed.notify_all();
    }

    pub(crate) fn register_pool(&self, close: Arc<dyn Fn() + Send + Sync>) {
        self.pools
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(close.clone());
        if self.device_failure().is_some() {
            close();
        }
    }

    pub(crate) fn device_failure(&self) -> Option<GpuContextError> {
        if let Some(message) = self.loss.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            return Some(GpuContextError::DeviceLost { message });
        }
        self.uncaptured
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .map(|message| GpuContextError::UncapturedError { message })
    }

    fn close_pools(&self) {
        let pools = self.pools.lock().unwrap_or_else(|p| p.into_inner()).clone();
        for close in pools {
            close();
        }
    }

    pub(crate) fn device_lost(&self, message: String) {
        self.closed.store(true, Ordering::Release);
        *self.loss.lock().unwrap_or_else(|p| p.into_inner()) = Some(message.clone());
        self.close_pools();
        let states: Vec<_> = self
            .pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .cloned()
            .collect();
        for state in states {
            state.fail(format!("device lost: {message}"), true);
        }
        self.changed.notify_all();
    }

    pub(crate) fn shutdown(&self, wait: WaitPolicy) -> Result<(), GpuSubmissionError> {
        {
            let _pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            self.closed.store(true, Ordering::Release);
        }
        let started = Instant::now();
        loop {
            if let Some(message) = self.loss.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                return Err(GpuSubmissionError::Failed(format!(
                    "device lost: {message}"
                )));
            }
            if let Some(message) = self
                .uncaptured
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
            {
                return Err(GpuSubmissionError::Failed(message.clone()));
            }
            self.device.poll(wgpu::Maintain::Poll);
            let pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            if pending.is_empty() && self.transfers.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            let remaining = match wait {
                WaitPolicy::Immediate => return Err(GpuSubmissionError::WouldBlock),
                WaitPolicy::Timeout(limit) => limit.saturating_sub(started.elapsed()),
            };
            if remaining.is_zero() {
                return Err(GpuSubmissionError::Timeout(match wait {
                    WaitPolicy::Timeout(limit) => limit,
                    WaitPolicy::Immediate => unreachable!(),
                }));
            }
            let _ = self
                .changed
                .wait_timeout(pending, remaining.min(Duration::from_millis(1)));
        }
    }

    pub(crate) fn stop_driver(&self, limit: Duration) {
        self.stopped.store(true, Ordering::Release);
        self.changed.notify_all();
        if let Some(driver) = self.driver.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let started = Instant::now();
            while !driver.is_finished() && started.elapsed() < limit {
                thread::sleep(Duration::from_millis(1));
            }
            if driver.is_finished() {
                let _ = driver.join();
            }
            // An unresponsive driver is detached; accepted callback-held
            // resources and the closed pool still prevent unsafe reuse.
        }
    }
}

/// A receipt for accepted work. Its drop is deliberately passive: the queue
/// completion callback, not this handle, owns the submitted leases.
#[derive(Clone)]
pub struct GpuSubmission {
    state: Arc<SubmissionState>,
}

impl GpuSubmission {
    pub fn try_wait(&self, wait: WaitPolicy) -> Result<(), GpuSubmissionError> {
        self.state.try_wait(wait)
    }
}

/// Consume a prepared recording. Every output must have appeared in a
/// dispatched bind group before admission; callers still own the shader's
/// obligation to write every visible texel of those output planes.
pub fn submit(recording: GpuRecording<'_>) -> Result<GpuSubmission, GpuSubmissionError> {
    submit_prepared(recording.finish(), None)
}

pub(crate) fn submit_with_operation(
    recording: GpuRecording<'_>,
    attempt: FrameOperationAttempt,
    bytes: FrameOperationBytes,
) -> Result<GpuSubmission, GpuSubmissionError> {
    submit_prepared(recording.finish(), Some((attempt, bytes)))
}

fn submit_prepared(
    prepared: PreparedRecording<'_>,
    operation: Option<(FrameOperationAttempt, FrameOperationBytes)>,
) -> Result<GpuSubmission, GpuSubmissionError> {
    if prepared.writes.is_empty() {
        return Err(GpuSubmissionError::NoOutputs);
    }
    for (frame, planes) in prepared.written.iter().enumerate() {
        for (plane, written) in planes.iter().enumerate() {
            if !written {
                return Err(GpuSubmissionError::UnwrittenOutput { frame, plane });
            }
        }
    }

    let PreparedRecording {
        context,
        command_buffer,
        reads,
        writes,
        ..
    } = prepared;
    // Clone every lease before queue acceptance so the callback can own them
    // independently of both the receipt and the published Frame handles.
    let retained_outputs: Vec<_> = writes
        .iter()
        .map(|write| write.retained_candidate())
        .collect();
    let state = Arc::new(SubmissionState::new(
        context.device().clone(),
        Retained {
            reads,
            outputs: retained_outputs,
        },
        operation,
    ));
    let _scope_guard = DEVICE_ERROR_SCOPE_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    // Registration precedes queue acceptance. No cancellation or fallible
    // operation separates acceptance from publication of pending versions.
    let id = context.progress.register(state.clone())?;
    for filter in [
        wgpu::ErrorFilter::Internal,
        wgpu::ErrorFilter::OutOfMemory,
        wgpu::ErrorFilter::Validation,
    ] {
        context.device().push_error_scope(filter);
    }
    context.queue().submit([command_buffer]);
    for write in writes {
        write.publish(state.clone());
    }
    let completed = state.clone();
    let progress = Arc::downgrade(&context.progress);
    context.queue().on_submitted_work_done(move || {
        if completed.queue_done()
            && let Some(progress) = progress.upgrade()
        {
            progress.retire(id);
        }
    });
    // wgpu-core's pop futures are ready after removing the device-local
    // scopes. Resolve them under the same admission lock so a different
    // recording cannot interleave its error scopes. Queue/device callbacks
    // still report failures after this immediate validation check.
    let validation = context.device().pop_error_scope();
    let out_of_memory = context.device().pop_error_scope();
    let internal = context.device().pop_error_scope();
    let error = pollster::block_on(async {
        let validation = validation.await;
        let out_of_memory = out_of_memory.await;
        let internal = internal.await;
        validation.or(out_of_memory).or(internal)
    })
    .map(|error| error.to_string());
    drop(_scope_guard);
    if state.validation_done(error) {
        context.progress.retire(id);
    }
    Ok(GpuSubmission { state })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use crate::frame::PixelFormat;
    use crate::frame::factory::FrameFactory;
    use crate::frame::layout::FrameSpec;
    use crate::gpu::WgpuContext;
    use crate::gpu::pool::{GpuPoolError, GpuPoolLimits, GpuPoolManager};
    use crate::gpu::recording::{GpuBindingSpec, GpuComputeProgram, GpuPlaneBinding};
    use crate::gpu::transfer::{GpuTransfer, GpuTransferError};
    use crate::gpu::{GpuContextError, WgpuContextOptions};
    use crate::statistics::{FrameOperationBytes, FrameOperationKind};
    use crate::statistics::{PipelineStatistics, StatsContext};

    const COPY_SHADER: &str = r#"
        @group(0) @binding(0) var source: texture_2d<u32>;
        @group(0) @binding(1) var output: texture_storage_2d<rgba8uint, write>;
        @compute @workgroup_size(1, 1, 1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let xy = vec2<i32>(id.xy);
            textureStore(output, xy, textureLoad(source, xy, 0));
        }
    "#;

    fn fixture(resources: usize) -> Option<(Arc<WgpuContext>, GpuPoolManager, GpuTransfer)> {
        let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Arc::new(context),
            Err(GpuContextError::Unavailable) => {
                eprintln!("submission GPU tests unverified: no compatible adapter");
                return None;
            }
            Err(error) => panic!("submission hardware gate failed: {error}"),
        };
        eprintln!(
            "submission adapter={} backend={:?}",
            context.adapter_info().name,
            context.adapter_info().backend
        );
        let pool = GpuPoolManager::new(
            context.clone(),
            GpuPoolLimits::new(resources, 1024 * 1024).unwrap(),
        );
        let stats = StatsContext::new(Arc::new(PipelineStatistics::new())).for_stage("submission");
        let transfer = GpuTransfer::new(context.clone(), pool.clone(), stats).unwrap();
        Some((context, pool, transfer))
    }

    fn spec() -> FrameSpec {
        FrameSpec::new(2, 2, PixelFormat::Luma8).unwrap()
    }

    fn wait() -> WaitPolicy {
        WaitPolicy::Timeout(Duration::from_secs(3))
    }

    #[test]
    fn conversion_ticket_tracks_completion_not_caller_wait() {
        let Some((context, _pool, _transfer)) = fixture(2) else {
            return;
        };
        let stats = Arc::new(PipelineStatistics::new());
        let context_stats = StatsContext::new(stats.clone()).for_stage("ticket");
        let kind = FrameOperationKind::Conversion {
            from: PixelFormat::Luma8,
            to: PixelFormat::Nv12,
            location: crate::statistics::ConversionLocation::Gpu,
        };
        let state = SubmissionState::new(
            context.device().clone(),
            Retained {
                reads: vec![],
                outputs: vec![],
            },
            Some((
                context_stats.begin_frame_operation(kind, None),
                FrameOperationBytes {
                    visible: 12,
                    ..Default::default()
                },
            )),
        );
        assert_eq!(
            state.try_wait(WaitPolicy::Immediate),
            Err(GpuSubmissionError::WouldBlock)
        );
        let before = stats.snapshot().frame_operations["ticket"]["convert_gpu_luma8_to_nv12"];
        assert_eq!(
            (before.attempted, before.failed, before.completed),
            (1, 0, 0)
        );
        state.fail("injected conversion failure".into(), false);
        assert!(matches!(
            state.try_wait(wait()),
            Err(GpuSubmissionError::Failed(_))
        ));
        let failed = stats.snapshot().frame_operations["ticket"]["convert_gpu_luma8_to_nv12"];
        assert_eq!(
            (
                failed.attempted,
                failed.failed,
                failed.completed,
                failed.visible_bytes
            ),
            (1, 1, 0, 0)
        );

        let completed = SubmissionState::new(
            context.device().clone(),
            Retained {
                reads: vec![],
                outputs: vec![],
            },
            Some((
                context_stats.begin_frame_operation(kind, None),
                FrameOperationBytes {
                    visible: 12,
                    ..Default::default()
                },
            )),
        );
        completed.validation_done(None);
        completed.queue_done();
        assert_eq!(completed.try_wait(wait()), Ok(()));
        let final_count = stats.snapshot().frame_operations["ticket"]["convert_gpu_luma8_to_nv12"];
        assert_eq!(
            (
                final_count.attempted,
                final_count.failed,
                final_count.completed,
                final_count.visible_bytes
            ),
            (2, 1, 1, 12)
        );
    }

    fn copy_recording<'a>(
        context: Arc<WgpuContext>,
        source: &mut Frame,
        destination: &'a mut Frame,
        transfer: &GpuTransfer,
    ) -> GpuRecording<'a> {
        let factory = FrameFactory::default();
        let read = source
            .try_read_gpu_as(&factory, transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let write = destination
            .try_write_gpu_as(transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let program = GpuComputeProgram::new(
            &context,
            COPY_SHADER,
            "main",
            &[
                GpuBindingSpec::read(0, PixelFormat::Luma8, 0),
                GpuBindingSpec::write(1, PixelFormat::Luma8, 0),
            ],
        )
        .unwrap();
        let mut recording = GpuRecording::new(context);
        let input = recording.add_read(read, spec()).unwrap()[0];
        let output = recording.add_write(write, spec()).unwrap()[0];
        let group = recording
            .bind_group(
                &program,
                &[
                    (0, GpuPlaneBinding::Read(input)),
                    (1, GpuPlaneBinding::Write(output)),
                ],
            )
            .unwrap();
        recording.dispatch(&program, group, [2, 2, 1]).unwrap();
        recording
    }

    #[test]
    fn submitted_copy_is_readable_after_bounded_wait() {
        let Some((context, _pool, transfer)) = fixture(3) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        let receipt = submit(copy_recording(context, &mut source, &mut output, &transfer)).unwrap();
        receipt.try_wait(wait()).unwrap();
        let factory = FrameFactory::default();
        let read = output
            .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert_eq!(read.luma8_bytes().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn canceled_or_rejected_recording_preserves_original_frame() {
        let Some((context, pool, transfer)) = fixture(2) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        let baseline = pool.snapshot().unwrap().active.len();
        drop(copy_recording(
            context.clone(),
            &mut source,
            &mut output,
            &transfer,
        ));
        assert_eq!(
            output.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[9; 4]
        );
        assert_eq!(pool.snapshot().unwrap().active.len(), baseline + 1);

        let write = output
            .try_write_gpu_as(&transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let mut recording = GpuRecording::new(context);
        recording.add_write(write, spec()).unwrap();
        assert!(matches!(
            submit(recording),
            Err(GpuSubmissionError::UnwrittenOutput { frame: 0, plane: 0 })
        ));
        assert_eq!(
            output.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[9; 4]
        );
        assert_eq!(pool.snapshot().unwrap().active.len(), baseline + 1);
    }

    #[test]
    fn dropped_receipt_and_retained_clone_keep_pool_leases_until_safe() {
        let Some((context, pool, transfer)) = fixture(2) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        let receipt = submit(copy_recording(context, &mut source, &mut output, &transfer)).unwrap();
        let clone = output.clone_shared();
        assert!(matches!(
            transfer.reserve_write(spec(), WaitPolicy::Immediate),
            Err(GpuTransferError::Pool(GpuPoolError::Exhausted))
        ));
        receipt.try_wait(wait()).unwrap();
        drop(receipt);
        drop(output);
        assert!(matches!(
            transfer.reserve_write(spec(), WaitPolicy::Immediate),
            Err(GpuTransferError::Pool(GpuPoolError::Exhausted))
        ));
        drop(clone);
        let reused = transfer
            .reserve_write(spec(), WaitPolicy::Immediate)
            .unwrap();
        assert_eq!(reused.plane_count(), 1);
        assert_eq!(pool.snapshot().unwrap().allocated_resources, 2);
    }

    #[test]
    fn pending_submission_keeps_leases_after_all_handles_drop() {
        let Some((context, _pool, transfer)) = fixture(2) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        let receipt = submit(copy_recording(context, &mut source, &mut output, &transfer)).unwrap();
        let pending =
            receipt.try_wait(WaitPolicy::Immediate) == Err(GpuSubmissionError::WouldBlock);
        drop(output);
        drop(source);
        drop(receipt);
        if pending {
            assert!(matches!(
                transfer.reserve_write(spec(), WaitPolicy::Immediate),
                Err(GpuTransferError::Pool(GpuPoolError::Exhausted))
            ));
        }
    }

    #[test]
    fn post_acceptance_error_poisons_receipt_and_published_frame() {
        let Some((context, _pool, transfer)) = fixture(3) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        context.progress.pause_driver.store(true, Ordering::Release);
        let receipt = submit(copy_recording(
            context.clone(),
            &mut source,
            &mut output,
            &transfer,
        ))
        .unwrap();
        // Drive the same state path as wgpu's uncaptured-error callback. The
        // hardware need not produce a validation fault on this machine.
        context
            .progress
            .fail_pending("injected validation failure".into());
        assert_eq!(
            receipt.try_wait(wait()),
            Err(GpuSubmissionError::Failed(
                "injected validation failure".into()
            ))
        );
        let factory = FrameFactory::default();
        assert!(matches!(
            output.try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait()),
            Err(crate::frame::FrameError::GpuSubmission(GpuSubmissionError::Failed(message)))
                if message == "injected validation failure"
        ));
        assert!(matches!(
            output.try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait()),
            Err(crate::frame::FrameError::GpuSubmission(GpuSubmissionError::Failed(message)))
                if message == "injected validation failure"
        ));
        assert!(matches!(
            transfer.reserve_write(spec(), WaitPolicy::Immediate),
            Err(GpuTransferError::Pool(GpuPoolError::Context(
                crate::gpu::GpuContextError::UncapturedError { .. }
            )))
        ));
        assert!(
            matches!(context.shutdown(wait()), Err(GpuSubmissionError::Failed(message)) if message == "injected validation failure")
        );
    }

    #[test]
    fn device_loss_quarantines_leases_and_bounds_shutdown() {
        let Some((context, pool, transfer)) = fixture(2) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        context.progress.pause_driver.store(true, Ordering::Release);
        let receipt = submit(copy_recording(
            context.clone(),
            &mut source,
            &mut output,
            &transfer,
        ))
        .unwrap();
        context.progress.device_lost("injected loss".into());
        assert!(
            matches!(receipt.try_wait(wait()), Err(GpuSubmissionError::Failed(message)) if message.contains("device lost"))
        );
        let factory = FrameFactory::default();
        assert!(matches!(
            output.try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait()),
            Err(crate::frame::FrameError::GpuSubmission(GpuSubmissionError::Failed(message)))
                if message.contains("device lost")
        ));
        assert!(matches!(
            output.try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait()),
            Err(crate::frame::FrameError::GpuSubmission(GpuSubmissionError::Failed(message)))
                if message.contains("device lost")
        ));
        drop(receipt);
        drop(source);
        drop(output);
        assert!(pool.snapshot().unwrap().closed);
        assert_eq!(pool.snapshot().unwrap().active.len(), 2);
        assert!(matches!(
            transfer.reserve_write(spec(), WaitPolicy::Immediate),
            Err(GpuTransferError::Pool(GpuPoolError::Context(
                crate::gpu::GpuContextError::DeviceLost { .. }
            )))
        ));
        assert!(
            matches!(context.shutdown(WaitPolicy::Timeout(Duration::from_millis(20))), Err(GpuSubmissionError::Failed(message)) if message.contains("device lost"))
        );
    }

    #[test]
    fn dropped_receipt_retires_without_an_explicit_wait() {
        let Some((context, pool, transfer)) = fixture(2) else {
            return;
        };
        let mut source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut output = Frame::from_raw(2, 2, vec![9; 4]).unwrap();
        let receipt = submit(copy_recording(
            context.clone(),
            &mut source,
            &mut output,
            &transfer,
        ))
        .unwrap();
        drop(receipt);
        drop(source);
        drop(output);
        let started = Instant::now();
        while !pool.snapshot().unwrap().active.is_empty()
            && started.elapsed() < Duration::from_secs(3)
        {
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            pool.snapshot().unwrap().active.is_empty(),
            "driver did not retire dropped handles"
        );
        context.shutdown(wait()).unwrap();
    }

    #[test]
    fn shutdown_times_out_on_unfinished_accepted_state() {
        let Some((context, _pool, _transfer)) = fixture(2) else {
            return;
        };
        let state = Arc::new(SubmissionState::new(
            context.device().clone(),
            Retained {
                reads: Vec::new(),
                outputs: Vec::new(),
            },
            None,
        ));
        let id = context.progress.register(state.clone()).unwrap();
        assert_eq!(
            context.shutdown(WaitPolicy::Timeout(Duration::from_millis(20))),
            Err(GpuSubmissionError::Timeout(Duration::from_millis(20)))
        );
        assert!(matches!(
            context.progress.register(state.clone()),
            Err(GpuSubmissionError::Closed)
        ));
        assert!(!state.validation_done(None));
        assert!(state.queue_done());
        context.progress.retire(id);
        context.shutdown(wait()).unwrap();
    }
}
