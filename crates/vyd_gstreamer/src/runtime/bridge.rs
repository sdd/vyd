//! Bounded context sidecar for a caller-declared one-to-one ordered native
//! section. Pixel data always comes from the section's actual output buffer.
use super::{
    error,
    native::{AnnotationUpdate, NativeGeometry},
    transport, Shared,
};
use gst::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use vyd_core::VydError;

struct BridgeState {
    pending: VecDeque<(u64, transport::ContextData)>,
    capacity: usize,
    high_watermark: usize,
    epoch: u64,
    // A flush crosses two pads. Neither side may admit buffers until both
    // pads have seen its stop event, including when events arrive out of order.
    input_ready: bool,
    output_ready: bool,
    flush_active: bool,
    closed: bool,
}

impl BridgeState {
    fn new(capacity: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            capacity,
            high_watermark: 0,
            epoch: 0,
            input_ready: true,
            output_ready: true,
            flush_active: false,
            closed: false,
        }
    }

    fn ready(&self) -> bool {
        !self.closed && !self.flush_active && self.input_ready && self.output_ready
    }
}

pub(super) struct BridgeHandle(Arc<Mutex<BridgeState>>);

fn report(shared: &Shared, message: impl std::fmt::Display) {
    let stats = {
        let mut runtime = shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if runtime.failure.is_none() {
            runtime.failure = Some(error(format!("native context bridge: {message}")));
        }
        runtime.stats.clone()
    };
    stats.cancellation_token().cancel();
}

fn flush(state: &Arc<Mutex<BridgeState>>, input: bool, starting: bool) {
    let mut state = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if starting {
        if !state.flush_active {
            state.epoch = state.epoch.wrapping_add(1);
            state.input_ready = false;
            state.output_ready = false;
        }
        state.flush_active = true;
        if input {
            state.input_ready = false;
        } else {
            state.output_ready = false;
        }
        state.pending.clear();
    } else if input {
        state.input_ready = true;
    } else {
        state.output_ready = true;
    }
    if state.input_ready && state.output_ready {
        state.flush_active = false;
    }
}

fn close(state: &Arc<Mutex<BridgeState>>) {
    let mut state = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.closed = true;
    state.pending.clear();
}

pub(super) fn abort(handles: &[BridgeHandle]) {
    for handle in handles {
        close(&handle.0);
    }
}

pub(super) fn install(
    element: &gst::Element,
    capacity: usize,
    shared: Shared,
    geometry: Option<NativeGeometry>,
    annotation_update: Option<AnnotationUpdate>,
) -> Result<BridgeHandle, VydError> {
    let input = element
        .sink_pads()
        .into_iter()
        .next()
        .ok_or_else(|| error("native bridge has no sink pad"))?;
    let output = element
        .src_pads()
        .into_iter()
        .next()
        .ok_or_else(|| error("native bridge has no source pad"))?;
    let state = Arc::new(Mutex::new(BridgeState::new(capacity)));
    let stats = shared
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .stats
        .clone();

    let entry_state = state.clone();
    let entry_runtime = shared.clone();
    let entry_stats = stats.clone();
    input
        .add_probe(gst::PadProbeType::BUFFER, move |pad, info| {
            let Some(buffer) = info.buffer() else {
                return gst::PadProbeReturn::Ok;
            };
            let result = {
                let mut sidecar = entry_state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if sidecar.closed || entry_stats.cancellation_token().is_cancelled() {
                    sidecar.closed = true;
                    sidecar.pending.clear();
                    return gst::PadProbeReturn::Drop;
                }
                if !sidecar.ready() {
                    return gst::PadProbeReturn::Drop;
                }
                if let Some(geometry) = geometry {
                    if let Err(failure) =
                        transport::validate_native_buffer(pad, buffer, geometry.input)
                    {
                        drop(sidecar);
                        close(&entry_state);
                        report(&entry_runtime, failure);
                        return gst::PadProbeReturn::Drop;
                    }
                }
                if sidecar.pending.len() == sidecar.capacity {
                    Err(error(format!(
                        "capacity {} exhausted in epoch {}",
                        sidecar.capacity, sidecar.epoch
                    )))
                } else {
                    match transport::take_context(buffer) {
                        Ok(context) => {
                            let epoch = sidecar.epoch;
                            sidecar.pending.push_back((epoch, context));
                            sidecar.high_watermark =
                                sidecar.high_watermark.max(sidecar.pending.len());
                            Ok(())
                        }
                        Err(failure) => Err(failure),
                    }
                }
            };
            if let Err(failure) = result {
                close(&entry_state);
                report(&entry_runtime, failure);
                return gst::PadProbeReturn::Drop;
            }
            gst::PadProbeReturn::Ok
        })
        .ok_or_else(|| error("failed to install native bridge input probe"))?;

    let exit_state = state.clone();
    let exit_runtime = shared.clone();
    let exit_stats = stats;
    output
        .add_probe(gst::PadProbeType::BUFFER, move |pad, info| {
            if let Some(geometry) = geometry {
                if let Some(buffer) = info.buffer() {
                    if let Err(failure) =
                        transport::validate_native_buffer(pad, buffer, geometry.output)
                    {
                        close(&exit_state);
                        report(&exit_runtime, failure);
                        return gst::PadProbeReturn::Drop;
                    }
                }
            }
            let context = {
                let mut sidecar = exit_state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if sidecar.closed || exit_stats.cancellation_token().is_cancelled() {
                    sidecar.closed = true;
                    sidecar.pending.clear();
                    return gst::PadProbeReturn::Drop;
                }
                if !sidecar.ready() {
                    return gst::PadProbeReturn::Drop;
                }
                sidecar.pending.pop_front()
            };
            let Some((epoch, mut context)) = context else {
                close(&exit_state);
                report(&exit_runtime, "output has no corresponding input context");
                return gst::PadProbeReturn::Drop;
            };
            let Some(buffer) = info.buffer_mut() else {
                close(&exit_state);
                report(&exit_runtime, "output probe has no buffer");
                return gst::PadProbeReturn::Drop;
            };
            if let Some(update) = &annotation_update {
                let declared = geometry.expect("annotation update requires declared geometry");
                let actual = match super::caps::negotiated_gray8(pad) {
                    Ok(info) => NativeGeometry {
                        input: declared.input,
                        output: (info.width(), info.height()),
                    },
                    Err(failure) => {
                        close(&exit_state);
                        report(&exit_runtime, failure);
                        return gst::PadProbeReturn::Drop;
                    }
                };
                if let Err(failure) = update(&mut context.annotations(), actual) {
                    close(&exit_state);
                    report(&exit_runtime, failure);
                    return gst::PadProbeReturn::Drop;
                }
            }
            if let Err(failure) = transport::attach_context(buffer.make_mut(), context) {
                close(&exit_state);
                report(&exit_runtime, failure);
                return gst::PadProbeReturn::Drop;
            }
            let sidecar = exit_state.lock().unwrap_or_else(|p| p.into_inner());
            if sidecar.ready() && sidecar.epoch == epoch {
                gst::PadProbeReturn::Ok
            } else {
                // A concurrent flush invalidated the reservation while the
                // output buffer was being made writable.
                gst::PadProbeReturn::Drop
            }
        })
        .ok_or_else(|| error("failed to install native bridge output probe"))?;

    let entry_state = state.clone();
    input
        .add_probe(
            gst::PadProbeType::EVENT_DOWNSTREAM | gst::PadProbeType::EVENT_FLUSH,
            move |_, info| {
                if let Some(gst::PadProbeData::Event(ref event)) = info.data {
                    match event.view() {
                        gst::EventView::FlushStart(_) => flush(&entry_state, true, true),
                        gst::EventView::FlushStop(_) => flush(&entry_state, true, false),
                        _ => {}
                    }
                }
                gst::PadProbeReturn::Ok
            },
        )
        .ok_or_else(|| error("failed to install native bridge input event probe"))?;

    let exit_state = state.clone();
    let exit_runtime = shared;
    output
        .add_probe(
            gst::PadProbeType::EVENT_DOWNSTREAM | gst::PadProbeType::EVENT_FLUSH,
            move |_, info| {
                if let Some(gst::PadProbeData::Event(ref event)) = info.data {
                    match event.view() {
                        gst::EventView::FlushStart(_) => flush(&exit_state, false, true),
                        gst::EventView::FlushStop(_) => flush(&exit_state, false, false),
                        gst::EventView::Eos(_) => {
                            let (pending, incomplete_flush) = {
                                let mut state = exit_state
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                                let pending = state.pending.len();
                                let incomplete_flush = state.flush_active;
                                state.pending.clear();
                                state.closed = true;
                                (pending, incomplete_flush)
                            };
                            if incomplete_flush {
                                report(&exit_runtime, "EOS before both flush stops");
                                return gst::PadProbeReturn::Drop;
                            }
                            if pending != 0 {
                                report(
                                    &exit_runtime,
                                    format!("EOS with {pending} unmatched input contexts"),
                                );
                                return gst::PadProbeReturn::Drop;
                            }
                        }
                        _ => {}
                    }
                }
                gst::PadProbeReturn::Ok
            },
        )
        .ok_or_else(|| error("failed to install native bridge output event probe"))?;

    Ok(BridgeHandle(state))
}

pub(super) fn install_geometry_validation(
    element: &gst::Element,
    geometry: NativeGeometry,
    shared: Shared,
    annotation_update: Option<AnnotationUpdate>,
) -> Result<(), VydError> {
    for (pad, dimensions, output) in [
        (element.sink_pads().remove(0), geometry.input, false),
        (element.src_pads().remove(0), geometry.output, true),
    ] {
        let runtime = shared.clone();
        let update = annotation_update.clone();
        pad.add_probe(gst::PadProbeType::BUFFER, move |pad, info| {
            if let Some(buffer) = info.buffer() {
                if let Err(failure) = transport::validate_native_buffer(pad, buffer, dimensions) {
                    report(&runtime, failure);
                    return gst::PadProbeReturn::Drop;
                }
            }
            if output {
                if let Some(update) = &update {
                    let result = (|| {
                        let negotiated = super::caps::negotiated_gray8(pad)?;
                        let actual = NativeGeometry {
                            input: geometry.input,
                            output: (negotiated.width(), negotiated.height()),
                        };
                        let buffer = info
                            .buffer_mut()
                            .ok_or_else(|| error("native output probe has no buffer"))?;
                        let mut context = transport::take_context(buffer)?;
                        update(&mut context.annotations(), actual)?;
                        transport::attach_context(buffer.make_mut(), context)
                    })();
                    if let Err(failure) = result {
                        report(&runtime, failure);
                        return gst::PadProbeReturn::Drop;
                    }
                }
            }
            gst::PadProbeReturn::Ok
        })
        .ok_or_else(|| error("failed to install native geometry probe"))?;
    }
    Ok(())
}

pub(super) fn finish(handles: &[BridgeHandle], normal_eos: bool) -> Option<VydError> {
    let mut failure = None;
    for handle in handles {
        let mut state = handle
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        tracing::debug!(
            capacity = state.capacity,
            high_watermark = state.high_watermark,
            epoch = state.epoch,
            pending = state.pending.len(),
            "Native context bridge finished"
        );
        if normal_eos && failure.is_none() {
            if state.flush_active {
                failure = Some(error("native context bridge: EOS before both flush stops"));
            } else if !state.pending.is_empty() {
                failure = Some(error(format!(
                    "native context bridge: EOS with {} unmatched input contexts",
                    state.pending.len()
                )));
            }
        }
        state.pending.clear();
        state.closed = true;
    }
    failure
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use vyd_core::statistics::PipelineStatistics;
    use vyd_core::{Frame, FrameContext};

    struct DropWitness(Arc<AtomicUsize>);
    impl Drop for DropWitness {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn context(drops: &Arc<AtomicUsize>) -> transport::ContextData {
        gst::init().unwrap();
        transport::register();
        let mut frame = FrameContext::new(Frame::from_raw(2, 2, vec![1; 4]).unwrap(), 0);
        frame.put_typed(DropWitness(drops.clone()));
        let info = transport::info(&frame.frame).unwrap();
        let buffer =
            transport::pack(frame, Instant::now(), &info, &PipelineStatistics::new()).unwrap();
        transport::take_context(&buffer).unwrap()
    }

    #[test]
    fn both_flush_stops_are_required_before_the_next_epoch_can_pair() {
        let drops = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(Mutex::new(BridgeState::new(2)));
        state
            .lock()
            .unwrap()
            .pending
            .push_back((0, context(&drops)));
        flush(&state, true, true);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!state.lock().unwrap().ready());
        flush(&state, true, false);
        assert!(!state.lock().unwrap().ready());
        flush(&state, false, true);
        assert!(!state.lock().unwrap().ready());
        flush(&state, false, false);
        assert!(state.lock().unwrap().ready());
        assert_eq!(state.lock().unwrap().epoch, 1);
        state
            .lock()
            .unwrap()
            .pending
            .push_back((1, context(&drops)));
        flush(&state, false, true);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(state.lock().unwrap().epoch, 2);
        flush(&state, false, false);
        assert!(!state.lock().unwrap().ready());
        flush(&state, true, false);
        assert!(state.lock().unwrap().ready());
    }

    #[test]
    fn abort_and_eos_release_all_pending_contexts() {
        let drops = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(Mutex::new(BridgeState::new(2)));
        state
            .lock()
            .unwrap()
            .pending
            .push_back((0, context(&drops)));
        let handles = [BridgeHandle(state.clone())];
        abort(&handles);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(state.lock().unwrap().closed);
        assert!(finish(&handles, false).is_none());

        let state = Arc::new(Mutex::new(BridgeState::new(2)));
        state
            .lock()
            .unwrap()
            .pending
            .push_back((0, context(&drops)));
        let handles = [BridgeHandle(state.clone())];
        let error = finish(&handles, true).unwrap();
        assert!(error.to_string().contains("unmatched input contexts"));
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(state.lock().unwrap().closed);

        let state = Arc::new(Mutex::new(BridgeState::new(2)));
        flush(&state, true, true);
        flush(&state, true, false);
        let error = finish(&[BridgeHandle(state)], true).unwrap();
        assert!(error.to_string().contains("before both flush stops"));
    }
}
