use std::sync::{Arc, Mutex};

use vyd_core::pipeline::{PayloadRequest, Pipeline, PipelineContext, Stage, run_pipeline};
use vyd_core::statistics::ProcessingType;
use vyd_core::{
    Frame, FrameContext, FrameProcessor, FrameProcessorResult, FrameSink, FrameSource, VydError,
};

struct Geometry;
struct OtherPayload;

#[test]
fn requests_are_independent_by_requester_and_payload_type() {
    let context = PipelineContext::new();
    let other_context = PipelineContext::new();
    assert!(!context.is_requested::<Geometry>());

    let first = context.request::<Geometry>();
    let second = context.request::<Geometry>();
    let other = context.request::<OtherPayload>();
    assert!(context.is_requested::<Geometry>());
    assert!(context.is_requested::<OtherPayload>());
    assert!(!other_context.is_requested::<Geometry>());

    drop(first);
    assert!(context.is_requested::<Geometry>());
    drop(second);
    assert!(!context.is_requested::<Geometry>());
    assert!(context.is_requested::<OtherPayload>());
    drop(other);
    assert!(!context.is_requested::<OtherPayload>());
}

struct Source(u64);

impl FrameSource for Source {
    fn next_frame(&mut self, _: &mut PipelineContext) -> Option<FrameContext> {
        if self.0 == 2 {
            return None;
        }
        let index = self.0;
        self.0 += 1;
        Some(FrameContext::new(
            Frame::from_raw(1, 1, vec![0]).unwrap(),
            index,
        ))
    }

    fn name(&self) -> &str {
        "source"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

struct Provider(Arc<Mutex<Vec<bool>>>);

impl FrameProcessor for Provider {
    fn process(
        &mut self,
        _: &mut FrameContext,
        context: &mut PipelineContext,
    ) -> FrameProcessorResult {
        self.0
            .lock()
            .unwrap()
            .push(context.is_requested::<Geometry>());
        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "provider"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

struct Consumer(Option<PayloadRequest<Geometry>>);

impl FrameProcessor for Consumer {
    fn pipeline_ctx_init(&mut self, context: &mut PipelineContext) -> Result<(), VydError> {
        self.0 = Some(context.request::<Geometry>());
        Ok(())
    }

    fn process(
        &mut self,
        frame: &mut FrameContext,
        _: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if frame.frame_index == 0 {
            self.0.take();
        }
        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "consumer"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

struct Sink;

impl FrameSink for Sink {
    fn consume(&mut self, _: &mut FrameContext, _: &mut PipelineContext) {}

    fn name(&self) -> &str {
        "sink"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

#[test]
fn downstream_initial_demand_is_visible_upstream_and_withdrawn_next_frame() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let pipeline = Pipeline {
        source: Stage::new(Box::new(Source(0)), "source"),
        stages: vec![
            Stage::new(Box::new(Provider(Arc::clone(&observations))), "provider"),
            Stage::new(Box::new(Consumer(None)), "consumer"),
        ],
        sink: Stage::new(Box::new(Sink), "sink"),
    };

    run_pipeline(PipelineContext::new(), pipeline, None).unwrap();
    assert_eq!(*observations.lock().unwrap(), [true, false]);
}
