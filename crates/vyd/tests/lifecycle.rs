use std::sync::{Arc, Mutex};
use vyd::pipeline::{Pipeline, PipelineContext, PipelineObserver, Stage, run_pipeline};
use vyd::statistics::ProcessingType;
use vyd::{
    Frame, FrameContext, FrameProcessor, FrameProcessorResult, FrameSink, FrameSource, VydError,
};

type Events = Arc<Mutex<Vec<&'static str>>>;
fn record(events: &Events, event: &'static str) {
    events.lock().unwrap().push(event);
}
fn error(message: &str) -> VydError {
    VydError::GeneralPluginError(message.into())
}

struct Source(Events, usize);
impl FrameSource for Source {
    fn pipeline_ctx_init(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.0, "source.init");
        Ok(())
    }
    fn next_frame(&mut self, _: &mut PipelineContext) -> Option<FrameContext> {
        if self.1 == 0 {
            return None;
        }
        self.1 -= 1;
        Some(FrameContext::new(
            Frame::from_raw(1, 1, vec![0]).unwrap(),
            self.1,
        ))
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.0, "source.finish");
        Ok(())
    }
    fn name(&self) -> &str {
        "source"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
struct Processor {
    events: Events,
    fail_init: bool,
    skip: bool,
}
impl FrameProcessor for Processor {
    fn pipeline_ctx_init(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "processor.init");
        if self.fail_init {
            return Err(error("init failed"));
        }
        Ok(())
    }
    fn process(&mut self, _: &mut FrameContext, _: &mut PipelineContext) -> FrameProcessorResult {
        record(&self.events, "processor.process");
        if self.skip {
            FrameProcessorResult::Skip
        } else {
            FrameProcessorResult::Continue
        }
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "processor.finish");
        Err(error("finalizer failed"))
    }
    fn name(&self) -> &str {
        "processor"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
struct Sink(Events);
impl FrameSink for Sink {
    fn pipeline_ctx_init(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.0, "sink.init");
        Ok(())
    }
    fn consume(&mut self, _: &mut FrameContext, _: &mut PipelineContext) {
        record(&self.0, "sink.consume");
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.0, "sink.finish");
        Ok(())
    }
    fn name(&self) -> &str {
        "sink"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
struct Observer {
    events: Events,
    fail_frame: bool,
}
impl PipelineObserver for Observer {
    fn pipeline_started(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "observer.start");
        Ok(())
    }
    fn frame_processed(
        &mut self,
        _: &FrameContext,
        _: &mut PipelineContext,
    ) -> Result<(), VydError> {
        record(&self.events, "observer.frame");
        if self.fail_frame {
            return Err(error("observer failed"));
        }
        Ok(())
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        record(&self.events, "observer.finish");
        Ok(())
    }
}
fn pipeline(events: &Events, fail_init: bool, skip: bool) -> Pipeline {
    Pipeline {
        source: Stage::new(Box::new(Source(events.clone(), 2)), "source"),
        stages: vec![Stage::new(
            Box::new(Processor {
                events: events.clone(),
                fail_init,
                skip,
            }),
            "processor",
        )],
        sink: Stage::new(Box::new(Sink(events.clone())), "sink"),
    }
}
fn context(events: &Events, fail_frame: bool) -> PipelineContext {
    let mut ctx = PipelineContext::new();
    ctx.add_observer(Observer {
        events: events.clone(),
        fail_frame,
    });
    ctx
}
#[test]
fn initialization_failure_finalizes_attempted_stages_and_observers() {
    let events = Events::default();
    let result = run_pipeline(
        context(&events, false),
        pipeline(&events, true, false),
        None,
    );
    assert!(matches!(result, Err(VydError::GeneralPluginError(s)) if s == "init failed"));
    assert_eq!(
        *events.lock().unwrap(),
        [
            "observer.start",
            "source.init",
            "processor.init",
            "source.finish",
            "processor.finish",
            "observer.finish"
        ]
    );
}
#[test]
fn observer_failure_preserves_original_error_and_runs_every_finalizer() {
    let events = Events::default();
    let result = run_pipeline(
        context(&events, true),
        pipeline(&events, false, false),
        None,
    );
    assert!(matches!(result, Err(VydError::GeneralPluginError(s)) if s == "observer failed"));
    assert_eq!(
        *events.lock().unwrap(),
        [
            "observer.start",
            "source.init",
            "processor.init",
            "sink.init",
            "processor.process",
            "sink.consume",
            "observer.frame",
            "source.finish",
            "processor.finish",
            "sink.finish",
            "observer.finish"
        ]
    );
}
#[test]
fn skipped_frames_reach_observers_and_finalizer_errors_reach_caller() {
    let events = Events::default();
    let result = run_pipeline(
        context(&events, false),
        pipeline(&events, false, true),
        None,
    );
    assert!(matches!(result, Err(VydError::GeneralPluginError(s)) if s == "finalizer failed"));
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|&&event| event == "observer.frame")
            .count(),
        2
    );
    assert!(!events.contains(&"sink.consume"));
    assert_eq!(events.last(), Some(&"observer.finish"));
}
