use crate::frame::CpuFrame;
use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::traits::FrameSource;
use crate::{Frame, FrameContext, VydError, register_vyd_frame_source};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct DummySource {
    frame_count: usize,
}

impl DummySource {
    pub fn new(_params: Option<&toml::Value>) -> Result<Self, VydError> {
        Ok(Self { frame_count: 5 })
    }
}

impl FrameSource for DummySource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        if self.frame_count == 0 {
            None
        } else {
            self.frame_count -= 1;

            // Track frames generated in pipeline context
            let counter = ctx
                .entry("frames_sourced".to_string())
                .or_insert_with(|| Box::new(AtomicUsize::new(0)));

            let frame_index = if let Some(atomic_counter) = counter.downcast_ref::<AtomicUsize>() {
                atomic_counter.fetch_add(1, Ordering::SeqCst)
            } else {
                0
            };

            Some(FrameContext::new(
                Frame::Cpu(
                    CpuFrame::from_vec(10, 10, vec![0; 100]).expect("Failed to create dummy frame"),
                ),
                frame_index,
            ))
        }
    }

    fn name(&self) -> &str {
        "dummy_source"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

register_vyd_frame_source!(DummySource);
