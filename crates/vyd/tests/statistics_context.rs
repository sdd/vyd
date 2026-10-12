use std::sync::{Arc, Barrier};
use vyd::statistics::{MemoryOperation, PipelineStatistics, StatsContext};

#[test]
fn worker_contexts_keep_concurrent_runs_separate() {
    let barrier = Arc::new(Barrier::new(2));
    let runs: Vec<_> = (0..2)
        .map(|_| Arc::new(PipelineStatistics::new()))
        .collect();
    let workers: Vec<_> = runs
        .iter()
        .enumerate()
        .map(|(i, stats)| {
            let _scope = StatsContext::new(stats.clone())
                .for_stage("processor")
                .enter();
            let captured = StatsContext::capture().unwrap();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                assert!(StatsContext::capture().is_none());
                let scope = captured.enter();
                barrier.wait();
                StatsContext::record_memory_operation(MemoryOperation::CpuCopy(i + 1));
                barrier.wait();
                drop(scope);
                assert!(StatsContext::capture().is_none());
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    for (i, stats) in runs.iter().enumerate() {
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["processor_cpu_copy_bytes"], (i + 1) as u64);
        assert_eq!(memory["processor_cpu_copy_count"], 1);
    }
}

#[test]
fn nesting_and_unwinding_restore_run_and_stage_without_holding_borrows() {
    let outer = Arc::new(PipelineStatistics::new());
    let inner = Arc::new(PipelineStatistics::new());
    let outer_scope = StatsContext::new(outer.clone()).for_stage("outer").enter();
    StatsContext::with_stats(|_| {
        let _scope = StatsContext::new(inner.clone()).for_stage("inner").enter();
        StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(7));
    })
    .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = StatsContext::new(inner.clone())
            .for_stage("panicking")
            .enter();
        panic!("test unwind");
    }));
    assert!(result.is_err());
    StatsContext::record_memory_operation(MemoryOperation::CpuAllocation(11));
    assert_eq!(
        outer.snapshot().memory_operations["outer_cpu_alloc_bytes"],
        11
    );
    assert_eq!(
        inner.snapshot().memory_operations["inner_cpu_alloc_bytes"],
        7
    );
    assert!(
        !inner
            .snapshot()
            .memory_operations
            .contains_key("outer_cpu_alloc_bytes")
    );
    drop(outer_scope);
    assert!(StatsContext::capture().is_none());
}

#[test]
fn concurrent_pipeline_runners_attribute_frames_and_finalizers_to_their_own_run() {
    use vyd::pipeline::{Pipeline, PipelineContext, Stage, run_pipeline};
    use vyd::statistics::ProcessingType;
    use vyd::{FrameContext, FrameSource, VydError};

    struct Source {
        barrier: Arc<Barrier>,
        bytes: usize,
        done: bool,
    }
    impl FrameSource for Source {
        fn next_frame(&mut self, _: &mut PipelineContext) -> Option<FrameContext> {
            if self.done {
                return None;
            }
            self.done = true;
            self.barrier.wait();
            // Force both runs to be active before recording. The old global
            // slot sends both operations into whichever run entered last.
            StatsContext::record_memory_operation(MemoryOperation::CpuCopy(self.bytes));
            self.barrier.wait();
            Some(FrameContext::new(
                vyd::Frame::from_raw(1, 1, vec![0]).unwrap(),
                0,
            ))
        }
        fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
            StatsContext::record_memory_operation(MemoryOperation::CpuCopy(self.bytes));
            Ok(())
        }
        fn name(&self) -> &str {
            "source"
        }
        fn processing_type(&self) -> ProcessingType {
            ProcessingType::Cpu
        }
    }

    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = [3usize, 5]
        .into_iter()
        .map(|bytes| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let stats = Arc::new(PipelineStatistics::new());
                let config = toml::from_str(
                    r#"
                [source]
                stage_type = "vyd::DummySource"
                [sink]
                stage_type = "vyd::DummySink"
            "#,
                )
                .unwrap();
                let Pipeline { stages, sink, .. } = vyd::pipeline::build_pipeline(&config).unwrap();
                let pipeline = Pipeline {
                    source: Stage::new(
                        Box::new(Source {
                            barrier,
                            bytes,
                            done: false,
                        }),
                        "source",
                    ),
                    stages,
                    sink,
                };
                let ctx =
                    run_pipeline(PipelineContext::new(), pipeline, Some(stats.clone())).unwrap();
                assert!(Arc::ptr_eq(
                    &ctx.services()
                        .latest::<PipelineStatistics>()
                        .unwrap()
                        .unwrap(),
                    &stats
                ));
                let snapshot = stats.snapshot();
                assert_eq!(snapshot.frame_count, 1);
                assert_eq!(
                    snapshot.memory_operations["source_cpu_copy_bytes"],
                    (2 * bytes) as u64
                );
                assert_eq!(snapshot.memory_operations["source_cpu_copy_count"], 2);
                assert!(StatsContext::capture().is_none());
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
