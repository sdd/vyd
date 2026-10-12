use std::sync::{Arc, Barrier};
use vyd::pipeline::PipelineContext;

#[derive(Debug, PartialEq)]
struct Model {
    version: usize,
    samples: Vec<usize>,
}

#[test]
fn publication_across_workers_keeps_existing_snapshots_and_runs_isolated() {
    let context = PipelineContext::new();
    let services = context.services().clone();
    let original = Arc::new(Model {
        version: 1,
        samples: vec![1; 8],
    });
    assert!(services.publish(original.clone()).unwrap().is_none());
    let retained = services.latest::<Model>().unwrap().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let ready = barrier.clone();
    let writer = services.clone();
    let worker = std::thread::spawn(move || {
        ready.wait();
        let previous = writer
            .publish(Arc::new(Model {
                version: 2,
                samples: vec![2; 8],
            }))
            .unwrap()
            .unwrap();
        assert_eq!(previous.version, 1);
        ready.wait();
    });
    barrier.wait();
    barrier.wait();
    let latest = context.services().latest::<Model>().unwrap().unwrap();
    assert_eq!(latest.samples, vec![2; 8]);
    assert_eq!(latest.version, 2);
    assert!(Arc::ptr_eq(&original, &retained));
    assert_eq!(retained.samples, vec![1; 8]);
    assert!(
        PipelineContext::new()
            .services()
            .latest::<Model>()
            .unwrap()
            .is_none()
    );
    worker.join().unwrap();
}

#[test]
fn service_destructor_can_reenter_registry_after_replacement() {
    struct Service(vyd::pipeline::PipelineServices);
    impl Drop for Service {
        fn drop(&mut self) {
            // This deadlocks if publish destroys the old value under its lock.
            self.0.publish(Arc::new(42usize)).unwrap();
        }
    }
    let context = PipelineContext::new();
    let services = context.services();
    services
        .publish(Arc::new(Service(services.clone())))
        .unwrap();
    // Keep the replacement free of a self-cycle. The first service is released
    // here, while the registry still contains a value of its type.
    drop(
        services
            .publish(Arc::new(Service(Default::default())))
            .unwrap(),
    );
    assert_eq!(*services.latest::<usize>().unwrap().unwrap(), 42);
}
