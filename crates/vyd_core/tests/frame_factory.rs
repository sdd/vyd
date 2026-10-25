use std::sync::Arc;
use std::time::Duration;
use vyd_core::Frame;
use vyd_core::frame::CpuAllocationOrigin;
use vyd_core::frame::FrameError;
use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::pool::{BufferPoolManager, PoolError, PoolLimits, WaitPolicy};
use vyd_core::pipeline::PipelineContext;
use vyd_core::statistics::{PipelineStatistics, StatsContext};

#[test]
fn context_factory_creates_normal_frames_with_bounded_pool_lifetime() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(4)).unwrap());
    let mut context = PipelineContext::new();
    context.set_frame_factory(FrameFactory::new(pool.clone()));
    let frame = context
        .frame_factory()
        .create_luma8_with(2, 2, WaitPolicy::Immediate, |bytes| {
            bytes.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
    let clone = frame.clone();
    assert_eq!(frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
    assert!(matches!(
        context
            .frame_factory()
            .create_luma8(2, 2, WaitPolicy::Immediate),
        Err(FrameError::Pool(PoolError::Exhausted))
    ));
    drop(frame);
    assert_eq!(pool.snapshot().unwrap().active.len(), 1);
    drop(clone);
    let reused = context
        .frame_factory()
        .create_luma8(2, 2, WaitPolicy::Immediate)
        .unwrap();
    assert_eq!(reused.as_cpu_ref().unwrap(), &[0, 0, 0, 0]);
}

#[test]
fn failed_fill_returns_lease_and_legacy_mutation_detaches() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(4)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let result = factory.create_luma8_with(2, 2, WaitPolicy::Immediate, |_| {
        Err(FrameError::BufferUnavailable)
    });
    assert!(matches!(result, Err(FrameError::BufferUnavailable)));
    let original = factory.create_luma8(2, 2, WaitPolicy::Immediate).unwrap();
    let mut changed = original.clone();
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone())
        .for_stage("legacy-image")
        .enter();
    let Frame::Cpu(cpu) = &mut changed else {
        panic!("expected CPU frame")
    };
    cpu.img.put_pixel(0, 0, image::Luma([9]));
    assert_eq!(original.as_cpu_ref().unwrap(), &[0, 0, 0, 0]);
    assert_eq!(changed.as_cpu_ref().unwrap(), &[9, 0, 0, 0]);
    let memory = stats.snapshot().memory_operations;
    assert_eq!(memory["legacy-image_cpu_copy_bytes"], 4);
    assert_eq!(memory["legacy-image_cpu_copy_count"], 1);
    assert_eq!(memory["legacy-image_cpu_alloc_bytes"], 4);
    drop(original);
    assert_eq!(pool.snapshot().unwrap().active.len(), 0);
}

#[test]
fn malformed_dimensions_fail_before_pool_acquisition() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(4)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    assert!(matches!(
        factory.create_luma8(0, 1, WaitPolicy::Immediate),
        Err(FrameError::Layout(_))
    ));
    assert!(matches!(
        Frame::from_raw(0, 1, vec![]),
        Err(FrameError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        Frame::from_raw(u32::MAX, u32::MAX, vec![]),
        Err(FrameError::InvalidDimensions { .. })
    ));
    assert_eq!(pool.snapshot().unwrap().allocated_buffers, 0);
}

#[test]
fn pooled_frame_write_keeps_backing_in_issuing_pool_and_isolates_clones() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(2), Some(8)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let original = factory
        .create_luma8_with(2, 2, WaitPolicy::Immediate, |bytes| {
            bytes.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
    let mut changed = original.clone();
    changed
        .try_update_pooled_luma8(WaitPolicy::Immediate, |bytes| bytes[0] = 9)
        .unwrap();
    assert_eq!(original.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
    assert_eq!(changed.as_cpu_ref().unwrap(), &[9, 2, 3, 4]);
    assert_eq!(pool.snapshot().unwrap().active.len(), 2);
    drop(original);
    drop(changed);
    assert_eq!(pool.snapshot().unwrap().active.len(), 0);
    assert_eq!(pool.snapshot().unwrap().idle_buffers, 2);
}

#[test]
fn pooled_frame_write_exhaustion_preserves_frame_and_unique_write_reuses_lease() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(4)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let mut frame = factory.create_luma8(2, 2, WaitPolicy::Immediate).unwrap();
    let retained = frame.clone();
    assert!(matches!(
        frame.try_update_pooled_luma8(WaitPolicy::Immediate, |bytes| bytes[0] = 9),
        Err(FrameError::Pool(PoolError::Exhausted))
    ));
    assert_eq!(frame.as_cpu_ref().unwrap(), &[0, 0, 0, 0]);
    drop(retained);
    let original_id = pool.snapshot().unwrap().active[0].id;
    frame
        .try_update_pooled_luma8(WaitPolicy::Immediate, |bytes| bytes[0] = 7)
        .unwrap();
    assert_eq!(frame.as_cpu_ref().unwrap(), &[7, 0, 0, 0]);
    assert_eq!(pool.snapshot().unwrap().active[0].id, original_id);
    assert_eq!(pool.snapshot().unwrap().allocated_buffers, 1);
}

#[test]
fn pooled_write_restores_image_after_panic_and_rejects_other_storage() {
    let factory = FrameFactory::default();
    let mut frame = factory.create_luma8(2, 2, WaitPolicy::Immediate).unwrap();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        frame
            .try_update_pooled_luma8(WaitPolicy::Immediate, |bytes| {
                bytes[0] = 5;
                panic!("callback failed")
            })
            .unwrap();
    }));
    assert!(caught.is_err());
    assert_eq!(frame.dimensions(), (2, 2));
    assert_eq!(frame.as_cpu_ref().unwrap(), &[5, 0, 0, 0]);

    let mut ordinary = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
    assert!(matches!(
        ordinary.try_update_pooled_luma8(WaitPolicy::Immediate, |_| ()),
        Err(FrameError::NotPooledCpuFrame)
    ));
    assert_eq!(ordinary.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
}

#[test]
fn fallible_reads_and_writes_keep_pool_provenance_and_isolate_retained_views() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(2), Some(8)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let mut frame = factory
        .create_luma8_with(2, 2, WaitPolicy::Immediate, |bytes| {
            bytes.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        frame.as_cpu_image().unwrap().as_raw().allocation_origin(),
        CpuAllocationOrigin::Pool
    );
    let retained = frame.clone();
    let read = retained
        .try_read_cpu_luma8(WaitPolicy::Timeout(Duration::ZERO))
        .unwrap();
    assert_eq!(read, &[1, 2, 3, 4]);
    {
        let mut write = frame.try_write_cpu_luma8(WaitPolicy::Immediate).unwrap();
        write.try_bytes_mut().unwrap()[0] = 9;
    }
    assert_eq!(read, &[1, 2, 3, 4]);
    assert_eq!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[9, 2, 3, 4]
    );
    assert_eq!(
        frame.as_cpu_image().unwrap().as_raw().allocation_origin(),
        CpuAllocationOrigin::Pool
    );
    assert_eq!(pool.snapshot().unwrap().active.len(), 2);
}

#[test]
fn fallible_write_exhaustion_and_timeout_preserve_original_bytes() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(4)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let mut frame = factory
        .create_luma8_with(2, 2, WaitPolicy::Immediate, |bytes| {
            bytes.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
    let retained = frame.clone();
    assert!(matches!(
        frame.try_write_cpu_luma8(WaitPolicy::Immediate),
        Err(FrameError::Pool(PoolError::Exhausted))
    ));
    assert!(matches!(
        frame.try_write_cpu_luma8(WaitPolicy::Timeout(Duration::from_millis(5))),
        Err(FrameError::Pool(PoolError::Timeout(_)))
    ));
    assert_eq!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4]
    );
    assert_eq!(
        retained.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4]
    );
    assert_eq!(pool.snapshot().unwrap().active.len(), 1);
    drop(retained);
    let original_id = pool.snapshot().unwrap().active[0].id;
    {
        let mut write = frame.try_write_cpu_luma8(WaitPolicy::Immediate).unwrap();
        write.try_bytes_mut().unwrap()[0] = 7;
    }
    assert_eq!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[7, 2, 3, 4]
    );
    assert_eq!(pool.snapshot().unwrap().active[0].id, original_id);
}

#[test]
fn heap_origin_is_separate_from_exclusive_mutability() {
    let mut frame = Frame::from_raw(2, 1, vec![1, 2]).unwrap();
    let retained = frame.clone();
    assert_eq!(
        frame.as_cpu_image().unwrap().as_raw().allocation_origin(),
        CpuAllocationOrigin::Heap
    );
    assert!(!frame.as_cpu_image().unwrap().as_raw().is_owned());
    {
        let mut write = frame.try_write_cpu_luma8(WaitPolicy::Immediate).unwrap();
        write.try_bytes_mut().unwrap()[0] = 8;
    }
    assert_eq!(
        retained.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2]
    );
    assert_eq!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[8, 2]
    );
    assert!(frame.as_cpu_image().unwrap().as_raw().is_owned());
}
