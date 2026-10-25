use std::num::NonZeroUsize;
use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::layout::{CpuLayout, FrameSpec};
use vyd_core::frame::pool::{BufferPoolManager, PoolError, PoolLimits, WaitPolicy};
use vyd_core::frame::{Frame, FrameError, Nv12Encoding, PixelFormat};

#[test]
fn pooled_luma8_write_copies_for_clone_and_failed_acquisition_preserves_bytes() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(2), Some(128)).unwrap());
    let factory = FrameFactory::new(pool);
    let mut frame = factory.create_luma8(2, 2, WaitPolicy::Immediate).unwrap();
    frame
        .try_write_cpu_luma8(WaitPolicy::Immediate)
        .unwrap()
        .try_bytes_mut()
        .unwrap()
        .copy_from_slice(&[1, 2, 3, 4]);
    let retained = frame.clone();
    frame
        .try_write_cpu_as(&factory, PixelFormat::Luma8, WaitPolicy::Immediate)
        .unwrap()
        .luma8_bytes_mut()
        .unwrap()[0] = 9;
    assert_eq!(
        retained.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4]
    );
    assert_eq!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[9, 2, 3, 4]
    );
    let latest = frame.clone();
    assert!(matches!(
        frame.try_write_cpu_as(&factory, PixelFormat::Luma8, WaitPolicy::Immediate),
        Err(FrameError::Pool(PoolError::Exhausted))
    ));
    assert_eq!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[9, 2, 3, 4]
    );
    assert_eq!(
        latest.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[9, 2, 3, 4]
    );
}

#[test]
fn requested_nv12_rows_preserve_old_clone_and_publish_partial_write_on_drop() {
    let factory = FrameFactory::default();
    let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
    let layout = CpuLayout::row_aligned(spec, NonZeroUsize::new(8).unwrap()).unwrap();
    let mut frame = factory
        .create_nv12_with_layout(layout, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    let retained = frame.clone();
    {
        let mut write = frame
            .try_write_cpu_as(&factory, PixelFormat::Nv12, WaitPolicy::Immediate)
            .unwrap();
        assert_eq!(write.pixel_format(), PixelFormat::Nv12);
        write
            .nv12_row_mut(0, 2)
            .unwrap()
            .copy_from_slice(&[7, 8, 9]);
        write
            .nv12_row_mut(1, 1)
            .unwrap()
            .copy_from_slice(&[1, 2, 3, 4]);
    }
    let read = frame.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
    assert_eq!(read.row(0, 2).unwrap(), &[7, 8, 9]);
    assert_eq!(read.row(1, 1).unwrap(), &[1, 2, 3, 4]);
    let old = retained.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
    assert_eq!(old.row(0, 2).unwrap(), &[0, 0, 0]);
    assert_eq!(old.row(1, 1).unwrap(), &[0, 0, 0, 0]);
}

#[test]
fn nv12_pool_exhaustion_preserves_both_regions() {
    let factory = FrameFactory::new(BufferPoolManager::new(
        PoolLimits::new(Some(2), Some(128)).unwrap(),
    ));
    let layout = CpuLayout::row_aligned(
        FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap(),
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap();
    let mut frame = factory
        .create_nv12_with_layout(layout, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    let retained = frame.clone();
    {
        let mut write = frame.try_write_cpu_nv12(WaitPolicy::Immediate).unwrap();
        assert!(matches!(
            write.row_mut(0, 0),
            Err(FrameError::Pool(PoolError::Exhausted))
        ));
    }
    for source in [&frame, &retained] {
        let read = source.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
        assert_eq!(read.row(0, 0).unwrap(), &[0, 0, 0]);
        assert_eq!(read.row(1, 0).unwrap(), &[0, 0, 0, 0]);
    }
}

#[test]
fn cross_format_cpu_write_materializes_requested_version() {
    let factory = FrameFactory::default();
    let mut frame = Frame::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let retained = frame.clone();
    {
        let mut write = frame
            .try_write_cpu_as(&factory, PixelFormat::Nv12, WaitPolicy::Immediate)
            .unwrap();
        write.nv12_row_mut(0, 0).unwrap()[1] = 9;
    }
    assert_eq!(frame.pixel_format(), PixelFormat::Nv12);
    assert_eq!(
        frame
            .try_read_cpu_nv12(WaitPolicy::Immediate)
            .unwrap()
            .row(0, 0)
            .unwrap(),
        &[1, 9, 3]
    );
    assert_eq!(
        retained.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4, 5, 6]
    );
}

#[cfg(feature = "gpu-wgpu")]
mod gpu {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use vyd_core::gpu::pool::{GpuPoolError, GpuPoolLimits, GpuPoolManager};
    use vyd_core::gpu::transfer::{GpuTransfer, GpuTransferError};
    use vyd_core::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};
    use vyd_core::statistics::{PipelineStatistics, StatsContext};

    fn fixture(resources: usize) -> Option<(GpuTransfer, GpuPoolManager, Arc<PipelineStatistics>)> {
        let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Arc::new(context),
            Err(GpuContextError::Unavailable) => {
                eprintln!("requested-write GPU tests unverified: no compatible adapter");
                return None;
            }
            Err(error) => panic!("requested-write hardware gate failed: {error}"),
        };
        eprintln!(
            "requested-write adapter={} backend={:?}",
            context.adapter_info().name,
            context.adapter_info().backend
        );
        let pool = GpuPoolManager::new(
            context.clone(),
            GpuPoolLimits::new(resources, 1024 * 1024).unwrap(),
        );
        let stats = Arc::new(PipelineStatistics::new());
        let transfer = GpuTransfer::new(
            context,
            pool.clone(),
            StatsContext::new(stats.clone()).for_stage("write"),
        )
        .unwrap();
        Some((transfer, pool, stats))
    }

    fn wait() -> WaitPolicy {
        WaitPolicy::Timeout(Duration::from_secs(3))
    }

    #[test]
    fn abandoned_gpu_candidate_keeps_source_cache_and_readers() {
        let Some((transfer, pool, stats)) = fixture(4) else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let reader = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let old = reader.clone().into_frame();
        let before = pool.snapshot().unwrap().active.len();
        {
            let candidate = frame
                .try_write_gpu_as(&transfer, PixelFormat::Nv12, wait())
                .unwrap();
            assert_eq!(candidate.plane_count(), 2);
            assert_eq!(candidate.spec().pixel_format(), PixelFormat::Nv12);
            assert_eq!(pool.snapshot().unwrap().active.len(), before + 2);
        }
        assert_eq!(pool.snapshot().unwrap().active.len(), before);
        assert_eq!(
            frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[1, 2, 3, 4]
        );
        let cached = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert_eq!(cached.device_key(), reader.device_key());
        assert_eq!(old.pixel_format(), PixelFormat::Luma8);
        assert_eq!(
            stats.snapshot().frame_operations["write"]
                .iter()
                .filter(|(name, _)| name.starts_with("upload_luma8"))
                .map(|(_, operation)| operation.completed)
                .sum::<u64>(),
            1
        );
    }

    #[test]
    fn gpu_candidate_exhaustion_preserves_source_and_clone() {
        let Some((transfer, pool, _)) = fixture(1) else {
            return;
        };
        let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        let old = frame.clone();
        let first = frame
            .try_write_gpu_as(&transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert_eq!(pool.snapshot().unwrap().active.len(), 1);
        let mut other = old.clone();
        assert!(matches!(
            other.try_write_gpu_as(&transfer, PixelFormat::Luma8, WaitPolicy::Immediate),
            Err(FrameError::GpuTransfer(GpuTransferError::Pool(
                GpuPoolError::Exhausted
            )))
        ));
        drop(first);
        assert!(pool.snapshot().unwrap().active.is_empty());
        assert_eq!(
            frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[1, 2, 3, 4]
        );
        assert_eq!(
            old.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn gpu_source_cpu_write_detaches_and_preserves_reader() {
        let Some((transfer, _, _)) = fixture(8) else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        frame.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
        let old = frame.clone();
        frame
            .try_write_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap()
            .luma8_bytes_mut()
            .unwrap()[0] = 9;
        assert_eq!(
            frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[9, 2, 3, 4]
        );
        let mut old = old;
        assert_eq!(
            old.try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
                .unwrap()
                .luma8_bytes()
                .unwrap(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn gpu_source_cross_format_write_converts_once() {
        let Some((transfer, _, stats)) = fixture(8) else {
            return;
        };
        let _scope = StatsContext::new(stats.clone())
            .for_stage("gpu_write_conversion")
            .enter();
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        frame.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
        let old = frame.clone();
        {
            let mut write = frame
                .try_write_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Nv12, wait())
                .unwrap();
            write.nv12_row_mut(0, 0).unwrap()[0] = 9;
        }
        assert_eq!(
            frame
                .try_read_cpu_nv12(WaitPolicy::Immediate)
                .unwrap()
                .row(0, 0)
                .unwrap(),
            &[9, 2, 3]
        );
        assert_eq!(old.pixel_format(), PixelFormat::Luma8);
        assert_eq!(
            stats.snapshot().frame_operations["gpu_write_conversion"]["convert_cpu_luma8_to_nv12"]
                .completed,
            1
        );
    }
}
