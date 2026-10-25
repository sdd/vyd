use std::sync::Arc;

use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::pool::WaitPolicy;
use vyd_core::frame::{CpuFrame, Frame, PixelFormat};
use vyd_core::statistics::{PipelineStatistics, StatsContext};

#[test]
fn cpu_requested_conversion_is_cached_until_legacy_mutation_or_replacement() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone())
        .for_stage("requested")
        .enter();
    let factory = FrameFactory::default();
    let mut frame = Frame::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
    for _ in 0..2 {
        let read = frame
            .try_read_cpu_as(&factory, PixelFormat::Nv12, WaitPolicy::Immediate)
            .unwrap();
        let nv12 = read.nv12().unwrap();
        assert_eq!(nv12.row(0, 0).unwrap(), &[1, 2, 3]);
        assert_eq!(nv12.row(1, 0).unwrap(), &[128; 4]);
    }
    assert_eq!(
        stats.snapshot().frame_operations["requested"]["convert_cpu_luma8_to_nv12"].completed,
        1
    );

    let retained = frame.clone();
    let Frame::Cpu(cpu) = &mut frame else {
        unreachable!()
    };
    cpu.img.as_mut()[0] = 9;
    assert_eq!(
        frame
            .try_read_cpu_as(&factory, PixelFormat::Nv12, WaitPolicy::Immediate)
            .unwrap()
            .nv12()
            .unwrap()
            .row(0, 0)
            .unwrap(),
        &[9, 2, 3]
    );
    assert_eq!(
        retained.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4, 5, 6]
    );

    let Frame::Cpu(cpu) = &mut frame else {
        unreachable!()
    };
    cpu.img = CpuFrame::from_vec(3, 2, vec![7; 6]).unwrap().img;
    assert_eq!(
        frame
            .try_read_cpu_as(&factory, PixelFormat::Nv12, WaitPolicy::Immediate)
            .unwrap()
            .nv12()
            .unwrap()
            .row(0, 0)
            .unwrap(),
        &[7, 7, 7]
    );
    assert_eq!(
        stats.snapshot().frame_operations["requested"]["convert_cpu_luma8_to_nv12"].completed,
        3
    );
}

#[cfg(feature = "gpu-wgpu")]
mod gpu_tests {
    use super::*;
    use std::time::Duration;
    use vyd_core::frame::FrameError;
    use vyd_core::gpu::pool::{GpuPoolLimits, GpuPoolManager};
    use vyd_core::gpu::transfer::{GpuTransfer, GpuTransferError};
    use vyd_core::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};

    fn fixture(limits: GpuPoolLimits) -> Option<(GpuTransfer, Arc<PipelineStatistics>)> {
        let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Arc::new(context),
            Err(GpuContextError::Unavailable) => {
                eprintln!("requested-read GPU tests unverified: no compatible adapter");
                return None;
            }
            Err(error) => panic!("requested-read hardware gate failed: {error}"),
        };
        eprintln!(
            "requested-read adapter={} backend={:?}",
            context.adapter_info().name,
            context.adapter_info().backend
        );
        let stats = Arc::new(PipelineStatistics::new());
        let pool = GpuPoolManager::new(context.clone(), limits);
        let transfer = GpuTransfer::new(
            context,
            pool,
            StatsContext::new(stats.clone()).for_stage("requested_gpu"),
        )
        .unwrap();
        Some((transfer, stats))
    }

    fn wait() -> WaitPolicy {
        WaitPolicy::Timeout(Duration::from_secs(3))
    }

    fn completed(stats: &PipelineStatistics, prefix: &str) -> u64 {
        stats.snapshot().frame_operations["requested_gpu"]
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(_, value)| value.completed)
            .sum()
    }

    #[test]
    fn upload_readback_and_legacy_mutation_keep_old_readers() {
        let Some((transfer, stats)) = fixture(GpuPoolLimits::new(16, 1024 * 1024).unwrap()) else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let old = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        let again = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert_eq!(old.device_key(), again.device_key());
        assert_eq!(completed(&stats, "upload_luma8"), 1);

        let mut gpu_frame = old.clone().into_frame();
        for _ in 0..2 {
            assert_eq!(
                gpu_frame
                    .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
                    .unwrap()
                    .luma8_bytes()
                    .unwrap(),
                &[1, 2, 3, 4, 5, 6]
            );
        }
        assert_eq!(completed(&stats, "download_luma8"), 1);

        let Frame::Cpu(cpu) = &mut frame else {
            unreachable!()
        };
        cpu.img.as_mut()[0] = 9;
        frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap();
        assert_eq!(completed(&stats, "upload_luma8"), 2);
        assert_eq!(
            gpu_frame
                .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
                .unwrap()
                .luma8_bytes()
                .unwrap(),
            &[1, 2, 3, 4, 5, 6]
        );
        let mut latest = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap()
            .into_frame();
        assert_eq!(
            latest
                .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
                .unwrap()
                .luma8_bytes()
                .unwrap(),
            &[9, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn wrong_device_and_failed_upload_preserve_source() {
        let Some((transfer, _)) = fixture(GpuPoolLimits::new(16, 1024 * 1024).unwrap()) else {
            return;
        };
        let Some((other, _)) = fixture(GpuPoolLimits::new(16, 1024 * 1024).unwrap()) else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mut gpu = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap()
            .into_frame();
        assert!(matches!(
            gpu.try_read_gpu_as(&factory, &other, PixelFormat::Luma8, wait()),
            Err(FrameError::GpuTransfer(GpuTransferError::Context(
                GpuContextError::IncompatibleDevice { .. }
            )))
        ));

        let Some((small, _)) = fixture(GpuPoolLimits::new(1, 64).unwrap()) else {
            return;
        };
        assert!(
            frame
                .try_read_gpu_as(&factory, &small, PixelFormat::Luma8, WaitPolicy::Immediate)
                .is_err()
        );
        assert_eq!(
            frame.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn cross_format_gpu_request_converts_once_and_nv12_readback_reuses_cpu() {
        let Some((transfer, stats)) = fixture(GpuPoolLimits::new(16, 1024 * 1024).unwrap()) else {
            return;
        };
        let _scope = StatsContext::new(stats.clone())
            .for_stage("conversion")
            .enter();
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(3, 3, (1..=9).collect()).unwrap();
        let first = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Nv12, wait())
            .unwrap();
        let second = frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Nv12, wait())
            .unwrap();
        assert_eq!(first.plane_count(), 2);
        assert_eq!(second.plane_count(), 2);
        assert_eq!(completed(&stats, "upload_nv12"), 1);
        assert_eq!(
            stats.snapshot().frame_operations["conversion"]["convert_cpu_luma8_to_nv12"].completed,
            1
        );

        let mut gpu = first.into_frame();
        for _ in 0..2 {
            let read = gpu
                .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Nv12, wait())
                .unwrap();
            assert_eq!(read.nv12().unwrap().row(0, 2).unwrap(), &[7, 8, 9]);
            assert_eq!(read.nv12().unwrap().row(1, 1).unwrap(), &[128; 4]);
        }
        assert_eq!(completed(&stats, "download_nv12"), 1);
    }

    #[test]
    fn ensure_gpu_with_retains_the_coherent_cpu_source() {
        let Some((transfer, stats)) = fixture(GpuPoolLimits::new(16, 1024 * 1024).unwrap()) else {
            return;
        };
        let factory = FrameFactory::default();
        let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        frame.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
        assert!(frame.is_gpu());
        assert_eq!(
            frame
                .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
                .unwrap()
                .luma8_bytes()
                .unwrap(),
            &[1, 2, 3, 4]
        );
        assert_eq!(completed(&stats, "upload_luma8"), 1);
        assert_eq!(completed(&stats, "download_luma8"), 0);
    }
}
