#![cfg(feature = "gpu-wgpu")]
#![recursion_limit = "256"]

use std::sync::Arc;
use std::time::Duration;
use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::pool::WaitPolicy;
use vyd_core::frame::{Frame, Nv12Encoding, PixelFormat};
use vyd_core::gpu::pool::{GpuPoolLimits, GpuPoolManager};
use vyd_core::gpu::submission::GpuSubmissionError;
use vyd_core::gpu::transfer::{GpuTransfer, GpuTransferError};
use vyd_core::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};
use vyd_core::statistics::{FrameOperationSnapshot, PipelineStatistics, StatsContext};

fn fixture(resources: usize) -> Option<(GpuTransfer, Arc<PipelineStatistics>)> {
    let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
        Ok(context) => Arc::new(context),
        Err(GpuContextError::Unavailable) => {
            eprintln!("GPU conversion hardware tests unverified: no compatible adapter");
            return None;
        }
        Err(error) => panic!("GPU conversion hardware gate failed: {error}"),
    };
    eprintln!(
        "GPU conversion adapter={} backend={:?}",
        context.adapter_info().name,
        context.adapter_info().backend
    );
    let stats = Arc::new(PipelineStatistics::new());
    let pool = GpuPoolManager::new(
        context.clone(),
        GpuPoolLimits::new(resources, 1024 * 1024).unwrap(),
    );
    let transfer = GpuTransfer::new(
        context,
        pool,
        StatsContext::new(stats.clone()).for_stage("gpu_conversion"),
    )
    .unwrap();
    Some((transfer, stats))
}

fn wait() -> WaitPolicy {
    WaitPolicy::Timeout(Duration::from_secs(3))
}

fn operation(stats: &PipelineStatistics, prefix: &str) -> FrameOperationSnapshot {
    *stats.snapshot().frame_operations["gpu_conversion"]
        .iter()
        .find(|(name, _)| name.starts_with(prefix))
        .unwrap_or_else(|| panic!("missing operation {prefix}"))
        .1
}

#[test]
fn gpu_luma_to_nv12_matches_cpu_for_even_and_odd_dimensions_and_caches() {
    let Some((transfer, stats)) = fixture(16) else {
        return;
    };
    let factory = FrameFactory::default();
    for (width, height) in [(4, 2), (3, 3), (1, 5)] {
        let pixels: Vec<_> = (0..width * height).map(|n| (n * 17 + 3) as u8).collect();
        let cpu = Frame::from_raw(width, height, pixels).unwrap();
        let expected = factory
            .convert_luma8_to_nv12(&cpu, Nv12Encoding::FullRange, wait())
            .unwrap();
        let mut gpu = cpu.clone_shared();
        gpu.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
        let first = gpu
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Nv12, wait())
            .unwrap();
        let second = gpu
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Nv12, wait())
            .unwrap();
        assert_eq!(first.plane_count(), 2);
        assert_eq!(second.plane_count(), 2);
        let mut result = first.into_frame();
        let actual_read = result
            .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Nv12, wait())
            .unwrap();
        let actual = actual_read.nv12().unwrap();
        let expected = expected.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
        for plane in 0..2 {
            let (_, rows) = expected.layout().spec().plane_shape(plane).unwrap();
            for row in 0..rows {
                assert_eq!(
                    actual.row(plane, row).unwrap(),
                    expected.row(plane, row).unwrap()
                );
            }
        }
    }
    let op = operation(&stats, "convert_gpu_luma8_to_nv12");
    assert_eq!((op.attempted, op.completed, op.failed), (3, 3, 0));
    assert_eq!(op.visible_bytes, 12 + 17 + 11);
}

#[test]
fn resident_gpu_round_trip_has_two_conversions_and_no_intermediate_transfer() {
    let Some((transfer, stats)) = fixture(8) else {
        return;
    };
    let factory = FrameFactory::default();
    let pixels: Vec<_> = (1..=9).collect();
    let mut frame = Frame::from_raw(3, 3, pixels.clone()).unwrap();
    frame.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
    let mut nv12 = frame
        .try_read_gpu_as(&factory, &transfer, PixelFormat::Nv12, wait())
        .unwrap()
        .into_frame();
    let mut luma = nv12
        .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
        .unwrap()
        .into_frame();
    let ops = &stats.snapshot().frame_operations["gpu_conversion"];
    assert_eq!(operation(&stats, "convert_gpu_luma8_to_nv12").completed, 1);
    assert_eq!(operation(&stats, "convert_gpu_nv12_to_luma8").completed, 1);
    assert_eq!(operation(&stats, "upload_luma8").completed, 1);
    assert!(!ops.contains_key("download_luma8"));
    assert!(!ops.contains_key("download_nv12"));
    assert_eq!(
        luma.try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap()
            .luma8_bytes()
            .unwrap(),
        pixels
    );
    assert_eq!(operation(&stats, "download_luma8").completed, 1);
}

#[test]
fn gpu_nv12_to_luma_discards_chroma_like_cpu() {
    let Some((transfer, stats)) = fixture(8) else {
        return;
    };
    let factory = FrameFactory::default();
    let mut cpu = factory
        .create_nv12(3, 3, Nv12Encoding::FullRange, wait())
        .unwrap();
    {
        let mut write = cpu.try_write_cpu_nv12(WaitPolicy::Immediate).unwrap();
        for row in 0..3 {
            write.row_mut(0, row).unwrap().copy_from_slice(&[
                (row * 3 + 1) as u8,
                (row * 3 + 2) as u8,
                (row * 3 + 3) as u8,
            ]);
        }
        write.row_mut(1, 0).unwrap().copy_from_slice(&[7, 8, 9, 10]);
        write
            .row_mut(1, 1)
            .unwrap()
            .copy_from_slice(&[11, 12, 13, 14]);
    }
    let expected = factory.convert_nv12_to_luma8(&cpu, wait()).unwrap();
    cpu.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
    let mut actual = cpu
        .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
        .unwrap()
        .into_frame();
    assert_eq!(
        actual
            .try_read_cpu_as_with_transfer(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap()
            .luma8_bytes()
            .unwrap(),
        expected.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap()
    );
    let op = operation(&stats, "convert_gpu_nv12_to_luma8");
    assert_eq!(
        (op.attempted, op.completed, op.failed, op.visible_bytes),
        (1, 1, 0, 9)
    );
}

#[test]
fn exhausted_destination_counts_failed_conversion_without_completed_bytes() {
    let Some((transfer, stats)) = fixture(2) else {
        return;
    };
    let factory = FrameFactory::default();
    let mut frame = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
    frame.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
    assert!(matches!(
        frame.try_read_gpu_as(
            &factory,
            &transfer,
            PixelFormat::Nv12,
            WaitPolicy::Immediate
        ),
        Err(vyd_core::frame::FrameError::GpuTransfer(
            GpuTransferError::Pool(_)
        ))
    ));
    let op = operation(&stats, "convert_gpu_luma8_to_nv12");
    assert_eq!(
        (op.attempted, op.failed, op.completed, op.visible_bytes),
        (1, 1, 0, 0)
    );
    assert_eq!(
        frame
            .try_read_gpu_as(&factory, &transfer, PixelFormat::Luma8, wait())
            .unwrap()
            .plane_count(),
        1
    );
}

#[test]
fn immediate_caller_can_retry_pending_conversion_without_duplicate_work() {
    let Some((transfer, stats)) = fixture(8) else {
        return;
    };
    let factory = FrameFactory::default();
    let mut frame = Frame::from_raw(3, 3, (1..=9).collect()).unwrap();
    frame.ensure_gpu_with(&factory, &transfer, wait()).unwrap();
    let first = frame.try_read_gpu_as(
        &factory,
        &transfer,
        PixelFormat::Nv12,
        WaitPolicy::Immediate,
    );
    assert!(matches!(
        first,
        Ok(_)
            | Err(vyd_core::frame::FrameError::GpuSubmission(
                GpuSubmissionError::WouldBlock
            ))
    ));
    let read = frame
        .try_read_gpu_as(&factory, &transfer, PixelFormat::Nv12, wait())
        .unwrap();
    assert_eq!(read.plane_count(), 2);
    let op = operation(&stats, "convert_gpu_luma8_to_nv12");
    assert_eq!(
        (op.attempted, op.failed, op.completed, op.visible_bytes),
        (1, 0, 1, 17)
    );
}
