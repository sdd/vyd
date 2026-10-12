use std::sync::Arc;

use vyd::frame::factory::FrameFactory;
use vyd::frame::layout::{CpuLayout, CpuPlaneLayout, FrameSpec};
use vyd::frame::pool::{BufferPoolManager, PoolError, PoolLimits, WaitPolicy};
use vyd::frame::{CpuNv12Frame, CpuStorage, Frame, FrameError, Nv12Encoding, PixelFormat};
use vyd::statistics::{PipelineStatistics, StatsContext};

#[test]
fn even_luma8_to_nv12_copies_y_and_neutralizes_every_uv_byte() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone())
        .for_stage("convert")
        .enter();
    let source = Frame::from_raw(4, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let factory = FrameFactory::default();
    let target = factory
        .convert_luma8_to_nv12(&source, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    let read = target.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
    assert_eq!(read.row(0, 0).unwrap(), &[1, 2, 3, 4]);
    assert_eq!(read.row(0, 1).unwrap(), &[5, 6, 7, 8]);
    assert_eq!(read.row(1, 0).unwrap(), &[128; 4]);
    assert_eq!(
        source.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4, 5, 6, 7, 8]
    );

    let op = stats.snapshot().frame_operations["convert"]["convert_cpu_luma8_to_nv12"];
    assert_eq!((op.attempted, op.completed, op.failed), (1, 1, 0));
    // One destination payload: 8 Y bytes + 4 neutral UV bytes.
    assert_eq!(
        (op.visible_bytes, op.staging_bytes, op.transfer_bytes),
        (12, 0, 0)
    );
}

#[test]
fn odd_luma8_nv12_round_trip_counts_each_destination_payload() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone()).for_stage("odd").enter();
    let source = Frame::from_raw(3, 3, (1..=9).collect()).unwrap();
    let factory = FrameFactory::default();
    let nv12 = factory
        .convert_luma8_to_nv12(&source, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    let nv12_read = nv12.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
    for row in 0..3 {
        assert_eq!(
            nv12_read.row(0, row).unwrap(),
            &[
                (row * 3 + 1) as u8,
                (row * 3 + 2) as u8,
                (row * 3 + 3) as u8
            ]
        );
    }
    assert_eq!(nv12_read.row(1, 0).unwrap(), &[128; 4]);
    assert_eq!(nv12_read.row(1, 1).unwrap(), &[128; 4]);
    let luma = factory
        .convert_nv12_to_luma8(&nv12, WaitPolicy::Immediate)
        .unwrap();
    assert_eq!(
        luma.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    let ops = &stats.snapshot().frame_operations["odd"];
    assert_eq!(ops["convert_cpu_luma8_to_nv12"].visible_bytes, 17);
    assert_eq!(ops["convert_cpu_nv12_to_luma8"].visible_bytes, 9);
}

#[test]
fn padded_independent_nv12_regions_only_copy_visible_y_and_discard_chroma() {
    let layout = CpuLayout::new(
        FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap(),
        &[CpuPlaneLayout::new(0, 1, 5), CpuPlaneLayout::new(1, 2, 6)],
        &[14, 12],
    )
    .unwrap();
    let mut source = Frame::from_cpu_nv12(
        CpuNv12Frame::from_regions(
            layout,
            Nv12Encoding::FullRange,
            vec![
                CpuStorage::from_vec(vec![99; 14]),
                CpuStorage::from_vec(vec![77; 12]),
            ],
        )
        .unwrap(),
    );
    {
        let mut write = source.try_write_cpu_nv12(WaitPolicy::Immediate).unwrap();
        for (row, y) in [[1, 2, 3], [4, 5, 6], [7, 8, 9]].into_iter().enumerate() {
            write.row_mut(0, row).unwrap().copy_from_slice(&y);
        }
        write
            .row_mut(1, 0)
            .unwrap()
            .copy_from_slice(&[10, 20, 30, 40]);
        write
            .row_mut(1, 1)
            .unwrap()
            .copy_from_slice(&[50, 60, 70, 80]);
    }
    let factory = FrameFactory::default();
    let destination = factory
        .convert_nv12_to_luma8(&source, WaitPolicy::Immediate)
        .unwrap();
    assert_eq!(
        destination
            .try_read_cpu_luma8(WaitPolicy::Immediate)
            .unwrap(),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(
        source
            .try_read_cpu_nv12(WaitPolicy::Immediate)
            .unwrap()
            .row(1, 1)
            .unwrap(),
        &[50, 60, 70, 80]
    );
}

#[test]
fn invalid_encoding_and_failed_allocation_count_attempts_without_completed_bytes() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone())
        .for_stage("failures")
        .enter();
    let source = Frame::from_raw(3, 3, vec![3; 9]).unwrap();
    let pool = BufferPoolManager::new(PoolLimits::new(1, 17).unwrap());
    let factory = FrameFactory::new(pool);
    let _held = factory.create_luma8(1, 1, WaitPolicy::Immediate).unwrap();
    for encoding in [Nv12Encoding::Unknown, Nv12Encoding::LimitedRange] {
        assert!(matches!(
            factory.convert_luma8_to_nv12(&source, encoding, WaitPolicy::Immediate),
            Err(FrameError::UnsupportedNv12Encoding)
        ));
    }
    assert!(matches!(
        factory.convert_luma8_to_nv12(&source, Nv12Encoding::FullRange, WaitPolicy::Immediate),
        Err(FrameError::Pool(PoolError::Exhausted))
    ));
    let op = stats.snapshot().frame_operations["failures"]["convert_cpu_luma8_to_nv12"];
    assert_eq!(
        (op.attempted, op.failed, op.completed, op.visible_bytes),
        (3, 3, 0, 0)
    );
    assert_eq!((op.staging_bytes, op.transfer_bytes), (0, 0));
}

#[test]
fn wrong_source_format_is_a_failed_conversion_attempt() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone()).for_stage("format").enter();
    let source = Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
    assert!(matches!(
        FrameFactory::default().convert_nv12_to_luma8(&source, WaitPolicy::Immediate),
        Err(FrameError::FormatConversionRequired { .. })
    ));
    let op = stats.snapshot().frame_operations["format"]["convert_cpu_nv12_to_luma8"];
    assert_eq!(
        (op.attempted, op.failed, op.completed, op.visible_bytes),
        (1, 1, 0, 0)
    );
}
