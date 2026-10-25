use vyd_core::frame::factory::FrameFactory;
use vyd_core::frame::layout::{CpuLayout, CpuPlaneLayout, FrameSpec, LayoutError};
use vyd_core::frame::pool::{BufferPoolManager, PoolError, PoolLimits, WaitPolicy};
use vyd_core::frame::{CpuNv12Frame, CpuStorage, Frame, FrameError, Nv12Encoding, PixelFormat};

#[test]
fn packed_odd_nv12_has_visible_y_and_uv_rows() {
    let factory = FrameFactory::default();
    let mut frame = factory
        .create_nv12(3, 3, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    assert!(frame.is_cpu());
    assert_eq!(frame.dimensions(), (3, 3));
    assert_eq!(frame.pixel_format(), PixelFormat::Nv12);
    assert_eq!(frame.pixel_count(), 9);
    {
        let mut write = frame.try_write_cpu_nv12(WaitPolicy::Immediate).unwrap();
        assert_eq!(write.layout().region_lengths(), &[17]);
        write.row_mut(0, 2).unwrap().copy_from_slice(&[7, 8, 9]);
        write
            .row_mut(1, 1)
            .unwrap()
            .copy_from_slice(&[10, 11, 12, 13]);
        assert!(matches!(
            write.row_mut(1, 2),
            Err(FrameError::Layout(LayoutError::InvalidRow))
        ));
    }
    let read = frame.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
    assert_eq!(read.encoding(), Nv12Encoding::FullRange);
    assert_eq!(read.row(0, 2).unwrap(), &[7, 8, 9]);
    assert_eq!(read.row(1, 1).unwrap(), &[10, 11, 12, 13]);
    assert!(matches!(
        read.row(2, 0),
        Err(FrameError::Layout(LayoutError::WrongPlaneCount))
    ));
    assert!(matches!(
        frame.try_read_cpu_luma8(WaitPolicy::Immediate),
        Err(FrameError::FormatConversionRequired { .. })
    ));
    assert!(matches!(
        frame.to_cpu(None),
        Err(FrameError::FormatConversionRequired { .. })
    ));
}

#[test]
fn distinct_padded_regions_expose_no_padding_and_reject_short_data() {
    let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
    let layout = CpuLayout::new(
        spec,
        &[CpuPlaneLayout::new(0, 1, 5), CpuPlaneLayout::new(1, 2, 6)],
        &[14, 12],
    )
    .unwrap();
    let y = vec![99; 14];
    let uv = vec![77; 12];
    assert!(matches!(
        CpuNv12Frame::from_regions(
            layout.clone(),
            Nv12Encoding::FullRange,
            vec![
                CpuStorage::from_vec(y.clone()),
                CpuStorage::from_vec(uv[..11].to_vec())
            ]
        ),
        Err(FrameError::Layout(LayoutError::ShortRegion))
    ));
    let mut frame = Frame::from_cpu_nv12(
        CpuNv12Frame::from_regions(
            layout,
            Nv12Encoding::FullRange,
            vec![CpuStorage::from_vec(y), CpuStorage::from_vec(uv)],
        )
        .unwrap(),
    );
    {
        let mut write = frame.try_write_cpu_nv12(WaitPolicy::Immediate).unwrap();
        write.row_mut(0, 0).unwrap().copy_from_slice(&[1, 2, 3]);
        write.row_mut(0, 2).unwrap().copy_from_slice(&[4, 5, 6]);
        write.row_mut(1, 1).unwrap().copy_from_slice(&[7, 8, 9, 10]);
    }
    let read = frame.try_read_cpu_nv12(WaitPolicy::Immediate).unwrap();
    assert_eq!(read.row(0, 0).unwrap(), &[1, 2, 3]);
    assert_eq!(read.row(0, 2).unwrap(), &[4, 5, 6]);
    assert_eq!(read.row(1, 1).unwrap(), &[7, 8, 9, 10]);
    // The checked layout still points at independent regions and skips the
    // leading and inter-row padding by construction.
    assert_eq!(read.layout().plane(0).unwrap().region(), 0);
    assert_eq!(read.layout().plane(1).unwrap().region(), 1);
}

#[test]
fn encoding_is_explicit_and_invalid_imports_never_claim_full_range() {
    for encoding in [Nv12Encoding::Unknown, Nv12Encoding::LimitedRange] {
        assert!(matches!(
            CpuNv12Frame::from_packed(2, 2, encoding, vec![0; 6]),
            Err(FrameError::UnsupportedNv12Encoding)
        ));
        assert!(matches!(
            FrameFactory::default().create_nv12(2, 2, encoding, WaitPolicy::Immediate),
            Err(FrameError::UnsupportedNv12Encoding)
        ));
    }
    let luma = CpuLayout::packed(FrameSpec::new(2, 2, PixelFormat::Luma8).unwrap()).unwrap();
    assert!(matches!(
        CpuNv12Frame::from_regions(
            luma,
            Nv12Encoding::FullRange,
            vec![CpuStorage::from_vec(vec![0; 4])]
        ),
        Err(FrameError::FormatConversionRequired { .. })
    ));
}

#[test]
fn pooled_clone_write_is_bounded_and_keeps_retained_rows() {
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(12)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let original = factory
        .create_nv12(2, 2, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    let mut changed = original.clone_shared();
    {
        let mut write = changed.try_write_cpu_nv12(WaitPolicy::Immediate).unwrap();
        let result = write.row_mut(0, 0);
        assert!(matches!(
            result,
            Err(FrameError::Pool(PoolError::Exhausted))
        ));
    }
    assert_eq!(
        original
            .try_read_cpu_nv12(WaitPolicy::Immediate)
            .unwrap()
            .row(0, 0)
            .unwrap(),
        &[0, 0]
    );
    drop(original);
    changed
        .try_write_cpu_nv12(WaitPolicy::Immediate)
        .unwrap()
        .row_mut(0, 0)
        .unwrap()
        .copy_from_slice(&[3, 4]);
    assert_eq!(
        changed
            .try_read_cpu_nv12(WaitPolicy::Immediate)
            .unwrap()
            .row(0, 0)
            .unwrap(),
        &[3, 4]
    );
    assert_eq!(pool.snapshot().unwrap().active.len(), 1);
}

#[test]
fn independent_factory_regions_are_pooled_and_released() {
    let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
    let layout = CpuLayout::new(
        spec,
        &[CpuPlaneLayout::new(0, 0, 5), CpuPlaneLayout::new(1, 0, 6)],
        &[13, 10],
    )
    .unwrap();
    let pool = BufferPoolManager::new(PoolLimits::new(Some(2), Some(23)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    let frame = factory
        .create_nv12_with_layout(layout, Nv12Encoding::FullRange, WaitPolicy::Immediate)
        .unwrap();
    assert_eq!(pool.snapshot().unwrap().active.len(), 2);
    assert_eq!(
        frame
            .try_read_cpu_nv12(WaitPolicy::Immediate)
            .unwrap()
            .row(1, 1)
            .unwrap(),
        &[0; 4]
    );
    drop(frame);
    assert_eq!(pool.snapshot().unwrap().active.len(), 0);
}

#[test]
fn failed_second_region_acquisition_returns_first_lease() {
    let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
    let layout = CpuLayout::new(
        spec,
        &[CpuPlaneLayout::new(0, 0, 5), CpuPlaneLayout::new(1, 0, 6)],
        &[13, 10],
    )
    .unwrap();
    let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(23)).unwrap());
    let factory = FrameFactory::new(pool.clone());
    assert!(matches!(
        factory.create_nv12_with_layout(layout, Nv12Encoding::FullRange, WaitPolicy::Immediate),
        Err(FrameError::Pool(PoolError::Exhausted))
    ));
    let snapshot = pool.snapshot().unwrap();
    assert!(snapshot.active.is_empty());
    assert_eq!(snapshot.idle_buffers, 1);
}
