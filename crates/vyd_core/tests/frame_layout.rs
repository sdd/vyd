use std::num::NonZeroUsize;
use vyd_core::frame::PixelFormat;
use vyd_core::frame::layout::{CpuLayout, CpuPlaneLayout, FrameSpec, LayoutError};

#[test]
fn padded_luma8_rows_accept_short_final_padding_but_never_expose_it_as_pixels() {
    let spec = FrameSpec::new(3, 2, PixelFormat::Luma8).unwrap();
    let layout = CpuLayout::new(spec, &[CpuPlaneLayout::new(0, 0, 4)], &[7]).unwrap();
    let bytes = [1, 2, 3, 99, 4, 5, 6];
    assert_eq!(&bytes[layout.row_range(0, 0).unwrap()], &[1, 2, 3]);
    assert_eq!(&bytes[layout.row_range(0, 1).unwrap()], &[4, 5, 6]);
    assert_eq!(layout.row(&[&bytes], 0, 1).unwrap(), &[4, 5, 6]);
    assert_eq!(
        layout.row(&[&bytes[..6]], 0, 1),
        Err(LayoutError::ShortRegion)
    );
    assert_eq!(layout.row_range(0, 2), None);
    assert_eq!(layout.row(&[&bytes], 0, 2), Err(LayoutError::InvalidRow));
    let mut writable = bytes;
    layout
        .row_mut(&mut [&mut writable], 0, 0)
        .unwrap()
        .copy_from_slice(&[8, 9, 10]);
    assert_eq!(&writable[..4], &[8, 9, 10, 99]);
    assert_eq!(CpuLayout::packed(spec).unwrap().region_lengths(), &[6]);
}

#[test]
fn odd_nv12_accepts_independently_mapped_y_and_interleaved_uv() {
    let spec = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
    assert_eq!(spec.plane_shape(0).unwrap(), (3, 3));
    assert_eq!(spec.plane_shape(1).unwrap(), (4, 2));
    let layout = CpuLayout::new(
        spec,
        &[CpuPlaneLayout::new(0, 1, 5), CpuPlaneLayout::new(1, 2, 6)],
        &[14, 12],
    )
    .unwrap();
    assert_eq!(layout.row_range(0, 2), Some(11..14));
    assert_eq!(layout.row_range(1, 1), Some(8..12));
    assert_eq!(layout.plane(1).unwrap().region(), 1);
    let y = [0u8; 14];
    let uv = [0u8; 12];
    assert_eq!(layout.row(&[&y, &uv], 1, 1).unwrap().len(), 4);
    assert_eq!(CpuLayout::packed(spec).unwrap().region_lengths(), &[17]);
    let aligned = CpuLayout::row_aligned(spec, NonZeroUsize::new(8).unwrap()).unwrap();
    assert_eq!(aligned.region_lengths(), &[24, 16]);
    assert_eq!(aligned.plane(0).unwrap().row_stride(), 8);
    assert_eq!(aligned.plane(1).unwrap().row_stride(), 8);
}

#[test]
fn malformed_layouts_fail_before_slices_can_be_created() {
    assert_eq!(
        FrameSpec::new(0, 1, PixelFormat::Luma8),
        Err(LayoutError::ZeroDimensions)
    );
    let luma = FrameSpec::new(3, 2, PixelFormat::Luma8).unwrap();
    assert_eq!(
        CpuLayout::new(luma, &[], &[6]),
        Err(LayoutError::WrongPlaneCount)
    );
    assert_eq!(
        CpuLayout::new(luma, &[CpuPlaneLayout::new(1, 0, 3)], &[6]),
        Err(LayoutError::MissingRegion)
    );
    assert_eq!(
        CpuLayout::new(luma, &[CpuPlaneLayout::new(0, 0, 2)], &[6]),
        Err(LayoutError::ShortStride)
    );
    assert_eq!(
        CpuLayout::new(luma, &[CpuPlaneLayout::new(0, 0, 4)], &[6]),
        Err(LayoutError::ShortRegion)
    );
    assert_eq!(
        CpuLayout::new(
            luma,
            &[CpuPlaneLayout::new(0, usize::MAX, 4)],
            &[usize::MAX]
        ),
        Err(LayoutError::Overflow)
    );
    let nv12 = FrameSpec::new(3, 3, PixelFormat::Nv12).unwrap();
    assert_eq!(
        CpuLayout::new(
            nv12,
            &[CpuPlaneLayout::new(0, 0, 3), CpuPlaneLayout::new(0, 8, 4)],
            &[17]
        ),
        Err(LayoutError::OverlappingPlanes)
    );
    assert_eq!(
        CpuLayout::row_aligned(luma, NonZeroUsize::new(usize::MAX).unwrap()),
        Err(LayoutError::Overflow)
    );
}
