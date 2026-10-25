//! Explicit CPU conversions between packed Luma8 and full-range grayscale NV12.
//! Requested-format replica selection and caching belong to the managed frame
//! access path, not to these conversion functions.

use super::factory::FrameFactory;
use super::layout::{CpuLayout, FrameSpec, LayoutError};
use super::pool::WaitPolicy;
use super::{Frame, FrameError, Nv12Encoding, PixelFormat};
use crate::statistics::{
    ConversionLocation, FrameOperationBytes, FrameOperationKind, StatsContext,
};

impl FrameFactory {
    /// Copy packed Luma8 samples to NV12 Y and set every visible UV sample to
    /// neutral 128. The caller explicitly declares full-range grayscale NV12.
    /// The returned frame owns a managed destination allocation.
    ///
    /// A completed conversion counts its *destination* visible payload bytes,
    /// including the newly written UV plane. Padding and source bytes are not
    /// added to that count. Failed attempts count no completed bytes.
    pub fn convert_luma8_to_nv12(
        &self,
        source: &Frame,
        encoding: Nv12Encoding,
        wait: WaitPolicy,
    ) -> Result<Frame, FrameError> {
        let attempt = StatsContext::begin_current_frame_operation(
            FrameOperationKind::Conversion {
                from: PixelFormat::Luma8,
                to: PixelFormat::Nv12,
                location: ConversionLocation::Cpu,
            },
            None,
        );
        if encoding != Nv12Encoding::FullRange {
            return Err(FrameError::UnsupportedNv12Encoding);
        }

        let (width, height) = source.dimensions();
        let source_bytes = source.try_read_cpu_luma8(WaitPolicy::Immediate)?;
        let source_layout = CpuLayout::packed(FrameSpec::new(width, height, PixelFormat::Luma8)?)?;

        let target_spec = FrameSpec::new(width, height, PixelFormat::Nv12)?;
        let target_bytes = visible_bytes(target_spec)?;

        let mut destination = self.create_nv12(width, height, encoding, wait)?;

        {
            let mut write = destination.try_write_cpu_nv12(WaitPolicy::Immediate)?;

            let (_, y_rows) = target_spec.plane_shape(0)?;
            for row in 0..y_rows {
                let y = source_layout.row(&[source_bytes], 0, row)?;
                write.row_mut(0, row)?.copy_from_slice(y);
            }

            let (_, uv_rows) = target_spec.plane_shape(1)?;
            for row in 0..uv_rows {
                write.row_mut(1, row)?.fill(128);
            }
        }

        if let Some(attempt) = attempt {
            attempt.complete(FrameOperationBytes {
                visible: target_bytes,
                ..Default::default()
            });
        }
        Ok(destination)
    }

    /// Copy only NV12's visible Y rows into a managed packed Luma8 frame.
    /// Chroma is intentionally discarded; no RGB reconstruction is implied.
    /// Completed visible bytes are the destination Luma8 payload only.
    pub fn convert_nv12_to_luma8(
        &self,
        source: &Frame,
        wait: WaitPolicy,
    ) -> Result<Frame, FrameError> {
        let attempt = StatsContext::begin_current_frame_operation(
            FrameOperationKind::Conversion {
                from: PixelFormat::Nv12,
                to: PixelFormat::Luma8,
                location: ConversionLocation::Cpu,
            },
            None,
        );
        let read = source.try_read_cpu_nv12(WaitPolicy::Immediate)?;
        if read.encoding() != Nv12Encoding::FullRange {
            return Err(FrameError::UnsupportedNv12Encoding);
        }
        let (width, height) = read.layout().spec().dimensions();
        let target_spec = FrameSpec::new(width, height, PixelFormat::Luma8)?;
        let target_layout = CpuLayout::packed(target_spec)?;
        let target_bytes = visible_bytes(target_spec)?;
        let destination = self.create_luma8_with(width, height, wait, |bytes| {
            let mut regions = [bytes];
            let (_, rows) = target_spec.plane_shape(0)?;
            for row in 0..rows {
                target_layout
                    .row_mut(&mut regions, 0, row)?
                    .copy_from_slice(read.row(0, row)?);
            }
            Ok(())
        })?;
        if let Some(attempt) = attempt {
            attempt.complete(FrameOperationBytes {
                visible: target_bytes,
                ..Default::default()
            });
        }
        Ok(destination)
    }
}

fn visible_bytes(spec: FrameSpec) -> Result<u64, LayoutError> {
    let mut total = 0usize;
    for plane in 0..spec.plane_count() {
        let (row_bytes, rows) = spec.plane_shape(plane)?;
        let bytes = row_bytes.checked_mul(rows).ok_or(LayoutError::Overflow)?;
        total = total.checked_add(bytes).ok_or(LayoutError::Overflow)?;
    }
    u64::try_from(total).map_err(|_| LayoutError::Overflow)
}
