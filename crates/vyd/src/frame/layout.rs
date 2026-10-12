//! Checked CPU image layouts. A plane addresses one mapped region, which may
//! be a separate allocation or a nonoverlapping span of another region.
use super::PixelFormat;
use std::num::NonZeroUsize;
use std::ops::Range;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LayoutError {
    #[error("frame dimensions must be nonzero")]
    ZeroDimensions,
    #[error("frame layout arithmetic overflow")]
    Overflow,
    #[error("plane region index is out of bounds")]
    MissingRegion,
    #[error("plane row stride is shorter than its visible row")]
    ShortStride,
    #[error("plane bytes exceed their mapped region")]
    ShortRegion,
    #[error("planes address overlapping spans of the same region")]
    OverlappingPlanes,
    #[error("plane count does not match pixel format")]
    WrongPlaneCount,
    #[error("row index is out of bounds")]
    InvalidRow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSpec {
    width: u32,
    height: u32,
    format: PixelFormat,
}

impl FrameSpec {
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Result<Self, LayoutError> {
        if width == 0 || height == 0 {
            return Err(LayoutError::ZeroDimensions);
        }
        let spec = Self {
            width,
            height,
            format,
        };
        // A valid spec must have representable visible plane sizes, even when
        // actual regions are supplied later by an importing adapter.
        let (y_width, y_height) = spec.plane_shape(0)?;
        let y_len = y_width.checked_mul(y_height).ok_or(LayoutError::Overflow)?;
        if format == PixelFormat::Nv12 {
            let (uv_width, uv_height) = spec.plane_shape(1)?;
            let uv = uv_width
                .checked_mul(uv_height)
                .ok_or(LayoutError::Overflow)?;
            y_len.checked_add(uv).ok_or(LayoutError::Overflow)?;
        }
        Ok(spec)
    }

    pub fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn pixel_format(self) -> PixelFormat {
        self.format
    }
    pub fn plane_count(self) -> usize {
        if self.format == PixelFormat::Nv12 {
            2
        } else {
            1
        }
    }

    /// Visible bytes per row and number of rows. NV12 chroma is interleaved
    /// UV with ceil(width/2) pairs in each of ceil(height/2) rows.
    pub fn plane_shape(self, index: usize) -> Result<(usize, usize), LayoutError> {
        let width = usize::try_from(self.width).map_err(|_| LayoutError::Overflow)?;
        let height = usize::try_from(self.height).map_err(|_| LayoutError::Overflow)?;
        match (self.format, index) {
            (PixelFormat::Luma8, 0) | (PixelFormat::Nv12, 0) => Ok((width, height)),
            (PixelFormat::Nv12, 1) => {
                let pairs = width / 2 + width % 2;
                Ok((
                    pairs.checked_mul(2).ok_or(LayoutError::Overflow)?,
                    height / 2 + height % 2,
                ))
            }
            _ => Err(LayoutError::WrongPlaneCount),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuPlaneLayout {
    region: usize,
    offset: usize,
    row_stride: usize,
}

impl CpuPlaneLayout {
    pub const fn new(region: usize, offset: usize, row_stride: usize) -> Self {
        Self {
            region,
            offset,
            row_stride,
        }
    }
    pub const fn region(self) -> usize {
        self.region
    }
    pub const fn offset(self) -> usize {
        self.offset
    }
    pub const fn row_stride(self) -> usize {
        self.row_stride
    }
}

/// A checked description, not a claim that the corresponding bytes are
/// initialized. Importers must keep mappings alive and prove initialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuLayout {
    spec: FrameSpec,
    planes: Box<[CpuPlaneLayout]>,
    region_lengths: Box<[usize]>,
}

impl CpuLayout {
    pub fn new(
        spec: FrameSpec,
        planes: &[CpuPlaneLayout],
        region_lengths: &[usize],
    ) -> Result<Self, LayoutError> {
        if planes.len() != spec.plane_count() {
            return Err(LayoutError::WrongPlaneCount);
        }
        let mut spans = Vec::with_capacity(planes.len());
        for (index, plane) in planes.iter().copied().enumerate() {
            let length = *region_lengths
                .get(plane.region)
                .ok_or(LayoutError::MissingRegion)?;
            let (row_bytes, rows) = spec.plane_shape(index)?;
            if plane.row_stride < row_bytes {
                return Err(LayoutError::ShortStride);
            }
            let last = rows
                .checked_sub(1)
                .ok_or(LayoutError::Overflow)?
                .checked_mul(plane.row_stride)
                .ok_or(LayoutError::Overflow)?
                .checked_add(plane.offset)
                .ok_or(LayoutError::Overflow)?
                .checked_add(row_bytes)
                .ok_or(LayoutError::Overflow)?;
            if last > length {
                return Err(LayoutError::ShortRegion);
            }
            spans.push((plane.region, plane.offset, last));
        }
        for i in 0..spans.len() {
            for j in i + 1..spans.len() {
                let a = spans[i];
                let b = spans[j];
                if a.0 == b.0 && a.1 < b.2 && b.1 < a.2 {
                    return Err(LayoutError::OverlappingPlanes);
                }
            }
        }
        Ok(Self {
            spec,
            planes: planes.into(),
            region_lengths: region_lengths.into(),
        })
    }

    pub fn packed(spec: FrameSpec) -> Result<Self, LayoutError> {
        let (row_bytes, rows) = spec.plane_shape(0)?;
        let y_len = row_bytes.checked_mul(rows).ok_or(LayoutError::Overflow)?;
        if spec.plane_count() == 1 {
            return Self::new(spec, &[CpuPlaneLayout::new(0, 0, row_bytes)], &[y_len]);
        }
        let (uv_bytes, uv_rows) = spec.plane_shape(1)?;
        let uv_len = uv_bytes.checked_mul(uv_rows).ok_or(LayoutError::Overflow)?;
        let total = y_len.checked_add(uv_len).ok_or(LayoutError::Overflow)?;
        Self::new(
            spec,
            &[
                CpuPlaneLayout::new(0, 0, row_bytes),
                CpuPlaneLayout::new(0, y_len, uv_bytes),
            ],
            &[total],
        )
    }

    pub fn row_aligned(spec: FrameSpec, alignment: NonZeroUsize) -> Result<Self, LayoutError> {
        let mut planes = Vec::with_capacity(spec.plane_count());
        let mut lengths = Vec::with_capacity(spec.plane_count());
        for index in 0..spec.plane_count() {
            let (row_bytes, rows) = spec.plane_shape(index)?;
            let stride = align_up(row_bytes, alignment.get())?;
            let size = stride.checked_mul(rows).ok_or(LayoutError::Overflow)?;
            planes.push(CpuPlaneLayout::new(index, 0, stride));
            lengths.push(size);
        }
        Self::new(spec, &planes, &lengths)
    }

    pub fn spec(&self) -> FrameSpec {
        self.spec
    }
    pub fn plane(&self, index: usize) -> Option<CpuPlaneLayout> {
        self.planes.get(index).copied()
    }
    pub fn region_lengths(&self) -> &[usize] {
        &self.region_lengths
    }

    /// Range of visible bytes in one row; excludes padding. The range is
    /// checked against the declared region length, not an arbitrary byte slice.
    pub fn row_range(&self, plane_index: usize, row: usize) -> Option<Range<usize>> {
        let plane = *self.planes.get(plane_index)?;
        let (row_bytes, rows) = self.spec.plane_shape(plane_index).ok()?;
        if row >= rows {
            return None;
        }
        let start = plane.offset + row * plane.row_stride;
        Some(start..start + row_bytes)
    }

    /// Borrow visible row bytes only when the supplied mapped region matches
    /// the checked layout. Distinct NV12 planes may use distinct slices.
    pub fn row<'a>(
        &self,
        regions: &[&'a [u8]],
        plane_index: usize,
        row: usize,
    ) -> Result<&'a [u8], LayoutError> {
        let plane = self
            .plane(plane_index)
            .ok_or(LayoutError::WrongPlaneCount)?;
        let region = *regions
            .get(plane.region)
            .ok_or(LayoutError::MissingRegion)?;
        if region.len() < self.region_lengths[plane.region] {
            return Err(LayoutError::ShortRegion);
        }
        let range = self
            .row_range(plane_index, row)
            .ok_or(LayoutError::InvalidRow)?;
        Ok(&region[range])
    }

    pub fn row_mut<'a>(
        &self,
        regions: &'a mut [&mut [u8]],
        plane_index: usize,
        row: usize,
    ) -> Result<&'a mut [u8], LayoutError> {
        let plane = self
            .plane(plane_index)
            .ok_or(LayoutError::WrongPlaneCount)?;
        let region = regions
            .get_mut(plane.region)
            .ok_or(LayoutError::MissingRegion)?;
        if region.len() < self.region_lengths[plane.region] {
            return Err(LayoutError::ShortRegion);
        }
        let range = self
            .row_range(plane_index, row)
            .ok_or(LayoutError::InvalidRow)?;
        Ok(&mut region[range])
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, LayoutError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(LayoutError::Overflow)
    }
}
