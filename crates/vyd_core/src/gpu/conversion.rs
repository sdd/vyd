//! Same-device requested-format conversion of managed RGBA plane textures.
//! Only logical Y and UV bytes are converted; spare RGBA channels are always
//! initialized to the same deterministic values used by upload.

use super::recording::{GpuBindingSpec, GpuComputeProgram, GpuPlaneBinding, GpuRecording};
use super::submission::{self, GpuSubmission};
use super::transfer::GpuTransfer;
use crate::frame::layout::FrameSpec;
use crate::frame::pool::WaitPolicy;
use crate::frame::{Frame, FrameError, GpuFrameRead, PixelFormat};
use crate::statistics::FrameOperationBytes;
use std::time::Instant;

const COPY_Y: &str = r#"
    @group(0) @binding(0) var source: texture_2d<u32>;
    @group(0) @binding(1) var output: texture_storage_2d<rgba8uint, write>;
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let size = textureDimensions(output);
        if (id.x >= size.x || id.y >= size.y) { return; }
        let y = textureLoad(source, vec2<i32>(id.xy), 0).r;
        textureStore(output, vec2<i32>(id.xy), vec4<u32>(y, 0u, 0u, 255u));
    }
"#;

const NEUTRAL_UV: &str = r#"
    @group(0) @binding(0) var output: texture_storage_2d<rgba8uint, write>;
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let size = textureDimensions(output);
        if (id.x >= size.x || id.y >= size.y) { return; }
        textureStore(output, vec2<i32>(id.xy), vec4<u32>(128u, 128u, 0u, 255u));
    }
"#;

fn remaining(wait: WaitPolicy, started: Instant) -> WaitPolicy {
    match wait {
        WaitPolicy::Immediate => WaitPolicy::Immediate,
        WaitPolicy::Timeout(limit) => WaitPolicy::Timeout(limit.saturating_sub(started.elapsed())),
    }
}

fn dispatch_extent(width: u32, height: u32) -> [u32; 3] {
    [
        width / 8 + u32::from(width % 8 != 0),
        height / 8 + u32::from(height % 8 != 0),
        1,
    ]
}

fn destination_visible_bytes(spec: FrameSpec) -> Result<u64, FrameError> {
    let mut total = 0_u64;
    for index in 0..spec.plane_count() {
        let (row, rows) = spec.plane_shape(index)?;
        let row = u64::try_from(row).map_err(|_| crate::frame::layout::LayoutError::Overflow)?;
        let rows = u64::try_from(rows).map_err(|_| crate::frame::layout::LayoutError::Overflow)?;
        let bytes = row
            .checked_mul(rows)
            .ok_or(crate::frame::layout::LayoutError::Overflow)?;
        total = total
            .checked_add(bytes)
            .ok_or(crate::frame::layout::LayoutError::Overflow)?;
    }
    Ok(total)
}

/// Admit a GPU conversion and return its pending managed destination. The
/// ticket is owned by the submission until it finishes or fails, regardless
/// of whether the caller waits or drops the receipt.
pub(crate) fn convert(
    source: GpuFrameRead,
    transfer: &GpuTransfer,
    to: PixelFormat,
    wait: WaitPolicy,
) -> Result<(Frame, GpuSubmission), FrameError> {
    let from = source.spec().pixel_format();
    debug_assert_ne!(from, to);
    let started = Instant::now();
    let mut attempt = transfer.conversion_attempt(from, to);
    let (width, height) = source.spec().dimensions();
    let destination_spec = FrameSpec::new(width, height, to)?;
    let visible = destination_visible_bytes(destination_spec)?;
    let context = transfer.context();
    let mut destination = source.clone().into_frame();
    let write = destination.try_write_gpu_as(transfer, to, remaining(wait, started))?;
    attempt.record_wait(started.elapsed());

    let copy = GpuComputeProgram::new(
        &context,
        COPY_Y,
        "main",
        &[
            GpuBindingSpec::read(0, from, 0),
            GpuBindingSpec::write(1, to, 0),
        ],
    )?;
    let fill = if to == PixelFormat::Nv12 {
        Some(GpuComputeProgram::new(
            &context,
            NEUTRAL_UV,
            "main",
            &[GpuBindingSpec::write(0, PixelFormat::Nv12, 1)],
        )?)
    } else {
        None
    };
    let mut recording = GpuRecording::new(context);
    let input = recording.add_read(source.clone(), source.spec())?;
    let output = recording.add_write(write, destination_spec)?;
    let group = recording.bind_group(
        &copy,
        &[
            (0, GpuPlaneBinding::Read(input[0])),
            (1, GpuPlaneBinding::Write(output[0])),
        ],
    )?;
    recording.dispatch(&copy, group, dispatch_extent(width, height))?;
    if let Some(fill) = fill {
        let group = recording.bind_group(&fill, &[(0, GpuPlaneBinding::Write(output[1]))])?;
        let (uv_row, uv_height) = destination_spec.plane_shape(1)?;
        let uv_width =
            u32::try_from(uv_row / 2).map_err(|_| crate::frame::layout::LayoutError::Overflow)?;
        let uv_height =
            u32::try_from(uv_height).map_err(|_| crate::frame::layout::LayoutError::Overflow)?;
        recording.dispatch(&fill, group, dispatch_extent(uv_width, uv_height))?;
    }
    let receipt = submission::submit_with_operation(
        recording,
        attempt,
        FrameOperationBytes {
            visible,
            ..Default::default()
        },
    )?;
    Ok((destination, receipt))
}
