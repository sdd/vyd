//! Packed-Luma8 transport for the linear GST graph. Compatible CPU storage is
//! lent to GST read-only; a mapped GST buffer can in turn back a Vyd frame.
//! The envelope contains no pixel owners or buffer handles.
use super::error;
use gst::glib;
use std::any::Any;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};
use vyd::frame::layout::{CpuLayout, CpuPlaneLayout, FrameSpec};
use vyd::frame::{CpuFrame, CpuStorageShared};
use vyd::frame_metadata::TypedFrameMetadata;
use vyd::statistics::{MemoryOperation, PipelineStatistics};
use vyd::{Frame, FrameContext, FrameTiming, VydError};

const META: &str = "VydSerialFrameContextMeta";

pub(super) struct ContextData {
    metadata: HashMap<String, Box<dyn Any + Send + Sync>>,
    typed_metadata: TypedFrameMetadata,
    index: usize,
    timing: FrameTiming,
    pub started: Instant,
}

/// Metadata available to a native output annotation policy. Edits to typed
/// values use the same copy-on-write isolation as ordinary frame metadata.
pub struct NativeFrameAnnotations<'a> {
    metadata: &'a mut HashMap<String, Box<dyn Any + Send + Sync>>,
    typed: &'a mut TypedFrameMetadata,
}

impl NativeFrameAnnotations<'_> {
    pub fn get_typed<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.typed.get::<T>()
    }
    pub fn put_typed<T: Any + Send + Sync>(&mut self, value: T) {
        self.typed.insert(value);
    }
    pub fn remove_typed<T: Any + Send + Sync>(&mut self) -> Option<Arc<T>> {
        self.typed.remove::<T>()
    }
    pub fn remove_named(&mut self, key: &str) -> bool {
        self.metadata.remove(key).is_some()
    }
}

impl ContextData {
    pub(super) fn annotations(&mut self) -> NativeFrameAnnotations<'_> {
        NativeFrameAnnotations {
            metadata: &mut self.metadata,
            typed: &mut self.typed_metadata,
        }
    }
}

// Consume-once transport is deliberately private to a linear graph. It is not
// branchable metadata: copying a GST buffer does not clone arbitrary Any values.
#[derive(Clone, glib::Boxed)]
#[boxed_type(name = "VydSerialContextEnvelope")]
struct Envelope(Arc<Mutex<Option<ContextData>>>);

pub(super) fn register() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| gst::meta::CustomMeta::register(META, &[]));
}

pub(super) fn take_context(buffer: &gst::BufferRef) -> Result<ContextData, VydError> {
    let meta = gst::meta::CustomMeta::from_buffer(buffer, META).map_err(|_| {
        error("buffer has no Vyd frame context; a native section may have stripped metadata and needs a correspondence bridge")
    })?;
    let envelope = meta.structure().get::<Envelope>("context").map_err(error)?;
    let context = envelope
        .0
        .lock()
        .map_err(|_| error("context envelope poisoned"))?
        .take()
        .ok_or_else(|| {
            error("context already consumed; branching is not supported by this adapter")
        });
    context
}

pub(super) fn attach_context(
    buffer: &mut gst::BufferRef,
    data: ContextData,
) -> Result<(), VydError> {
    validate_no_self_buffer_owner(buffer, &data)?;
    let mut meta = match gst::meta::CustomMeta::from_mut_buffer(buffer, META) {
        Ok(meta) => meta,
        Err(_) => gst::meta::CustomMeta::add(buffer, META).map_err(error)?,
    };
    meta.mut_structure()
        .set("context", Envelope(Arc::new(Mutex::new(Some(data)))));
    Ok(())
}

// A generic Any value can hide a frame in an application-defined wrapper. We
// reject direct pixel owners and identifiable references to this very buffer;
// stage authors must keep other wrappers free of the attached buffer handle.
fn validate_no_self_buffer_owner(
    buffer: &gst::BufferRef,
    data: &ContextData,
) -> Result<(), VydError> {
    let attached_ptr = (buffer.n_memory() == 1)
        .then(|| {
            buffer
                .peek_memory(0)
                .map_readable()
                .ok()
                .map(|map| map.as_ptr())
        })
        .flatten();
    fn same_cpu(cpu: &CpuFrame, attached_ptr: Option<*const u8>) -> bool {
        attached_ptr.is_some_and(|ptr| cpu.img.as_raw().as_ptr() == ptr)
    }
    fn same_frame(frame: &Frame, attached_ptr: Option<*const u8>) -> bool {
        matches!(frame, Frame::Cpu(cpu) if same_cpu(cpu, attached_ptr))
    }
    fn owns_attached_buffer(value: &dyn Any, attached_ptr: Option<*const u8>) -> bool {
        value.is::<Frame>()
            || value.is::<CpuFrame>()
            || value.is::<FrameContext>()
            || value
                .downcast_ref::<Arc<Frame>>()
                .is_some_and(|frame| same_frame(frame, attached_ptr))
            || value
                .downcast_ref::<Arc<CpuFrame>>()
                .is_some_and(|cpu| same_cpu(cpu, attached_ptr))
            || value
                .downcast_ref::<Arc<FrameContext>>()
                .is_some_and(|ctx| same_frame(&ctx.frame, attached_ptr))
    }
    if data
        .metadata
        .values()
        .any(|value| owns_attached_buffer(value.as_ref(), attached_ptr))
        || data.typed_metadata.get::<Frame>().is_some()
        || data.typed_metadata.get::<CpuFrame>().is_some()
        || data.typed_metadata.get::<FrameContext>().is_some()
        || data
            .typed_metadata
            .get::<Arc<Frame>>()
            .is_some_and(|frame| same_frame(frame, attached_ptr))
        || data
            .typed_metadata
            .get::<Arc<CpuFrame>>()
            .is_some_and(|cpu| same_cpu(cpu, attached_ptr))
        || data
            .typed_metadata
            .get::<Arc<FrameContext>>()
            .is_some_and(|ctx| same_frame(&ctx.frame, attached_ptr))
    {
        return Err(error("attached frame metadata cannot own its own pixel buffer; store a distinct derived value instead"));
    }
    Ok(())
}

pub(super) fn info(frame: &Frame) -> Result<gst_video::VideoInfo, VydError> {
    let Frame::Cpu(cpu) = frame else {
        return Err(error("serial GST adapter requires a CPU Luma8 frame"));
    };
    let (width, height) = cpu.img.dimensions();
    let spec = FrameSpec::new(width, height, vyd::frame::PixelFormat::Luma8).map_err(error)?;
    let size = CpuLayout::packed(spec).map_err(error)?.region_lengths()[0];
    if width > i32::MAX as u32 || height > i32::MAX as u32 || cpu.img.as_raw().len() != size {
        return Err(error("invalid packed Luma8 frame layout"));
    }
    gst_video::VideoInfo::builder(gst_video::VideoFormat::Gray8, width, height)
        .build()
        .map_err(error)
}

pub(super) fn pack(
    frame: FrameContext,
    started: Instant,
    info: &gst_video::VideoInfo,
    stats: &PipelineStatistics,
) -> Result<gst::Buffer, VydError> {
    let actual = self::info(&frame.frame)?;
    if actual.width() != info.width() || actual.height() != info.height() {
        return Err(error(
            "unsupported midstream format change in serial GST adapter",
        ));
    }
    let FrameContext {
        frame,
        frame_index,
        metadata,
        typed_metadata,
        timing,
    } = frame;
    let Frame::Cpu(cpu) = frame else {
        unreachable!()
    };
    let width = info.width() as usize;
    let layout = video_layout(info)?;
    let mut buffer = if is_packed(info) && cpu.img.as_raw().len() == info.size() {
        // GStreamer cannot write this memory while a Vyd clone may still read
        // it. A native writer must negotiate or create its own writable output.
        let mut buffer = gst::Buffer::new();
        buffer
            .get_mut()
            .ok_or_else(|| error("new GST buffer is not writable"))?
            .append_memory(gst::Memory::from_slice(FrameBytes(cpu)));
        stats.record_memory_operation("gst-adapter/pack", MemoryOperation::ZeroCopyReference);
        buffer
    } else {
        // Initialize padding as well as visible pixels before exposing safe slices.
        let mut bytes = vec![0; info.size()];
        let mut regions = [bytes.as_mut_slice()];
        for (row, src) in cpu.img.as_raw().chunks_exact(width).enumerate() {
            layout
                .row_mut(&mut regions, 0, row)
                .map_err(error)?
                .copy_from_slice(src);
        }
        stats.record_memory_operation(
            "gst-adapter/pack",
            MemoryOperation::CpuAllocation(bytes.len()),
        );
        stats.record_memory_operation(
            "gst-adapter/pack",
            MemoryOperation::CpuCopy(width * info.height() as usize),
        );
        gst::Buffer::from_mut_slice(bytes)
    };
    let writable = buffer
        .get_mut()
        .ok_or_else(|| error("new GST buffer is not writable"))?;
    writable.set_offset(frame_index as u64);
    writable.set_pts(clock_time(timing.presentation_timestamp)?);
    writable.set_dts(clock_time(timing.decode_timestamp)?);
    writable.set_duration(clock_time(timing.duration)?);
    gst_video::VideoMeta::add(
        writable,
        gst_video::VideoFrameFlags::empty(),
        gst_video::VideoFormat::Gray8,
        info.width(),
        info.height(),
    )
    .map_err(error)?;
    attach_context(
        writable,
        ContextData {
            metadata,
            typed_metadata,
            index: frame_index,
            timing,
            started,
        },
    )?;
    Ok(buffer)
}

fn clock_time(time: Option<Duration>) -> Result<Option<gst::ClockTime>, VydError> {
    time.map(|time| {
        u64::try_from(time.as_nanos())
            .ok()
            .filter(|nanos| *nanos < u64::MAX)
            .map(gst::ClockTime::from_nseconds)
            .ok_or_else(|| error("frame timing exceeds GST clock range"))
    })
    .transpose()
}

pub(super) fn unpack(
    buffer: &gst::BufferRef,
    info: &gst_video::VideoInfo,
    stats: &PipelineStatistics,
) -> Result<(FrameContext, Instant), VydError> {
    validate_layout(buffer, info)?;
    let layout = video_layout(info)?;
    let width = info.width() as usize;
    let packed_bytes = width * info.height() as usize;
    let cpu = if is_packed(info) && buffer.n_memory() == 1 {
        let memory = buffer.peek_memory(0);
        let memory_map = memory.map_readable().map_err(error)?;
        // Keep the same GST miniobject referenced while Vyd exposes its
        // mapped bytes. BufferRef::to_owned() would make a shallow copy.
        let retained = unsafe { gst::Buffer::from_glib_none(buffer.as_ptr()) };
        let buffer_map = retained
            .into_mapped_buffer_readable()
            .map_err(|_| error("GST buffer cannot be mapped for a shared CPU read"))?;
        // A buffer map can coalesce or download storage. Share it only when
        // the single system-memory map aliases the actual GST memory bytes.
        // GstMappedBytes owns the readable map and thus a buffer reference;
        // another element cannot write that buffer in place while the frame
        // lives. GST must make a writable copy or replace its output memory.
        if memory.is_type("SystemMemory")
            && memory_map.as_ptr() == buffer_map.as_ptr()
            && buffer_map.len() == packed_bytes
        {
            stats.record_memory_operation("gst-adapter/unpack", MemoryOperation::ZeroCopyReference);
            CpuFrame::from_shared(
                info.width(),
                info.height(),
                GstMappedBytes(Arc::new(buffer_map)),
            )
            .map_err(error)?
        } else {
            copy_pixels(buffer_map.as_slice(), &layout, info, stats)?
        }
    } else {
        let map = buffer.map_readable().map_err(error)?;
        copy_pixels(map.as_slice(), &layout, info, stats)?
    };
    let data = take_context(buffer)?;
    Ok((
        FrameContext {
            frame: Frame::Cpu(cpu),
            frame_index: data.index,
            timing: data.timing,
            metadata: data.metadata,
            typed_metadata: data.typed_metadata,
        },
        data.started,
    ))
}

pub(super) fn validate_native_buffer(
    pad: &gst::Pad,
    buffer: &gst::BufferRef,
    dimensions: (u32, u32),
) -> Result<(), VydError> {
    let info = super::caps::negotiated_gray8(pad)?;
    if (info.width(), info.height()) != dimensions {
        return Err(error(format!(
            "native boundary expected {}x{} GRAY8, negotiated {}x{}",
            dimensions.0,
            dimensions.1,
            info.width(),
            info.height()
        )));
    }
    if !is_packed(&info) || info.size() != dimensions.0 as usize * dimensions.1 as usize {
        return Err(error(
            "native boundary requires zero-offset packed GRAY8 layout",
        ));
    }
    validate_layout(buffer, &info)?;
    if buffer.n_memory() != 1 || !buffer.peek_memory(0).is_type("SystemMemory") {
        return Err(error("native boundary requires one SystemMemory buffer"));
    }
    Ok(())
}

fn validate_layout(buffer: &gst::BufferRef, info: &gst_video::VideoInfo) -> Result<(), VydError> {
    if info.format() != gst_video::VideoFormat::Gray8 || buffer.size() != info.size() {
        return Err(error("unsupported GST buffer format or size"));
    }
    if let Some(video) = buffer.meta::<gst_video::VideoMeta>() {
        if video.format() != info.format()
            || video.width() != info.width()
            || video.height() != info.height()
            || video.stride() != info.stride()
            || video.offset() != info.offset()
        {
            return Err(error(
                "unsupported GST buffer layout or midstream format change",
            ));
        }
    }
    Ok(())
}

struct FrameBytes(CpuFrame);

impl AsRef<[u8]> for FrameBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.img.as_raw()
    }
}

struct GstMappedBytes(Arc<gst::MappedBuffer<gst::buffer::Readable>>);

impl Deref for GstMappedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CpuStorageShared for GstMappedBytes {
    fn clone_shared(&self) -> Box<dyn CpuStorageShared> {
        Box::new(Self(self.0.clone()))
    }

    fn try_into_owned(self: Box<Self>) -> Result<Vec<u8>, Box<dyn CpuStorageShared>> {
        Err(self)
    }
}

fn is_packed(info: &gst_video::VideoInfo) -> bool {
    info.offset()[0] == 0 && info.stride()[0] == info.width() as i32
}

fn copy_pixels(
    bytes: &[u8],
    layout: &CpuLayout,
    info: &gst_video::VideoInfo,
    stats: &PipelineStatistics,
) -> Result<CpuFrame, VydError> {
    let width = info.width() as usize;
    let mut pixels = Vec::with_capacity(width * info.height() as usize);
    let regions = [bytes];
    for row in 0..info.height() as usize {
        pixels.extend_from_slice(layout.row(&regions, 0, row).map_err(error)?);
    }
    stats.record_memory_operation(
        "gst-adapter/unpack",
        MemoryOperation::CpuAllocation(pixels.len()),
    );
    stats.record_memory_operation("gst-adapter/unpack", MemoryOperation::CpuCopy(pixels.len()));
    CpuFrame::from_vec(info.width(), info.height(), pixels).map_err(error)
}

fn video_layout(info: &gst_video::VideoInfo) -> Result<CpuLayout, VydError> {
    let spec = FrameSpec::new(info.width(), info.height(), vyd::frame::PixelFormat::Luma8)
        .map_err(error)?;
    let stride = usize::try_from(info.stride()[0])
        .map_err(|_| error("GST Luma8 row stride must be positive"))?;
    CpuLayout::new(
        spec,
        &[CpuPlaneLayout::new(0, info.offset()[0], stride)],
        &[info.size()],
    )
    .map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vyd::frame::factory::FrameFactory;
    use vyd::frame::pool::{BufferPoolManager, PoolLimits, WaitPolicy};

    struct DropValue(Arc<AtomicUsize>);

    impl Drop for DropValue {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct DerivedFrame(Frame);

    #[test]
    fn context_meta_copy_is_consume_once_and_releases_values_after_last_buffer() {
        gst::init().unwrap();
        register();
        register();
        let drops = Arc::new(AtomicUsize::new(0));
        let mut context = FrameContext::new(Frame::from_raw(4, 2, vec![1; 8]).unwrap(), 31);
        context.timing = FrameTiming {
            presentation_timestamp: Some(Duration::from_millis(40)),
            decode_timestamp: Some(Duration::from_millis(30)),
            duration: Some(Duration::from_millis(10)),
        };
        context.put_typed(DropValue(drops.clone()));
        let info = info(&context.frame).unwrap();
        let stats = PipelineStatistics::new();
        let buffer = pack(context, Instant::now(), &info, &stats).unwrap();
        assert_eq!(buffer.pts(), Some(gst::ClockTime::from_mseconds(40)));
        assert_eq!(buffer.dts(), Some(gst::ClockTime::from_mseconds(30)));
        assert_eq!(buffer.duration(), Some(gst::ClockTime::from_mseconds(10)));
        let copy = buffer.copy();
        assert!(gst::meta::CustomMeta::from_buffer(&copy, META).is_ok());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let (restored, _) = unpack(&copy, &info, &stats).unwrap();
        assert_eq!(restored.frame_index, 31);
        assert_eq!(
            restored.timing.presentation_timestamp,
            Some(Duration::from_millis(40))
        );
        assert_eq!(restored.frame.as_cpu_ref().unwrap(), &[1; 8]);
        assert!(take_context(&buffer).is_err());
        drop(copy);
        drop(buffer);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(restored);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn attached_metadata_rejects_direct_frame_owners_but_allows_distinct_derived_frames() {
        gst::init().unwrap();
        register();
        let frame = Frame::from_raw(4, 2, vec![1; 8]).unwrap();
        let info = info(&frame).unwrap();
        let stats = PipelineStatistics::new();
        let mut direct = FrameContext::new(frame.clone(), 0);
        direct.put_typed(frame.clone());
        assert!(pack(direct, Instant::now(), &info, &stats).is_err());
        let mut legacy = FrameContext::new(frame.clone(), 0);
        legacy.put("pixel_owner", frame.clone());
        assert!(pack(legacy, Instant::now(), &info, &stats).is_err());
        let mut same_arc = FrameContext::new(frame.clone(), 0);
        same_arc.put("pixel_owner", Arc::new(frame.clone()));
        assert!(pack(same_arc, Instant::now(), &info, &stats).is_err());
        let mut allowed = FrameContext::new(frame, 0);
        allowed.put_typed(DerivedFrame(Frame::from_raw(4, 2, vec![2; 8]).unwrap()));
        allowed.put(
            "distinct_median",
            Arc::new(Frame::from_raw(4, 2, vec![3; 8]).unwrap()),
        );
        let buffer = pack(allowed, Instant::now(), &info, &stats).unwrap();
        let (restored, _) = unpack(&buffer, &info, &stats).unwrap();
        assert_eq!(
            restored
                .get_typed::<DerivedFrame>()
                .unwrap()
                .0
                .as_cpu_ref()
                .unwrap(),
            &[2; 8]
        );
        assert_eq!(
            restored
                .get_as::<Arc<Frame>>("distinct_median")
                .as_cpu_ref()
                .unwrap(),
            &[3; 8]
        );
    }

    #[test]
    fn abandoned_context_meta_drops_its_annotation_without_a_pixel_cycle() {
        gst::init().unwrap();
        register();
        let drops = Arc::new(AtomicUsize::new(0));
        let frame = Frame::from_raw(4, 2, vec![1; 8]).unwrap();
        let info = info(&frame).unwrap();
        let mut context = FrameContext::new(frame, 0);
        context.put_typed(DropValue(drops.clone()));
        let buffer = pack(context, Instant::now(), &info, &PipelineStatistics::new()).unwrap();
        drop(buffer);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn gst_clock_range_rejects_overflow_and_reserved_none_value() {
        assert!(clock_time(Some(Duration::from_nanos(u64::MAX))).is_err());
        assert!(clock_time(Some(Duration::from_secs(u64::MAX))).is_err());
    }

    #[test]
    fn native_output_layout_rejects_padded_offset_and_short_buffers() {
        gst::init().unwrap();
        let info = gst_video::VideoInfo::builder(gst_video::VideoFormat::Gray8, 8, 2)
            .build()
            .unwrap();
        assert!(is_packed(&info));
        assert!(validate_layout(&gst::Buffer::from_mut_slice(vec![0; 15]), &info).is_err());
        for (offset, stride, size) in [(0, 9, 18), (1, 8, 17)] {
            let mut buffer = gst::Buffer::from_mut_slice(vec![0; size]);
            gst_video::VideoMeta::add_full(
                buffer.get_mut().unwrap(),
                gst_video::VideoFrameFlags::empty(),
                gst_video::VideoFormat::Gray8,
                8,
                2,
                &[offset],
                &[stride],
            )
            .unwrap();
            assert!(
                validate_layout(&buffer, &info).is_err(),
                "offset={offset} stride={stride}"
            );
        }
    }

    #[test]
    fn packed_gst_map_keeps_issuing_pool_lease_alive_and_reads_actual_pixels() {
        gst::init().unwrap();
        register();
        let pool = BufferPoolManager::new(PoolLimits::new(1, 8).unwrap());
        let factory = FrameFactory::new(pool.clone());
        let frame = factory
            .create_luma8_with(4, 2, WaitPolicy::Immediate, |bytes| {
                bytes.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
                Ok(())
            })
            .unwrap();
        let original_ptr = frame.as_cpu_ref().unwrap().as_ptr();
        let info = info(&frame).unwrap();
        assert!(is_packed(&info));
        let stats = PipelineStatistics::new();
        let buffer = pack(FrameContext::new(frame, 7), Instant::now(), &info, &stats).unwrap();
        assert_eq!(
            buffer.peek_memory(0).map_readable().unwrap().as_ptr(),
            original_ptr
        );
        assert!(buffer
            .peek_memory(0)
            .flags()
            .contains(gst::MemoryFlags::READONLY));
        assert_eq!(pool.snapshot().unwrap().active.len(), 1);
        let (unpacked, _) = unpack(&buffer, &info, &stats).unwrap();
        assert_eq!(unpacked.frame_index, 7);
        assert_eq!(
            unpacked.frame.as_cpu_ref().unwrap(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(unpacked.frame.as_cpu_ref().unwrap().as_ptr(), original_ptr);
        drop(buffer);
        assert_eq!(pool.snapshot().unwrap().active.len(), 1);
        drop(unpacked);
        assert_eq!(pool.snapshot().unwrap().idle_buffers, 1);
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["gst-adapter/pack_zero_copy"], 1);
        assert_eq!(memory["gst-adapter/unpack_zero_copy"], 1);
        assert!(!memory.contains_key("gst-adapter/unpack_cpu_copy_bytes"));
    }

    #[test]
    fn padded_gst_rows_are_copied_and_counted() {
        gst::init().unwrap();
        register();
        let frame = Frame::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let info = info(&frame).unwrap();
        assert!(!is_packed(&info));
        let stats = PipelineStatistics::new();
        let buffer = pack(FrameContext::new(frame, 1), Instant::now(), &info, &stats).unwrap();
        assert_eq!(
            buffer.map_readable().unwrap().as_slice(),
            &[1, 2, 3, 0, 4, 5, 6, 0]
        );
        let (unpacked, _) = unpack(&buffer, &info, &stats).unwrap();
        assert_eq!(unpacked.frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4, 5, 6]);
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["gst-adapter/pack_cpu_copy_bytes"], 6);
        assert_eq!(memory["gst-adapter/pack_cpu_alloc_bytes"], 8);
        assert_eq!(memory["gst-adapter/unpack_cpu_copy_bytes"], 6);
        assert_eq!(memory["gst-adapter/unpack_cpu_alloc_bytes"], 6);
    }

    #[test]
    fn split_gst_memory_uses_counted_fallback_even_for_packed_caps() {
        gst::init().unwrap();
        register();
        let frame = Frame::from_raw(4, 2, vec![1; 8]).unwrap();
        let info = info(&frame).unwrap();
        let stats = PipelineStatistics::new();
        let mut buffer = pack(FrameContext::new(frame, 2), Instant::now(), &info, &stats).unwrap();
        let writable = buffer.make_mut();
        writable.replace_all_memory(gst::Memory::from_mut_slice(vec![2; 4]));
        writable.append_memory(gst::Memory::from_mut_slice(vec![3; 4]));
        assert_eq!(writable.n_memory(), 2);
        let (unpacked, _) = unpack(&buffer, &info, &stats).unwrap();
        assert_eq!(
            unpacked.frame.as_cpu_ref().unwrap(),
            &[2, 2, 2, 2, 3, 3, 3, 3]
        );
        let memory = stats.snapshot().memory_operations;
        assert_eq!(memory["gst-adapter/unpack_cpu_copy_bytes"], 8);
        assert_eq!(memory["gst-adapter/unpack_cpu_alloc_bytes"], 8);
        assert!(!memory.contains_key("gst-adapter/unpack_zero_copy"));
    }

    #[test]
    fn mapped_frame_write_detaches_from_gst_output() {
        gst::init().unwrap();
        register();
        let frame = Frame::from_raw(4, 2, vec![1; 8]).unwrap();
        let info = info(&frame).unwrap();
        let stats = PipelineStatistics::new();
        let buffer = pack(FrameContext::new(frame, 3), Instant::now(), &info, &stats).unwrap();
        let (mut unpacked, _) = unpack(&buffer, &info, &stats).unwrap();
        let original_ptr = unpacked.frame.as_cpu_ref().unwrap().as_ptr();
        let mut write = unpacked
            .frame
            .try_write_cpu_luma8(WaitPolicy::Immediate)
            .unwrap();
        write.try_bytes_mut().unwrap()[0] = 9;
        drop(write);
        assert_ne!(unpacked.frame.as_cpu_ref().unwrap().as_ptr(), original_ptr);
        assert_eq!(unpacked.frame.as_cpu_ref().unwrap()[0], 9);
        assert_eq!(buffer.map_readable().unwrap().as_slice(), &[1; 8]);
    }
}
