use crate::CancellationToken;
use crate::frame::PixelFormat;
use dashmap::DashMap;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The stable pipeline, device, or pool that owns an operation's resources.
/// Do not use a fresh ID for every frame: that would make statistics grow
/// without bound. Individual lease IDs belong in pool diagnostics and traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceOwnerId(pub u64);

/// A physical transfer or a requested-format conversion. Cache hits do not
/// create an attempt. The location names where conversion bytes are processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameOperationKind {
    Upload(PixelFormat),
    Download(PixelFormat),
    Conversion {
        from: PixelFormat,
        to: PixelFormat,
        location: ConversionLocation,
    },
    PlanePacking {
        format: PixelFormat,
        direction: PackingDirection,
    },
    GpuCopy(PixelFormat),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionLocation {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackingDirection {
    Pack,
    Unpack,
}

impl FrameOperationKind {
    fn label(self) -> String {
        fn format_label(format: PixelFormat) -> &'static str {
            match format {
                PixelFormat::Luma8 => "luma8",
                PixelFormat::Nv12 => "nv12",
            }
        }
        match self {
            Self::Upload(format) => format!("upload_{}", format_label(format)),
            Self::Download(format) => format!("download_{}", format_label(format)),
            Self::GpuCopy(format) => format!("gpu_copy_{}", format_label(format)),
            Self::PlanePacking { format, direction } => format!(
                "{}_{}",
                match direction {
                    PackingDirection::Pack => "pack",
                    PackingDirection::Unpack => "unpack",
                },
                format_label(format)
            ),
            Self::Conversion { from, to, location } => format!(
                "convert_{}_{}_to_{}",
                match location {
                    ConversionLocation::Cpu => "cpu",
                    ConversionLocation::Gpu => "gpu",
                },
                format_label(from),
                format_label(to)
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, PartialEq, Eq)]
pub struct FrameOperationSnapshot {
    pub attempted: u64,
    pub failed: u64,
    pub completed: u64,
    /// Visible pixel payload processed by completed operations only.
    pub visible_bytes: u64,
    /// Physical staging bytes processed by completed operations only.
    pub staging_bytes: u64,
    /// Bytes physically copied across the transfer or texture-copy boundary.
    pub transfer_bytes: u64,
    pub wait_count: u64,
    pub wait_time_us: u64,
    pub owner_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameOperationBytes {
    pub visible: u64,
    pub staging: u64,
    pub transferred: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FrameOperationKey {
    kind: FrameOperationKind,
    owner: Option<ResourceOwnerId>,
}

type FrameOperationMap = DashMap<FrameOperationKey, Arc<Mutex<FrameOperationSnapshot>>>;

/// An attempted operation. Dropping it without `complete` records failure,
/// including early validation errors, cancellation, and unwinding. A caller
/// must complete only after bytes have actually transferred or converted.
#[must_use = "complete the operation or let it record a failure on drop"]
pub struct FrameOperationAttempt {
    counters: Arc<Mutex<FrameOperationSnapshot>>,
    completed: bool,
}

impl FrameOperationAttempt {
    pub fn record_wait(&mut self, duration: Duration) {
        let mut counters = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        counters.wait_count = counters.wait_count.saturating_add(1);
        counters.wait_time_us = counters
            .wait_time_us
            .saturating_add(duration.as_micros().min(u64::MAX as u128) as u64);
    }

    pub fn complete(mut self, bytes: FrameOperationBytes) {
        let mut counters = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        counters.completed = counters.completed.saturating_add(1);
        counters.visible_bytes = counters.visible_bytes.saturating_add(bytes.visible);
        counters.staging_bytes = counters.staging_bytes.saturating_add(bytes.staging);
        counters.transfer_bytes = counters.transfer_bytes.saturating_add(bytes.transferred);
        self.completed = true;
    }
}

impl Drop for FrameOperationAttempt {
    fn drop(&mut self) {
        if !self.completed {
            let mut counters = self.counters.lock().unwrap_or_else(|e| e.into_inner());
            counters.failed = counters.failed.saturating_add(1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageTimingStats {
    pub total_time_us: Arc<AtomicU64>,
    pub call_count: Arc<AtomicU64>,
    pub min_time_us: Arc<AtomicU64>,
    pub max_time_us: Arc<AtomicU64>,
    // Store individual measurements for percentile calculations
    pub measurements: Arc<Mutex<Vec<u64>>>,
}

impl Default for StageTimingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ProcessingType {
    Cpu,
    Gpu,
}

impl ProcessingType {
    fn as_str(&self) -> &'static str {
        match self {
            ProcessingType::Cpu => "CPU",
            ProcessingType::Gpu => "GPU",
        }
    }
}

#[derive(Debug)]
pub enum MemoryOperation {
    ZeroCopyReference,
    GpuDownload(usize),
    CpuAllocation(usize),
    CpuZero(usize),
    GpuUpload(usize),
    CpuCopy(usize),
}

// Helper struct for enhanced stage data
#[derive(Debug)]
struct EnhancedStageData {
    #[allow(unused)]
    name: String,
    display_name: String,
    stats: StageTimingStats,
    median_time_us: f64,
    processing_type: Option<ProcessingType>,
    memory_summary: String,
}

impl StageTimingStats {
    pub fn new() -> Self {
        Self {
            total_time_us: Arc::new(AtomicU64::new(0)),
            call_count: Arc::new(AtomicU64::new(0)),
            min_time_us: Arc::new(AtomicU64::new(u64::MAX)),
            max_time_us: Arc::new(AtomicU64::new(0)),
            measurements: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn record_timing(&self, duration_us: u64) {
        self.total_time_us.fetch_add(duration_us, Ordering::Relaxed);
        self.call_count.fetch_add(1, Ordering::Relaxed);

        // Store individual measurement for percentile calculations
        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.push(duration_us);
        }

        // Update min
        loop {
            let current_min = self.min_time_us.load(Ordering::Relaxed);
            if duration_us >= current_min {
                break;
            }
            if self
                .min_time_us
                .compare_exchange_weak(
                    current_min,
                    duration_us,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        // Update max
        loop {
            let current_max = self.max_time_us.load(Ordering::Relaxed);
            if duration_us <= current_max {
                break;
            }
            if self
                .max_time_us
                .compare_exchange_weak(
                    current_max,
                    duration_us,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    pub fn get_avg_time_us(&self) -> f64 {
        let total = self.total_time_us.load(Ordering::Relaxed);
        let count = self.call_count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Calculate percentile (0.0 to 1.0) of measurements
    /// Returns None if there are no measurements
    pub fn get_percentile(&self, percentile: f64) -> Option<f64> {
        if let Ok(mut measurements) = self.measurements.lock() {
            if measurements.is_empty() {
                return None;
            }

            measurements.sort_unstable();
            let index = (percentile * (measurements.len() - 1) as f64).round() as usize;
            Some(measurements[index] as f64)
        } else {
            None
        }
    }

    /// Calculate trimmed mean excluding outliers outside the given percentile range
    /// e.g., trimmed_mean(0.05, 0.95) excludes bottom 5% and top 5%
    pub fn get_trimmed_mean(&self, lower_percentile: f64, upper_percentile: f64) -> Option<f64> {
        if let Ok(mut measurements) = self.measurements.lock() {
            if measurements.is_empty() {
                return None;
            }

            measurements.sort_unstable();
            let len = measurements.len();
            let lower_index = ((lower_percentile * (len - 1) as f64).round() as usize).min(len - 1);
            let upper_index = ((upper_percentile * (len - 1) as f64).round() as usize).min(len - 1);

            if lower_index >= upper_index {
                return Some(measurements[lower_index] as f64);
            }

            let trimmed_slice = &measurements[lower_index..=upper_index];
            let sum: u64 = trimmed_slice.iter().sum();
            Some(sum as f64 / trimmed_slice.len() as f64)
        } else {
            None
        }
    }

    /// Get median (50th percentile)
    pub fn get_median(&self) -> Option<f64> {
        self.get_percentile(0.5)
    }
}

#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    pub start_time: Instant,
    pub frame_count: Arc<AtomicU64>,
    pub total_processing_time_us: Arc<AtomicU64>,
    pub stage_timings: Arc<DashMap<String, StageTimingStats>>,
    cancellation: CancellationToken,

    pub stage_processing_types: Arc<DashMap<String, ProcessingType>>,
    pub memory_operations: Arc<DashMap<String, Arc<AtomicU64>>>,
    frame_operations: Arc<DashMap<String, Arc<FrameOperationMap>>>,
}

#[derive(Debug, serde::Serialize)]
pub struct PipelineStatisticsSnapshot {
    pub elapsed_us: u64,
    pub frame_count: u64,
    pub total_processing_time_us: u64,
    pub stages: std::collections::BTreeMap<String, StageStatisticsSnapshot>,
    pub memory_operations: std::collections::BTreeMap<String, u64>,
    /// Omitted when empty so CPU-only exports retain their existing JSON form.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub frame_operations: BTreeMap<String, BTreeMap<String, FrameOperationSnapshot>>,
}

#[derive(Debug, serde::Serialize)]
pub struct StageStatisticsSnapshot {
    pub call_count: u64,
    pub total_time_us: u64,
    pub min_time_us: Option<u64>,
    pub max_time_us: Option<u64>,
    pub median_time_us: Option<f64>,
}

impl Default for PipelineStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStatistics {
    /// Capture counters for export. Values may advance during an active run.
    pub fn snapshot(&self) -> PipelineStatisticsSnapshot {
        PipelineStatisticsSnapshot {
            elapsed_us: self.start_time.elapsed().as_micros() as u64,
            frame_count: self.frame_count.load(Ordering::Relaxed),
            total_processing_time_us: self.total_processing_time_us.load(Ordering::Relaxed),
            stages: self
                .stage_timings
                .iter()
                .map(|entry| {
                    let stats = entry.value();
                    let calls = stats.call_count.load(Ordering::Relaxed);
                    (
                        entry.key().clone(),
                        StageStatisticsSnapshot {
                            call_count: calls,
                            total_time_us: stats.total_time_us.load(Ordering::Relaxed),
                            min_time_us: (calls > 0)
                                .then(|| stats.min_time_us.load(Ordering::Relaxed)),
                            max_time_us: (calls > 0)
                                .then(|| stats.max_time_us.load(Ordering::Relaxed)),
                            median_time_us: stats.get_median(),
                        },
                    )
                })
                .collect(),
            memory_operations: self
                .memory_operations
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
                .collect(),
            frame_operations: self
                .frame_operations
                .iter()
                .map(|stage| {
                    let operations = stage
                        .value()
                        .iter()
                        .map(|entry| {
                            let key = entry.key();
                            let label = match key.owner {
                                Some(owner) => format!("{}@owner_{}", key.kind.label(), owner.0),
                                None => key.kind.label(),
                            };
                            let counters = *entry.value().lock().unwrap_or_else(|e| e.into_inner());
                            (label, counters)
                        })
                        .collect();
                    (stage.key().clone(), operations)
                })
                .collect(),
        }
    }

    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            frame_count: Arc::new(AtomicU64::new(0)),
            total_processing_time_us: Arc::new(AtomicU64::new(0)),
            stage_timings: Arc::new(DashMap::new()),
            cancellation: CancellationToken::new(),
            stage_processing_types: Arc::new(DashMap::new()),
            memory_operations: Arc::new(DashMap::new()),
            frame_operations: Arc::new(DashMap::new()),
        }
    }

    /// Begin one physical operation. Keep the returned ticket alive until it
    /// succeeds; dropping it records a failed attempt without completed bytes.
    pub fn begin_frame_operation(
        &self,
        stage: &str,
        kind: FrameOperationKind,
        owner: Option<ResourceOwnerId>,
    ) -> FrameOperationAttempt {
        let stage_operations = self
            .frame_operations
            .entry(stage.to_string())
            .or_insert_with(|| Arc::new(DashMap::new()))
            .clone();
        let counters = stage_operations
            .entry(FrameOperationKey { kind, owner })
            .or_insert_with(|| {
                Arc::new(Mutex::new(FrameOperationSnapshot {
                    owner_id: owner.map(|owner| owner.0),
                    ..FrameOperationSnapshot::default()
                }))
            })
            .clone();
        let mut snapshot = counters.lock().unwrap_or_else(|e| e.into_inner());
        snapshot.attempted = snapshot.attempted.saturating_add(1);
        drop(snapshot);
        FrameOperationAttempt {
            counters,
            completed: false,
        }
    }

    pub fn register_stage(&self, stage_name: &str, processing_type: ProcessingType) {
        self.stage_processing_types
            .insert(stage_name.to_string(), processing_type);
        tracing::debug!(
            "Registered stage '{}' with processing type {:?}",
            stage_name,
            processing_type
        );
    }

    pub fn record_memory_operation(&self, stage_name: &str, operation: MemoryOperation) {
        let operation_key = match operation {
            MemoryOperation::ZeroCopyReference => format!("{}_zero_copy", stage_name),
            MemoryOperation::GpuDownload(bytes) => {
                let key = format!("{}_gpu_download_bytes", stage_name);
                self.memory_operations
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                format!("{}_gpu_download_count", stage_name)
            }
            MemoryOperation::CpuAllocation(bytes) => {
                let key = format!("{}_cpu_alloc_bytes", stage_name);
                self.memory_operations
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                format!("{}_cpu_alloc_count", stage_name)
            }
            MemoryOperation::CpuZero(bytes) => {
                let key = format!("{}_cpu_zero_bytes", stage_name);
                self.memory_operations
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                format!("{}_cpu_zero_count", stage_name)
            }
            MemoryOperation::GpuUpload(bytes) => {
                let key = format!("{}_gpu_upload_bytes", stage_name);
                self.memory_operations
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                format!("{}_gpu_upload_count", stage_name)
            }
            MemoryOperation::CpuCopy(bytes) => {
                let key = format!("{}_cpu_copy_bytes", stage_name);
                self.memory_operations
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                format!("{}_cpu_copy_count", stage_name)
            }
        };

        self.memory_operations
            .entry(operation_key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_frame(&self, processing_time_us: u64) {
        self.frame_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_us
            .fetch_add(processing_time_us, Ordering::Relaxed);
    }

    pub fn record_stage_timing(&self, stage_name: &str, timing_us: u64) {
        let stats = self
            .stage_timings
            .entry(stage_name.to_string())
            .or_default();
        stats.record_timing(timing_us);
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    #[deprecated(note = "request cancellation through Pipeline::cancellation_token")]
    pub fn stop(&self) {
        self.cancellation.cancel();
    }

    #[deprecated(note = "query the pipeline cancellation token")]
    pub fn is_running(&self) -> bool {
        !self.cancellation.is_cancelled()
    }

    /// Extract the unqualified name from a fully qualified stage name
    /// e.g., "vyd_source_gstreamer::GstSource" -> "GstSource"
    fn extract_stage_name(full_name: &str) -> &str {
        if let Some(pos) = full_name.rfind("::") {
            &full_name[pos + 2..]
        } else {
            full_name
        }
    }

    /// Get memory operation summary for a stage
    fn get_memory_summary(&self, stage_name: &str) -> String {
        let mut operations = Vec::new();

        // Check for zero-copy operations
        if let Some(zero_copy_count) = self
            .memory_operations
            .get(&format!("{}_zero_copy", stage_name))
        {
            let count = zero_copy_count.load(Ordering::Relaxed);
            if count > 0 {
                operations.push(format!("{}×Zero", count));
            }
        }

        // Check for CPU allocations
        if let Some(alloc_count) = self
            .memory_operations
            .get(&format!("{}_cpu_alloc_count", stage_name))
        {
            let count = alloc_count.load(Ordering::Relaxed);
            if count > 0 {
                let bytes = self
                    .memory_operations
                    .get(&format!("{}_cpu_alloc_bytes", stage_name))
                    .map(|b| b.load(Ordering::Relaxed))
                    .unwrap_or(0);
                operations.push(format!("{}×CPU({})", count, Self::format_bytes(bytes)));
            }
        }

        if let Some(zero_count) = self
            .memory_operations
            .get(&format!("{}_cpu_zero_count", stage_name))
        {
            let count = zero_count.load(Ordering::Relaxed);
            if count > 0 {
                let bytes = self
                    .memory_operations
                    .get(&format!("{}_cpu_zero_bytes", stage_name))
                    .map(|b| b.load(Ordering::Relaxed))
                    .unwrap_or(0);
                operations.push(format!("{}×Clear({})", count, Self::format_bytes(bytes)));
            }
        }

        // Check for GPU downloads
        if let Some(dl_count) = self
            .memory_operations
            .get(&format!("{}_gpu_download_count", stage_name))
        {
            let count = dl_count.load(Ordering::Relaxed);
            if count > 0 {
                let bytes = self
                    .memory_operations
                    .get(&format!("{}_gpu_download_bytes", stage_name))
                    .map(|b| b.load(Ordering::Relaxed))
                    .unwrap_or(0);
                operations.push(format!("{}×GPU↓({})", count, Self::format_bytes(bytes)));
            }
        }

        // Check for GPU uploads
        if let Some(ul_count) = self
            .memory_operations
            .get(&format!("{}_gpu_upload_count", stage_name))
        {
            let count = ul_count.load(Ordering::Relaxed);
            if count > 0 {
                let bytes = self
                    .memory_operations
                    .get(&format!("{}_gpu_upload_bytes", stage_name))
                    .map(|b| b.load(Ordering::Relaxed))
                    .unwrap_or(0);
                operations.push(format!("{}×GPU↑({})", count, Self::format_bytes(bytes)));
            }
        }

        // Check for CPU copies
        if let Some(copy_count) = self
            .memory_operations
            .get(&format!("{}_cpu_copy_count", stage_name))
        {
            let count = copy_count.load(Ordering::Relaxed);
            if count > 0 {
                let bytes = self
                    .memory_operations
                    .get(&format!("{}_cpu_copy_bytes", stage_name))
                    .map(|b| b.load(Ordering::Relaxed))
                    .unwrap_or(0);
                operations.push(format!("{}×Copy({})", count, Self::format_bytes(bytes)));
            }
        }

        // Managed frame transfers use frame-operation tickets rather than the
        // older memory-operation counters. Show their logical payload here so
        // GPU boundary and processor rows do not appear transfer-free.
        if let Some(frame_operations) = self.frame_operations.get(stage_name) {
            let mut uploaded = (0_u64, 0_u64);
            let mut downloaded = (0_u64, 0_u64);
            for operation in frame_operations.iter() {
                let counters = *operation
                    .value()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let totals = match operation.key().kind {
                    FrameOperationKind::Upload(_) => &mut uploaded,
                    FrameOperationKind::Download(_) => &mut downloaded,
                    _ => continue,
                };
                totals.0 = totals.0.saturating_add(counters.completed);
                totals.1 = totals.1.saturating_add(counters.visible_bytes);
            }
            let legacy_upload = self
                .memory_operations
                .get(&format!("{}_gpu_upload_count", stage_name))
                .is_some_and(|count| count.load(Ordering::Relaxed) > 0);
            let legacy_download = self
                .memory_operations
                .get(&format!("{}_gpu_download_count", stage_name))
                .is_some_and(|count| count.load(Ordering::Relaxed) > 0);
            if uploaded.0 > 0 && !legacy_upload {
                operations.push(format!(
                    "{}×GPU↑({})",
                    uploaded.0,
                    Self::format_bytes(uploaded.1)
                ));
            }
            if downloaded.0 > 0 && !legacy_download {
                operations.push(format!(
                    "{}×GPU↓({})",
                    downloaded.0,
                    Self::format_bytes(downloaded.1)
                ));
            }
        }

        if operations.is_empty() {
            "-".to_string()
        } else {
            operations.join(" ")
        }
    }

    /// Enhanced stage data structure for table rendering
    fn get_enhanced_stage_data(
        &self,
    ) -> (
        Vec<EnhancedStageData>,
        Vec<EnhancedStageData>,
        Vec<EnhancedStageData>,
        Vec<EnhancedStageData>,
    ) {
        let mut external_stages = Vec::new();
        let mut boundary_timings = Vec::new();
        let mut background_timings = Vec::new();
        let mut pipeline_stages = Vec::new();

        for entry in self.stage_timings.iter() {
            let stage_name = entry.key();
            let stats = entry.value().clone();
            let median_time_us = stats.get_median().unwrap_or(0.0);

            // External elements are not registered as Vyd CPU/GPU stages.
            // Leave their execution type unknown rather than implying CPU.
            let processing_type = self
                .stage_processing_types
                .get(stage_name)
                .map(|pt| *pt.value());

            // Get memory operations summary
            let memory_summary = self.get_memory_summary(stage_name);

            let enhanced_data = EnhancedStageData {
                name: stage_name.clone(),
                display_name: if let Some(label) = stage_name
                    .strip_prefix("external/")
                    .or_else(|| stage_name.strip_prefix("boundary/"))
                    .or_else(|| stage_name.strip_prefix("background/"))
                {
                    label.to_string()
                } else {
                    Self::extract_stage_name(stage_name).to_string()
                },
                stats,
                median_time_us,
                processing_type,
                memory_summary,
            };

            if stage_name.starts_with("external/") {
                external_stages.push(enhanced_data);
            } else if stage_name.starts_with("boundary/") {
                boundary_timings.push(enhanced_data);
            } else if stage_name.starts_with("background/") {
                background_timings.push(enhanced_data);
            } else {
                pipeline_stages.push(enhanced_data);
            }
        }

        // Sort by median time (descending)
        external_stages.sort_by(|a, b| b.median_time_us.partial_cmp(&a.median_time_us).unwrap());
        boundary_timings.sort_by(|a, b| b.median_time_us.partial_cmp(&a.median_time_us).unwrap());
        background_timings.sort_by(|a, b| b.median_time_us.partial_cmp(&a.median_time_us).unwrap());
        pipeline_stages.sort_by(|a, b| b.median_time_us.partial_cmp(&a.median_time_us).unwrap());

        (
            external_stages,
            boundary_timings,
            background_timings,
            pipeline_stages,
        )
    }

    fn print_stage_table(&self, title: &str, stages: &[EnhancedStageData]) {
        if stages.is_empty() {
            return;
        }

        println!("\n--- {} ---", title);
        println!(
            "{:<25} {:<4} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<50}",
            "Stage", "Type", "Calls", "p50", "p95", "Mean", "Min", "Max", "Memory Operations"
        );
        println!(
            "{:<25} {:<4} {:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<50}",
            "", "", "", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)", ""
        );
        println!("{}", "─".repeat(140));

        for stage_data in stages {
            let call_count = stage_data.stats.call_count.load(Ordering::Relaxed);
            let min_time_us = stage_data.stats.min_time_us.load(Ordering::Relaxed);
            let max_time_us = stage_data.stats.max_time_us.load(Ordering::Relaxed);
            let min_time_ms = if min_time_us == u64::MAX {
                0.0
            } else {
                min_time_us as f64 / 1000.0
            };
            let max_time_ms = max_time_us as f64 / 1000.0;
            let p50_ms = stage_data.stats.get_percentile(0.5).unwrap_or(0.0) / 1000.0;
            let p95_ms = stage_data.stats.get_percentile(0.95).unwrap_or(0.0) / 1000.0;
            let mean_ms = stage_data.stats.get_avg_time_us() / 1000.0;

            // Truncate memory summary if too long
            let memory_display = if stage_data.memory_summary.len() > 50 {
                format!(
                    "{}...",
                    stage_data
                        .memory_summary
                        .chars()
                        .take(47)
                        .collect::<String>()
                )
            } else {
                stage_data.memory_summary.clone()
            };

            println!(
                "{:<25} {:<4} {:<6} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<50}",
                stage_data.display_name,
                stage_data.processing_type.map_or("?", |kind| kind.as_str()),
                call_count,
                p50_ms,
                p95_ms,
                mean_ms,
                min_time_ms,
                max_time_ms,
                memory_display
            );
        }
    }

    pub fn print_final_stats(&self) {
        let total_duration = self.start_time.elapsed();
        let frames = self.frame_count.load(Ordering::Relaxed);
        let total_processing_us = self.total_processing_time_us.load(Ordering::Relaxed);

        println!("\n=== Final Pipeline Statistics ===");
        println!(
            "Wall runtime (including startup/shutdown): {:.2}s",
            total_duration.as_secs_f64()
        );
        println!("Frames processed: {}", frames);

        if frames > 0 {
            let avg_fps = frames as f64 / total_duration.as_secs_f64();
            let avg_frame_time_us = total_processing_us / frames;

            println!("Average FPS: {:.2}", avg_fps);
            println!(
                "Mean Vyd processing span per frame (after source): {:.2}ms",
                avg_frame_time_us as f64 / 1000.0
            );
        }

        // Print detailed stage-specific timings with enhanced information
        if !self.stage_timings.is_empty() {
            let (external_stages, boundary_timings, background_timings, pipeline_stages) =
                self.get_enhanced_stage_data();

            println!(
                "Stage times are per-call elapsed latencies; calls can overlap across frames. Type is declared by Vyd stages; ? means unclassified."
            );
            if !boundary_timings.is_empty() {
                println!(
                    "Boundary times include automatic GPU upload and readback outside processor callbacks."
                );
            }
            if !background_timings.is_empty() {
                println!(
                    "Background work runs outside frame callbacks and is not part of per-frame stage latency."
                );
            }
            self.print_stage_table("External Stage Timings", &external_stages);
            self.print_stage_table("Pipeline Boundary Timings", &boundary_timings);
            self.print_stage_table("Background Work", &background_timings);
            self.print_stage_table("Pipeline Stages", &pipeline_stages);

            // Print memory operation legend
            println!("\nMemory Operation Legend:");
            println!("  Zero    = Zero-copy reference");
            println!("  CPU(s)  = CPU allocation (size)");
            println!("  Clear(s) = recycled CPU buffer clearing (size)");
            println!("  GPU↓(s) = GPU download (size)");
            println!("  GPU↑(s) = GPU upload (size)");
            println!("  Copy(s) = CPU copy (size)");

            println!("{}", "─".repeat(140));
        }
        println!("=================================\n");
    }

    fn format_bytes(bytes: u64) -> String {
        if bytes == 0 {
            return "0B".to_string();
        }

        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{}B", bytes)
        } else {
            format!("{:.1}{}", size, UNITS[unit_index])
        }
    }
}

// Legacy fallback for callers outside managed run/invocation scopes. Pipeline
// execution never installs a global run: concurrent runs enter explicit scopes.
static GLOBAL_STATS: OnceLock<Arc<Mutex<Option<Arc<PipelineStatistics>>>>> = OnceLock::new();

std::thread_local! {
    static CURRENT_STAGE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static CURRENT_CONTEXT: std::cell::RefCell<Option<StatsContext>> = const { std::cell::RefCell::new(None) };
}

/// Owned run/stage attribution that can be captured and moved to a worker.
/// Enter it around synchronous work on that worker. It does not automatically
/// propagate through arbitrary thread spawns or async tasks.
#[derive(Clone)]
pub struct StatsContext {
    stats: Arc<PipelineStatistics>,
    stage: Option<Arc<str>>,
}

/// Restores the previous context on scope exit, including unwinding.
/// Guards must be nested and dropped in reverse order, on the entering thread.
/// Do not hold one across an async suspension; enter for each synchronous poll
/// or work item instead. The owned `StatsContext` is the transferable token.
#[must_use = "keep the guard alive for the duration of the attributed work"]
pub struct StatsScope {
    previous: Option<StatsContext>,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for StatsScope {
    fn drop(&mut self) {
        CURRENT_CONTEXT.with(|current| current.replace(self.previous.take()));
    }
}

impl StatsContext {
    pub fn new(stats: Arc<PipelineStatistics>) -> Self {
        Self { stats, stage: None }
    }

    pub fn for_stage(&self, stage: impl Into<Arc<str>>) -> Self {
        Self {
            stats: self.stats.clone(),
            stage: Some(stage.into()),
        }
    }

    /// Keep a caller's explicit attribution. A pipeline-wide service can
    /// instead borrow the current synchronous stage scope for this operation.
    #[cfg(feature = "gpu-wgpu")]
    pub(crate) fn with_current_stage(&self) -> Self {
        if self.stage.is_some() {
            self.clone()
        } else {
            Self::capture().unwrap_or_else(|| self.clone())
        }
    }

    /// Explicit attribution token for work that may move between threads or
    /// async polls. An unqualified token records under the pipeline stage.
    pub fn begin_frame_operation(
        &self,
        kind: FrameOperationKind,
        owner: Option<ResourceOwnerId>,
    ) -> FrameOperationAttempt {
        self.stats
            .begin_frame_operation(self.stage.as_deref().unwrap_or("pipeline"), kind, owner)
    }

    /// Convenience for synchronous stage code. Capture an explicit context
    /// before moving work to another worker or across an async suspension.
    pub fn begin_current_frame_operation(
        kind: FrameOperationKind,
        owner: Option<ResourceOwnerId>,
    ) -> Option<FrameOperationAttempt> {
        if let Some(context) = Self::capture() {
            return Some(context.begin_frame_operation(kind, owner));
        }
        let stage = CURRENT_STAGE.with(|name| name.borrow().clone())?;
        Self::with_stats(|stats| stats.begin_frame_operation(&stage, kind, owner))
    }

    pub fn enter(&self) -> StatsScope {
        StatsScope {
            previous: CURRENT_CONTEXT.with(|current| current.replace(Some(self.clone()))),
            _thread_bound: std::marker::PhantomData,
        }
    }

    /// Capture explicit attribution for a child worker. No scope means no token.
    pub fn capture() -> Option<Self> {
        CURRENT_CONTEXT.with(|current| current.borrow().clone())
    }

    /// Legacy process-global fallback. Prefer `new(...).enter()` for run isolation.
    pub fn set_global_stats(stats: Arc<PipelineStatistics>) {
        let global = GLOBAL_STATS.get_or_init(|| Arc::new(Mutex::new(None)));
        if let Ok(mut guard) = global.lock() {
            *guard = Some(stats);
        }
    }

    pub fn clear_global_stats() {
        let global = GLOBAL_STATS.get_or_init(|| Arc::new(Mutex::new(None)));
        if let Ok(mut guard) = global.lock() {
            *guard = None;
        }
    }

    pub fn with_stats<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&PipelineStatistics) -> R,
    {
        if let Some(context) = Self::capture() {
            return Some(f(&context.stats));
        }
        let global = GLOBAL_STATS.get_or_init(|| Arc::new(Mutex::new(None)));
        let stats = global.lock().ok()?.clone()?;
        // Release the registry lock before invoking caller code (which may
        // enter another scope or call with_stats recursively).
        Some(f(&stats))
    }

    /// Set stage attribution for the legacy fallback only. Explicit scopes take
    /// precedence; use `for_stage(...).enter()` inside a managed run.
    pub fn set_current_stage(stage_name: String) {
        CURRENT_STAGE.with(|name| {
            *name.borrow_mut() = Some(stage_name);
        });
    }

    pub fn clear_current_stage() {
        CURRENT_STAGE.with(|name| {
            *name.borrow_mut() = None;
        });
    }

    pub fn record_memory_operation(operation: MemoryOperation) {
        if let Some(context) = Self::capture() {
            if let Some(stage) = context.stage {
                context.stats.record_memory_operation(&stage, operation);
            }
            return;
        }
        let stage = CURRENT_STAGE.with(|name| name.borrow().clone());
        if let Some(stage) = stage {
            Self::with_stats(|stats| stats.record_memory_operation(&stage, operation));
        }
    }
}
