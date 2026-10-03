use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

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
    processing_type: ProcessingType,
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
    pub is_running: Arc<AtomicBool>,

    pub stage_processing_types: Arc<DashMap<String, ProcessingType>>,
    pub memory_operations: Arc<DashMap<String, Arc<AtomicU64>>>,
}

impl Default for PipelineStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStatistics {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            frame_count: Arc::new(AtomicU64::new(0)),
            total_processing_time_us: Arc::new(AtomicU64::new(0)),
            stage_timings: Arc::new(DashMap::new()),
            is_running: Arc::new(AtomicBool::new(true)),
            stage_processing_types: Arc::new(DashMap::new()),
            memory_operations: Arc::new(DashMap::new()),
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
            .or_insert_with(StageTimingStats::new);
        stats.record_timing(timing_us);
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
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

        if operations.is_empty() {
            "-".to_string()
        } else {
            operations.join(" ")
        }
    }

    /// Enhanced stage data structure for table rendering
    fn get_enhanced_stage_data(&self) -> (Vec<EnhancedStageData>, Vec<EnhancedStageData>) {
        let mut external_stages = Vec::new();
        let mut pipeline_stages = Vec::new();

        for entry in self.stage_timings.iter() {
            let stage_name = entry.key();
            let stats = entry.value().clone();
            let median_time_us = stats.get_median().unwrap_or(0.0);

            // Get processing type
            let processing_type = self
                .stage_processing_types
                .get(stage_name)
                .map(|pt| *pt.value())
                .unwrap_or(ProcessingType::Cpu);

            // Get memory operations summary
            let memory_summary = self.get_memory_summary(stage_name);

            let enhanced_data = EnhancedStageData {
                name: stage_name.clone(),
                display_name: if stage_name.starts_with("external/") {
                    stage_name
                        .strip_prefix("external/")
                        .unwrap_or(stage_name)
                        .to_string()
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
            } else {
                pipeline_stages.push(enhanced_data);
            }
        }

        // Sort by median time (descending)
        external_stages.sort_by(|a, b| b.median_time_us.partial_cmp(&a.median_time_us).unwrap());
        pipeline_stages.sort_by(|a, b| b.median_time_us.partial_cmp(&a.median_time_us).unwrap());

        (external_stages, pipeline_stages)
    }

    fn print_stage_table(
        &self,
        title: &str,
        stages: &[EnhancedStageData],
        total_stage_time_us: u64,
    ) {
        if stages.is_empty() {
            return;
        }

        println!("\n--- {} ---", title);
        println!(
            "{:<25} {:<4} {:<6} {:<8} {:<8} {:<8} {:<8} {:<10} {:<8} {:<25}",
            "Stage",
            "Type",
            "Calls",
            "Median",
            "95%Mean",
            "Min",
            "Max",
            "Total",
            "%Pipe",
            "Memory Operations"
        );
        println!(
            "{:<25} {:<4} {:<6} {:<8} {:<8} {:<8} {:<8} {:<10} {:<8} {:<25}",
            "", "", "", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)", "", ""
        );
        println!("{}", "─".repeat(145));

        for stage_data in stages {
            let call_count = stage_data.stats.call_count.load(Ordering::Relaxed);
            let min_time_us = stage_data.stats.min_time_us.load(Ordering::Relaxed);
            let max_time_us = stage_data.stats.max_time_us.load(Ordering::Relaxed);
            let total_time_us = stage_data.stats.total_time_us.load(Ordering::Relaxed);

            let min_time_ms = if min_time_us == u64::MAX {
                0.0
            } else {
                min_time_us as f64 / 1000.0
            };
            let max_time_ms = max_time_us as f64 / 1000.0;
            let total_time_ms = total_time_us as f64 / 1000.0;

            // Calculate robust statistics
            let median_ms = stage_data.stats.get_median().unwrap_or(0.0) / 1000.0;
            let trimmed_mean_ms = stage_data
                .stats
                .get_trimmed_mean(0.025, 0.975)
                .unwrap_or(0.0)
                / 1000.0;

            let percentage = if total_stage_time_us > 0 {
                (total_time_us as f64 / total_stage_time_us as f64) * 100.0
            } else {
                0.0
            };

            // Truncate memory summary if too long
            let memory_display = if stage_data.memory_summary.len() > 25 {
                format!("{}...", &stage_data.memory_summary[..22])
            } else {
                stage_data.memory_summary.clone()
            };

            println!(
                "{:<25} {:<4} {:<6} {:<8.2} {:<8.2} {:<8.2} {:<8.2} {:<10.1} {:<7.1}% {:<25}",
                stage_data.display_name,
                stage_data.processing_type.as_str(),
                call_count,
                median_ms,
                trimmed_mean_ms,
                min_time_ms,
                max_time_ms,
                total_time_ms,
                percentage,
                memory_display
            );
        }
    }

    pub fn print_final_stats(&self) {
        let total_duration = self.start_time.elapsed();
        let frames = self.frame_count.load(Ordering::Relaxed);
        let total_processing_us = self.total_processing_time_us.load(Ordering::Relaxed);

        println!("\n=== Final Pipeline Statistics ===");
        println!("Total runtime: {:.2}s", total_duration.as_secs_f64());
        println!("Frames processed: {}", frames);

        if frames > 0 {
            let avg_fps = frames as f64 / total_duration.as_secs_f64();
            let avg_frame_time_us = total_processing_us / frames;

            println!("Average FPS: {:.2}", avg_fps);
            println!(
                "Average frame processing time: {:.2}ms",
                avg_frame_time_us as f64 / 1000.0
            );

            // Calculate processing overhead vs. total runtime
            let processing_ratio =
                (total_processing_us as f64 / 1_000_000.0) / total_duration.as_secs_f64();
            println!(
                "Processing time ratio: {:.1}% of total runtime",
                processing_ratio * 100.0
            );
        }

        // Print detailed stage-specific timings with enhanced information
        if !self.stage_timings.is_empty() {
            let (external_stages, pipeline_stages) = self.get_enhanced_stage_data();

            let total_stage_time_us: u64 = self
                .stage_timings
                .iter()
                .map(|entry| entry.value().total_time_us.load(Ordering::Relaxed))
                .sum();

            self.print_stage_table(
                "External Stage Timings",
                &external_stages,
                total_stage_time_us,
            );
            self.print_stage_table("Pipeline Stages", &pipeline_stages, total_stage_time_us);

            // Print memory operation legend
            println!("\nMemory Operation Legend:");
            println!("  Zero    = Zero-copy reference");
            println!("  CPU(s)  = CPU allocation (size)");
            println!("  GPU↓(s) = GPU download (size)");
            println!("  GPU↑(s) = GPU upload (size)");
            println!("  Copy(s) = CPU copy (size)");

            println!("{}", "─".repeat(145));
            println!(
                "Total stage processing time: {:.1}ms",
                total_stage_time_us as f64 / 1000.0
            );

            if frames > 0 {
                let total_wall_clock_us = (total_duration.as_secs_f64() * 1_000_000.0) as u64;
                let pipeline_overhead_us = total_wall_clock_us.saturating_sub(total_stage_time_us);
                let avg_overhead_per_frame = pipeline_overhead_us as f64 / frames as f64 / 1000.0;

                println!(
                    "Average pipeline overhead per frame: {:.2}ms",
                    avg_overhead_per_frame
                );
            }
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

// Global statistics context
static GLOBAL_STATS: OnceLock<Arc<Mutex<Option<Arc<PipelineStatistics>>>>> = OnceLock::new();

// Properly declare thread-local storage
std::thread_local! {
    static CURRENT_STAGE: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

pub struct StatsContext;

impl StatsContext {
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
        let global = GLOBAL_STATS.get_or_init(|| Arc::new(Mutex::new(None)));
        if let Ok(guard) = global.lock() {
            if let Some(ref stats) = *guard {
                Some(f(stats))
            } else {
                None
            }
        } else {
            None
        }
    }

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
        CURRENT_STAGE.with(|name| {
            if let Some(ref stage_name) = *name.borrow() {
                Self::with_stats(|stats| {
                    stats.record_memory_operation(stage_name, operation);
                });
            }
        });
    }
}
