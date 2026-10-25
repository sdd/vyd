use std::sync::Arc;
use std::time::Duration;
use vyd_core::frame::PixelFormat;
use vyd_core::statistics::{
    ConversionLocation, FrameOperationBytes, FrameOperationKind, PackingDirection,
    PipelineStatistics, ResourceOwnerId, StatsContext,
};

#[test]
fn failed_attempts_do_not_count_completed_bytes() {
    let stats = Arc::new(PipelineStatistics::new());
    let context = StatsContext::new(stats.clone()).for_stage("decoder");
    let owner = Some(ResourceOwnerId(7));
    let mut failed =
        context.begin_frame_operation(FrameOperationKind::Upload(PixelFormat::Nv12), owner);
    failed.record_wait(Duration::from_micros(15));
    drop(failed);

    let key = "upload_nv12@owner_7";
    let operation = stats.snapshot().frame_operations["decoder"][key];
    assert_eq!(
        (operation.attempted, operation.failed, operation.completed),
        (1, 1, 0)
    );
    assert_eq!(
        (
            operation.visible_bytes,
            operation.staging_bytes,
            operation.transfer_bytes
        ),
        (0, 0, 0)
    );
    assert_eq!((operation.wait_count, operation.wait_time_us), (1, 15));
    assert_eq!(operation.owner_id, Some(7));

    context
        .begin_frame_operation(FrameOperationKind::Upload(PixelFormat::Nv12), owner)
        .complete(FrameOperationBytes {
            visible: 6,
            staging: 256,
            transferred: 16,
        });
    let operation = stats.snapshot().frame_operations["decoder"][key];
    assert_eq!(
        (operation.attempted, operation.failed, operation.completed),
        (2, 1, 1)
    );
    assert_eq!(
        (
            operation.visible_bytes,
            operation.staging_bytes,
            operation.transfer_bytes
        ),
        (6, 256, 16)
    );
}

#[test]
fn logical_conversion_and_physical_packing_have_separate_counters() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone())
        .for_stage("convert")
        .enter();
    StatsContext::begin_current_frame_operation(
        FrameOperationKind::Conversion {
            from: PixelFormat::Luma8,
            to: PixelFormat::Nv12,
            location: ConversionLocation::Cpu,
        },
        None,
    )
    .unwrap()
    .complete(FrameOperationBytes {
        visible: 6,
        ..Default::default()
    });
    StatsContext::begin_current_frame_operation(
        FrameOperationKind::PlanePacking {
            format: PixelFormat::Nv12,
            direction: PackingDirection::Pack,
        },
        None,
    )
    .unwrap()
    .complete(FrameOperationBytes {
        visible: 6,
        staging: 256,
        transferred: 0,
    });
    let operations = &stats.snapshot().frame_operations["convert"];
    assert_eq!(operations["convert_cpu_luma8_to_nv12"].visible_bytes, 6);
    assert_eq!(operations["pack_nv12"].staging_bytes, 256);
    assert_eq!(operations.len(), 2);
}

#[test]
fn cpu_only_statistics_keep_the_legacy_json_shape() {
    let stats = PipelineStatistics::new();
    let value = serde_json::to_value(stats.snapshot()).unwrap();
    assert!(value.get("frame_operations").is_none());
    for key in [
        "elapsed_us",
        "frame_count",
        "total_processing_time_us",
        "stages",
        "memory_operations",
    ] {
        assert!(value.get(key).is_some(), "missing {key}");
    }
}
