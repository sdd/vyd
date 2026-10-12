use std::sync::Arc;
use vyd::frame::{CpuStorage, SharedVec};
use vyd::statistics::{PipelineStatistics, StatsContext};

#[test]
fn nontransferable_storage_makes_one_counted_copy() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone()).for_stage("cow").enter();
    let backing: Arc<[u8]> = vec![1, 2, 3, 4].into();
    let source_ptr = backing.as_ptr();
    let storage = CpuStorage::from_shared(backing);
    let owned = storage.to_owned();
    assert_eq!(&*owned, &[1, 2, 3, 4]);
    assert_ne!(owned.as_ptr(), source_ptr);
    let memory = stats.snapshot().memory_operations;
    assert_eq!(memory["cow_cpu_copy_bytes"], 4);
    assert_eq!(memory["cow_cpu_copy_count"], 1);
    assert_eq!(memory["cow_cpu_alloc_bytes"], 4);
    assert!(!memory.contains_key("cow_zero_copy"));
}

#[test]
fn unique_vec_transfer_keeps_its_allocation_without_copy() {
    let stats = Arc::new(PipelineStatistics::new());
    let _scope = StatsContext::new(stats.clone()).for_stage("cow").enter();
    let values = vec![5, 6, 7];
    let pointer = values.as_ptr();
    let storage = CpuStorage::from_shared(SharedVec::new(values));
    let owned = storage.to_owned();
    assert_eq!(owned.as_ptr(), pointer);
    assert_eq!(&*owned, &[5, 6, 7]);
    let memory = stats.snapshot().memory_operations;
    assert_eq!(memory["cow_zero_copy"], 1);
    assert!(!memory.contains_key("cow_cpu_copy_count"));
}
