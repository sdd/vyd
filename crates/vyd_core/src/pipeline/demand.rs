use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Aggregate demand for payloads identified by their Rust type.
#[derive(Default)]
pub(super) struct PayloadDemand {
    counts: Mutex<HashMap<TypeId, usize>>,
}

impl PayloadDemand {
    pub(super) fn request<T: 'static>(self: &Arc<Self>) -> PayloadRequest<T> {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *counts.entry(TypeId::of::<T>()).or_default() += 1;
        PayloadRequest {
            demand: Arc::clone(self),
            payload: PhantomData,
        }
    }

    pub(super) fn is_requested<T: 'static>(&self) -> bool {
        self.counts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains_key(&TypeId::of::<T>())
    }
}

/// A stage's request for a payload type. Dropping it withdraws only this request.
///
/// Keep the handle while the stage needs the payload. It is deliberately not
/// cloneable: each call to [`PipelineContext::request`](super::PipelineContext::request)
/// represents one independent requester.
#[must_use = "dropping the handle immediately withdraws the payload request"]
pub struct PayloadRequest<T: 'static> {
    demand: Arc<PayloadDemand>,
    payload: PhantomData<fn() -> T>,
}

impl<T: 'static> Drop for PayloadRequest<T> {
    fn drop(&mut self) {
        let mut counts = self
            .demand
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let payload = TypeId::of::<T>();
        if let Some(count) = counts.get_mut(&payload) {
            if *count > 1 {
                *count -= 1;
            } else {
                counts.remove(&payload);
            }
        }
    }
}
