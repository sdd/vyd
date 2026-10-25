//! Immutable per-frame values with cheap sharing and copy-on-write map edits.
//! A value attached to a GST buffer must not retain that same buffer's pixel
//! storage, including through application-defined wrappers. A distinct derived
//! frame, such as a median image, is a valid annotation.
//! Mutable cross-frame services belong in `PipelineContext` instead.
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

type Value = Arc<dyn Any + Send + Sync>;

#[derive(Clone, Default)]
pub struct TypedFrameMetadata {
    values: Arc<HashMap<TypeId, Value>>,
}

impl TypedFrameMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the value of this type in this frame's metadata map.
    /// Existing clones retain their previous map and values.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) {
        Arc::make_mut(&mut self.values).insert(TypeId::of::<T>(), Arc::new(value));
    }

    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.values
            .get(&TypeId::of::<T>())?
            .as_ref()
            .downcast_ref::<T>()
    }

    /// Keep an immutable value beyond a borrow of the metadata map.
    pub fn get_shared<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::downcast(self.values.get(&TypeId::of::<T>())?.clone()).ok()
    }

    pub fn remove<T: Any + Send + Sync>(&mut self) -> Option<Arc<T>> {
        Arc::downcast(Arc::make_mut(&mut self.values).remove(&TypeId::of::<T>())?).ok()
    }
}
