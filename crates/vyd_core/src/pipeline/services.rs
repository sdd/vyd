use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::VydError;

type Values = HashMap<TypeId, Arc<dyn Any + Send + Sync>>;

/// Shareable pipeline services and latest published values.
///
/// There is one slot per concrete Rust type. Use distinct newtypes when several
/// values have the same representation. Clones share the registry; independent
/// pipelines start with separate registries. Reads return owned `Arc`s, so a
/// replacement cannot invalidate an existing reader or hold up publication.
///
/// Publish immutable values for feedback snapshots. Explicitly mutable services
/// must supply their own synchronization (for example, `Mutex<Service>`).
/// Separate reads are not an atomic snapshot of all slots: publish one aggregate
/// when several values must belong to the same version.
#[derive(Clone, Default)]
pub struct PipelineServices {
    values: Arc<RwLock<Values>>,
}

impl PipelineServices {
    /// Publish a value, returning the previous value of the same type, if any.
    /// User destructors run after the registry lock has been released.
    pub fn publish<T: Any + Send + Sync>(&self, value: Arc<T>) -> Result<Option<Arc<T>>, VydError> {
        let previous = self
            .values
            .write()
            .map_err(|_| VydError::PipelineServicesPoisoned)?
            .insert(TypeId::of::<T>(), value);
        Ok(previous.map(|value| {
            value
                .downcast::<T>()
                .expect("service slot type matches its TypeId")
        }))
    }

    /// Read the latest value of a type without retaining a registry lock.
    pub fn latest<T: Any + Send + Sync>(&self) -> Result<Option<Arc<T>>, VydError> {
        let value = self
            .values
            .read()
            .map_err(|_| VydError::PipelineServicesPoisoned)?
            .get(&TypeId::of::<T>())
            .cloned();
        Ok(value.map(|value| {
            value
                .downcast::<T>()
                .expect("service slot type matches its TypeId")
        }))
    }
}
