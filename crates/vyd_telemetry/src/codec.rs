use anyhow::{Context, anyhow};
use arrow_array::RecordBatch;
use arrow_schema::FieldRef;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_arrow::schema::{SchemaLike, TracingOptions};
use std::sync::Arc;
use vyd_core::caps::ProcessorPortSpecs;
use vyd_core::{FrameContext, FrameProcessorResult, VydError};

/// A processor-owned description of its metadata output. `Row` must serialize
/// as a struct whose fields become independent Vortex columns.
pub trait TypedStageCodec: Default + Send + Sync + 'static {
    type Row: Serialize + DeserializeOwned + Send + Sync + 'static;
    const VERSION: u32;

    fn capture(
        &self,
        frame: &FrameContext,
        result: FrameProcessorResult,
    ) -> Result<Self::Row, VydError>;
    fn restore(
        &self,
        frame: &mut FrameContext,
        result: FrameProcessorResult,
        row: Self::Row,
    ) -> Result<(), VydError>;

    fn ports(&self) -> ProcessorPortSpecs {
        ProcessorPortSpecs::default()
    }
}

pub trait StageCodec: Send + Sync {
    fn version(&self) -> u32;
    fn fields(&self) -> Result<Vec<FieldRef>, VydError>;
    fn capture_batch(
        &self,
        frame: &FrameContext,
        result: FrameProcessorResult,
    ) -> Result<RecordBatch, VydError>;
    fn restore_batch(
        &self,
        frame: &mut FrameContext,
        result: FrameProcessorResult,
        batch: &RecordBatch,
    ) -> Result<(), VydError>;
    fn ports(&self) -> ProcessorPortSpecs;
}

struct CodecAdapter<C: TypedStageCodec> {
    codec: C,
    fields: Vec<FieldRef>,
}

impl<C: TypedStageCodec> CodecAdapter<C> {
    fn new() -> Result<Self, VydError> {
        let fields = Vec::<FieldRef>::from_type::<C::Row>(TracingOptions::default())
            .context("derive stage codec schema")?;
        if fields.is_empty() {
            return Err(VydError::PersistenceError(anyhow!(
                "output row must have at least one named field"
            )));
        }
        Ok(Self {
            codec: C::default(),
            fields,
        })
    }
}

impl<C: TypedStageCodec> StageCodec for CodecAdapter<C> {
    fn version(&self) -> u32 {
        C::VERSION
    }

    fn fields(&self) -> Result<Vec<FieldRef>, VydError> {
        Ok(self.fields.clone())
    }

    fn capture_batch(
        &self,
        frame: &FrameContext,
        result: FrameProcessorResult,
    ) -> Result<RecordBatch, VydError> {
        let row = self.codec.capture(frame, result)?;
        Ok(serde_arrow::to_record_batch(&self.fields, &[row])
            .context("serialize stage codec output")?)
    }

    fn restore_batch(
        &self,
        frame: &mut FrameContext,
        result: FrameProcessorResult,
        batch: &RecordBatch,
    ) -> Result<(), VydError> {
        let mut rows: Vec<C::Row> =
            serde_arrow::from_record_batch(batch).context("deserialize stage codec output")?;
        if rows.len() != 1 {
            return Err(VydError::PersistenceError(anyhow!(
                "expected one output row, got {}",
                rows.len()
            )));
        }
        self.codec.restore(frame, result, rows.remove(0))
    }

    fn ports(&self) -> ProcessorPortSpecs {
        self.codec.ports()
    }
}

pub trait StageCodecFactory: Send + Sync {
    fn stage_type(&self) -> &'static str;
    fn create(&self) -> Result<Arc<dyn StageCodec>, VydError>;
}

vyd_core::inventory::collect!(&'static dyn StageCodecFactory);

pub fn codec_for(stage_type: &str) -> Result<Option<Arc<dyn StageCodec>>, VydError> {
    vyd_core::inventory::iter::<&dyn StageCodecFactory>()
        .find(|factory| factory.stage_type() == stage_type)
        .map(|factory| factory.create())
        .transpose()
}

/// Register a typed codec for the exact stage type used in TOML.
#[macro_export]
macro_rules! register_vyd_stage_codec {
    ($codec:ty, $stage_type:literal) => {
        const _: () = {
            struct RegisteredCodec;
            impl $crate::StageCodecFactory for RegisteredCodec {
                fn stage_type(&self) -> &'static str {
                    $stage_type
                }

                fn create(
                    &self,
                ) -> Result<std::sync::Arc<dyn $crate::StageCodec>, $crate::VydError> {
                    Ok(std::sync::Arc::new($crate::codec::new_adapter::<$codec>()?))
                }
            }
            static REGISTERED: RegisteredCodec = RegisteredCodec;
            $crate::inventory::submit! {
                &REGISTERED as &'static dyn $crate::StageCodecFactory
            }
        };
    };
}

pub fn new_adapter<C: TypedStageCodec>() -> Result<impl StageCodec, VydError> {
    CodecAdapter::<C>::new()
}
