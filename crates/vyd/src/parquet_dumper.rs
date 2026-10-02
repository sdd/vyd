use crate::traits::Dumpable;
use anyhow::Result;
use arrow_schema::{Field, FieldRef, Schema as ArrowSchema};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub struct ParquetDumper<T: Dumpable> {
    writer: Option<parquet::arrow::ArrowWriter<File>>,
    run_id: u64,
    buffer: Vec<T::Row>,
    file: Option<File>,
    fields: Option<Vec<Arc<Field>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Dumpable> ParquetDumper<T>
where
    T::Row: serde::Serialize + Send + Sync,
{
    pub fn new<P: AsRef<Path>>(path: P, run_id: u64) -> Result<Self> {
        let file = File::create(path)?;

        Ok(Self {
            writer: None,
            run_id,
            buffer: Vec::new(),
            file: Some(file),
            fields: None,
            _phantom: Default::default(),
        })
    }

    fn infer_schema_from_data(&mut self) -> Result<()> {
        if self.fields.is_none() && !self.buffer.is_empty() {
            self.fields = Some(Vec::<FieldRef>::from_type::<T::Row>(
                TracingOptions::default(),
            )?);
        }
        Ok(())
    }

    fn ensure_writer(&mut self) -> Result<()> {
        if self.writer.is_none() {
            self.infer_schema_from_data()?;

            if let Some(ref fields) = self.fields {
                let schema = Arc::new(ArrowSchema::new(fields.clone()));
                let file = self.file.take().unwrap();
                let writer = parquet::arrow::ArrowWriter::try_new(file, schema, None)?;
                self.writer = Some(writer);
            }
        }
        Ok(())
    }

    pub fn dump(&mut self, item: &T, frame_index: usize) -> Result<()> {
        let row = item.to_row(self.run_id, frame_index);
        self.buffer.push(row);

        // Write in batches for efficiency
        if self.buffer.len() >= 1000 {
            self.flush_buffer()?;
        }

        Ok(())
    }

    pub fn dumps<'a>(
        &mut self,
        items: impl Iterator<Item = &'a T>,
        frame_index: usize,
    ) -> Result<()> {
        items.for_each(|item| {
            let row = item.to_row(self.run_id, frame_index);
            self.buffer.push(row);
        });

        // Write in batches for efficiency
        if self.buffer.len() >= 1000 {
            self.flush_buffer()?;
        }

        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        self.ensure_writer()?;

        if let Some(ref mut writer) = self.writer {
            if let Some(ref fields) = self.fields {
                let batch = serde_arrow::to_record_batch(fields.as_slice(), &self.buffer)?;
                writer.write(&batch)?;
            }
        }

        self.buffer.clear();
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.flush_buffer()?;
        if let Some(ref mut writer) = self.writer {
            writer.flush()?;
        }
        Ok(())
    }

    pub fn close(&mut self) -> Result<()> {
        self.flush()?;

        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }

        Ok(())
    }
}

impl<T: Dumpable> Drop for ParquetDumper<T>
where
    T::Row: serde::Serialize + Send + Sync,
{
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("Error flushing ParquetDumper on drop: {}", e);
        }
        if let Err(e) = self.close() {
            eprintln!("Error closing ParquetDumper on drop: {}", e);
        }
    }
}
