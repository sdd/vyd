/// Represents a type that can be dumped to parquet for offline analysis
pub trait Dumpable: Send + Sync + 'static {
    /// Flat row type suitable for Parquet export
    type Row: for<'a> serde::Deserialize<'a> + serde::Serialize + Send + Sync;

    /// Identifier fragment used for output filenames
    const TABLE_NAME: &'static str;

    /// Schema version (bump if layout of Row changes)
    const VERSION: u32;

    /// Convert into a row, always including run_id and frame_index
    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row;
}
