use crate::clean::{clean_batch, clean_text, CleanConfig};
use crate::dedup::{DedupConfig, Deduplicator};
use pyo3::prelude::*;

/// Fast text cleaning (HTML tag removal, entity decoding, unicode norm)
/// Returns an empty string if the text is filtered out (too short/long)
#[pyfunction]
pub fn fast_clean_text(text: String) -> String {
    let config = CleanConfig::default();
    clean_text(&text, &config).unwrap_or_default()
}

/// Fast batched text cleaning
#[pyfunction]
pub fn fast_clean_batch(texts: Vec<String>) -> Vec<String> {
    let config = CleanConfig::default();
    clean_batch(texts, &config)
}

/// Deduplicator utilizing MinHash + LSH with Rayon parallel processing
#[pyclass(name = "Deduplicator", module = "hbllm_data_tools_rs")]
pub struct PyDeduplicator {
    inner: Deduplicator,
}

#[pymethods]
impl PyDeduplicator {
    #[new]
    #[pyo3(signature = (num_perm=64, num_bands=8, threshold=0.5, shingle_size=3))]
    pub fn new(num_perm: usize, num_bands: usize, threshold: f64, shingle_size: usize) -> Self {
        let config = DedupConfig {
            num_perm,
            num_bands,
            threshold,
            shingle_size,
        };
        Self {
            inner: Deduplicator::new(config),
        }
    }

    /// Run the full deduplication pipeline on a list of texts.
    /// Returns a list of unique texts.
    pub fn deduplicate(&self, texts: Vec<String>) -> Vec<String> {
        self.inner.deduplicate(texts)
    }
}
