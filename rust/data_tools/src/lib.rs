//! HBLLM Data Tools — High-performance text processing.
//!
//! This crate provides fast, parallelized data processing tools for
//! preparing training data:
//!
//! - **Text Cleaning** (`clean`): HTML removal, Unicode normalization,
//!   whitespace normalization, control character removal, and length filtering.
//! - **Deduplication** (`dedup`): MinHash + Locality-Sensitive Hashing (LSH)
//!   for finding and removing near-duplicate documents at scale.
//!
//! These tools are called from the Python data pipeline but run in Rust
//! for 10-100× speedup on CPU-bound operations.
//!
//! ## Usage
//!
//! ```rust
//! use hbllm_data_tools::clean::{clean_text, CleanConfig};
//! use hbllm_data_tools::dedup::{Deduplicator, DedupConfig};
//!
//! // Clean text
//! let config = CleanConfig::default();
//! let cleaned = clean_text("<p>Hello &amp; world</p>", &config);
//!
//! // Deduplicate
//! let dedup = Deduplicator::new(DedupConfig::default());
//! let docs = vec!["doc one".to_string(), "doc one".to_string(), "doc two".to_string()];
//! let unique_docs = dedup.deduplicate(docs);
//! ```

pub mod clean;
pub mod dedup;
pub mod python;

use pyo3::prelude::*;

/// Python module entry point
#[pymodule]
fn hbllm_data_tools_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::fast_clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(python::fast_clean_batch, m)?)?;
    m.add_class::<python::PyDeduplicator>()?;
    Ok(())
}

// Re-export main types
pub use clean::{clean_batch, clean_text, CleanConfig};
pub use dedup::{DedupConfig, Deduplicator, MinHashSignature};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_clean_then_dedup_pipeline() {
        // Simulate a real pipeline: clean → deduplicate
        let raw_docs = vec![
            "<p>The quick brown fox jumps over the lazy dog in the sunny meadow today and every day</p>".to_string(),
            "<div>The quick brown fox jumps over the lazy dog in the sunny meadow today and every day</div>".to_string(), // Same content, different HTML
            "A completely different document about machine learning and artificial intelligence research papers".to_string(),
        ];

        // Step 1: Clean
        let config = CleanConfig {
            min_length: 10,
            max_length: 10000,
            ..Default::default()
        };
        let cleaned = clean_batch(raw_docs, &config);
        assert_eq!(cleaned.len(), 3); // All pass length filter

        // Step 2: Deduplicate
        let dedup = Deduplicator::new(DedupConfig {
            num_perm: 64,
            num_bands: 8,
            threshold: 0.5,
            shingle_size: 3,
        });
        let unique = dedup.deduplicate(cleaned);

        // The two fox documents should be deduplicated
        assert_eq!(unique.len(), 2);
    }
}
