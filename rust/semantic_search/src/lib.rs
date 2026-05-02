#![allow(non_local_definitions)]
//! HBLLM Semantic Search — fast cosine similarity and TF-IDF encoding.
//!
//! Provides:
//! - Batch cosine similarity scoring with optional SIMD
//! - TF-IDF vocabulary building and encoding
//! - Content hashing (MD5) for deduplication

use md5::{Digest, Md5};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\w+\b").unwrap());

// ── Cosine Similarity ───────────────────────────────────────────────────────

fn cosine_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute cosine similarity between a query vector and a matrix of document vectors.
/// Returns a Vec of similarity scores.
fn cosine_similarity_batch(query: &[f64], matrix: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let query_norm = norm(query);
    if query_norm == 0.0 {
        return vec![0.0; n_rows];
    }

    (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = &matrix[i * n_cols..(i + 1) * n_cols];
            let row_norm = norm(row);
            if row_norm == 0.0 {
                0.0
            } else {
                cosine_dot(query, row) / (query_norm * row_norm + 1e-9)
            }
        })
        .collect()
}

// ── TF-IDF ──────────────────────────────────────────────────────────────────

#[pyclass]
struct TfIdfEncoder {
    vocab: HashMap<String, usize>,
    idf: HashMap<String, f64>,
    doc_freqs: HashMap<String, usize>,
    doc_count: usize,
}

#[pymethods]
impl TfIdfEncoder {
    #[new]
    fn new() -> Self {
        TfIdfEncoder {
            vocab: HashMap::new(),
            idf: HashMap::new(),
            doc_freqs: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Add a document to the vocabulary and update IDF.
    fn fit_token(&mut self, text: &str) {
        let tokens: Vec<String> = WORD_RE
            .find_iter(&text.to_lowercase())
            .map(|m| m.as_str().to_string())
            .collect();

        // Update vocabulary
        for token in &tokens {
            let next_idx = self.vocab.len();
            self.vocab.entry(token.clone()).or_insert(next_idx);
        }

        // Update document frequencies (unique tokens per doc)
        let unique: std::collections::HashSet<&String> = tokens.iter().collect();
        for token in &unique {
            *self.doc_freqs.entry((*token).clone()).or_insert(0) += 1;
        }
        self.doc_count += 1;

        // Recompute IDF
        for (term, df) in &self.doc_freqs {
            self.idf.insert(
                term.clone(),
                ((self.doc_count as f64 + 1.0) / (*df as f64 + 1.0)).ln() + 1.0,
            );
        }
    }

    /// Encode a single text into a TF-IDF vector (returned as numpy array).
    fn encode_one<'py>(&self, py: Python<'py>, text: &str) -> &'py PyArray1<f64> {
        let dim = self.vocab.len().max(1);
        let mut vec = vec![0.0f64; dim];

        let tokens: Vec<String> = WORD_RE
            .find_iter(&text.to_lowercase())
            .map(|m| m.as_str().to_string())
            .collect();

        let total = tokens.len().max(1) as f64;

        // Compute TF
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }

        // TF-IDF
        for (term, count) in &tf {
            if let Some(&idx) = self.vocab.get(*term) {
                let idf = self.idf.get(*term).copied().unwrap_or(1.0);
                vec[idx] = (*count as f64 / total) * idf;
            }
        }

        // L2 normalize
        let n: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.0 {
            for v in &mut vec {
                *v /= n;
            }
        }

        PyArray1::from_vec(py, vec)
    }

    /// Get current vocabulary size.
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ── Python-Facing Functions ─────────────────────────────────────────────────

/// Batch cosine similarity: query (1D) × matrix (2D) → scores (1D).
#[pyfunction]
fn batch_cosine_similarity<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f64>,
    matrix: PyReadonlyArray2<f64>,
) -> &'py PyArray1<f64> {
    let q = query.as_slice().unwrap();
    let shape = matrix.shape();
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let mat = matrix.as_slice().unwrap();

    let scores = cosine_similarity_batch(q, mat, n_rows, n_cols);
    PyArray1::from_vec(py, scores)
}

/// MD5 content hash for deduplication.
#[pyfunction]
fn content_hash(text: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[pymodule]
fn hbllm_semantic_search(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TfIdfEncoder>()?;
    m.add_function(wrap_pyfunction!(batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(content_hash, m)?)?;
    Ok(())
}
