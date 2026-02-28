use pyo3::prelude::*;
use std::path::Path;

use crate::trainer::{Trainer, TrainerConfig};
use crate::vocab::Vocab;

/// Python wrapper for the BPE Vocabulary
#[pyclass(name = "Vocab", module = "hbllm_tokenizer_rs")]
pub struct PyVocab {
    pub(crate) inner: Vocab,
}

#[pymethods]
impl PyVocab {
    #[new]
    pub fn new(vocab_size: u32) -> Self {
        Self {
            inner: Vocab::new(vocab_size),
        }
    }

    /// Encode text to a list of token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text.as_bytes())
    }

    /// Decode a list of token IDs to a string
    pub fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode_to_string(&ids)
    }

    /// Save vocab to a JSON file
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load vocab from a JSON file
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let vocab = Vocab::load(Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: vocab })
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Python wrapper for the BPE Trainer
#[pyclass(name = "Trainer", module = "hbllm_tokenizer_rs")]
pub struct PyTrainer {
    pub(crate) inner: Trainer,
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (vocab_size=32768, min_frequency=2))]
    pub fn new(vocab_size: u32, min_frequency: u32) -> Self {
        let mut config = TrainerConfig::default();
        config.vocab_size = vocab_size;
        config.min_frequency = min_frequency;
        Self {
            inner: Trainer::new(config),
        }
    }

    /// Train a new vocabulary from a block of text
    pub fn train_from_text(&self, text: &str) -> PyVocab {
        let vocab = self.inner.train_from_text(text);
        PyVocab { inner: vocab }
    }

    /// Train a new vocabulary from multiple files
    pub fn train_from_files(&self, files: Vec<String>) -> PyResult<PyVocab> {
        let paths: Vec<&Path> = files.iter().map(|s| Path::new(s)).collect();
        let vocab = self
            .inner
            .train_from_files(&paths)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyVocab { inner: vocab })
    }
}
