//! HBLLM Tokenizer — Custom byte-level BPE tokenizer.
//!
//! This crate implements a fast, byte-level Byte Pair Encoding (BPE) tokenizer
//! designed for the HBLLM project. It supports:
//!
//! - Byte-level BPE (no OOV tokens, handles any language/binary data)
//! - Custom special tokens for domain routing (`<|domain:code|>`, etc.)
//! - Parallel training on large corpora via rayon
//! - Serialization to/from JSON vocab files
//! - Encode/decode with learned merge rules
//!
//! ## Architecture
//!
//! The tokenizer uses the same algorithm as GPT-2/LLaMA tokenizers:
//! 1. Start with a base vocabulary of 256 byte tokens
//! 2. Iteratively merge the most frequent byte pairs
//! 3. Continue until reaching the target vocabulary size
//!
//! ## Usage
//!
//! ```rust
//! use hbllm_tokenizer::trainer::{Trainer, TrainerConfig};
//!
//! let config = TrainerConfig {
//!     vocab_size: 32768,
//!     min_frequency: 2,
//!     ..Default::default()
//! };
//! let trainer = Trainer::new(config);
//! let vocab = trainer.train_from_text("your training corpus here...");
//!
//! let encoded = vocab.encode(b"hello world");
//! let decoded = vocab.decode_to_string(&encoded);
//! assert_eq!(decoded, "hello world");
//! ```

use pyo3::prelude::*;

pub mod trainer;
pub mod vocab;

/// Python module entry point
#[pymodule]
fn hbllm_tokenizer_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<python::PyVocab>()?;
    m.add_class::<python::PyTrainer>()?;
    Ok(())
}

// Re-export main types
pub use trainer::{Trainer, TrainerConfig};
pub use vocab::Vocab;

pub mod python;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        // Train → Encode → Decode end-to-end test
        let config = TrainerConfig {
            vocab_size: 300,
            min_frequency: 1,
            special_tokens: vec!["<|bos|>".into(), "<|eos|>".into()],
            log_interval: 0,
        };

        let trainer = Trainer::new(config);
        let corpus = "the quick brown fox jumps over the lazy dog \
                       the quick brown fox jumps over the lazy dog \
                       the quick brown fox jumps over the lazy dog";
        let vocab = trainer.train_from_text(corpus);

        // Verify special tokens exist
        assert!(vocab.special_tokens.len() == 2);

        // Verify encode/decode roundtrip
        for word in corpus.split_whitespace() {
            let encoded = vocab.encode(word.as_bytes());
            let decoded = vocab.decode_to_string(&encoded);
            assert_eq!(decoded, word);
        }

        // Should compress the text (fewer tokens than bytes)
        let full_encoded = vocab.encode(b"the quick");
        assert!(full_encoded.len() < b"the quick".len());
    }
}
