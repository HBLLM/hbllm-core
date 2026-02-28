//! BPE Tokenizer Training
//!
//! Trains a byte-level BPE tokenizer by iteratively finding and merging
//! the most frequent adjacent byte pairs in a text corpus.
//!
//! Algorithm:
//! 1. Represent the entire corpus as byte sequences
//! 2. Count all adjacent byte-pair frequencies (parallelized with rayon)
//! 3. Merge the most frequent pair, creating a new token
//! 4. Update all sequences and counts
//! 5. Repeat until target vocabulary size is reached

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::vocab::Vocab;

/// Configuration for BPE training.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Target vocabulary size (including 256 base bytes + special tokens)
    pub vocab_size: u32,
    /// Minimum frequency for a pair to be considered for merging
    pub min_frequency: u32,
    /// Special tokens to reserve (e.g., `<|bos|>`, `<|domain:code|>`)
    pub special_tokens: Vec<String>,
    /// Log progress every N merges
    pub log_interval: u32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32768,
            min_frequency: 2,
            special_tokens: vec![
                "<|bos|>".into(),
                "<|eos|>".into(),
                "<|pad|>".into(),
                "<|sep|>".into(),
            ],
            log_interval: 1000,
        }
    }
}

/// A word in the corpus, represented as a sequence of token IDs
/// along with its frequency count.
#[derive(Debug, Clone)]
struct Word {
    tokens: Vec<u32>,
    count: u64,
}

/// BPE Trainer — learns merge rules from a text corpus.
pub struct Trainer {
    config: TrainerConfig,
}

impl Trainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// Train a BPE vocabulary from text files.
    ///
    /// Reads all files, splits into words (whitespace-delimited),
    /// and learns merge rules up to the target vocabulary size.
    pub fn train_from_files(
        &self,
        file_paths: &[&Path],
    ) -> Result<Vocab, Box<dyn std::error::Error>> {
        // Read and tokenize all files into words
        let mut word_counts: HashMap<Vec<u8>, u64> = HashMap::new();

        for path in file_paths {
            let content = fs::read_to_string(path)?;
            for word in content.split_whitespace() {
                let bytes = word.as_bytes().to_vec();
                *word_counts.entry(bytes).or_insert(0) += 1;
            }
        }

        self.train_from_word_counts(&word_counts)
    }

    /// Train a BPE vocabulary from raw text string.
    pub fn train_from_text(&self, text: &str) -> Vocab {
        let mut word_counts: HashMap<Vec<u8>, u64> = HashMap::new();

        for word in text.split_whitespace() {
            let bytes = word.as_bytes().to_vec();
            *word_counts.entry(bytes).or_insert(0) += 1;
        }

        self.train_from_word_counts(&word_counts)
            .expect("Training from a string should not fail")
    }

    /// Train from pre-counted words.
    fn train_from_word_counts(
        &self,
        word_counts: &HashMap<Vec<u8>, u64>,
    ) -> Result<Vocab, Box<dyn std::error::Error>> {
        let mut vocab = Vocab::new(self.config.vocab_size);

        // Add special tokens first (reserves IDs at the end)
        for token in &self.config.special_tokens {
            vocab.add_special_token(token);
        }

        // Convert words to token sequences
        let mut words: Vec<Word> = word_counts
            .iter()
            .map(|(bytes, &count)| Word {
                tokens: bytes.iter().map(|&b| b as u32).collect(),
                count,
            })
            .collect();

        // Calculate how many merges we need
        let num_special = self.config.special_tokens.len() as u32;
        let max_merges = self.config.vocab_size.saturating_sub(256 + num_special);

        eprintln!(
            "Training BPE: {} unique words, target {} merges",
            words.len(),
            max_merges
        );

        // Iteratively learn merge rules
        for merge_num in 0..max_merges {
            // Count pair frequencies (parallelized)
            let pair_counts = self.count_pairs(&words);

            // Find the most frequent pair
            let best_pair = pair_counts.iter().max_by_key(|&(_, &count)| count);

            match best_pair {
                Some((&(left, right), &count)) if count >= self.config.min_frequency as u64 => {
                    // Add the merge to vocabulary
                    let new_id = vocab.add_merge(left, right);

                    // Apply the merge to all words
                    self.apply_merge(&mut words, left, right, new_id);

                    if self.config.log_interval > 0
                        && (merge_num + 1) % self.config.log_interval == 0
                    {
                        eprintln!(
                            "  Merge {}/{}: ({}, {}) → {} (freq={})",
                            merge_num + 1,
                            max_merges,
                            left,
                            right,
                            new_id,
                            count
                        );
                    }
                }
                _ => {
                    eprintln!(
                        "  Stopping early at merge {}: no pairs above min_frequency={}",
                        merge_num, self.config.min_frequency
                    );
                    break;
                }
            }
        }

        eprintln!(
            "Training complete: {} merges learned, vocab size = {}",
            vocab.merges.len(),
            vocab.len()
        );

        Ok(vocab)
    }

    /// Count all adjacent token pair frequencies across all words.
    /// Uses rayon for parallel counting on large corpora.
    fn count_pairs(&self, words: &[Word]) -> HashMap<(u32, u32), u64> {
        // Parallel count: each chunk produces its own HashMap, then we merge
        let chunk_counts: Vec<HashMap<(u32, u32), u64>> = words
            .par_chunks(1000)
            .map(|chunk| {
                let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
                for word in chunk {
                    for window in word.tokens.windows(2) {
                        let pair = (window[0], window[1]);
                        *counts.entry(pair).or_insert(0) += word.count;
                    }
                }
                counts
            })
            .collect();

        // Merge chunk counts
        let mut total_counts: HashMap<(u32, u32), u64> = HashMap::new();
        for chunk in chunk_counts {
            for (pair, count) in chunk {
                *total_counts.entry(pair).or_insert(0) += count;
            }
        }

        total_counts
    }

    /// Apply a merge to all words: replace all occurrences of (left, right) with new_id.
    fn apply_merge(&self, words: &mut [Word], left: u32, right: u32, new_id: u32) {
        words.par_iter_mut().for_each(|word| {
            let mut i = 0;
            let mut new_tokens = Vec::with_capacity(word.tokens.len());

            while i < word.tokens.len() {
                if i + 1 < word.tokens.len()
                    && word.tokens[i] == left
                    && word.tokens[i + 1] == right
                {
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(word.tokens[i]);
                    i += 1;
                }
            }

            word.tokens = new_tokens;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_basic() {
        let config = TrainerConfig {
            vocab_size: 300,
            min_frequency: 2,
            special_tokens: vec!["<|bos|>".into(), "<|eos|>".into()],
            log_interval: 0,
        };

        let trainer = Trainer::new(config);
        let vocab = trainer.train_from_text("the cat sat on the mat the cat");

        // Should have learned at least some merges
        assert!(vocab.merges.len() > 0);
        assert!(vocab.len() > 256);
    }

    #[test]
    fn test_train_encode_roundtrip() {
        let config = TrainerConfig {
            vocab_size: 300,
            min_frequency: 1,
            special_tokens: vec![],
            log_interval: 0,
        };

        let trainer = Trainer::new(config);
        let text = "hello world hello world hello";
        let vocab = trainer.train_from_text(text);

        // Encode and decode should roundtrip each word
        for word in text.split_whitespace() {
            let encoded = vocab.encode(word.as_bytes());
            let decoded = vocab.decode_to_string(&encoded);
            assert_eq!(decoded, word, "Roundtrip failed for '{}'", word);
        }
    }

    #[test]
    fn test_frequent_pairs_merge_first() {
        let config = TrainerConfig {
            vocab_size: 258, // 256 base + 2 merges
            min_frequency: 1,
            special_tokens: vec![],
            log_interval: 0,
        };

        let trainer = Trainer::new(config);
        // "ab" appears 5 times, "cd" appears 2 times
        let vocab = trainer.train_from_text("ab ab ab ab ab cd cd");

        // First merge should be 'a'+'b' since it's most frequent
        assert!(vocab.merges.len() >= 1);
        let first_merge = &vocab.merges[0];
        assert_eq!(first_merge.left, b'a' as u32);
        assert_eq!(first_merge.right, b'b' as u32);
    }

    #[test]
    fn test_min_frequency_cutoff() {
        let config = TrainerConfig {
            vocab_size: 300,
            min_frequency: 100, // Very high threshold
            special_tokens: vec![],
            log_interval: 0,
        };

        let trainer = Trainer::new(config);
        let vocab = trainer.train_from_text("hello world");

        // No merges should be learned with such high min_frequency
        assert_eq!(vocab.merges.len(), 0);
    }
}
