//! Vocabulary management for the byte-level BPE tokenizer.
//!
//! The vocabulary starts with 256 byte tokens (0x00-0xFF) and grows
//! by merging frequent byte pairs during training. Special tokens
//! (like `<|bos|>`, `<|domain:code|>`) are reserved at the end.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A single merge rule: (left_token, right_token) → merged_token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRule {
    pub left: u32,
    pub right: u32,
    pub merged: u32,
    pub rank: u32,
}

/// Special tokens for routing and control flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken {
    pub token: String,
    pub id: u32,
}

/// The complete BPE vocabulary.
///
/// Structure:
/// - IDs 0-255: raw byte tokens
/// - IDs 256..(256+num_merges): merged tokens from BPE training
/// - IDs (vocab_size-num_special)..vocab_size: special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// Target vocabulary size
    pub vocab_size: u32,

    /// Ordered list of merge rules (applied in order during encoding)
    pub merges: Vec<MergeRule>,

    /// Token ID → byte sequence mapping
    #[serde(skip)]
    pub id_to_bytes: HashMap<u32, Vec<u8>>,

    /// Byte sequence → token ID mapping (for encoding)
    #[serde(skip)]
    pub bytes_to_id: HashMap<Vec<u8>, u32>,

    /// Special tokens
    pub special_tokens: Vec<SpecialToken>,

    /// Merge pair → rank (for fast lookup during encoding)
    #[serde(skip)]
    pub merge_ranks: HashMap<(u32, u32), u32>,
}

impl Vocab {
    /// Create a new vocabulary with the base 256 byte tokens.
    pub fn new(vocab_size: u32) -> Self {
        let mut id_to_bytes = HashMap::with_capacity(vocab_size as usize);
        let mut bytes_to_id = HashMap::with_capacity(vocab_size as usize);

        // Initialize with 256 byte tokens
        for byte_val in 0u32..256 {
            let bytes = vec![byte_val as u8];
            id_to_bytes.insert(byte_val, bytes.clone());
            bytes_to_id.insert(bytes, byte_val);
        }

        Vocab {
            vocab_size,
            merges: Vec::new(),
            id_to_bytes,
            bytes_to_id,
            special_tokens: Vec::new(),
            merge_ranks: HashMap::new(),
        }
    }

    /// Current number of tokens (base bytes + merges + specials).
    pub fn len(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Whether the vocabulary is empty (it shouldn't be after init).
    pub fn is_empty(&self) -> bool {
        self.id_to_bytes.is_empty()
    }

    /// Add a merge rule: merging `left` + `right` creates a new token.
    pub fn add_merge(&mut self, left: u32, right: u32) -> u32 {
        let new_id = 256 + self.merges.len() as u32;
        let rank = self.merges.len() as u32;

        // Build the byte sequence for the merged token
        let mut merged_bytes = self.id_to_bytes[&left].clone();
        merged_bytes.extend_from_slice(&self.id_to_bytes[&right]);

        // Store the merge
        let rule = MergeRule {
            left,
            right,
            merged: new_id,
            rank,
        };
        self.merges.push(rule);
        self.merge_ranks.insert((left, right), rank);

        // Update mappings
        self.id_to_bytes.insert(new_id, merged_bytes.clone());
        self.bytes_to_id.insert(merged_bytes, new_id);

        new_id
    }

    /// Add a special token and return its ID.
    pub fn add_special_token(&mut self, token: &str) -> u32 {
        // Check if already exists
        for st in &self.special_tokens {
            if st.token == token {
                return st.id;
            }
        }

        let id = self.vocab_size - 1 - self.special_tokens.len() as u32;
        let special = SpecialToken {
            token: token.to_string(),
            id,
        };

        let token_bytes = token.as_bytes().to_vec();
        self.id_to_bytes.insert(id, token_bytes.clone());
        self.bytes_to_id.insert(token_bytes, id);
        self.special_tokens.push(special);

        id
    }

    /// Encode a byte sequence into token IDs using the learned merges.
    ///
    /// Algorithm:
    /// 1. Start with one token per byte
    /// 2. Repeatedly find the highest-priority merge pair and apply it
    /// 3. Stop when no more merges apply
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with byte-level tokens
        let mut tokens: Vec<u32> = text.iter().map(|&b| b as u32).collect();

        // Iteratively apply merges (lowest rank = highest priority)
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the merge pair with the lowest rank (highest priority)
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges applicable
            }

            // Apply the merge
            let left = tokens[best_idx];
            let right = tokens[best_idx + 1];
            let merged = self.merges[best_rank as usize].merged;
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }

        tokens
    }

    /// Decode token IDs back to bytes.
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        bytes
    }

    /// Decode token IDs to a UTF-8 string (lossy).
    pub fn decode_to_string(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode(ids)).into_owned()
    }

    /// Save vocabulary to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load vocabulary from a JSON file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let vocab: Self = serde_json::from_str(&json)?;
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vocab_has_256_bytes() {
        let vocab = Vocab::new(32768);
        assert_eq!(vocab.len(), 256);
    }

    #[test]
    fn test_add_merge() {
        let mut vocab = Vocab::new(32768);
        let merged_id = vocab.add_merge(b'h' as u32, b'e' as u32);
        assert_eq!(merged_id, 256); // First merge gets ID 256
        assert_eq!(vocab.len(), 257);
        assert_eq!(vocab.id_to_bytes[&merged_id], b"he".to_vec());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut vocab = Vocab::new(32768);
        // Add merge: 'h' + 'e' → 256
        vocab.add_merge(b'h' as u32, b'e' as u32);
        // Add merge: 256('he') + 'l' → 257
        vocab.add_merge(256, b'l' as u32);

        let text = b"hello";
        let encoded = vocab.encode(text);
        let decoded = vocab.decode(&encoded);
        assert_eq!(decoded, text.to_vec());
    }

    #[test]
    fn test_encode_applies_merges() {
        let mut vocab = Vocab::new(32768);
        // 'h' + 'e' → 256
        vocab.add_merge(b'h' as u32, b'e' as u32);

        let encoded = vocab.encode(b"he");
        assert_eq!(encoded, vec![256]); // Should be merged
    }

    #[test]
    fn test_special_tokens() {
        let mut vocab = Vocab::new(1000);
        let bos_id = vocab.add_special_token("<|bos|>");
        let eos_id = vocab.add_special_token("<|eos|>");
        assert_ne!(bos_id, eos_id);
        assert!(bos_id >= 990); // Near the end of vocab
    }

    #[test]
    fn test_encode_empty() {
        let vocab = Vocab::new(32768);
        assert_eq!(vocab.encode(b""), Vec::<u32>::new());
    }

    #[test]
    fn test_decode_to_string() {
        let vocab = Vocab::new(32768);
        let ids: Vec<u32> = b"hello".iter().map(|&b| b as u32).collect();
        assert_eq!(vocab.decode_to_string(&ids), "hello");
    }
}
