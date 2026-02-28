//! MinHash + LSH deduplication at scale.
//!
//! Efficiently finds and removes near-duplicate documents from the
//! training corpus using:
//! - **Shingling**: Convert documents to sets of character n-grams
//! - **MinHash**: Generate compact signatures that preserve Jaccard similarity
//! - **Locality-Sensitive Hashing (LSH)**: Group similar documents into buckets
//!
//! The algorithm runs in O(n) per document (amortized), making it practical
//! for multi-billion token corpora. Parallelized via rayon.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Configuration for deduplication.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Number of MinHash permutations (more = more accurate, slower)
    pub num_perm: usize,
    /// Number of LSH bands (num_perm must be divisible by num_bands)
    pub num_bands: usize,
    /// Jaccard similarity threshold for considering documents as duplicates
    pub threshold: f64,
    /// Shingle (n-gram) size in characters
    pub shingle_size: usize,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            num_perm: 128,
            num_bands: 16,
            threshold: 0.8,
            shingle_size: 5,
        }
    }
}

/// A MinHash signature — compact representation of a document's content.
#[derive(Debug, Clone)]
pub struct MinHashSignature {
    /// The MinHash values (one per permutation)
    pub values: Vec<u64>,
}

/// The deduplication engine.
pub struct Deduplicator {
    config: DedupConfig,
    /// Rows per LSH band
    rows_per_band: usize,
}

impl Deduplicator {
    /// Create a new deduplicator with the given config.
    pub fn new(config: DedupConfig) -> Self {
        assert!(
            config.num_perm % config.num_bands == 0,
            "num_perm ({}) must be divisible by num_bands ({})",
            config.num_perm,
            config.num_bands
        );
        let rows_per_band = config.num_perm / config.num_bands;
        Self {
            config,
            rows_per_band,
        }
    }

    /// Generate shingles (character n-grams) from a text.
    fn shingle(&self, text: &str) -> HashSet<u64> {
        let chars: Vec<char> = text.chars().collect();
        let mut shingles = HashSet::new();

        if chars.len() < self.config.shingle_size {
            // Document too short for shingling — hash the whole thing
            shingles.insert(xxh3_64_with_seed(text.as_bytes(), 0));
            return shingles;
        }

        for window in chars.windows(self.config.shingle_size) {
            let s: String = window.iter().collect();
            let hash = xxh3_64_with_seed(s.as_bytes(), 0);
            shingles.insert(hash);
        }

        shingles
    }

    /// Compute the MinHash signature for a document.
    ///
    /// For each "permutation" (simulated via different hash seeds),
    /// the MinHash value is the minimum hash of all shingles.
    pub fn compute_signature(&self, text: &str) -> MinHashSignature {
        let shingles = self.shingle(text);

        let values: Vec<u64> = (0..self.config.num_perm as u64)
            .map(|seed| {
                shingles
                    .iter()
                    .map(|&shingle| xxh3_64_with_seed(&shingle.to_le_bytes(), seed))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect();

        MinHashSignature { values }
    }

    /// Compute MinHash signatures for a batch of documents in parallel.
    pub fn compute_signatures_batch(&self, texts: &[String]) -> Vec<MinHashSignature> {
        texts
            .par_iter()
            .map(|text| self.compute_signature(text))
            .collect()
    }

    /// Estimate Jaccard similarity between two signatures.
    pub fn estimate_similarity(sig_a: &MinHashSignature, sig_b: &MinHashSignature) -> f64 {
        assert_eq!(sig_a.values.len(), sig_b.values.len());
        let matches = sig_a
            .values
            .iter()
            .zip(sig_b.values.iter())
            .filter(|(&a, &b)| a == b)
            .count();
        matches as f64 / sig_a.values.len() as f64
    }

    /// Find duplicate document indices using LSH banding.
    ///
    /// Returns a set of indices to REMOVE (keeps the first document
    /// in each duplicate cluster).
    pub fn find_duplicates(&self, signatures: &[MinHashSignature]) -> HashSet<usize> {
        // LSH: hash each band of the signature into buckets
        // Documents that land in the same bucket for ANY band are candidates
        let mut candidate_pairs: HashSet<(usize, usize)> = HashSet::new();

        for band_idx in 0..self.config.num_bands {
            let start = band_idx * self.rows_per_band;
            let end = start + self.rows_per_band;

            // Build bucket → document indices map for this band
            let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();

            for (doc_idx, sig) in signatures.iter().enumerate() {
                // Hash the band slice
                let band_slice = &sig.values[start..end];
                let mut band_bytes = Vec::with_capacity(band_slice.len() * 8);
                for &val in band_slice {
                    band_bytes.extend_from_slice(&val.to_le_bytes());
                }
                let bucket_hash = xxh3_64_with_seed(&band_bytes, band_idx as u64);

                buckets.entry(bucket_hash).or_default().push(doc_idx);
            }

            // All documents in the same bucket are candidates
            for (_bucket, doc_indices) in &buckets {
                if doc_indices.len() > 1 {
                    for i in 0..doc_indices.len() {
                        for j in (i + 1)..doc_indices.len() {
                            let a = doc_indices[i].min(doc_indices[j]);
                            let b = doc_indices[i].max(doc_indices[j]);
                            candidate_pairs.insert((a, b));
                        }
                    }
                }
            }
        }

        // Verify candidates with actual similarity check
        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut removed_check: HashSet<usize> = HashSet::new();

        for (idx_a, idx_b) in &candidate_pairs {
            if removed_check.contains(idx_a) || removed_check.contains(idx_b) {
                continue;
            }

            let similarity = Self::estimate_similarity(&signatures[*idx_a], &signatures[*idx_b]);

            if similarity >= self.config.threshold {
                // Keep idx_a (earlier document), remove idx_b
                to_remove.insert(*idx_b);
                removed_check.insert(*idx_b);
            }
        }

        to_remove
    }

    /// Deduplicate a batch of documents, returning only unique ones.
    ///
    /// The first occurrence of each document cluster is kept.
    pub fn deduplicate(&self, texts: Vec<String>) -> Vec<String> {
        if texts.len() <= 1 {
            return texts;
        }

        let signatures = self.compute_signatures_batch(&texts);
        let to_remove = self.find_duplicates(&signatures);

        texts
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !to_remove.contains(idx))
            .map(|(_, text)| text)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dedup() -> Deduplicator {
        Deduplicator::new(DedupConfig {
            num_perm: 64,
            num_bands: 8,
            threshold: 0.5,
            shingle_size: 3,
        })
    }

    #[test]
    fn test_identical_documents() {
        let dedup = make_dedup();
        let docs = vec![
            "the quick brown fox jumps over the lazy dog".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
            "completely different text about programming languages".to_string(),
        ];

        let result = dedup.deduplicate(docs);
        assert_eq!(result.len(), 2); // One duplicate removed
    }

    #[test]
    fn test_similar_documents() {
        let dedup = make_dedup();
        let sig_a = dedup.compute_signature("the quick brown fox jumps over the lazy dog today");
        let sig_b = dedup.compute_signature("the quick brown fox jumps over the lazy dog tomorrow");

        let similarity = Deduplicator::estimate_similarity(&sig_a, &sig_b);
        assert!(
            similarity > 0.5,
            "Similar documents should have high similarity: {}",
            similarity
        );
    }

    #[test]
    fn test_different_documents() {
        let dedup = make_dedup();
        let sig_a = dedup.compute_signature(
            "machine learning algorithms for natural language processing and deep neural networks",
        );
        let sig_b = dedup.compute_signature(
            "gardening tips for growing roses and maintaining a beautiful lawn in summer season",
        );

        let similarity = Deduplicator::estimate_similarity(&sig_a, &sig_b);
        assert!(
            similarity < 0.5,
            "Different documents should have low similarity: {}",
            similarity
        );
    }

    #[test]
    fn test_unique_documents_preserved() {
        let dedup = make_dedup();
        let docs = vec![
            "machine learning and artificial intelligence research papers from recent conferences"
                .to_string(),
            "cooking recipes from italian cuisine including pasta and pizza preparation methods"
                .to_string(),
            "software engineering best practices and design patterns for large scale systems"
                .to_string(),
        ];

        let result = dedup.deduplicate(docs.clone());
        assert_eq!(result.len(), 3); // All unique, none removed
    }

    #[test]
    fn test_empty_and_single() {
        let dedup = make_dedup();

        let empty: Vec<String> = vec![];
        assert_eq!(dedup.deduplicate(empty).len(), 0);

        let single = vec!["one document".to_string()];
        assert_eq!(dedup.deduplicate(single).len(), 1);
    }

    #[test]
    fn test_shingle_generation() {
        let dedup = make_dedup();
        let shingles = dedup.shingle("abcdef");
        // "abcdef" with shingle_size=3: "abc", "bcd", "cde", "def" = 4 shingles
        assert_eq!(shingles.len(), 4);
    }
}
