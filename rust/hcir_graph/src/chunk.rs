//! Chunked persistent storage with structural sharing.
//!
//! Provides O(1) root snapshot creation and O(chunk_size) branch mutation by
//! partitioning hash maps into independent Arc-wrapped chunks.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

pub const NUM_CHUNKS: usize = 32;

#[derive(Clone, Debug)]
pub struct ChunkedStore<V: Clone> {
    chunks: Vec<Arc<HashMap<String, V>>>,
    total_count: usize,
}

impl<V: Clone> Default for ChunkedStore<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone> ChunkedStore<V> {
    pub fn new() -> Self {
        let mut chunks = Vec::with_capacity(NUM_CHUNKS);
        for _ in 0..NUM_CHUNKS {
            chunks.push(Arc::new(HashMap::new()));
        }
        Self {
            chunks,
            total_count: 0,
        }
    }

    #[inline]
    fn chunk_index(key: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % NUM_CHUNKS
    }

    pub fn len(&self) -> usize {
        self.total_count
    }

    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        let idx = Self::chunk_index(key);
        self.chunks[idx].get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let idx = Self::chunk_index(key);
        self.chunks[idx].contains_key(key)
    }

    /// Insert or update a value, cloning ONLY the affected chunk if shared.
    pub fn with_inserted(&self, key: String, value: V) -> Self {
        let idx = Self::chunk_index(&key);
        let mut new_chunks = self.chunks.clone(); // O(NUM_CHUNKS) Arc pointer clones
        let chunk = Arc::make_mut(&mut new_chunks[idx]); // Clones only the specific chunk

        let is_new = !chunk.contains_key(&key);
        chunk.insert(key, value);

        Self {
            chunks: new_chunks,
            total_count: if is_new { self.total_count + 1 } else { self.total_count },
        }
    }

    /// Remove a key, cloning ONLY the affected chunk if shared.
    pub fn with_removed(&self, key: &str) -> Self {
        let idx = Self::chunk_index(key);
        if !self.chunks[idx].contains_key(key) {
            return self.clone();
        }

        let mut new_chunks = self.chunks.clone();
        let chunk = Arc::make_mut(&mut new_chunks[idx]);
        chunk.remove(key);

        Self {
            chunks: new_chunks,
            total_count: self.total_count.saturating_sub(1),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.chunks.iter().flat_map(|chunk| chunk.keys())
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.chunks.iter().flat_map(|chunk| chunk.values())
    }
}
