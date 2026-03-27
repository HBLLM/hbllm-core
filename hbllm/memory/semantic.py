"""
Semantic Memory (RAG Vector Database).

Embeds text using Sentence-Transformers and stores them for cosine-similarity 
semantic search. This allows the agent to recall long-term context that falls
out of the immediate rolling episodic window.

Falls back to TF-IDF when sentence-transformers is not installed, so the
system works out of the box without heavy dependencies.

Usage::

    mem = SemanticMemory()
    mem.store("Python is a programming language", {"topic": "coding"})
    mem.store("The weather is sunny today", {"topic": "weather"})
    results = mem.search("programming")
    # [{"content": "Python is a programming language", "score": 0.87, ...}]
"""

import hashlib
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import threading

logger = logging.getLogger(__name__)

# ── Fallback Embedder (TF-IDF) ──────────────────────────────────────────────

class _TfIdfEmbedder:
    """
    Lightweight TF-IDF embedder used when sentence-transformers is not installed.
    
    Builds a vocabulary from stored documents and computes sparse TF-IDF vectors.
    Not as good as dense embeddings, but works without any ML dependencies.
    """
    
    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._doc_freqs: Counter = Counter()
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _update_vocab(self, tokens: list[str]) -> None:
        """Add new tokens to vocabulary."""
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
    
    def _compute_idf(self) -> None:
        """Recompute IDF values based on document frequencies."""
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log((self._doc_count + 1) / (df + 1)) + 1
    
    def fit_token(self, text: str) -> None:
        """Update vocabulary and document frequencies with a new document."""
        tokens = self._tokenize(text)
        old_vocab_size = len(self._vocab)
        self._update_vocab(tokens)
        unique_tokens = set(tokens)
        self._doc_freqs.update(unique_tokens)
        self._doc_count += 1
        self._compute_idf()
        # Track whether vocabulary changed (new dimensions added)
        self._vocab_changed = len(self._vocab) != old_vocab_size
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into TF-IDF vectors."""
        if not self._vocab:
            return np.zeros((len(texts), 1))
        
        dim = len(self._vocab)
        vectors = np.zeros((len(texts), dim))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            
            for term, count in tf.items():
                if term in self._vocab:
                    idx = self._vocab[term]
                    idf = self._idf.get(term, 1.0)
                    vectors[i, idx] = (count / total) * idf
        
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms
        
        return vectors


class SemanticMemory:
    """
    A lightweight in-memory vector database for the Modular Brain.
    
    Uses sentence-transformers to compute dense embeddings and NumPy
    for exact k-Nearest Neighbors cosine similarity search.
    
    If sentence-transformers is not installed, falls back to TF-IDF
    which works without any ML dependencies (lower quality but functional).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", hybrid_alpha: float = 0.7):
        """
        Args:
            model_name: Sentence-transformers model name for dense embeddings.
            hybrid_alpha: Blending weight (0.0 = pure sparse/TF-IDF, 1.0 = pure dense).
                          Only used when both dense + sparse indexes are available.
        """
        self.model_name = model_name
        self.model = None
        self._use_tfidf = False
        self._tfidf = _TfIdfEmbedder()  # Always maintained for hybrid sparse index
        self.hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))
        self.documents: list[dict[str, Any]] = []
        self._vector_list: list[np.ndarray] = []  # Dense vectors (lazy stacking)
        self._sparse_list: list[np.ndarray] = []  # Sparse TF-IDF vectors (lazy stacking)
        self._vectors_dirty = True
        self._vectors_cache: np.ndarray | None = None
        self._sparse_dirty = True
        self._sparse_cache: np.ndarray | None = None
        self._content_hashes: set[str] = set()
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        """Number of stored documents."""
        return len(self.documents)

    def _load_model(self) -> None:
        if self.model is None and not self._use_tfidf:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model %s...", self.model_name)
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                logger.info(
                    "sentence-transformers unavailable — using TF-IDF fallback. "
                    "Install with: pip install sentence-transformers"
                )
                self._use_tfidf = True

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the best available method."""
        self._load_model()
        if self._use_tfidf:
            return self._tfidf.encode(texts)
        return self.model.encode(texts)

    @property
    def vectors(self) -> np.ndarray | None:
        """Lazily stack dense vectors only when needed."""
        if not self._vector_list:
            return None
        if self._vectors_dirty:
            self._vectors_cache = np.vstack(self._vector_list)
            self._vectors_dirty = False
        return self._vectors_cache

    @property
    def sparse_vectors(self) -> np.ndarray | None:
        """Lazily stack sparse TF-IDF vectors only when needed."""
        if not self._sparse_list:
            return None
        if self._sparse_dirty:
            self._sparse_cache = np.vstack(self._sparse_list)
            self._sparse_dirty = False
        return self._sparse_cache

    @staticmethod
    def _content_hash(content: str) -> str:
        """Fast hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def store(self, content: str, metadata: dict[str, Any] | None = None, is_priority: bool = False) -> int:
        """
        Embed and store a document.
        
        Args:
            content: Text to embed and store.
            metadata: Optional metadata dict.
            is_priority: Whether this is a high-salience priority document.
            
        Returns:
            Index of the stored document, or -1 if skipped.
        """
        with self._lock:
            return self._store_unsafe(content, metadata, is_priority)

    def _store_unsafe(self, content: str, metadata: dict[str, Any] | None = None, is_priority: bool = False) -> int:
        if not content or not content.strip():
            logger.warning("Attempted to store empty content — skipping")
            return -1
        
        # Deduplication: skip if this exact content was already stored
        content_hash = self._content_hash(content)
        if content_hash in self._content_hashes:
            logger.debug("Duplicate content detected — skipping store")
            return -1
        self._content_hashes.add(content_hash)
        
        if self._use_tfidf or self.model is None:
            self._load_model()
            
        meta = metadata or {}
        if is_priority:
            meta["is_priority"] = True
        
        # Always update TF-IDF vocabulary (needed for hybrid sparse index)
        self._tfidf.fit_token(content)
        
        if self._use_tfidf:
            # TF-IDF only mode (no sentence-transformers)
            # Must re-encode all documents when vocab changes since vector dimensions change.
            # Once vocabulary stabilizes (after initial ramp-up), this path is rarely hit.
            doc = {"content": content, "metadata": meta}
            self.documents.append(doc)
            
            if self._tfidf._vocab_changed:
                all_texts = [d["content"] for d in self.documents]
                all_vectors = self._tfidf.encode(all_texts)
                self._vector_list = [all_vectors[i:i+1] for i in range(len(all_vectors))]
            else:
                new_vec = self._tfidf.encode([content])
                self._vector_list.append(new_vec)
        else:
            # Dense embeddings + sparse TF-IDF for hybrid search
            embedding = self.model.encode([content])[0]
            doc = {"content": content, "metadata": meta}
            self.documents.append(doc)
            self._vector_list.append(np.array([embedding]))
            
            # Also maintain sparse index for hybrid scoring
            if self._tfidf._vocab_changed:
                all_texts = [d["content"] for d in self.documents]
                all_sparse = self._tfidf.encode(all_texts)
                self._sparse_list = [all_sparse[i:i+1] for i in range(len(all_sparse))]
            else:
                sparse_vec = self._tfidf.encode([content])
                self._sparse_list.append(sparse_vec)
            self._sparse_dirty = True
        
        self._vectors_dirty = True
        logger.debug("Stored semantic document (priority=%s): %s...", is_priority, content[:50])
        return len(self.documents) - 1

    def search(
        self,
        query: str,
        top_k: int = 3,
        reward_scores: dict[int, float] | None = None,
        reward_boost: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Search for the most semantically similar documents to the query.
        
        Uses hybrid scoring (dense + sparse) when both indexes are available,
        and optionally boosts results using reward scores from ValueMemory.
        
        Args:
            query: Search text.
            top_k: Number of results to return.
            reward_scores: Optional dict mapping doc index → reward score.
                           Positive rewards boost documents, negative ones penalize.
            reward_boost: How much to weight reward scores (default 0.1).
            
        Returns:
            List of document dicts with "score" field, sorted by relevance.
        """
        if not self.documents or self.vectors is None:
            return []
        
        if not query or not query.strip():
            return []
        
        # --- Compute dense similarity ---
        query_vec = self._encode([query])[0]
        norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        dense_scores = np.dot(self.vectors, query_vec) / (norms * query_norm + 1e-9)
        
        # --- Hybrid: blend with sparse TF-IDF scores if available ---
        if not self._use_tfidf and self.sparse_vectors is not None and len(self._sparse_list) == len(self.documents):
            sparse_query = self._tfidf.encode([query])[0]
            sparse_norms = np.linalg.norm(self.sparse_vectors, axis=1)
            sparse_query_norm = np.linalg.norm(sparse_query)
            
            if sparse_query_norm > 0:
                sparse_scores = np.dot(self.sparse_vectors, sparse_query) / (sparse_norms * sparse_query_norm + 1e-9)
                # Blend: alpha * dense + (1 - alpha) * sparse
                final_scores = self.hybrid_alpha * dense_scores + (1 - self.hybrid_alpha) * sparse_scores
            else:
                final_scores = dense_scores
        else:
            final_scores = dense_scores
        
        # --- Reward boosting ---
        if reward_scores:
            for idx, reward in reward_scores.items():
                if 0 <= idx < len(final_scores):
                    final_scores[idx] += reward_boost * reward
        
        # Get top-k indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if final_scores[idx] < 0.1:
                continue
                
            res = self.documents[idx].copy()
            res["score"] = float(final_scores[idx])
            results.append(res)
            
        return results

    def delete(self, index: int) -> bool:
        """
        Delete a document by index.
        
        Args:
            index: Index of the document to remove.
            
        Returns:
            True if deleted, False if index out of range.
        """
        if index < 0 or index >= len(self.documents):
            return False
        
        # Remove content hash
        removed_doc = self.documents[index]
        self._content_hashes.discard(self._content_hash(removed_doc["content"]))
        
        self.documents.pop(index)
        
        if self._vector_list and len(self.documents) > 0:
            self._vector_list.pop(index)
            self._vectors_dirty = True
        else:
            self._vector_list.clear()
            self._vectors_cache = None
        
        return True

    def clear(self) -> int:
        """Clear all documents. Returns count of removed docs."""
        count = len(self.documents)
        self.documents.clear()
        self._vector_list.clear()
        self._vectors_cache = None
        self._vectors_dirty = True
        self._sparse_list.clear()
        self._sparse_cache = None
        self._sparse_dirty = True
        self._content_hashes.clear()
        return count

    def get_all(self) -> list[dict[str, Any]]:
        """Return all stored documents (without vectors)."""
        return [doc.copy() for doc in self.documents]

    def save_to_disk(self, path: str | Path) -> None:
        """Save semantic memory to disk (metadata + vectors)."""
        from pathlib import Path as _Path
        save_dir = _Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save document metadata
        import json
        meta_path = save_dir / "documents.json"
        with open(meta_path, "w") as f:
            json.dump(self.documents, f)

        # Save dense vectors
        if self._vector_list:
            vectors = np.array(self._vector_list)
            np.save(save_dir / "dense_vectors.npy", vectors)

        # Save sparse vectors
        if self._sparse_list:
            sparse = np.array(self._sparse_list)
            np.save(save_dir / "sparse_vectors.npy", sparse)

        # Save content hashes
        with open(save_dir / "hashes.json", "w") as f:
            json.dump(list(self._content_hashes), f)

        logger.info("SemanticMemory saved to %s (%d docs)", save_dir, len(self.documents))

    @classmethod
    def load_from_disk(cls, path: str | Path, **kwargs) -> "SemanticMemory":
        """Load semantic memory from disk."""
        from pathlib import Path as _Path
        import json

        load_dir = _Path(path)
        if not load_dir.exists() or not (load_dir / "documents.json").exists():
            logger.info("No SemanticMemory data at %s, starting empty", load_dir)
            return cls(**kwargs)

        mem = cls(**kwargs)

        # Load documents
        with open(load_dir / "documents.json") as f:
            mem.documents = json.load(f)

        # Load dense vectors
        dense_path = load_dir / "dense_vectors.npy"
        if dense_path.exists():
            vectors = np.load(dense_path)
            mem._vector_list = [vectors[i] for i in range(len(vectors))]
            mem._vectors_dirty = True

        # Load sparse vectors
        sparse_path = load_dir / "sparse_vectors.npy"
        if sparse_path.exists():
            sparse = np.load(sparse_path)
            mem._sparse_list = [sparse[i] for i in range(len(sparse))]
            mem._sparse_dirty = True

        # Load content hashes
        hashes_path = load_dir / "hashes.json"
        if hashes_path.exists():
            with open(hashes_path) as f:
                mem._content_hashes = set(json.load(f))

        logger.info("SemanticMemory loaded from %s (%d docs)", load_dir, len(mem.documents))
        return mem

