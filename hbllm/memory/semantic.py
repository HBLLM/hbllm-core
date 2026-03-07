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
from typing import Any

import numpy as np

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
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._use_tfidf = False
        self._tfidf = _TfIdfEmbedder()
        self.documents: list[dict[str, Any]] = []
        self._vector_list: list[np.ndarray] = []  # Accumulate vectors lazily
        self._vectors_dirty = True  # Whether _vector_list needs stacking
        self._vectors_cache: np.ndarray | None = None  # Stacked array cache
        self._content_hashes: set[str] = set()  # Deduplication hashes

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
        """Lazily stack vectors only when needed (e.g., for search)."""
        if not self._vector_list:
            return None
        if self._vectors_dirty:
            self._vectors_cache = np.vstack(self._vector_list)
            self._vectors_dirty = False
        return self._vectors_cache

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
        
        # For TF-IDF, update vocabulary first
        if self._use_tfidf:
            self._tfidf.fit_token(content)
            
            if self._tfidf._vocab_changed:
                # Vocabulary grew: must re-encode all docs for consistent dimensions
                all_texts = [d["content"] for d in self.documents] + [content]
                all_vectors = self._tfidf.encode(all_texts)
                doc = {"content": content, "metadata": meta}
                self.documents.append(doc)
                self._vector_list = [all_vectors[i:i+1] for i in range(len(all_vectors))]
            else:
                # Vocabulary unchanged: only encode the new document
                new_vec = self._tfidf.encode([content])
                doc = {"content": content, "metadata": meta}
                self.documents.append(doc)
                self._vector_list.append(new_vec)
        else:
            embedding = self.model.encode([content])[0]
            doc = {"content": content, "metadata": meta}
            self.documents.append(doc)
            self._vector_list.append(np.array([embedding]))
        
        self._vectors_dirty = True
        logger.debug("Stored semantic document (priority=%s): %s...", is_priority, content[:50])
        return len(self.documents) - 1

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Search for the most semantically similar documents to the query.
        
        Args:
            query: Search text.
            top_k: Number of results to return.
            
        Returns:
            List of document dicts with "score" field, sorted by relevance.
        """
        if not self.documents or self.vectors is None:
            return []
        
        if not query or not query.strip():
            return []
            
        query_vec = self._encode([query])[0]
        
        # Compute cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        similarities = np.dot(self.vectors, query_vec) / (norms * query_norm + 1e-9)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] < 0.1:
                continue
                
            res = self.documents[idx].copy()
            res["score"] = float(similarities[idx])
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
        self._content_hashes.clear()
        return count

    def get_all(self) -> list[dict[str, Any]]:
        """Return all stored documents (without vectors)."""
        return [doc.copy() for doc in self.documents]
