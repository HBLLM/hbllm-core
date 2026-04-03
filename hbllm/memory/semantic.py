"""
Semantic Memory (RAG Vector Database).

Embeds text using Sentence-Transformers and stores them for cosine-similarity
semantic search. This allows the agent to recall long-term context that falls
out of the immediate rolling episodic window.

Falls back to TF-IDF when sentence-transformers is not installed, so the
system works out of the box without heavy dependencies.

When ``qdrant-client`` is installed, an optional Qdrant backend can be enabled
for production-grade HNSW approximate nearest-neighbor search.  Pass
``use_qdrant=True`` to the constructor to activate this.

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
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional Qdrant Import ───────────────────────────────────────────────────

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False

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

    When ``use_qdrant=True`` is passed and ``qdrant-client`` is installed,
    a Qdrant HNSW index is maintained alongside the NumPy index for
    production-grade sub-millisecond search at scale.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        hybrid_alpha: float = 0.7,
        use_qdrant: bool = False,
        qdrant_path: str | None = None,
    ):
        """
        Args:
            model_name: Sentence-transformers model name for dense embeddings.
            hybrid_alpha: Blending weight (0.0 = pure sparse/TF-IDF, 1.0 = pure dense).
                          Only used when both dense + sparse indexes are available.
            use_qdrant: If True and qdrant-client is installed, use Qdrant HNSW backend.
            qdrant_path: Optional path for persistent Qdrant disk storage.
        """
        self.model_name = model_name
        self.model = None
        self._use_tfidf = False
        self._tfidf = _TfIdfEmbedder()  # Always maintained for hybrid sparse index
        self.hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))
        self.documents: dict[str, dict[str, Any]] = {}
        self.ids: list[str] = []
        self._vector_list: list[np.ndarray] = []  # Dense vectors (lazy stacking)
        self._sparse_list: list[np.ndarray] = []  # Sparse TF-IDF vectors (lazy stacking)
        self._vectors_dirty = True
        self._vectors_cache: np.ndarray | None = None
        self._sparse_dirty = True
        self._sparse_cache: np.ndarray | None = None
        self._content_hashes: set[str] = set()
        self._lock = threading.Lock()
        self._tfidf_timer: threading.Timer | None = None

        # ── Optional Qdrant Backend ──────────────────────────────────────
        self._use_qdrant = use_qdrant and _HAS_QDRANT
        self._qdrant: QdrantClient | None = None  # type: ignore[name-defined]
        self._qdrant_collection = "semantic_memory"
        self._qdrant_initialized = False
        if self._use_qdrant:
            if qdrant_path:
                self._qdrant = QdrantClient(path=str(qdrant_path))
            else:
                self._qdrant = QdrantClient(":memory:")

    @property
    def count(self) -> int:
        """Number of stored documents."""
        return len(self.ids)

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

    def _ensure_qdrant_collection(self) -> None:
        """Lazily create the Qdrant collection on first write."""
        if not self._use_qdrant or self._qdrant is None or self._qdrant_initialized:
            return

        if self._qdrant.collection_exists(self._qdrant_collection):
            self._qdrant_initialized = True
            return

        dims = 384
        if self.model is not None:
            dims = self.model.get_sentence_embedding_dimension()

        self._qdrant.create_collection(
            collection_name=self._qdrant_collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
        self._qdrant_initialized = True

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the best available method."""
        self._load_model()
        if self._use_tfidf:
            return self._tfidf.encode(texts)
        return self.model.encode(texts)

    @property
    def vectors(self) -> np.ndarray | None:
        """Lazily stack dense vectors only when needed."""
        self._flush_tfidf()
        if not self._vector_list:
            return None
        if self._vectors_dirty:
            self._vectors_cache = np.vstack(self._vector_list)
            self._vectors_dirty = False
        return self._vectors_cache

    @property
    def sparse_vectors(self) -> np.ndarray | None:
        """Lazily stack sparse TF-IDF vectors only when needed."""
        self._flush_tfidf()
        if not self._sparse_list:
            return None
        if self._sparse_dirty:
            self._sparse_cache = np.vstack(self._sparse_list)
            self._sparse_dirty = False
        return self._sparse_cache

    def _flush_tfidf(self) -> None:
        """Forces pending TF-IDF encodes to complete synchronously."""
        if self._tfidf_timer is not None:
            self._tfidf_timer.cancel()
            func = self._tfidf_timer.function
            self._tfidf_timer = None
            func()

    @staticmethod
    def _content_hash(content: str) -> str:
        """Fast hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def store(self, content: str, metadata: dict[str, Any] | None = None, is_priority: bool = False) -> str | None:
        """
        Embed and store a document.

        Args:
            content: Text to embed and store.
            metadata: Optional metadata dict.
            is_priority: Whether this is a high-salience priority document.

        Returns:
            UUID of the stored document, or None if skipped.
        """
        with self._lock:
            return self._store_unsafe(content, metadata, is_priority)

    def _store_unsafe(self, content: str, metadata: dict[str, Any] | None = None, is_priority: bool = False) -> str | None:
        if not content or not content.strip():
            logger.warning("Attempted to store empty content — skipping")
            return None

        # Deduplication: skip if this exact content was already stored
        content_hash = self._content_hash(content)
        if content_hash in self._content_hashes:
            logger.debug("Duplicate content detected — skipping store")
            return None
        self._content_hashes.add(content_hash)

        if self._use_tfidf or self.model is None:
            self._load_model()

        meta = metadata or {}
        if is_priority:
            meta["is_priority"] = True

        doc_id = str(uuid.uuid4())
        doc = {"id": doc_id, "content": content, "metadata": meta}

        # Always update TF-IDF vocabulary (needed for hybrid sparse index)
        self._tfidf.fit_token(content)

        if self._use_tfidf:
            # TF-IDF only mode (no sentence-transformers)
            self.documents[doc_id] = doc
            self.ids.append(doc_id)
            self._schedule_tfidf_encode()
        else:
            # Dense embeddings + sparse TF-IDF for hybrid search
            embedding = self.model.encode([content])[0]
            self.documents[doc_id] = doc
            self.ids.append(doc_id)
            self._vector_list.append(np.array([embedding]))

            # Submits TF-IDF to background queue if vocab changed,
            # otherwise just appends to sparse index
            if self._tfidf._vocab_changed:
                self._schedule_tfidf_encode()
            else:
                sparse_vec = self._tfidf.encode([content])
                self._sparse_list.append(sparse_vec)
                self._sparse_dirty = True

            # ── Qdrant Sidecar Index ─────────────────────────────────────
            if self._use_qdrant and self._qdrant is not None:
                try:
                    self._ensure_qdrant_collection()
                    self._qdrant.upsert(
                        collection_name=self._qdrant_collection,
                        points=[PointStruct(
                            id=doc_id,
                            vector=embedding.tolist(),
                            payload=doc,
                        )],
                    )
                except Exception as e:
                    logger.warning("Qdrant upsert failed (falling back to NumPy): %s", e)

        self._vectors_dirty = True
        logger.debug("Stored semantic document (priority=%s): %s...", is_priority, content[:50])
        return doc_id

    def _schedule_tfidf_encode(self):
        """Debounces and schedules a full TF-IDF re-encoding."""
        if self._tfidf_timer is not None:
            self._tfidf_timer.cancel()

        def _do_encode():
            with self._lock:
                if not self.ids:
                    return
                all_texts = [self.documents[doc_id]["content"] for doc_id in self.ids]
                all_sparse = self._tfidf.encode(all_texts)
                if self._use_tfidf:
                    self._vector_list = [all_sparse[i:i+1] for i in range(len(all_sparse))]
                    self._vectors_dirty = True
                else:
                    self._sparse_list = [all_sparse[i:i+1] for i in range(len(all_sparse))]
                    self._sparse_dirty = True

        # Debounce for 2 seconds to coalesce rapid document insertions
        self._tfidf_timer = threading.Timer(2.0, _do_encode)
        self._tfidf_timer.start()

    def search(
        self,
        query: str,
        top_k: int = 3,
        reward_scores: dict[str, float] | None = None,
        reward_boost: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Search for the most semantically similar documents to the query.

        Uses hybrid scoring (dense + sparse) when both indexes are available,
        and optionally boosts results using reward scores from ValueMemory.

        Args:
            query: Search text.
            top_k: Number of results to return.
            reward_scores: Optional dict mapping doc UUID → reward score.
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
            for idx, doc_id in enumerate(self.ids):
                if doc_id in reward_scores:
                    final_scores[idx] += reward_boost * reward_scores[doc_id]

        # Get top-k indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if final_scores[idx] < 0.1:
                continue

            doc_id = self.ids[idx]
            res = self.documents[doc_id].copy()
            res["score"] = float(final_scores[idx])
            results.append(res)

        return results

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by UUID.

        Args:
            doc_id: UUID of the document to remove.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if doc_id not in self.documents:
                return False

            # Remove content hash
            removed_doc = self.documents[doc_id]
            self._content_hashes.discard(self._content_hash(removed_doc["content"]))

            index = self.ids.index(doc_id)
            self.ids.pop(index)
            del self.documents[doc_id]

            if self._vector_list and len(self.ids) > 0:
                self._vector_list.pop(index)
                self._vectors_dirty = True
            else:
                self._vector_list.clear()
                self._vectors_cache = None

            if self._sparse_list and len(self.ids) > 0:
                self._sparse_list.pop(index)
                self._sparse_dirty = True
            else:
                self._sparse_list.clear()
                self._sparse_cache = None

            # ── Qdrant Sidecar ───────────────────────────────────────────
            if self._use_qdrant and self._qdrant is not None and self._qdrant_initialized:
                try:
                    self._qdrant.delete(
                        collection_name=self._qdrant_collection,
                        points_selector=[doc_id],
                    )
                except Exception as e:
                    logger.warning("Qdrant delete failed: %s", e)

            return True

    def clear(self) -> int:
        """Clear all documents. Returns count of removed docs."""
        with self._lock:
            count = len(self.ids)
            self.documents.clear()
            self.ids.clear()
            self._vector_list.clear()
            self._vectors_cache = None
            self._vectors_dirty = True
            self._sparse_list.clear()
            self._sparse_cache = None
            self._sparse_dirty = True
            self._content_hashes.clear()
            if self._tfidf_timer is not None:
                self._tfidf_timer.cancel()

            # ── Qdrant Sidecar ───────────────────────────────────────────
            if self._use_qdrant and self._qdrant is not None and self._qdrant_initialized:
                try:
                    self._qdrant.delete_collection(self._qdrant_collection)
                    self._qdrant_initialized = False
                except Exception as e:
                    logger.warning("Qdrant clear failed: %s", e)

            return count

    def get_all(self) -> list[dict[str, Any]]:
        """Return all stored documents (without vectors)."""
        return [self.documents[doc_id].copy() for doc_id in self.ids]

    def save_to_disk(self, path: str | Path) -> None:
        """Save semantic memory to disk (metadata + vectors)."""
        from pathlib import Path as _Path
        save_dir = _Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._flush_tfidf()
        with self._lock:
            # Save document metadata
            import json
            meta_path = save_dir / "documents.json"
            with open(meta_path, "w") as f:
                json.dump({"documents": self.documents, "ids": self.ids}, f)

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

            logger.info("SemanticMemory saved to %s (%d docs)", save_dir, len(self.ids))

    @classmethod
    def load_from_disk(cls, path: str | Path, **kwargs) -> "SemanticMemory":
        """Load semantic memory from disk."""
        import json
        from pathlib import Path as _Path

        load_dir = _Path(path)
        if not load_dir.exists() or not (load_dir / "documents.json").exists():
            logger.info("No SemanticMemory data at %s, starting empty", load_dir)
            return cls(**kwargs)

        mem = cls(**kwargs)

        # Load documents
        with open(load_dir / "documents.json") as f:
            data = json.load(f)
            if isinstance(data, dict) and "documents" in data and "ids" in data:
                mem.documents = data["documents"]
                mem.ids = data["ids"]
            elif isinstance(data, list):
                # Backwards compatibility migration
                mem.documents = {}
                mem.ids = []
                for doc in data:
                    doc_id = str(uuid.uuid4())
                    doc["id"] = doc_id
                    mem.documents[doc_id] = doc
                    mem.ids.append(doc_id)

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

        logger.info("SemanticMemory loaded from %s (%d docs)", load_dir, len(mem.ids))
        return mem
