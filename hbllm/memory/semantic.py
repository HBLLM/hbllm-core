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

import asyncio
import hashlib
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hbllm.memory.latent_cluster import LatentClusterManager

logger = logging.getLogger(__name__)

# ── Optional Qdrant Import ───────────────────────────────────────────────────

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False

# ── Optional Rust Acceleration ───────────────────────────────────────────────

try:
    from hbllm_semantic_search import (  # type: ignore[import-not-found]
        batch_cosine_similarity as _rust_cosine,
    )
    from hbllm_semantic_search import (
        content_hash as _rust_hash,
    )

    _HAS_RUST_SEARCH = True
    logger.debug("Using Rust-accelerated semantic search")
except ImportError:
    _HAS_RUST_SEARCH = False

# ── Fallback Embedder (TF-IDF) ──────────────────────────────────────────────


class _TfIdfEmbedder:
    """
    Lightweight TF-IDF embedder used when sentence-transformers is not installed.

    Builds a vocabulary from stored documents and computes sparse TF-IDF vectors.
    Not as good as dense embeddings, but works without any ML dependencies.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._doc_freqs: Counter[str] = Counter()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

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

    def encode(self, texts: list[str]) -> np.ndarray[Any, Any]:
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

        from typing import cast

        return cast(np.ndarray[Any, Any], vectors)


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
        max_documents: int = 5000,
        safe_save: bool = True,
        dedup_threshold: float = 0.95,
        feedback_weight: float = 0.15,
    ):
        """
        Args:
            model_name: Sentence-transformers model name for dense embeddings.
            hybrid_alpha: Blending weight (0.0 = pure sparse/TF-IDF, 1.0 = pure dense).
                          Only used when both dense + sparse indexes are available.
            use_qdrant: If True and qdrant-client is installed, use Qdrant HNSW backend.
            qdrant_path: Optional path for persistent Qdrant disk storage.
            max_documents: Max documents to keep in memory in fallback mode.
        """
        self.model_name = model_name
        self.model: Any | None = None
        self._use_tfidf = False
        self._tfidf = _TfIdfEmbedder()  # Always maintained for hybrid sparse index
        self.hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))
        self.documents: dict[str, dict[str, Any]] = {}
        self.ids: list[str] = []
        self._vector_list: list[np.ndarray[Any, Any]] = []  # Dense vectors (lazy stacking)
        self._sparse_list: list[np.ndarray[Any, Any]] = []  # Sparse TF-IDF vectors (lazy stacking)
        self._vectors_dirty = True
        self._vectors_cache: np.ndarray[Any, Any] | None = None
        self._sparse_dirty = True
        self._sparse_cache: np.ndarray[Any, Any] | None = None
        self._content_hashes: set[str] = set()
        self._norms_cache: np.ndarray[Any, Any] | None = None  # Cached L2 norms of vectors
        self._lock = threading.RLock()
        self._tfidf_timer: threading.Timer | None = None
        self._closed = False
        self.max_documents = max_documents
        self.safe_save = safe_save
        self.dedup_threshold = dedup_threshold
        self.feedback_weight = feedback_weight

        # ── Hebbian Synaptic Plasticity Matrix ──────────────────────────
        self.synaptic_weights = {}
        categories = ["physics", "math", "coding", "finance", "personal", "general"]
        for cat in categories:
            self.synaptic_weights[cat] = {
                other: 1.0 if cat == other else 0.0 for other in categories
            }

        # ── Latent Memory Clusters ──────────────────────────────────────
        self.cluster_manager = LatentClusterManager()

        # Rolling history of priming boosts at search time to align credit assignment on feedback
        self._retrieval_priming_history = {}
        self._priming_history_keys = []

        # ── PostgreSQL/pgvector Backend ──────────────────────────────────
        from hbllm.persistence.db_pool import DBPool

        self._db_pool_class = DBPool
        self._pg_table_created = False

        # ── Optional Qdrant Backend ──────────────────────────────────────
        self._use_qdrant = use_qdrant and _HAS_QDRANT
        self._qdrant: QdrantClient | None = None
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
        if os.environ.get("HBLLM_TESTING") == "1":
            self._use_tfidf = True

        if self.model is None and not self._use_tfidf:
            # Import and load model outside the lock if possible, but use a local
            # check to avoid redundant loads.
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s...", self.model_name)
                model = SentenceTransformer(self.model_name)
                with self._lock:
                    if self.model is None:
                        self.model = model
            except (ImportError, OSError, RuntimeError):
                logger.info(
                    "sentence-transformers unavailable — using TF-IDF fallback. "
                    "Install with: pip install sentence-transformers"
                )
                with self._lock:
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

    async def _ensure_pg_table(self) -> Any:
        """Lazily create the PostgreSQL pgvector table on first write."""
        pool = await self._db_pool_class.get_pool()
        if pool is None:
            return None

        if not self._pg_table_created:
            dims = 384
            if self.model is not None:
                dims = self.model.get_sentence_embedding_dimension()

            async with pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS semantic_memory (
                        id UUID PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        embedding vector({dims}),
                        tenant_id TEXT,
                        user_id TEXT,
                        device_id TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_semantic_tenant ON semantic_memory(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memory(user_id);
                    CREATE INDEX IF NOT EXISTS idx_semantic_device ON semantic_memory(device_id);
                """)
            self._pg_table_created = True
        return pool

    def _encode(self, texts: list[str]) -> np.ndarray[Any, Any]:
        """Encode texts using the best available method."""
        self._load_model()
        if self._use_tfidf:
            return self._tfidf.encode(texts)
        from typing import cast

        if self.model is None:
            return self._tfidf.encode(texts)
        return cast(np.ndarray[Any, Any], self.model.encode(texts))

    @property
    def vectors(self) -> np.ndarray[Any, Any] | None:
        """Lazily stack dense vectors only when needed."""
        self._flush_tfidf()
        if not self._vector_list:
            return None
        if self._vectors_dirty:
            self._vectors_cache = np.vstack(self._vector_list)
            self._norms_cache = None  # Invalidate norms cache
            self._vectors_dirty = False
        return self._vectors_cache

    @property
    def sparse_vectors(self) -> np.ndarray[Any, Any] | None:
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

    async def _aflush_tfidf(self) -> None:
        """Forces pending TF-IDF encodes to complete asynchronously without blocking the event loop."""
        if self._tfidf_timer is not None:
            self._tfidf_timer.cancel()
            func = self._tfidf_timer.function
            self._tfidf_timer = None
            await asyncio.to_thread(func)

    @staticmethod
    def _content_hash(content: str) -> str:
        """Fast hash for deduplication."""
        if _HAS_RUST_SEARCH:
            return _rust_hash(content)
        return hashlib.md5(content.encode()).hexdigest()

    from hbllm.security.tenant_guard import require_tenant

    @require_tenant
    def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        is_priority: bool = False,
        tenant_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        vector_clock: dict[str, int] | None = None,
        authority_score: int = 50,
    ) -> str | None:
        """
        Embed and store a document.

        Args:
            content: Text to embed and store.
            metadata: Optional metadata dict.
            is_priority: Whether this is a high-salience priority document.
            tenant_id: Secure namespace to partition the data under.
            user_id: The specific user this memory belongs to.
            device_id: The device this memory originated from.

        Returns:
            UUID of the stored document, or None if skipped.
        """
        if self._closed:
            logger.warning("Attempted to store in a closed SemanticMemory instance")
            return None

        with self._lock:
            return self._store_unsafe(
                content,
                metadata,
                is_priority,
                tenant_id,
                user_id,
                device_id,
                vector_clock,
                authority_score,
            )

    def _store_unsafe(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        is_priority: bool = False,
        tenant_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        vector_clock: dict[str, int] | None = None,
        authority_score: int = 50,
    ) -> str | None:
        if not content or not content.strip():
            logger.warning("Attempted to store empty content — skipping")
            return None

        # Deduplication: skip if this exact content was already stored under this tenant
        content_hash = self._content_hash(content)
        tenant_hash_key = f"{tenant_id or 'default'}:{content_hash}"
        if tenant_hash_key in self._content_hashes or content_hash in self._content_hashes:
            logger.debug("Duplicate content detected — skipping store")
            return None

        # Fuzzy Deduplication
        if self.safe_save:
            dup_id = self._find_duplicate(content, tenant_id)
            if dup_id:
                logger.info(
                    "Fuzzy duplicate memory detected (similarity >= %s) — returning None",
                    self.dedup_threshold,
                )
                return None

        # Evict oldest document if we reached memory limit
        if len(self.ids) >= self.max_documents:
            oldest_id = self.ids.pop(0)
            oldest_doc = self.documents.pop(oldest_id, None)
            if oldest_doc:
                oldest_hash = self._content_hash(oldest_doc["content"])
                self._content_hashes.discard(oldest_hash)
            if self._vector_list:
                self._vector_list.pop(0)
            if self._sparse_list:
                self._sparse_list.pop(0)
            self._vectors_dirty = True
            self._sparse_dirty = True

        self._content_hashes.add(tenant_hash_key)

        if self._use_tfidf or self.model is None:
            self._load_model()

        meta = metadata or {}
        if is_priority:
            meta["is_priority"] = True
        meta["tenant_id"] = tenant_id
        meta["user_id"] = user_id
        meta["device_id"] = device_id
        meta["vector_clock"] = vector_clock
        meta["authority_score"] = authority_score

        doc_id = str(uuid.uuid4())
        doc = {"id": doc_id, "content": content, "metadata": meta}

        # Always update TF-IDF vocabulary (needed for hybrid sparse index)
        self._tfidf.fit_token(content)

        if self._use_tfidf:
            # TF-IDF only mode (no sentence-transformers)
            self.documents[doc_id] = doc
            self.ids.append(doc_id)
            self._schedule_tfidf_encode()
            embedding = self._tfidf.encode([content])[0]
        else:
            # Dense embeddings + sparse TF-IDF for hybrid search
            if self.model is None:
                raise RuntimeError("Embedding model not loaded")
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

        # Assign document to a latent cluster using LatentClusterManager
        cluster_id = self.cluster_manager.assign_to_cluster(doc_id, embedding, content)
        meta.setdefault("domain", f"cluster_{cluster_id}")
        meta.setdefault("category", f"cluster_{cluster_id}")

        # Initialize connection weights in synaptic_weights matrix for the new cluster
        cluster_key = f"cluster_{cluster_id}"
        if cluster_key not in self.synaptic_weights:
            self.synaptic_weights[cluster_key] = {cluster_key: 1.0}
            for other_cat in list(self.synaptic_weights.keys()):
                if other_cat != cluster_key:
                    self.synaptic_weights[other_cat][cluster_key] = 0.0
                    self.synaptic_weights[cluster_key][other_cat] = 0.0

        if not self._use_tfidf:
            # ── Qdrant Sidecar Index ─────────────────────────────────────
            if self._use_qdrant and self._qdrant is not None:
                try:
                    self._ensure_qdrant_collection()
                    self._qdrant.upsert(
                        collection_name=self._qdrant_collection,
                        points=[
                            PointStruct(
                                id=doc_id,
                                vector=embedding.tolist(),
                                payload=doc,
                            )
                        ],
                    )
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.warning("Qdrant upsert failed (falling back to NumPy): %s", e)

        self._vectors_dirty = True
        logger.debug("Stored semantic document (priority=%s): %s...", is_priority, content[:50])
        return doc_id

    from hbllm.security.tenant_guard import require_tenant

    @require_tenant
    async def astore(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        is_priority: bool = False,
        tenant_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        vector_clock: dict[str, int] | None = None,
        authority_score: int = 50,
    ) -> str | None:
        """Async version of store that persists to Postgres if configured."""
        # 1. Store in local fallback memory (also handles deduplication/TF-IDF)
        doc_id = self.store(
            content,
            metadata,
            is_priority,
            tenant_id,
            user_id,
            device_id,
            vector_clock,
            authority_score,
        )
        if not doc_id:
            return None

        # 2. Replicate to PostgreSQL pgvector
        pool = await self._ensure_pg_table()
        if pool:
            meta = metadata or {}
            if is_priority:
                meta["is_priority"] = True

            # Re-compute embedding (or fetch from memory)
            # We already loaded the model inside self.store()
            if self.model is not None:
                embedding = self.model.encode([content])[0]
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO semantic_memory (id, content, metadata, embedding, tenant_id, user_id, device_id)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            doc_id,
                            content,
                            meta,
                            embedding.tolist(),
                            tenant_id,
                            user_id,
                            device_id,
                        )
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.error("Failed to store vector in PostgreSQL: %s", e)

        return doc_id

    def _schedule_tfidf_encode(self) -> None:
        """Debounces and schedules a full TF-IDF re-encoding."""
        if self._tfidf_timer is not None:
            self._tfidf_timer.cancel()

        def _do_encode() -> None:
            with self._lock:
                if not self.ids:
                    return
                all_texts = [self.documents[doc_id]["content"] for doc_id in self.ids]
                all_sparse = self._tfidf.encode(all_texts)
                if self._use_tfidf:
                    self._vector_list = [all_sparse[i : i + 1] for i in range(len(all_sparse))]
                    self._vectors_dirty = True
                else:
                    self._sparse_list = [all_sparse[i : i + 1] for i in range(len(all_sparse))]
                    self._sparse_dirty = True

        # Debounce for 2 seconds to coalesce rapid document insertions
        self._tfidf_timer = threading.Timer(2.0, _do_encode)
        self._tfidf_timer.start()

    from hbllm.security.tenant_guard import require_tenant

    @require_tenant
    def search(
        self,
        query: str,
        top_k: int = 3,
        reward_scores: dict[str, float] | None = None,
        reward_boost: float = 0.1,
        priming_boosts: dict[str, float] | None = None,
        priming_boost_weight: float = 0.15,
        tenant_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        explain: bool = False,
        primer: Any | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
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
            tenant_id: Secure namespace to query strictly within.
            user_id: The specific user this memory belongs to.
            device_id: The device this memory originated from.

        Returns:
            List of document dicts with "score" field, sorted by relevance.
        """
        if self._closed:
            return []

        if not self.documents or self.vectors is None:
            return []

        if not query or not query.strip():
            return []

        # --- Compute dense similarity ---
        query_vec = self._encode([query])[0]

        # Project query vector onto centroids and stimulate SNN primer
        if primer is not None:
            primer.stimulate_by_vector(query_vec, self.cluster_manager.centroids)
            priming_boosts = primer.get_boosts()

        if _HAS_RUST_SEARCH:
            # Rust-accelerated cosine similarity
            dense_scores = np.asarray(
                _rust_cosine(
                    query_vec.astype(np.float64),
                    self.vectors.astype(np.float64),
                )
            )
        else:
            # Use cached norms (only recomputed when vectors change)
            if self._norms_cache is None or len(self._norms_cache) != len(self.vectors):
                self._norms_cache = np.linalg.norm(self.vectors, axis=1)
            query_norm = np.linalg.norm(query_vec)

            if query_norm == 0:
                return []

            dense_scores = np.dot(self.vectors, query_vec) / (self._norms_cache * query_norm + 1e-9)

        # --- Hybrid: blend with sparse TF-IDF scores if available ---
        if (
            not self._use_tfidf
            and self.sparse_vectors is not None
            and len(self._sparse_list) == len(self.documents)
        ):
            sparse_query = self._tfidf.encode([query])[0]
            sparse_norms = np.linalg.norm(self.sparse_vectors, axis=1)
            sparse_query_norm = np.linalg.norm(sparse_query)

            if sparse_query_norm > 0:
                sparse_scores = np.dot(self.sparse_vectors, sparse_query) / (
                    sparse_norms * sparse_query_norm + 1e-9
                )
                # Blend: alpha * dense + (1 - alpha) * sparse
                final_scores = (
                    self.hybrid_alpha * dense_scores + (1 - self.hybrid_alpha) * sparse_scores
                )
            else:
                final_scores = dense_scores
        else:
            final_scores = dense_scores

        # Save base similarity scores before applying any boosts
        base_similarities = final_scores.copy()

        # --- Usefulness boosting ---
        for idx, doc_id in enumerate(self.ids):
            doc = self.documents[doc_id]
            meta = doc.get("metadata") or {}
            usefulness = int(meta.get("usefulness", 0) or 0)
            boost = self.feedback_weight * (math.log1p(max(usefulness, 0)) / math.log1p(10))
            final_scores[idx] += boost

        # --- Reward boosting ---
        if reward_scores:
            for idx, doc_id in enumerate(self.ids):
                if doc_id in reward_scores:
                    final_scores[idx] += reward_boost * reward_scores[doc_id]

        # --- Synaptic Priming boost ---
        if priming_boosts:
            for idx, doc_id in enumerate(self.ids):
                doc = self.documents[doc_id]
                meta = doc.get("metadata") or {}
                doc_cats = []
                cluster_id = self.cluster_manager.cluster_assignments.get(doc_id)
                if cluster_id is not None:
                    doc_cats.append(f"cluster_{cluster_id}")
                orig_domain = meta.get("domain")
                if orig_domain and orig_domain not in doc_cats:
                    doc_cats.append(orig_domain)
                orig_category = meta.get("category")
                if orig_category and orig_category not in doc_cats:
                    doc_cats.append(orig_category)

                if doc_cats:
                    boost_sum = 0.0
                    for prime_cat, potential in priming_boosts.items():
                        if potential > 0.01:
                            max_w = 0.0
                            for d_cat in doc_cats:
                                w = self.synaptic_weights.setdefault(prime_cat, {}).setdefault(
                                    d_cat, 1.0 if prime_cat == d_cat else 0.0
                                )
                                if w > max_w:
                                    max_w = w
                            boost_sum += max_w * min(1.0, potential)
                    final_scores[idx] += priming_boost_weight * boost_sum

        # --- Security: Hard Partition Masking ---
        if tenant_id and tenant_id != "system":
            tenant_mask = np.array(
                [
                    self.documents[doc_id]["metadata"].get("tenant_id") == tenant_id
                    for doc_id in self.ids
                ]
            )
            final_scores[~tenant_mask] = -1.0  # Erase non-tenant similarity completely

        if user_id:
            user_mask = np.array(
                [
                    self.documents[doc_id]["metadata"].get("user_id") == user_id
                    for doc_id in self.ids
                ]
            )
            final_scores[~user_mask] = -1.0

        if device_id:
            device_mask = np.array(
                [
                    self.documents[doc_id]["metadata"].get("device_id") == device_id
                    for doc_id in self.ids
                ]
            )
            final_scores[~device_mask] = -1.0

        # Get top-k indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if final_scores[idx] < 0.1:
                continue

            doc_id = self.ids[idx]
            res = self.documents[doc_id].copy()
            res["score"] = float(final_scores[idx])

            # Increment activation count for the retrieved document's cluster
            meta = res.get("metadata") or {}
            cluster_id = self.cluster_manager.cluster_assignments.get(doc_id)
            if cluster_id is not None:
                if cluster_id in self.cluster_manager.cluster_stats:
                    stats = self.cluster_manager.cluster_stats[cluster_id]
                    stats["activation_count"] += 1
                    stats["last_used"] = time.time()

            # Calculate individual component scores
            usefulness = int(meta.get("usefulness", 0) or 0)
            u_boost = self.feedback_weight * (math.log1p(max(usefulness, 0)) / math.log1p(10))

            r_boost = 0.0
            if reward_scores and doc_id in reward_scores:
                r_boost = reward_boost * reward_scores[doc_id]

            p_boost = 0.0
            if priming_boosts:
                doc_cats = []
                if cluster_id is not None:
                    doc_cats.append(f"cluster_{cluster_id}")
                orig_domain = meta.get("domain")
                if orig_domain and orig_domain not in doc_cats:
                    doc_cats.append(orig_domain)
                orig_category = meta.get("category")
                if orig_category and orig_category not in doc_cats:
                    doc_cats.append(orig_category)

                if doc_cats:
                    boost_sum = 0.0
                    for prime_cat, potential in priming_boosts.items():
                        if potential > 0.01:
                            max_w = 0.0
                            for d_cat in doc_cats:
                                w = self.synaptic_weights.setdefault(prime_cat, {}).setdefault(
                                    d_cat, 1.0 if prime_cat == d_cat else 0.0
                                )
                                if w > max_w:
                                    max_w = w
                            boost_sum += max_w * min(1.0, potential)
                    p_boost = priming_boost_weight * boost_sum

            res["score_breakdown"] = {
                "similarity": float(base_similarities[idx]),
                "usefulness_boost": float(u_boost),
                "reward_boost": float(r_boost),
                "priming_boost": float(p_boost),
            }
            results.append(res)

        # Record search-time priming history for credit assignment during subsequent feedback
        if priming_boosts:
            if len(self._retrieval_priming_history) > 200:
                oldest_keys = self._priming_history_keys[:100]
                for k in oldest_keys:
                    self._retrieval_priming_history.pop(k, None)
                self._priming_history_keys = self._priming_history_keys[100:]

            for r in results:
                self._retrieval_priming_history[r["id"]] = priming_boosts.copy()
                self._priming_history_keys.append(r["id"])

        if explain:
            return {
                "results": results,
                "explanations": [
                    {
                        "doc_id": r["id"],
                        "score_breakdown": r["score_breakdown"],
                        "domain": r.get("metadata", {}).get("domain")
                        or r.get("metadata", {}).get("category"),
                    }
                    for r in results
                ],
                "global_stats": {
                    "query": query,
                    "total_documents": len(self.documents),
                    "priming_applied": bool(priming_boosts),
                },
            }

        return results

    @require_tenant
    async def asearch(
        self,
        query: str,
        top_k: int = 3,
        reward_scores: dict[str, float] | None = None,
        reward_boost: float = 0.1,
        priming_boosts: dict[str, float] | None = None,
        priming_boost_weight: float = 0.15,
        tenant_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        explain: bool = False,
        primer: Any | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Async version of search that queries Postgres pgvector if available."""
        await self._aflush_tfidf()
        pool = await self._ensure_pg_table()
        if not pool:
            return self.search(
                query,
                top_k,
                reward_scores,
                reward_boost,
                priming_boosts,
                priming_boost_weight,
                tenant_id,
                user_id,
                device_id,
                explain,
                primer=primer,
            )

        if not query or not query.strip():
            return []

        self._load_model()
        if self.model is None:
            return self.search(
                query,
                top_k,
                reward_scores,
                reward_boost,
                priming_boosts,
                priming_boost_weight,
                tenant_id,
                user_id,
                device_id,
                explain,
                primer=primer,
            )

        query_vec = self.model.encode([query])[0]

        # Project query vector onto centroids and stimulate SNN primer
        if primer is not None:
            primer.stimulate_by_vector(query_vec, self.cluster_manager.centroids)
            priming_boosts = primer.get_boosts()

        results = []
        try:
            async with pool.acquire() as conn:
                # We use the <=> operator for cosine distance.
                sql = """
                    SELECT id, content, metadata, 1 - (embedding <=> $1) AS score
                    FROM semantic_memory
                """
                args: list[Any] = [query_vec.tolist()]

                conditions = []
                if tenant_id and tenant_id != "system":
                    args.append(tenant_id)
                    conditions.append(f"(tenant_id = ${len(args)} OR tenant_id IS NULL)")

                if user_id:
                    args.append(user_id)
                    conditions.append(f"(user_id = ${len(args)} OR user_id IS NULL)")

                if device_id:
                    args.append(device_id)
                    conditions.append(f"(device_id = ${len(args)} OR device_id IS NULL)")

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

                sql += f" ORDER BY embedding <=> $1 ASC LIMIT {top_k}"

                rows = await conn.fetch(sql, *args)

                for row in rows:
                    doc_id = str(row["id"])
                    base_similarity = float(row["score"])
                    score = base_similarity

                    # Apply reward boosting if any
                    r_boost = 0.0
                    if reward_scores and doc_id in reward_scores:
                        r_boost = reward_boost * reward_scores[doc_id]
                        score += r_boost

                    # Apply usefulness boost if stored in metadata
                    meta = row["metadata"] or {}
                    usefulness = int(meta.get("usefulness", 0) or 0)
                    u_boost = self.feedback_weight * (
                        math.log1p(max(usefulness, 0)) / math.log1p(10)
                    )
                    score += u_boost

                    # Apply priming boost
                    p_boost = 0.0
                    cluster_id = self.cluster_manager.cluster_assignments.get(doc_id)
                    if priming_boosts:
                        doc_cats = []
                        if cluster_id is not None:
                            doc_cats.append(f"cluster_{cluster_id}")
                        orig_domain = meta.get("domain")
                        if orig_domain and orig_domain not in doc_cats:
                            doc_cats.append(orig_domain)
                        orig_category = meta.get("category")
                        if orig_category and orig_category not in doc_cats:
                            doc_cats.append(orig_category)

                        if doc_cats:
                            boost_sum = 0.0
                            for prime_cat, potential in priming_boosts.items():
                                if potential > 0.01:
                                    max_w = 0.0
                                    for d_cat in doc_cats:
                                        w = self.synaptic_weights.setdefault(
                                            prime_cat, {}
                                        ).setdefault(d_cat, 1.0 if prime_cat == d_cat else 0.0)
                                        if w > max_w:
                                            max_w = w
                                    boost_sum += max_w * min(1.0, potential)
                            p_boost = priming_boost_weight * boost_sum
                            score += p_boost

                    # Increment activation count for the retrieved document's cluster
                    if cluster_id is not None:
                        if cluster_id in self.cluster_manager.cluster_stats:
                            stats = self.cluster_manager.cluster_stats[cluster_id]
                            stats["activation_count"] += 1
                            stats["last_used"] = time.time()

                    results.append(
                        {
                            "id": doc_id,
                            "content": row["content"],
                            "metadata": row["metadata"],
                            "score": score,
                            "score_breakdown": {
                                "similarity": base_similarity,
                                "usefulness_boost": float(u_boost),
                                "reward_boost": float(r_boost),
                                "priming_boost": float(p_boost),
                            },
                        }
                    )

            # Sort again if rewards/priming changed the order
            if reward_scores or priming_boosts:
                results.sort(key=lambda x: x["score"], reverse=True)

            # Record search-time priming history for credit assignment during subsequent feedback
            if priming_boosts:
                if len(self._retrieval_priming_history) > 200:
                    oldest_keys = self._priming_history_keys[:100]
                    for k in oldest_keys:
                        self._retrieval_priming_history.pop(k, None)
                    self._priming_history_keys = self._priming_history_keys[100:]

                for r in results:
                    self._retrieval_priming_history[r["id"]] = priming_boosts.copy()
                    self._priming_history_keys.append(r["id"])

            if explain:
                return {
                    "results": results,
                    "explanations": [
                        {
                            "doc_id": r["id"],
                            "score_breakdown": r["score_breakdown"],
                            "domain": r.get("metadata", {}).get("domain")
                            or r.get("metadata", {}).get("category"),
                        }
                        for r in results
                    ],
                    "global_stats": {
                        "query": query,
                        "total_documents": -1,  # Database backend total is unqueried
                        "priming_applied": bool(priming_boosts),
                    },
                }

            return results
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("PostgreSQL pgvector search failed: %s", e)
            return self.search(
                query,
                top_k,
                reward_scores,
                reward_boost,
                priming_boosts,
                priming_boost_weight,
                tenant_id,
                user_id,
                device_id,
                explain,
            )

    def get_ranking_differential(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compute score differentials between consecutive documents in search results.

        Explains why the top documents outranked their immediate successors.
        """
        differentials = []
        if len(results) < 2:
            return differentials

        for i in range(len(results) - 1):
            doc_a = results[i]
            doc_b = results[i + 1]
            breakdown_a = doc_a.get("score_breakdown", {})
            breakdown_b = doc_b.get("score_breakdown", {})

            # Compute deltas (A - B)
            total_delta = doc_a["score"] - doc_b["score"]
            sim_delta = breakdown_a.get("similarity", 0.0) - breakdown_b.get("similarity", 0.0)
            usefulness_delta = breakdown_a.get("usefulness_boost", 0.0) - breakdown_b.get(
                "usefulness_boost", 0.0
            )
            reward_delta = breakdown_a.get("reward_boost", 0.0) - breakdown_b.get(
                "reward_boost", 0.0
            )
            priming_delta = breakdown_a.get("priming_boost", 0.0) - breakdown_b.get(
                "priming_boost", 0.0
            )

            # Generate natural language explanation of the win
            reasons = []
            if sim_delta > 0:
                reasons.append(f"higher base semantic similarity (+{sim_delta:.3f})")
            elif sim_delta < 0:
                reasons.append(f"lower base similarity ({sim_delta:.3f})")

            if priming_delta > 0:
                reasons.append(f"SNN priming bias advantage (+{priming_delta:.3f})")
            elif priming_delta < 0:
                reasons.append(f"SNN priming bias deficit ({priming_delta:.3f})")

            if reward_delta > 0:
                reasons.append(f"value preference boost (+{reward_delta:.3f})")
            elif reward_delta < 0:
                reasons.append(f"value preference deficit ({reward_delta:.3f})")

            if usefulness_delta > 0:
                reasons.append(f"usefulness feedback boost (+{usefulness_delta:.3f})")
            elif usefulness_delta < 0:
                reasons.append(f"usefulness feedback deficit ({usefulness_delta:.3f})")

            doc_a_id = doc_a.get("id", "unknown")
            doc_b_id = doc_b.get("id", "unknown")
            explanation_str = (
                f"Document '{doc_a_id[:8]}' outranked '{doc_b_id[:8]}' by {total_delta:.3f}. "
            )
            if reasons:
                explanation_str += "Key drivers: " + ", ".join(reasons) + "."
            else:
                explanation_str += "Minimal overall score difference."

            differentials.append(
                {
                    "rank_a": i + 1,
                    "rank_b": i + 2,
                    "doc_a_id": doc_a_id,
                    "doc_b_id": doc_b_id,
                    "deltas": {
                        "total": float(total_delta),
                        "similarity": float(sim_delta),
                        "usefulness": float(usefulness_delta),
                        "reward": float(reward_delta),
                        "priming": float(priming_delta),
                    },
                    "explanation": explanation_str,
                }
            )
        return differentials

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
            removed_tenant = (removed_doc.get("metadata") or {}).get("tenant_id")
            removed_hash = self._content_hash(removed_doc["content"])
            self._content_hashes.discard(f"{removed_tenant or 'default'}:{removed_hash}")
            self._content_hashes.discard(removed_hash)

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
                except (OSError, ConnectionError, RuntimeError) as e:
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
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.warning("Qdrant clear failed: %s", e)

            return count

    def close(self) -> None:
        """Gracefully shut down, cancelling timers and clearing caches."""
        with self._lock:
            self._closed = True
            if self._tfidf_timer is not None:
                self._tfidf_timer.cancel()
                self._tfidf_timer = None
            self._vectors_cache = None
            self._sparse_cache = None

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

            # Save Hebbian synaptic weights separately (Split Persistence)
            with open(save_dir / "synaptic_matrix.json", "w") as f:
                json.dump(self.synaptic_weights, f)

            # Save latent cluster manager details
            with open(save_dir / "latent_clusters.json", "w") as f:
                json.dump(self.cluster_manager.to_dict(), f)

            logger.info("SemanticMemory saved to %s (%d docs)", save_dir, len(self.ids))

    @classmethod
    def load_from_disk(cls, path: str | Path, **kwargs: Any) -> "SemanticMemory":
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

        # Load Hebbian synaptic weights (Split Persistence)
        synaptic_path = load_dir / "synaptic_matrix.json"
        if synaptic_path.exists():
            try:
                with open(synaptic_path) as f:
                    mem.synaptic_weights = json.load(f)
            except Exception as e:
                logger.warning("Failed to load synaptic matrix: %s", e)

        # Load latent cluster manager details
        clusters_path = load_dir / "latent_clusters.json"
        if clusters_path.exists():
            try:
                with open(clusters_path) as f:
                    mem.cluster_manager.from_dict(json.load(f))
            except Exception as e:
                logger.warning("Failed to load latent clusters: %s", e)

        logger.info("SemanticMemory loaded from %s (%d docs)", load_dir, len(mem.ids))
        return mem

    def _find_duplicate(self, content: str, tenant_id: str | None) -> str | None:
        """Find a duplicate document in the same tenant using cosine similarity."""
        if not self.documents or self.vectors is None:
            return None
        query_vec = self._encode([content])[0]
        if _HAS_RUST_SEARCH:
            dense_scores = np.asarray(
                _rust_cosine(
                    query_vec.astype(np.float64),
                    self.vectors.astype(np.float64),
                )
            )
        else:
            norms = np.linalg.norm(self.vectors, axis=1)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return None
            dense_scores = np.dot(self.vectors, query_vec) / (norms * query_norm + 1e-9)

        if tenant_id and tenant_id != "system":
            tenant_mask = np.array(
                [
                    self.documents[doc_id]["metadata"].get("tenant_id") == tenant_id
                    for doc_id in self.ids
                ]
            )
            dense_scores[~tenant_mask] = -1.0

        best_idx = int(np.argmax(dense_scores))
        if dense_scores[best_idx] >= self.dedup_threshold:
            return self.ids[best_idx]
        return None

    def feedback(self, doc_id: str, useful: bool = True) -> int | None:
        """Adjust a memory's usefulness (used by search re-ranking) and update Hebbian synaptic weights."""
        with self._lock:
            doc = self.documents.get(doc_id)
            if not doc:
                return None
            meta = doc.setdefault("metadata", {})
            usefulness = int(meta.get("usefulness", 0) or 0)
            usefulness = max(0, usefulness + (1 if useful else -1))
            meta["usefulness"] = usefulness
            doc["usefulness"] = usefulness

            # --- Hebbian Plasticity Update ---
            doc_cats = []
            cluster_id = self.cluster_manager.cluster_assignments.get(doc_id)
            if cluster_id is not None:
                doc_cats.append(f"cluster_{cluster_id}")
            orig_domain = meta.get("domain")
            if orig_domain and orig_domain not in doc_cats:
                doc_cats.append(orig_domain)
            orig_category = meta.get("category")
            if orig_category and orig_category not in doc_cats:
                doc_cats.append(orig_category)

            # Track positive feedback for success rate
            for d_cat in doc_cats:
                if d_cat.startswith("cluster_"):
                    try:
                        c_id = int(d_cat.split("_")[1])
                        if c_id in self.cluster_manager.cluster_stats:
                            stats = self.cluster_manager.cluster_stats[c_id]
                            if useful:
                                stats["positive_feedback_count"] += 1
                            # Recompute success rate: positive_feedback_count / (activation_count or 1)
                            if stats["activation_count"] > 0:
                                stats["success_rate"] = (
                                    float(stats["positive_feedback_count"])
                                    / stats["activation_count"]
                                )
                            else:
                                stats["success_rate"] = (
                                    1.0 if stats["positive_feedback_count"] > 0 else 0.0
                                )
                    except (ValueError, IndexError):
                        pass

            history = self._retrieval_priming_history.pop(doc_id, None)
            if history and doc_cats:
                learning_rate = 0.05
                decay_rate = 0.01  # Homeostatic regulation factor
                feedback_score = 1.0 if useful else -0.6

                for prime_cat, potential in history.items():
                    if potential > 0.01:
                        cat_weights = self.synaptic_weights.setdefault(prime_cat, {})

                        # 1. Homeostatic decay: apply to all connection weights for the active category
                        for domain in list(cat_weights.keys()):
                            cat_weights[domain] *= 1.0 - decay_rate

                        # 2. LTP / LTD: reinforce the connection to target document domains
                        for d_cat in doc_cats:
                            if d_cat not in cat_weights:
                                cat_weights[d_cat] = 1.0 if prime_cat == d_cat else 0.0

                            delta = learning_rate * potential * feedback_score
                            new_w = cat_weights[d_cat] + delta

                            # 3. Clip weights to [0.0, 1.0]
                            cat_weights[d_cat] = max(0.0, min(1.0, new_w))

            if self._use_qdrant and self._qdrant is not None and self._qdrant_initialized:
                try:
                    self._qdrant.set_payload(
                        collection_name=self._qdrant_collection,
                        payload={"metadata": meta},
                        points=[doc_id],
                    )
                except Exception as e:
                    logger.warning("Qdrant set_payload failed: %s", e)
            return usefulness

    def consolidate(
        self, tenant_id: str | None = None, threshold: float | None = None
    ) -> dict[str, Any]:
        """Self-optimization: merge near-duplicate memories within a tenant."""
        thr = threshold if threshold is not None else self.dedup_threshold
        with self._lock:
            if not self.documents or self.vectors is None:
                return {"tenant_id": tenant_id, "removed": 0, "merged_into": {}}

            target_ids = []
            target_vectors = []
            for idx, doc_id in enumerate(self.ids):
                doc_tenant = self.documents[doc_id].get("metadata", {}).get("tenant_id")
                if tenant_id is None or doc_tenant == tenant_id:
                    target_ids.append(doc_id)
                    target_vectors.append(self._vector_list[idx][0])

            if len(target_ids) <= 1:
                return {"tenant_id": tenant_id, "removed": 0, "merged_into": {}}

            removed = 0
            merged_into: dict[str, list[str]] = {}
            alive_ids = set(target_ids)

            for k_idx in reversed(range(len(target_ids))):
                keeper_id = target_ids[k_idx]
                if keeper_id not in alive_ids:
                    continue

                keeper_vector = target_vectors[k_idx]

                for d_idx in reversed(range(k_idx)):
                    dup_id = target_ids[d_idx]
                    if dup_id not in alive_ids:
                        continue

                    dup_vector = target_vectors[d_idx]
                    k_norm = np.linalg.norm(keeper_vector)
                    d_norm = np.linalg.norm(dup_vector)
                    if k_norm == 0 or d_norm == 0:
                        continue
                    sim = np.dot(keeper_vector, dup_vector) / (k_norm * d_norm + 1e-9)

                    if sim >= thr:
                        keeper_doc = self.documents[keeper_id]
                        dup_doc = self.documents[dup_id]

                        k_meta = keeper_doc.setdefault("metadata", {})
                        d_meta = dup_doc.get("metadata", {})

                        # Merge tags (handle lists safely)
                        k_tags = list(set(k_meta.get("tags", [])) | set(d_meta.get("tags", [])))
                        k_meta["tags"] = k_tags

                        # Merge usefulness
                        k_use = int(k_meta.get("usefulness", 0) or 0)
                        d_use = int(d_meta.get("usefulness", 0) or 0)
                        k_meta["usefulness"] = max(k_use, d_use)
                        keeper_doc["usefulness"] = max(k_use, d_use)

                        self.delete(dup_id)
                        alive_ids.discard(dup_id)
                        removed += 1
                        merged_into.setdefault(keeper_id, []).append(dup_id)

            # Perform latent cluster maintenance
            doc_vectors_dict = {}
            for idx, doc_id in enumerate(self.ids):
                if not self._use_tfidf and self._vector_list:
                    doc_vectors_dict[doc_id] = self._vector_list[idx][0]
                elif self._sparse_list:
                    doc_vectors_dict[doc_id] = self._sparse_list[idx][0]

            self.cluster_manager.maintain_clusters(doc_vectors_dict, self.synaptic_weights)

            # Update document metadata categories for any re-assigned documents
            for doc_id, c_id in self.cluster_manager.cluster_assignments.items():
                if doc_id in self.documents:
                    doc_meta = self.documents[doc_id].setdefault("metadata", {})
                    # Only overwrite if it is missing or starts with "cluster_"
                    if not doc_meta.get("domain") or doc_meta.get("domain").startswith("cluster_"):
                        doc_meta["domain"] = f"cluster_{c_id}"
                    if not doc_meta.get("category") or doc_meta.get("category").startswith(
                        "cluster_"
                    ):
                        doc_meta["category"] = f"cluster_{c_id}"

            return {"tenant_id": tenant_id, "removed": removed, "merged_into": merged_into}
