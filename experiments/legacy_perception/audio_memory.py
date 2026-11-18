"""Audio Memory — acoustic embedding index for one-shot learning.

Analogous to VisualMemory. Stores audio embeddings and supports
cosine-similarity search for observation matching and concept
prototype acceleration.

Architecture:
    Observation-first index:
        All embeddings are stored as observations.
        Concept prototypes are derived from observations.

    Memory never mutates HCIR. It is a local index that the
    AudioPerceptionRuntime queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from hbllm.perception.providers.audio_types import AudioEmbedding

logger = logging.getLogger(__name__)


@dataclass
class AudioObservationEntry:
    """A stored audio observation in the memory index.

    Attributes:
        observation_id: Unique observation ID.
        embedding: The audio embedding vector.
        concept_label: Label (if assigned to a concept).
        metadata: Additional metadata.

    """

    observation_id: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    concept_label: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AudioConceptPrototype:
    """Running-average prototype for an acoustic concept.

    Attributes:
        concept_label: The label for this concept.
        prototype: L2-normalized centroid of all exemplars.
        observation_count: Number of exemplars.
        exemplar_refs: References to exemplar observation IDs.

    """

    concept_label: str = ""
    prototype: np.ndarray = field(default_factory=lambda: np.array([]))
    observation_count: int = 0
    exemplar_refs: list[str] = field(default_factory=list)


@dataclass
class AudioSearchResult:
    """Result of a similarity search.

    Attributes:
        observation_id: Matched observation ID.
        similarity: Cosine similarity score.
        concept_label: Label of matched observation (if any).

    """

    observation_id: str = ""
    similarity: float = 0.0
    concept_label: str | None = None


class AudioMemory:
    """Audio embedding index — observation-first with prototype acceleration.

    Usage::

        memory = AudioMemory()
        memory.store_observation("obs1", embedding, concept_label="doorbell")
        results = memory.search_observations(query_embedding, top_k=5)

    """

    def __init__(self, max_exemplars_per_concept: int = 20) -> None:
        self._observations: dict[str, AudioObservationEntry] = {}
        self._prototypes: dict[str, AudioConceptPrototype] = {}
        self._max_exemplars = max_exemplars_per_concept

    # ── Observation storage ────────────────────────────────────────────

    def store_observation(
        self,
        observation_id: str,
        embedding: AudioEmbedding,
        concept_label: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Store an audio observation.

        Args:
            observation_id: Unique ID for this observation.
            embedding: Audio embedding to store.
            concept_label: Optional concept label.
            metadata: Optional metadata.

        """
        vec = np.array(embedding.vector, dtype=np.float32)
        entry = AudioObservationEntry(
            observation_id=observation_id,
            embedding=vec,
            concept_label=concept_label,
            metadata=metadata or {},
        )
        self._observations[observation_id] = entry

    def search_observations(
        self,
        query: AudioEmbedding,
        top_k: int = 5,
        concept_filter: str | None = None,
    ) -> list[AudioSearchResult]:
        """Search observations by cosine similarity.

        Args:
            query: Query embedding.
            top_k: Number of results to return.
            concept_filter: If set, only return observations with this label.

        Returns:
            Sorted list of search results (highest similarity first).

        """
        query_vec = np.array(query.vector, dtype=np.float32)
        results: list[AudioSearchResult] = []

        for entry in self._observations.values():
            if concept_filter and entry.concept_label != concept_filter:
                continue
            sim = float(np.dot(query_vec, entry.embedding))
            results.append(
                AudioSearchResult(
                    observation_id=entry.observation_id,
                    similarity=sim,
                    concept_label=entry.concept_label,
                ),
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    # ── Prototype management ───────────────────────────────────────────

    def store_prototype(
        self,
        concept_label: str,
        embedding: AudioEmbedding,
    ) -> None:
        """Store or initialize a concept prototype.

        Args:
            concept_label: The concept label.
            embedding: Initial prototype embedding.

        """
        vec = np.array(embedding.vector, dtype=np.float32)
        self._prototypes[concept_label] = AudioConceptPrototype(
            concept_label=concept_label,
            prototype=vec,
            observation_count=1,
            exemplar_refs=[],
        )

    def update_prototype(
        self,
        concept_label: str,
        new_embedding: AudioEmbedding,
    ) -> None:
        """Update prototype with running average.

        Args:
            concept_label: Concept to update.
            new_embedding: New exemplar embedding.

        """
        if concept_label not in self._prototypes:
            self.store_prototype(concept_label, new_embedding)
            return

        proto = self._prototypes[concept_label]
        new_vec = np.array(new_embedding.vector, dtype=np.float32)
        n = proto.observation_count
        updated = (proto.prototype * n + new_vec) / (n + 1)
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm
        proto.prototype = updated
        proto.observation_count = n + 1

    def search_prototypes(
        self,
        query: AudioEmbedding,
        top_k: int = 5,
    ) -> list[AudioSearchResult]:
        """Search concept prototypes by cosine similarity.

        Args:
            query: Query embedding.
            top_k: Number of results.

        Returns:
            Sorted list of prototype matches.

        """
        query_vec = np.array(query.vector, dtype=np.float32)
        results: list[AudioSearchResult] = []

        for proto in self._prototypes.values():
            sim = float(np.dot(query_vec, proto.prototype))
            results.append(
                AudioSearchResult(
                    observation_id="",
                    similarity=sim,
                    concept_label=proto.concept_label,
                ),
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def add_exemplar(
        self,
        concept_label: str,
        observation_id: str,
        embedding: AudioEmbedding,
        min_diversity: float = 0.1,
    ) -> bool:
        """Add an exemplar to a concept with diversity enforcement.

        Args:
            concept_label: Concept to add to.
            observation_id: The observation being added.
            embedding: Embedding of the new exemplar.
            min_diversity: Minimum distance from existing exemplars.

        Returns:
            True if added, False if too similar or at capacity.

        """
        if concept_label not in self._prototypes:
            return False

        proto = self._prototypes[concept_label]
        if len(proto.exemplar_refs) >= self._max_exemplars:
            return False

        # Check diversity against existing exemplars
        new_vec = np.array(embedding.vector, dtype=np.float32)
        for ref in proto.exemplar_refs:
            if ref in self._observations:
                existing = self._observations[ref].embedding
                sim = float(np.dot(new_vec, existing))
                if 1.0 - sim < min_diversity:
                    return False

        proto.exemplar_refs.append(observation_id)
        self.update_prototype(concept_label, embedding)
        return True

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def observation_count(self) -> int:
        """Total observations in memory."""
        return len(self._observations)

    @property
    def concept_count(self) -> int:
        """Total concepts with prototypes."""
        return len(self._prototypes)
