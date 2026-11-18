"""Visual Memory — vector index for visual embeddings.

A self-contained visual embedding store that manages observation and
prototype vectors separately from the text-oriented SemanticMemory.

Architecture:
    search_observations(): PRIMARY — raw evidence retrieval
    search_prototypes():   SECONDARY — fast coarse retrieval accelerator
    derive_concept_candidates(): group observations → concepts + ranking

Invariant:
    VisualMemory indexes evidence (vectors).
    HCIR owns identity/state (nodes, edges).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict

import numpy as np

from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    ObservationMatch,
)
from hbllm.perception.providers.policy import RecognitionPolicy
from hbllm.perception.providers.types import VisualEmbedding

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _VectorEntry:
    """Internal storage for a single embedding vector."""

    __slots__ = (
        "ref_id",
        "vector",
        "space_id",
        "concept_node_id",
        "label",
        "timestamp",
        "entry_type",
    )

    def __init__(
        self,
        ref_id: str,
        vector: np.ndarray,
        space_id: str,
        concept_node_id: str | None,
        label: str,
        timestamp: float,
        entry_type: str,  # "observation" | "prototype"
    ) -> None:
        self.ref_id = ref_id
        self.vector = vector
        self.space_id = space_id
        self.concept_node_id = concept_node_id
        self.label = label
        self.timestamp = timestamp
        self.entry_type = entry_type


class VisualMemory:
    """Visual embedding index — observations first, prototypes for acceleration.

    This is a self-contained vector store for visual embeddings, separate
    from the text-oriented SemanticMemory.  It manages two collections:

        observations: Individual visual evidence embeddings
        prototypes:   Concept centroid vectors for fast coarse retrieval

    Usage::

        memory = VisualMemory()

        # Store an observation
        ref = await memory.store_observation(embedding, concept_id, "cup")

        # Search for similar observations
        matches = await memory.search_observations(query_embedding, top_k=10)

        # Derive concept candidates from observation matches
        candidates, ranking = memory.derive_concept_candidates(matches)
    """

    def __init__(self) -> None:
        self._entries: dict[str, _VectorEntry] = {}  # ref_id → entry
        self._observations: list[str] = []  # ref_ids (ordered)
        self._prototypes: list[str] = []  # ref_ids (ordered)

    async def store_observation(
        self,
        embedding: VisualEmbedding,
        concept_node_id: str | None,
        label: str,
    ) -> str:
        """Store an observation embedding. Returns the ref_id."""
        ref_id = f"vobs_{uuid.uuid4().hex[:12]}"
        entry = _VectorEntry(
            ref_id=ref_id,
            vector=np.array(embedding.vector, dtype=np.float32),
            space_id=embedding.space_id,
            concept_node_id=concept_node_id,
            label=label,
            timestamp=time.time(),
            entry_type="observation",
        )
        self._entries[ref_id] = entry
        self._observations.append(ref_id)
        logger.debug("Stored observation %s (label=%s, concept=%s)", ref_id, label, concept_node_id)
        return ref_id

    async def store_prototype(
        self,
        concept_node_id: str,
        centroid: list[float],
        embedding: VisualEmbedding,
    ) -> str:
        """Store a prototype (centroid) vector for fast coarse retrieval."""
        ref_id = f"vproto_{uuid.uuid4().hex[:12]}"
        entry = _VectorEntry(
            ref_id=ref_id,
            vector=np.array(centroid, dtype=np.float32),
            space_id=embedding.space_id,
            concept_node_id=concept_node_id,
            label="",
            timestamp=time.time(),
            entry_type="prototype",
        )
        self._entries[ref_id] = entry
        self._prototypes.append(ref_id)
        logger.debug("Stored prototype %s for concept %s", ref_id, concept_node_id)
        return ref_id

    async def search_observations(
        self,
        embedding: VisualEmbedding,
        top_k: int = 10,
    ) -> list[ObservationMatch]:
        """Search for similar observations — primary search.

        Returns observations sorted by descending similarity.
        Only searches within the same embedding space.
        """
        query = np.array(embedding.vector, dtype=np.float32)
        matches: list[tuple[float, _VectorEntry]] = []

        for ref_id in self._observations:
            entry = self._entries[ref_id]
            if entry.space_id != embedding.space_id:
                continue
            sim = _cosine_similarity(query, entry.vector)
            matches.append((sim, entry))

        # Sort by descending similarity
        matches.sort(key=lambda x: x[0], reverse=True)

        return [
            ObservationMatch(
                observation_ref=entry.ref_id,
                similarity=sim,
                concept_node_id=entry.concept_node_id,
                label=entry.label,
                timestamp=entry.timestamp,
            )
            for sim, entry in matches[:top_k]
        ]

    async def search_prototypes(
        self,
        embedding: VisualEmbedding,
        top_k: int = 5,
    ) -> list[ObservationMatch]:
        """Search prototypes — fast coarse retrieval for scaling."""
        query = np.array(embedding.vector, dtype=np.float32)
        matches: list[tuple[float, _VectorEntry]] = []

        for ref_id in self._prototypes:
            entry = self._entries[ref_id]
            if entry.space_id != embedding.space_id:
                continue
            sim = _cosine_similarity(query, entry.vector)
            matches.append((sim, entry))

        matches.sort(key=lambda x: x[0], reverse=True)

        return [
            ObservationMatch(
                observation_ref=entry.ref_id,
                similarity=sim,
                concept_node_id=entry.concept_node_id,
                label=entry.label,
                timestamp=entry.timestamp,
            )
            for sim, entry in matches[:top_k]
        ]

    def derive_concept_candidates(
        self,
        observation_matches: list[ObservationMatch],
    ) -> tuple[list[ConceptCandidate], CandidateRanking]:
        """Group observation matches by concept and compute ranking.

        Observations without a concept_node_id are ignored for
        candidate derivation (they're unassigned evidence).

        Returns:
            (candidates sorted by best_similarity, ranking with ambiguity)

        """
        if not observation_matches:
            return [], CandidateRanking()

        # Group by concept_node_id
        groups: dict[str, list[ObservationMatch]] = defaultdict(list)
        for match in observation_matches:
            if match.concept_node_id:
                groups[match.concept_node_id].append(match)

        if not groups:
            return [], CandidateRanking()

        # Build candidates
        candidates: list[ConceptCandidate] = []
        for concept_id, obs_list in groups.items():
            sims = [o.similarity for o in obs_list]
            candidates.append(
                ConceptCandidate(
                    concept_node_id=concept_id,
                    label=obs_list[0].label,
                    mean_similarity=sum(sims) / len(sims),
                    best_similarity=max(sims),
                    matching_observations=len(obs_list),
                )
            )

        # Sort by best similarity descending
        candidates.sort(key=lambda c: c.best_similarity, reverse=True)

        # Build ranking with ambiguity signal
        scores = [c.best_similarity for c in candidates]
        ranking = CandidateRanking.from_scores(scores)

        return candidates, ranking

    async def add_exemplar(
        self,
        concept_node_id: str,
        embedding: VisualEmbedding,
        policy: RecognitionPolicy,
    ) -> str | None:
        """Add an exemplar if under limit and sufficiently diverse.

        Returns ref_id if stored, None if skipped (duplicate or over limit).
        """
        # Count existing exemplars for this concept
        existing = [
            self._entries[ref_id]
            for ref_id in self._observations
            if self._entries[ref_id].concept_node_id == concept_node_id
        ]

        if len(existing) >= policy.exemplar_limit:
            logger.debug(
                "Exemplar limit reached for concept %s (%d/%d)",
                concept_node_id,
                len(existing),
                policy.exemplar_limit,
            )
            return None

        # Check diversity — skip near-duplicates
        query = np.array(embedding.vector, dtype=np.float32)
        for entry in existing:
            if entry.space_id != embedding.space_id:
                continue
            sim = _cosine_similarity(query, entry.vector)
            if sim >= policy.exemplar_diversity_threshold:
                logger.debug("Near-duplicate exemplar (sim=%.3f) — skipping", sim)
                return None

        return await self.store_observation(
            embedding, concept_node_id, existing[0].label if existing else ""
        )

    async def update_prototype(
        self,
        concept_node_id: str,
        new_embedding: VisualEmbedding,
        count: int,
    ) -> None:
        """Update prototype centroid with running average."""
        for ref_id in self._prototypes:
            entry = self._entries[ref_id]
            if entry.concept_node_id == concept_node_id:
                new_vec = np.array(new_embedding.vector, dtype=np.float32)
                # Running average: centroid = (old * (n-1) + new) / n
                entry.vector = (entry.vector * (count - 1) + new_vec) / count
                # Re-normalize
                norm = float(np.linalg.norm(entry.vector))
                if norm > 0:
                    entry.vector = entry.vector / norm
                logger.debug("Updated prototype for concept %s (n=%d)", concept_node_id, count)
                return

    def get_observation_count(self, concept_node_id: str) -> int:
        """Count observations for a concept."""
        return sum(
            1
            for ref_id in self._observations
            if self._entries[ref_id].concept_node_id == concept_node_id
        )

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def prototype_count(self) -> int:
        return len(self._prototypes)
