"""Dentate Gyrus pattern separation and CA3 recurrent attractor networks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class PatternSeparationResult:
    """Result of Dentate Gyrus sparse projection and pattern separation."""

    sparse_code: np.ndarray
    sparsity_ratio: float
    expansion_dim: int
    orthogonal_distance: float


def _project_vector_to_dim(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Project arbitrary sensory or memory vector to target dimension preserving full profile (WP-C6)."""
    flat = vector.flatten().astype(np.float32)
    if len(flat) == target_dim:
        return flat
    if len(flat) == 0:
        return np.zeros(target_dim, dtype=np.float32)
    if len(flat) > target_dim:
        # 1D adaptive average pooling across sensory profile
        indices = np.linspace(0, len(flat), target_dim + 1, dtype=int)
        pooled = np.array(
            [
                float(np.mean(flat[indices[i] : max(indices[i] + 1, indices[i + 1])]))
                for i in range(target_dim)
            ],
            dtype=np.float32,
        )
        return pooled
    # Linear interpolation to expand lower-dimensional profile
    x_old = np.linspace(0.0, 1.0, len(flat))
    x_new = np.linspace(0.0, 1.0, target_dim)
    return np.interp(x_new, x_old, flat).astype(np.float32)


class DentateGyrusPatternSeparator:
    """Hippocampal Dentate Gyrus (DG) sparse expansion and pattern separator."""

    def __init__(self, input_dim: int = 64, expansion_dim: int = 256, k_active: int = 24) -> None:
        self.input_dim = input_dim
        self.expansion_dim = expansion_dim
        self.k_active = k_active
        rng = np.random.RandomState(42)
        self.weights = rng.randn(expansion_dim, input_dim).astype(np.float32)
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights /= np.maximum(norms, 1e-6)

    def separate_pattern(self, sensory_vector: np.ndarray) -> PatternSeparationResult:
        """Project input vector into high-dimensional sparse orthogonal representation."""
        flat = _project_vector_to_dim(sensory_vector, self.input_dim)

        proj = np.dot(self.weights, flat)
        top_k_indices = np.argsort(proj)[-self.k_active :]
        sparse = np.zeros(self.expansion_dim, dtype=np.float32)
        sparse[top_k_indices] = proj[top_k_indices]
        sparse_norm = float(np.linalg.norm(sparse))
        if sparse_norm > 1e-6:
            sparse /= sparse_norm

        sparsity = float(self.k_active / self.expansion_dim)
        return PatternSeparationResult(
            sparse_code=sparse,
            sparsity_ratio=sparsity,
            expansion_dim=self.expansion_dim,
            orthogonal_distance=float(np.mean(np.abs(sparse))),
        )


@dataclass
class PatternCompletionResult:
    """Result of CA3 auto-associative pattern completion."""

    completed_pattern: np.ndarray
    retrieved_memory_id: str | None
    retrieval_confidence: float
    iterations: int


class CA3RecurrentPatternCompleter:
    """Hippocampal CA3 recurrent auto-associative memory for episodic pattern completion."""

    def __init__(self, capacity: int = 64, feature_dim: int = 256) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.memories: dict[str, np.ndarray] = {}
        self.associated_plans: dict[str, list[Any]] = {}

    def store_pattern(
        self, memory_id: str, pattern: np.ndarray, plan: list[Any] | None = None
    ) -> None:
        """Store an episodic attractor pattern with associated winning plan."""
        pat = _project_vector_to_dim(pattern, self.feature_dim)
        norm = float(np.linalg.norm(pat))
        if norm > 1e-6:
            pat /= norm
        self.memories[memory_id] = pat
        if plan is not None:
            self.associated_plans[memory_id] = list(plan)

    def complete_pattern(
        self,
        partial_cue: np.ndarray,
        match_threshold: float = 0.35,
    ) -> PatternCompletionResult:
        """Retrieve full stored pattern and plan from noisy or partial sensory cue."""
        if not self.memories:
            return PatternCompletionResult(
                completed_pattern=partial_cue,
                retrieved_memory_id=None,
                retrieval_confidence=0.0,
                iterations=0,
            )

        cue = _project_vector_to_dim(partial_cue, self.feature_dim)
        cue_norm = float(np.linalg.norm(cue))
        if cue_norm > 1e-6:
            cue /= cue_norm

        best_id: str | None = None
        best_sim = -1.0
        for mem_id, stored_pat in self.memories.items():
            sim = float(np.dot(cue, stored_pat))
            if sim > best_sim:
                best_sim = sim
                best_id = mem_id

        if best_sim >= match_threshold and best_id is not None:
            return PatternCompletionResult(
                completed_pattern=self.memories[best_id].copy(),
                retrieved_memory_id=best_id,
                retrieval_confidence=best_sim,
                iterations=1,
            )

        return PatternCompletionResult(
            completed_pattern=cue,
            retrieved_memory_id=None,
            retrieval_confidence=max(0.0, best_sim),
            iterations=1,
        )


class HippocampalAttractorNetwork:
    """Hippocampal trisynaptic circuit integrating DG pattern separation and CA3 pattern completion."""

    def __init__(
        self,
        input_dim: int = 64,
        expansion_dim: int = 256,
        k_active: int = 24,
        capacity: int = 64,
    ) -> None:
        self.dg_separator = DentateGyrusPatternSeparator(
            input_dim=input_dim, expansion_dim=expansion_dim, k_active=k_active
        )
        self.ca3_completer = CA3RecurrentPatternCompleter(
            capacity=capacity, feature_dim=expansion_dim
        )

    def store_episode_attractor(
        self,
        memory_id: str,
        sensory_state: np.ndarray,
        action_plan: list[Any] | None = None,
    ) -> PatternSeparationResult:
        """Separate sensory pattern via Dentate Gyrus and store in CA3 auto-associative memory."""
        sep_res = self.dg_separator.separate_pattern(sensory_state)
        self.ca3_completer.store_pattern(
            memory_id=memory_id, pattern=sep_res.sparse_code, plan=action_plan
        )
        return sep_res

    def recall_from_partial_cue(
        self,
        partial_sensory_cue: np.ndarray,
        match_threshold: float = 0.35,
    ) -> tuple[list[Any] | None, PatternCompletionResult]:
        """Project partial cue via DG and retrieve complete episodic action sequence from CA3."""
        sep_cue = self.dg_separator.separate_pattern(partial_sensory_cue)
        comp_res = self.ca3_completer.complete_pattern(
            partial_cue=sep_cue.sparse_code, match_threshold=match_threshold
        )
        plan = None
        if comp_res.retrieved_memory_id is not None:
            plan = self.ca3_completer.associated_plans.get(comp_res.retrieved_memory_id)
        return plan, comp_res

    def to_causal_predicate(self) -> CausalPredicate:
        """Export attractor network status as a declarative causal predicate."""
        return CausalPredicate(
            variable="hippocampal_attractor_network",
            operator="==",
            value=f"stored_memories:{len(self.ca3_completer.memories)}",
            relation="dg_pattern_separation_ca3_completion",
            target_entity="hippocampus_trisynaptic",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting hippocampal attractor network state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="hippocampal_attractor_network",
                node_data={
                    "id": "hippocampal_attractor_network",
                    "node_type": "memory",
                    "category": "episodic",
                    "memory_count": len(self.ca3_completer.memories),
                },
            )
        ]


__all__ = [
    "CA3RecurrentPatternCompleter",
    "DentateGyrusPatternSeparator",
    "HippocampalAttractorNetwork",
    "PatternCompletionResult",
    "PatternSeparationResult",
    "_project_vector_to_dim",
]
