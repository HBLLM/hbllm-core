"""Cognitive Compaction Engine.

Handles memory importance scoring, attention-based forgetting, and
multi-stage semantic folding to manage graph entropy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """A generic wrapper for any graph node to track compaction metadata."""

    node_id: str
    content: Any

    # Importance Scoring Metrics
    emotional_weight: float = 0.0
    causal_centrality: float = 0.0
    recurrence: int = 1
    user_relevance: float = 0.0
    failure_association: float = 0.0
    novelty: float = 0.0

    # Attention-based forgetting
    activation_strength: float = 1.0


class MemoryImportanceScorer:
    """Scores nodes to determine survival during compaction."""

    def score(self, node: MemoryNode) -> float:
        """Calculate the importance score of a memory node."""
        score = (
            (node.emotional_weight * 2.0)
            + (node.causal_centrality * 1.5)
            + (min(10.0, node.recurrence) * 0.5)
            + (node.user_relevance * 1.5)
            + (node.failure_association * 2.5)  # Failures matter forever
            + (node.novelty * 1.0)
        )
        return score


class SemanticFolder:
    """Implements Multi-Stage Semantic Folding."""

    def __init__(self, llm_labeler: Any | None = None) -> None:
        self.llm_labeler = llm_labeler
        self.heuristic_clusters: dict[str, list[MemoryNode]] = {}

    def heuristic_clustering(self, events: list[MemoryNode]) -> None:
        """Stage 1: Safe, deterministic grouping of repeated events."""
        for event in events:
            # Example heuristic: hash the core event structure ignoring timestamp
            cluster_hash = self._generate_heuristic_hash(event.content)
            if cluster_hash not in self.heuristic_clusters:
                self.heuristic_clusters[cluster_hash] = []
            self.heuristic_clusters[cluster_hash].append(event)

    def _generate_heuristic_hash(self, content: Any) -> str:
        """Mock heuristic hash generation."""
        if isinstance(content, dict) and "action" in content:
            return content["action"]
        return "unknown_cluster"

    def label_clusters(self) -> dict[str, str]:
        """Stage 2: LLM simply names the deterministic clusters."""
        labels = {}
        for cluster_hash, nodes in self.heuristic_clusters.items():
            if len(nodes) >= 3:
                # Only label highly recurring clusters
                if self.llm_labeler:
                    labels[cluster_hash] = self.llm_labeler.label(nodes)
                else:
                    labels[cluster_hash] = f"Routine_{cluster_hash.capitalize()}"
        return labels


class CognitiveCompactionEngine:
    """Main garbage collection and memory optimization layer."""

    def __init__(self, decay_rate: float = 0.95) -> None:
        self.scorer = MemoryImportanceScorer()
        self.folder = SemanticFolder()
        self.decay_rate = decay_rate

        self.nodes: dict[str, MemoryNode] = {}
        self.cold_storage: dict[str, MemoryNode] = {}

    def apply_attention_decay(self) -> None:
        """Multiply activation_strength by decay_rate."""
        for node in self.nodes.values():
            node.activation_strength *= self.decay_rate

    def compact(self, current_memory_pressure: float) -> None:
        """Run the full compaction cycle."""
        logger.info("Starting Cognitive Compaction (Pressure: %.2f)", current_memory_pressure)

        self.apply_attention_decay()

        # Prune dead/unimportant nodes
        keys_to_remove = []
        for node_id, node in self.nodes.items():
            importance = self.scorer.score(node)

            # If importance is low and activation strength has decayed, archive it
            if importance < 3.0 and node.activation_strength < 0.2:
                self.cold_storage[node_id] = node
                keys_to_remove.append(node_id)

        for k in keys_to_remove:
            del self.nodes[k]

        logger.info("Compaction completed. Archived %d nodes.", len(keys_to_remove))
