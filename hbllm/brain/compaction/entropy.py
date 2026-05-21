"""Cognitive Entropy Engine.

Tracks the 'mental health' of the AI by monitoring graph density,
stale node ratios, unused memory, and semantic duplication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntropyMetrics:
    """Metrics tracking the degradation of the cognitive runtime."""

    graph_density: float = 0.0  # How interconnected the causal graph is
    stale_node_ratio: float = 0.0  # Nodes not accessed in > 7 days
    unused_memory_ratio: float = 0.0  # Memory nodes never verified
    causal_drift: float = 0.0  # Rate of unexpected causal outcomes
    simulation_branch_explosion: float = 0.0  # Average branches per plan


class CognitiveEntropyEngine:
    """Calculates and exposes the system's entropy score."""

    def __init__(self) -> None:
        self.metrics = EntropyMetrics()

    def update_metrics(self, metrics: EntropyMetrics) -> None:
        """Update the underlying entropy metrics."""
        self.metrics = metrics

    def get_system_entropy_score(self) -> float:
        """Calculate the global entropy score (0.0 to 1.0).

        High entropy (>0.7) means the system is cluttered, slow, or drifting,
        and requires aggressive cognitive compaction.
        """
        # Normalize and weight the metrics
        score = (
            (min(1.0, self.metrics.graph_density) * 0.1)
            + (min(1.0, self.metrics.stale_node_ratio) * 0.4)
            + (min(1.0, self.metrics.unused_memory_ratio) * 0.2)
            + (min(1.0, self.metrics.causal_drift) * 0.2)
            + (min(1.0, self.metrics.simulation_branch_explosion / 20.0) * 0.1)
        )
        return min(1.0, max(0.0, score))
