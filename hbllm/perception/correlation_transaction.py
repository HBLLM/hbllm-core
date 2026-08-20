"""Correlation Transaction — commits cross-modal associations to HCIR.

Takes CorrelationCandidate from the CorrelationEngine and creates
CORRELATES_WITH hyperedges between observation nodes in the cognitive graph.

CORRELATES_WITH semantics:
    - Temporal/spatial association ONLY
    - NOT causal ("the person made the footsteps")
    - NOT identity ("this is the same thing")
    - Edge metadata carries: confidence, temporal_overlap, spatial_overlap, created_by

The observations remain independent nodes. The edge says:
    "These observations are statistically aligned."

That's an important distinction. The epistemic layer later decides
whether the correlation implies causation, identity, or coincidence.
"""

from __future__ import annotations

import logging
import time

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType
from hbllm.perception.correlation_engine import CorrelationCandidate

logger = logging.getLogger(__name__)


class CorrelationTransaction:
    """Commits cross-modal correlation candidates to HCIR.

    Creates CORRELATES_WITH edges between existing observation nodes.
    Never creates new observation nodes — those are created by
    modality-specific transactions (AudioPerceptionTransaction, etc.).

    Usage::

        tx = CorrelationTransaction(graph=graph)
        edge = tx.commit_correlation(candidate)

    """

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph

    def commit_correlation(
        self,
        candidate: CorrelationCandidate,
    ) -> HCIREdge | None:
        """Commit a correlation candidate as a CORRELATES_WITH edge.

        Returns the created edge, or None if the observations
        don't exist in the graph.

        Args:
            candidate: The correlation candidate to commit.

        Returns:
            The created HCIREdge, or None.

        """
        # Verify both observation nodes exist
        source_node = self._graph.get_node(candidate.source_observation_id)
        target_node = self._graph.get_node(candidate.target_observation_id)

        if source_node is None or target_node is None:
            logger.debug(
                "Skipping correlation: node(s) not found (%s, %s)",
                candidate.source_observation_id,
                candidate.target_observation_id,
            )
            return None

        # Build edge properties
        properties: dict[str, object] = {
            "confidence": candidate.score,
            "temporal_overlap": candidate.temporal_overlap,
            "delta_time_ms": candidate.delta_time_ms,
            "source_modality": candidate.source_modality,
            "target_modality": candidate.target_modality,
            "created_by": "correlation_engine",
            "created_at": time.time(),
        }

        if candidate.spatial_overlap is not None:
            properties["spatial_overlap"] = candidate.spatial_overlap

        edge = HCIREdge(
            edge_type=HCIREdgeType.CORRELATES_WITH,
            sources=[candidate.source_observation_id],
            targets=[candidate.target_observation_id],
            properties=properties,
        )

        self._graph.add_edge(edge)

        logger.info(
            "Committed CORRELATES_WITH edge: %s (%s) → %s (%s), score=%.3f",
            candidate.source_observation_id,
            candidate.source_modality,
            candidate.target_observation_id,
            candidate.target_modality,
            candidate.score,
        )

        return edge

    def commit_batch(
        self,
        candidates: list[CorrelationCandidate],
        min_score: float = 0.3,
    ) -> list[HCIREdge]:
        """Commit multiple correlation candidates.

        Only commits candidates above min_score threshold.

        Args:
            candidates: List of correlation candidates.
            min_score: Minimum score threshold.

        Returns:
            List of created edges.

        """
        edges: list[HCIREdge] = []
        for candidate in candidates:
            if candidate.score >= min_score:
                edge = self.commit_correlation(candidate)
                if edge is not None:
                    edges.append(edge)
        return edges
