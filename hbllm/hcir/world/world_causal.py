"""
World Causal Graph — Causal Chains, Failure Hypotheses, & Relational Edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CausalEdgeType(str, Enum):
    """Types of causal relationships between world entities and variables."""

    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    CORRELATES_WITH = "correlates_with"


@dataclass
class CausalEdge:
    """Directed edge representing causal relationship."""

    source_id: str
    target_id: str
    relationship: CausalEdgeType
    weight: float = 1.0
    factors: list[str] = field(default_factory=list)


class WorldCausalGraph:
    """Graph tracking causal failure chains and hypothesis dependencies."""

    def __init__(self, world_id: str = "default_world") -> None:
        self.world_id = world_id
        self._edges: list[CausalEdge] = []

    def add_causal_relation(
        self,
        source_id: str,
        target_id: str,
        relationship: CausalEdgeType = CausalEdgeType.CAUSES,
        weight: float = 1.0,
        factors: list[str] | None = None,
    ) -> CausalEdge:
        """Record a causal relationship edge."""
        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            factors=list(factors) if factors else [],
        )
        self._edges.append(edge)
        logger.debug(
            "WorldCausalGraph [%s] added edge %s -[%s]-> %s (factors=%s)",
            self.world_id,
            source_id,
            relationship.value,
            target_id,
            edge.factors,
        )
        return edge

    def get_causes_for(self, target_id: str) -> list[CausalEdge]:
        """Retrieve incoming causal edges pointing to target node."""
        return [
            e
            for e in self._edges
            if e.target_id == target_id and e.relationship == CausalEdgeType.CAUSES
        ]

    def get_composite_causes_for(self, target_id: str) -> list[CausalEdge]:
        """Retrieve incoming causal edges pointing to target node, including composite factors."""
        return [
            e
            for e in self._edges
            if e.target_id == target_id and e.relationship == CausalEdgeType.CAUSES
        ]

    def get_effects_of(self, source_id: str) -> list[CausalEdge]:
        """Retrieve outgoing causal edges originating from source node."""
        return [e for e in self._edges if e.source_id == source_id]
