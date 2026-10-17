"""World Causal Graph — Causal Chains, Failure Hypotheses, & Relational Edges.

Tracks causal failure chains and hypothesis dependencies, lifted to declarative
Pearl CausalPredicates, HCIRSymbolicWorldTheory, and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from hbllm.hcir.transactions import TransactionOp, TransactionOperation

if TYPE_CHECKING:
    from hbllm.hcir.world.causal_discovery import CausalPredicate
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

logger = logging.getLogger(__name__)


class CausalEdgeType(StrEnum):
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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export causal edge as a declarative causal predicate."""
        from hbllm.hcir.world.causal_discovery import CausalPredicate

        return CausalPredicate(
            variable=f"causal_edge:{self.source_id}->{self.target_id}",
            operator="==",
            value=self.relationship.value,
            relation=",".join(self.factors) if self.factors else "direct",
            target_entity=self.target_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction adding causal edge to semantic graph."""
        return [
            TransactionOperation(
                op=TransactionOp.ADD_EDGE,
                node_id=f"edge_{self.source_id}_{self.target_id}",
                node_data={
                    "source_id": self.source_id,
                    "target_id": self.target_id,
                    "relationship": self.relationship.value,
                    "weight": self.weight,
                    "factors": list(self.factors),
                },
            )
        ]


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

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all edges as declarative causal predicates."""
        return [e.to_causal_predicate() for e in self._edges]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting the causal graph."""
        ops: list[TransactionOperation] = []
        for e in self._edges:
            ops.extend(e.to_transaction_operations())
        return ops

    def to_world_theory(self) -> HCIRSymbolicWorldTheory:
        """Synthesize an empirical HCIRSymbolicWorldTheory from causal graph edges."""
        from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

        theory = HCIRSymbolicWorldTheory()
        theory.goal_predicates.extend(self.export_causal_predicates())
        return theory
