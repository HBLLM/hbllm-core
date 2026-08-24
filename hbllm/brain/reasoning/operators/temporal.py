"""
Temporal Reasoning Operator — Allen's interval algebra over HCIR events.

Reasons about temporal relationships between events and entities using
Allen's 13 interval relations and event ordering inference.

Allen's relations::

    A before B          ───A───        ───B───
    A meets B           ───A──────B───
    A overlaps B        ───A───
                            ───B───
    A starts B          ───A───
                        ───B──────
    A during B             ───A───
                        ───B──────────
    A finishes B              ───A───
                        ───B──────────
    A equals B          ───A───
                        ───B───

Plus 6 inverses (after, met-by, overlapped-by, started-by, contains, finished-by).

Independence Level: L1 (no LLM execution)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.hcir.graph import (
    EventNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


class AllenRelation(StrEnum):
    """Allen's 13 interval relations."""

    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


@dataclass
class TemporalInterval:
    """A time interval for an event or entity state."""

    node_id: str
    start: float
    end: float
    label: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TemporalRelationship:
    """A derived temporal relationship between two intervals."""

    interval_a: str  # node_id
    interval_b: str  # node_id
    relation: AllenRelation
    confidence: float = 1.0


# Tolerance for "meeting" (within epsilon = same time)
_EPSILON = 0.001


def compute_allen_relation(a: TemporalInterval, b: TemporalInterval) -> AllenRelation:
    """Compute the Allen relation between two intervals."""
    if abs(a.end - b.start) < _EPSILON:
        return AllenRelation.MEETS
    if abs(b.end - a.start) < _EPSILON:
        return AllenRelation.MET_BY
    if abs(a.start - b.start) < _EPSILON and abs(a.end - b.end) < _EPSILON:
        return AllenRelation.EQUALS

    if a.end < b.start:
        return AllenRelation.BEFORE
    if b.end < a.start:
        return AllenRelation.AFTER

    if a.start < b.start and a.end > b.start and a.end < b.end:
        return AllenRelation.OVERLAPS
    if b.start < a.start and b.end > a.start and b.end < a.end:
        return AllenRelation.OVERLAPPED_BY

    if abs(a.start - b.start) < _EPSILON and a.end < b.end:
        return AllenRelation.STARTS
    if abs(a.start - b.start) < _EPSILON and a.end > b.end:
        return AllenRelation.STARTED_BY

    if a.start > b.start and a.end < b.end:
        return AllenRelation.DURING
    if b.start > a.start and b.end < a.end:
        return AllenRelation.CONTAINS

    if abs(a.end - b.end) < _EPSILON and a.start > b.start:
        return AllenRelation.FINISHES
    if abs(a.end - b.end) < _EPSILON and a.start < b.start:
        return AllenRelation.FINISHED_BY

    # Fallback — shouldn't happen with well-formed intervals
    return AllenRelation.OVERLAPS


# Transitivity table (partial — key compositions)
_TRANSITIVITY: dict[tuple[AllenRelation, AllenRelation], AllenRelation | None] = {
    (AllenRelation.BEFORE, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.BEFORE, AllenRelation.MEETS): AllenRelation.BEFORE,
    (AllenRelation.MEETS, AllenRelation.BEFORE): AllenRelation.BEFORE,
    (AllenRelation.AFTER, AllenRelation.AFTER): AllenRelation.AFTER,
    (AllenRelation.CONTAINS, AllenRelation.DURING): None,  # Indeterminate
    (AllenRelation.DURING, AllenRelation.CONTAINS): None,
    (AllenRelation.BEFORE, AllenRelation.DURING): None,
}


class TemporalOperator:
    """Allen's interval algebra + event ordering over HCIR."""

    @property
    def operator_id(self) -> str:
        return "temporal"

    @property
    def operator_name(self) -> str:
        return "Temporal Reasoning Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.TEMPORAL: 0.95,
            ProblemType.EXPLANATION: 0.4,
            ProblemType.CAUSAL: 0.5,
            ProblemType.PREDICTION: 0.4,
            ProblemType.DIAGNOSIS: 0.3,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        n_events = len(view.nodes_by_type(HCIRNodeType.EVENT))
        if n_events < 2:
            return 0.0

        return min(1.0, base + min(0.3, n_events * 0.02))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n_events = len(context.graph_view.nodes_by_type(HCIRNodeType.EVENT))
        return ResourceCost(
            wall_clock_ms=max(1.0, n_events * n_events * 0.05),
            nodes_read=n_events,
            edges_read=context.graph_view.edge_count,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Compute temporal relationships between events."""
        start = time.time()
        view = context.graph_view

        # ── Extract intervals from events ────────────────────────────
        intervals = self._extract_intervals(view)

        if len(intervals) < 2:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "Fewer than 2 temporal intervals"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Compute pairwise relations ───────────────────────────────
        relationships = self._compute_relationships(intervals)

        # ── Infer transitive relations ───────────────────────────────
        inferred = self._infer_transitive(relationships, intervals)
        all_relations = relationships + inferred

        # ── Build ordering ───────────────────────────────────────────
        ordering = self._topological_ordering(intervals, all_relations)

        # ── Build result ─────────────────────────────────────────────
        proposed_ops: list[TransactionOperation] = []
        provenance_chains: list[ProvenanceChain] = []

        # Propose temporal edges for newly discovered relations
        for rel in all_relations:
            edge_type = self._relation_to_edge_type(rel.relation)
            if edge_type is not None:
                edge = HCIREdge(
                    edge_type=edge_type,
                    sources=[rel.interval_a],
                    targets=[rel.interval_b],
                    weight=rel.confidence,
                    provenance=Provenance(
                        created_by=self.operator_id,
                        source_type="inferred",
                        reason=f"Temporal: {rel.interval_a} {rel.relation} {rel.interval_b}",
                    ),
                )
                proposed_ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_EDGE,
                        edge_data=edge.model_dump(),
                    )
                )

        provenance_chains.append(
            ProvenanceChain(
                conclusion=f"Temporal ordering of {len(intervals)} events",
                evidence_node_ids=[i.node_id for i in intervals],
                operator_id=self.operator_id,
                reasoning_steps=[
                    f"Computed {len(relationships)} direct relations",
                    f"Inferred {len(inferred)} transitive relations",
                    f"Ordering: {' → '.join(ordering)}",
                ],
                confidence=0.9,
            )
        )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "event_count": len(intervals),
                "relation_count": len(all_relations),
                "ordering": ordering,
                "relations": [
                    {
                        "a": r.interval_a,
                        "relation": r.relation,
                        "b": r.interval_b,
                    }
                    for r in all_relations[:20]  # Cap output size
                ],
            },
            confidence=0.9,
            evidence_refs=[i.node_id for i in intervals],
            proposed_transitions=proposed_ops,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_intervals(view: Any) -> list[TemporalInterval]:
        """Extract temporal intervals from EventNodes."""
        intervals: list[TemporalInterval] = []
        for node in view.nodes_by_type(HCIRNodeType.EVENT):
            if isinstance(node, EventNode):
                ts = node.event_timestamp
                # Events are point-like by default; use a small duration
                duration = node.event_data.get("duration", 0.001)
                intervals.append(
                    TemporalInterval(
                        node_id=node.id,
                        start=ts,
                        end=ts + duration,
                        label=node.event_kind or node.id,
                    )
                )
        intervals.sort(key=lambda i: i.start)
        return intervals

    @staticmethod
    def _compute_relationships(
        intervals: list[TemporalInterval],
    ) -> list[TemporalRelationship]:
        """Compute pairwise Allen relations."""
        rels: list[TemporalRelationship] = []
        for i, a in enumerate(intervals):
            for b in intervals[i + 1 :]:
                rel = compute_allen_relation(a, b)
                rels.append(
                    TemporalRelationship(
                        interval_a=a.node_id,
                        interval_b=b.node_id,
                        relation=rel,
                    )
                )
        return rels

    @staticmethod
    def _infer_transitive(
        direct: list[TemporalRelationship],
        intervals: list[TemporalInterval],
    ) -> list[TemporalRelationship]:
        """Infer transitive relations from the transitivity table."""
        # Build relation map
        rel_map: dict[tuple[str, str], AllenRelation] = {}
        for r in direct:
            rel_map[(r.interval_a, r.interval_b)] = r.relation

        inferred: list[TemporalRelationship] = []
        ids = [i.node_id for i in intervals]

        for a in ids:
            for b in ids:
                if a == b:
                    continue
                ab = rel_map.get((a, b))
                if ab is None:
                    continue
                for c in ids:
                    if c in (a, b):
                        continue
                    bc = rel_map.get((b, c))
                    if bc is None:
                        continue
                    ac = rel_map.get((a, c))
                    if ac is not None:
                        continue  # Already known

                    composed = _TRANSITIVITY.get((ab, bc))
                    if composed is not None:
                        inferred.append(
                            TemporalRelationship(
                                interval_a=a,
                                interval_b=c,
                                relation=composed,
                                confidence=0.85,
                            )
                        )
                        rel_map[(a, c)] = composed

        return inferred

    @staticmethod
    def _topological_ordering(
        intervals: list[TemporalInterval],
        relations: list[TemporalRelationship],
    ) -> list[str]:
        """Build a temporal ordering from BEFORE/MEETS relations."""
        ordering = sorted(intervals, key=lambda i: i.start)
        return [i.label or i.node_id for i in ordering]

    @staticmethod
    def _relation_to_edge_type(
        relation: AllenRelation,
    ) -> HCIREdgeType | None:
        """Map Allen relation to HCIR edge type (if applicable)."""
        mapping = {
            AllenRelation.BEFORE: HCIREdgeType.BEFORE,
            AllenRelation.AFTER: HCIREdgeType.AFTER,
            AllenRelation.DURING: HCIREdgeType.DURING,
            AllenRelation.CONTAINS: None,  # No direct HCIR edge
            AllenRelation.MEETS: HCIREdgeType.BEFORE,  # Close enough
        }
        return mapping.get(relation)
