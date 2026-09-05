"""
Spatial Reasoning Operator — qualitative spatial reasoning over HCIR.

Reasons about spatial relationships between physical entities using
Region Connection Calculus (RCC-8) and qualitative direction/distance.

RCC-8 relations::

    DC   - Disconnected
    EC   - Externally Connected
    PO   - Partial Overlap
    EQ   - Equal
    TPP  - Tangential Proper Part
    NTPP - Non-Tangential Proper Part
    TPPi - Tangential Proper Part inverse
    NTPPi- Non-Tangential Proper Part inverse

Qualitative spatial predicates::

    contains, inside, on_top_of, above, below, adjacent_to,
    left_of, right_of, near, far

Inference rules::

    A contains B, B contains C → A contains C
    A above B, B above C → A above C
    A adjacent_to B → B adjacent_to A  (symmetric)

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
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


class SpatialRelation(StrEnum):
    """Qualitative spatial relations."""

    CONTAINS = "contains"
    INSIDE = "inside"
    ON_TOP_OF = "on_top_of"
    ABOVE = "above"
    BELOW = "below"
    ADJACENT_TO = "adjacent_to"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    NEAR = "near"
    FAR = "far"
    SUPPORTS = "supports"
    SUPPORTED_BY = "supported_by"


# Transitive relations: if R(A,B) and R(B,C), then R(A,C)
_TRANSITIVE_RELATIONS = frozenset(
    {
        SpatialRelation.CONTAINS,
        SpatialRelation.INSIDE,
        SpatialRelation.ABOVE,
        SpatialRelation.BELOW,
        SpatialRelation.LEFT_OF,
        SpatialRelation.RIGHT_OF,
    }
)

# Symmetric relations: if R(A,B) then R(B,A)
_SYMMETRIC_RELATIONS = frozenset(
    {
        SpatialRelation.ADJACENT_TO,
        SpatialRelation.NEAR,
        SpatialRelation.FAR,
    }
)

# Inverse pairs
_INVERSES: dict[SpatialRelation, SpatialRelation] = {
    SpatialRelation.CONTAINS: SpatialRelation.INSIDE,
    SpatialRelation.INSIDE: SpatialRelation.CONTAINS,
    SpatialRelation.ABOVE: SpatialRelation.BELOW,
    SpatialRelation.BELOW: SpatialRelation.ABOVE,
    SpatialRelation.LEFT_OF: SpatialRelation.RIGHT_OF,
    SpatialRelation.RIGHT_OF: SpatialRelation.LEFT_OF,
    SpatialRelation.ON_TOP_OF: SpatialRelation.SUPPORTED_BY,
    SpatialRelation.SUPPORTED_BY: SpatialRelation.ON_TOP_OF,
    SpatialRelation.SUPPORTS: SpatialRelation.SUPPORTED_BY,
}


@dataclass
class SpatialFact:
    """A known spatial relationship between two entities."""

    entity_a: str
    entity_b: str
    relation: SpatialRelation
    confidence: float = 1.0
    source_id: str = ""


class SpatialOperator:
    """Qualitative spatial reasoning over HCIR physical entities."""

    @property
    def operator_id(self) -> str:
        return "spatial"

    @property
    def operator_name(self) -> str:
        return "Qualitative Spatial Reasoning Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.SPATIAL: 0.95,
            ProblemType.EXPLANATION: 0.3,
            ProblemType.PREDICTION: 0.3,
            ProblemType.PLANNING: 0.4,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        n_entities = len(view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY))
        if n_entities < 2:
            return 0.0

        return min(1.0, base + min(0.3, n_entities * 0.03))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = len(context.graph_view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY))
        return ResourceCost(
            wall_clock_ms=max(1.0, n * n * 0.02),
            nodes_read=n,
            edges_read=context.graph_view.edge_count,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Infer spatial relationships between physical entities."""
        start = time.time()
        view = context.graph_view

        # ── Extract known spatial facts ──────────────────────────────
        known_facts = self._extract_spatial_facts(view)

        if not known_facts:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No spatial facts found"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Infer new relations ──────────────────────────────────────
        inferred = self._infer_relations(known_facts)

        all_facts = known_facts + inferred

        # ── Build result ─────────────────────────────────────────────
        # Collect existing graph edges between physical entities to avoid duplicates
        existing_edges: set[tuple[str, HCIREdgeType, str]] = set()
        entity_ids = {n.id for n in view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)}
        for nid in entity_ids:
            for edge in view.edges_from(nid):
                for tgt in edge.targets:
                    if tgt in entity_ids:
                        existing_edges.add((nid, edge.edge_type, tgt))

        proposed_ops: list[TransactionOperation] = []
        proposed_edges: set[tuple[str, HCIREdgeType, str]] = set()

        for fact in inferred:
            spec = self._fact_to_edge_spec(fact)
            if spec is None:
                continue
            src, edge_type, tgt = spec

            # Never re-propose an existing edge or duplicate in this batch
            if (src, edge_type, tgt) in existing_edges or (src, edge_type, tgt) in proposed_edges:
                continue

            # For symmetric relations, avoid redundant reciprocal edge if already present
            if fact.relation in _SYMMETRIC_RELATIONS and (tgt, edge_type, src) in existing_edges:
                continue

            proposed_edges.add((src, edge_type, tgt))

            edge = HCIREdge(
                edge_type=edge_type,
                sources=[src],
                targets=[tgt],
                weight=fact.confidence,
                properties={"spatial_relation": fact.relation},
                provenance=Provenance(
                    created_by=self.operator_id,
                    source_type="inferred",
                    reason=f"Spatial inference: {src} {edge_type} {tgt} (derived via {fact.relation})",
                ),
            )
            proposed_ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_data=edge.model_dump(),
                )
            )

        provenance = ProvenanceChain(
            conclusion=f"Spatial analysis of {view.node_count} entities",
            evidence_node_ids=list(
                {f.entity_a for f in known_facts} | {f.entity_b for f in known_facts}
            ),
            operator_id=self.operator_id,
            reasoning_steps=[
                f"Extracted {len(known_facts)} known spatial facts",
                f"Inferred {len(inferred)} new relations via transitivity/symmetry",
            ],
            confidence=0.85,
        )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "known_facts": len(known_facts),
                "inferred_facts": len(inferred),
                "total_relations": len(all_facts),
                "spatial_graph": [
                    {
                        "a": f.entity_a,
                        "relation": f.relation,
                        "b": f.entity_b,
                        "confidence": round(f.confidence, 3),
                    }
                    for f in all_facts[:30]
                ],
            },
            confidence=0.85,
            evidence_refs=list({f.source_id for f in known_facts if f.source_id}),
            proposed_transitions=proposed_ops,
            provenance_chains=[provenance],
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_spatial_facts(view: Any) -> list[SpatialFact]:
        """Extract spatial relationships from HCIR edges and entity properties."""
        facts: list[SpatialFact] = []

        # From edges with spatial semantics
        spatial_edge_map: dict[HCIREdgeType, SpatialRelation] = {
            HCIREdgeType.PART_OF: SpatialRelation.INSIDE,
            HCIREdgeType.SUPPORTS: SpatialRelation.SUPPORTS,
        }

        entity_ids = {n.id for n in view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)}

        for nid in entity_ids:
            for edge in view.edges_from(nid):
                if edge.edge_type in spatial_edge_map:
                    for target in edge.targets:
                        if target in entity_ids:
                            facts.append(
                                SpatialFact(
                                    entity_a=nid,
                                    entity_b=target,
                                    relation=spatial_edge_map[edge.edge_type],
                                    source_id=edge.id,
                                )
                            )

                # Check edge properties for spatial relations
                if "spatial_relation" in edge.properties:
                    rel_str = edge.properties["spatial_relation"]
                    try:
                        rel = SpatialRelation(rel_str)
                        for target in edge.targets:
                            if target in entity_ids:
                                facts.append(
                                    SpatialFact(
                                        entity_a=nid,
                                        entity_b=target,
                                        relation=rel,
                                        source_id=edge.id,
                                    )
                                )
                    except ValueError:
                        pass

        return facts

    @staticmethod
    def _infer_relations(known: list[SpatialFact]) -> list[SpatialFact]:
        """Infer new relations via transitivity, symmetry, and inverses."""
        # Build relation index: (entity_a, entity_b) → relation
        rel_map: dict[tuple[str, str], SpatialRelation] = {}
        for fact in known:
            rel_map[(fact.entity_a, fact.entity_b)] = fact.relation

        inferred: list[SpatialFact] = []
        entities = list({f.entity_a for f in known} | {f.entity_b for f in known})

        # Symmetry: if R(A,B) and R is symmetric, then R(B,A)
        for fact in known:
            if fact.relation in _SYMMETRIC_RELATIONS:
                pair = (fact.entity_b, fact.entity_a)
                if pair not in rel_map:
                    inferred.append(
                        SpatialFact(
                            entity_a=fact.entity_b,
                            entity_b=fact.entity_a,
                            relation=fact.relation,
                            confidence=fact.confidence,
                        )
                    )
                    rel_map[pair] = fact.relation

        # Inverses: if R(A,B) then R_inv(B,A)
        for fact in known:
            if fact.relation in _INVERSES:
                inv = _INVERSES[fact.relation]
                pair = (fact.entity_b, fact.entity_a)
                if pair not in rel_map:
                    inferred.append(
                        SpatialFact(
                            entity_a=fact.entity_b,
                            entity_b=fact.entity_a,
                            relation=inv,
                            confidence=fact.confidence * 0.95,
                        )
                    )
                    rel_map[pair] = inv

        # Transitivity: if R(A,B) and R(B,C) and R is transitive, then R(A,C)
        changed = True
        max_iters = 10
        iteration = 0
        while changed and iteration < max_iters:
            changed = False
            iteration += 1
            for a in entities:
                for b in entities:
                    if a == b:
                        continue
                    ab = rel_map.get((a, b))
                    if ab is None or ab not in _TRANSITIVE_RELATIONS:
                        continue
                    for c in entities:
                        if c in (a, b):
                            continue
                        bc = rel_map.get((b, c))
                        if bc != ab:
                            continue
                        pair_ac = (a, c)
                        if pair_ac not in rel_map:
                            inferred.append(
                                SpatialFact(
                                    entity_a=a,
                                    entity_b=c,
                                    relation=ab,
                                    confidence=0.8,
                                )
                            )
                            rel_map[pair_ac] = ab
                            changed = True

        return inferred

    @staticmethod
    def _fact_to_edge_spec(fact: SpatialFact) -> tuple[str, HCIREdgeType, str] | None:
        """Map a spatial fact to canonical (source_id, edge_type, target_id).

        Preserves asymmetric edge directionality in HCIR:
          - A INSIDE B        => source=A, edge=PART_OF, target=B       [A is part of B]
          - A CONTAINS B      => source=B, edge=PART_OF, target=A       [B is part of A]
          - A SUPPORTS B      => source=A, edge=SUPPORTS, target=B      [A supports B]
          - A SUPPORTED_BY B  => source=B, edge=SUPPORTS, target=A      [B supports A]
          - A ON_TOP_OF B     => source=B, edge=SUPPORTS, target=A      [B supports A]
          - A ADJACENT_TO B   => source=A, edge=CORRELATES_WITH, target=B [symmetric]
          - A NEAR B          => source=A, edge=CORRELATES_WITH, target=B [symmetric]
        """
        if fact.relation == SpatialRelation.INSIDE:
            return (fact.entity_a, HCIREdgeType.PART_OF, fact.entity_b)
        elif fact.relation == SpatialRelation.CONTAINS:
            return (fact.entity_b, HCIREdgeType.PART_OF, fact.entity_a)
        elif fact.relation == SpatialRelation.SUPPORTS:
            return (fact.entity_a, HCIREdgeType.SUPPORTS, fact.entity_b)
        elif fact.relation in (SpatialRelation.SUPPORTED_BY, SpatialRelation.ON_TOP_OF):
            return (fact.entity_b, HCIREdgeType.SUPPORTS, fact.entity_a)
        elif fact.relation in (SpatialRelation.ADJACENT_TO, SpatialRelation.NEAR):
            return (fact.entity_a, HCIREdgeType.CORRELATES_WITH, fact.entity_b)
        return None

    @staticmethod
    def _relation_to_edge_type(relation: SpatialRelation) -> HCIREdgeType:
        """Map spatial relation to default HCIR edge type."""
        mapping: dict[SpatialRelation, HCIREdgeType] = {
            SpatialRelation.CONTAINS: HCIREdgeType.PART_OF,
            SpatialRelation.INSIDE: HCIREdgeType.PART_OF,
            SpatialRelation.SUPPORTS: HCIREdgeType.SUPPORTS,
            SpatialRelation.SUPPORTED_BY: HCIREdgeType.SUPPORTS,
            SpatialRelation.ADJACENT_TO: HCIREdgeType.CORRELATES_WITH,
            SpatialRelation.NEAR: HCIREdgeType.CORRELATES_WITH,
        }
        return mapping.get(relation, HCIREdgeType.CORRELATES_WITH)

