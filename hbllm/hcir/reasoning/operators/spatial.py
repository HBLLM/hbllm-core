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
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.hcir.fuzzy import RelationCompositionPolicy
from hbllm.hcir.graph import (
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.hcir.seed.loader import load_seeded_composition_policy
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
    semantic_degree: float = 1.0
    source_id: str = ""


class SpatialOperator:
    """Qualitative spatial reasoning over HCIR physical entities."""

    def __init__(
        self,
        composition_policy: RelationCompositionPolicy | None = None,
        *,
        enable_fuzzy: bool = False,
        contextual_scope: str = "default",
    ) -> None:
        self._composition_policy = composition_policy
        self._enable_fuzzy = bool(enable_fuzzy)
        self._contextual_scope = contextual_scope

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

        # ── Infer new relations via declarative RelationCompositionPolicy ──
        policy = self._resolve_policy(view, self._composition_policy)
        entity_scales = self._extract_entity_scales(view)
        inferred = self._infer_relations(
            known_facts,
            policy=policy,
            entity_scales=entity_scales,
            contextual_scope=self._contextual_scope,
        )

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
                weight=fact.semantic_degree if self._enable_fuzzy else fact.confidence,
                properties={
                    "spatial_relation": fact.relation.value,
                    "semantic_degree": round(float(fact.semantic_degree), 4),
                    "posterior_belief": round(float(fact.confidence), 4),
                },
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
                f"Inferred {len(inferred)} new relations via {policy.policy_id}",
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
                        "semantic_degree": round(f.semantic_degree, 3),
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
    def _resolve_policy(
        view: Any, fallback: RelationCompositionPolicy | None
    ) -> RelationCompositionPolicy:
        if fallback is not None:
            return fallback
        policy_node = view.get_node("prior_policy_relation_composition")
        if policy_node is not None:
            props = getattr(policy_node, "properties", None)
            if isinstance(props, dict) and isinstance(props.get("policy"), dict):
                return RelationCompositionPolicy.model_validate(props["policy"])
        return load_seeded_composition_policy()

    @staticmethod
    def _extract_entity_scales(view: Any) -> dict[str, float]:
        scales: dict[str, float] = {}
        for node in view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
            props = getattr(node, "properties", None)
            if not isinstance(props, dict):
                continue
            if isinstance(props.get("diameter"), (int, float)):
                scales[node.id] = max(1e-6, float(props["diameter"]))
            elif isinstance(props.get("extent"), (int, float)):
                scales[node.id] = max(1e-6, float(props["extent"]))
            elif isinstance(props.get("bbox"), (list, tuple)) and len(props["bbox"]) >= 4:
                scales[node.id] = max(
                    1e-6, math.hypot(float(props["bbox"][2]), float(props["bbox"][3]))
                )
            elif isinstance(props.get("size"), (int, float)):
                scales[node.id] = max(1e-6, math.sqrt(max(1.0, float(props["size"]))))
        return scales

    @staticmethod
    def _extract_spatial_facts(view: Any) -> list[SpatialFact]:
        """Extract spatial relationships from HCIR edges and entity properties."""
        facts: list[SpatialFact] = []

        # From edges with spatial semantics
        spatial_edge_map: dict[HCIREdgeType, SpatialRelation] = {
            HCIREdgeType.PART_OF: SpatialRelation.INSIDE,
            HCIREdgeType.LOCATED_IN: SpatialRelation.INSIDE,
            HCIREdgeType.CONTAINS: SpatialRelation.CONTAINS,
            HCIREdgeType.SUPPORTS: SpatialRelation.SUPPORTS,
            HCIREdgeType.ABOVE: SpatialRelation.ABOVE,
            HCIREdgeType.BELOW: SpatialRelation.BELOW,
            HCIREdgeType.LEFT_OF: SpatialRelation.LEFT_OF,
            HCIREdgeType.RIGHT_OF: SpatialRelation.RIGHT_OF,
            HCIREdgeType.NEAR: SpatialRelation.NEAR,
            HCIREdgeType.TOUCHING: SpatialRelation.ADJACENT_TO,
        }

        entity_ids = {n.id for n in view.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)}

        for nid in entity_ids:
            for edge in view.edges_from(nid):
                props = edge.properties if isinstance(edge.properties, dict) else {}
                sem_deg = float(props.get("semantic_degree", edge.weight))
                post_conf = float(
                    props.get("posterior_belief", props.get("confidence", edge.weight))
                )
                sem_deg = max(0.0, min(1.0, sem_deg))
                post_conf = max(0.0, min(1.0, post_conf))

                if "spatial_relation" in props:
                    rel_str = str(props["spatial_relation"]).lower()
                    try:
                        rel = SpatialRelation(rel_str)
                        for target in edge.targets:
                            if target in entity_ids:
                                facts.append(
                                    SpatialFact(
                                        entity_a=nid,
                                        entity_b=target,
                                        relation=rel,
                                        confidence=post_conf,
                                        semantic_degree=sem_deg,
                                        source_id=edge.id,
                                    )
                                )
                        continue
                    except ValueError:
                        pass

                if edge.edge_type in spatial_edge_map:
                    for target in edge.targets:
                        if target in entity_ids:
                            facts.append(
                                SpatialFact(
                                    entity_a=nid,
                                    entity_b=target,
                                    relation=spatial_edge_map[edge.edge_type],
                                    confidence=post_conf,
                                    semantic_degree=sem_deg,
                                    source_id=edge.id,
                                )
                            )

        return facts

    @staticmethod
    def _infer_relations(
        known: list[SpatialFact],
        *,
        policy: RelationCompositionPolicy | None = None,
        entity_scales: dict[str, float] | None = None,
        contextual_scope: str = "default",
    ) -> list[SpatialFact]:
        """Infer new relations via symmetry, inverses, and declarative `RelationCompositionPolicy`."""
        eff_policy = policy or load_seeded_composition_policy()
        scales = entity_scales or {}

        rel_map: dict[tuple[str, str], SpatialFact] = {}
        for fact in known:
            rel_map[(fact.entity_a, fact.entity_b)] = fact

        inferred: list[SpatialFact] = []
        entities = sorted({f.entity_a for f in known} | {f.entity_b for f in known})

        # Symmetry: if R(A,B) and R is symmetric, then R(B,A)
        for fact in known:
            if fact.relation in _SYMMETRIC_RELATIONS:
                pair = (fact.entity_b, fact.entity_a)
                if pair not in rel_map:
                    sym = SpatialFact(
                        entity_a=fact.entity_b,
                        entity_b=fact.entity_a,
                        relation=fact.relation,
                        confidence=fact.confidence,
                        semantic_degree=fact.semantic_degree,
                    )
                    inferred.append(sym)
                    rel_map[pair] = sym

        # Inverses: if R(A,B) then R_inv(B,A)
        for fact in known:
            if fact.relation in _INVERSES:
                inv = _INVERSES[fact.relation]
                pair = (fact.entity_b, fact.entity_a)
                if pair not in rel_map:
                    inv_fact = SpatialFact(
                        entity_a=fact.entity_b,
                        entity_b=fact.entity_a,
                        relation=inv,
                        confidence=fact.confidence * 0.95,
                        semantic_degree=fact.semantic_degree,
                    )
                    inferred.append(inv_fact)
                    rel_map[pair] = inv_fact

        # Policy-guided composition: R1(A, B) o R2(B, C) -> R3(A, C)
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
                    fact_ab = rel_map.get((a, b))
                    if fact_ab is None:
                        continue
                    for c in entities:
                        if c in (a, b):
                            continue
                        fact_bc = rel_map.get((b, c))
                        if fact_bc is None:
                            continue
                        pair_ac = (a, c)
                        if pair_ac in rel_map:
                            continue
                        diam_b = scales.get(b, 1.0)
                        diam_c = scales.get(c, 1.0)
                        comp = eff_policy.compose(
                            fact_ab.relation.value,
                            fact_bc.relation.value,
                            fact_ab.semantic_degree,
                            fact_bc.semantic_degree,
                            contextual_scope=contextual_scope,
                            geometry_context={
                                "intermediate_diameter": diam_b,
                                "target_diameter": diam_c,
                                "reference_scale": diam_c,
                                "same_axis_aligned": True,
                            },
                        )
                        if not comp.allowed or not comp.composed_relation or comp.degree <= 0.0:
                            continue
                        try:
                            composed_rel = SpatialRelation(comp.composed_relation.lower())
                        except ValueError:
                            continue
                        derived = SpatialFact(
                            entity_a=a,
                            entity_b=c,
                            relation=composed_rel,
                            confidence=round(min(fact_ab.confidence, fact_bc.confidence) * 0.90, 4),
                            semantic_degree=round(float(comp.degree), 4),
                        )
                        inferred.append(derived)
                        rel_map[pair_ac] = derived
                        changed = True

        return inferred

    @staticmethod
    def _fact_to_edge_spec(fact: SpatialFact) -> tuple[str, HCIREdgeType, str] | None:
        """Map a spatial fact to canonical (source_id, edge_type, target_id)."""
        if fact.relation == SpatialRelation.INSIDE:
            return (fact.entity_a, HCIREdgeType.PART_OF, fact.entity_b)
        elif fact.relation == SpatialRelation.CONTAINS:
            return (fact.entity_b, HCIREdgeType.PART_OF, fact.entity_a)
        elif fact.relation == SpatialRelation.SUPPORTS:
            return (fact.entity_a, HCIREdgeType.SUPPORTS, fact.entity_b)
        elif fact.relation in (SpatialRelation.SUPPORTED_BY, SpatialRelation.ON_TOP_OF):
            return (fact.entity_b, HCIREdgeType.SUPPORTS, fact.entity_a)
        elif fact.relation == SpatialRelation.ABOVE:
            return (fact.entity_a, HCIREdgeType.ABOVE, fact.entity_b)
        elif fact.relation == SpatialRelation.BELOW:
            return (fact.entity_a, HCIREdgeType.BELOW, fact.entity_b)
        elif fact.relation == SpatialRelation.LEFT_OF:
            return (fact.entity_a, HCIREdgeType.LEFT_OF, fact.entity_b)
        elif fact.relation == SpatialRelation.RIGHT_OF:
            return (fact.entity_a, HCIREdgeType.RIGHT_OF, fact.entity_b)
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
