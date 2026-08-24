"""
Analogy Operator — Structure-Mapping Theory over HCIR.

Finds structural correspondences between a source domain and a target
domain by comparing the relational structure of HCIR subgraphs rather
than surface features.

Based on Gentner's Structure-Mapping Theory:
    1. Identify relational structure in source and target.
    2. Find consistent one-to-one mappings between entities.
    3. Prefer mappings that preserve higher-order relations.
    4. Transfer candidate inferences from source to target.

Example::

    Source: solar system
        sun ATTRACTS planets, planets ORBIT sun

    Target: atom
        nucleus ??? electrons, electrons ??? nucleus

    Mapping: sun→nucleus, planets→electrons
    Transfer: nucleus ATTRACTS electrons, electrons ORBIT nucleus

Independence Level: L1 (no LLM execution)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


@dataclass
class StructuralMapping:
    """A mapping between source and target domain entities."""

    entity_map: dict[str, str]  # source_node_id → target_node_id
    relation_map: dict[str, str]  # source_edge_type → target_edge_type
    score: float = 0.0  # Quality of the mapping
    transferred_inferences: list[dict[str, Any]] = field(default_factory=list)


class AnalogyOperator:
    """Structure-mapping analogy over HCIR subgraphs.

    Given a problem specifying source and target domain nodes,
    finds structural correspondences and proposes transferred
    inferences as new HCIR edges.
    """

    @property
    def operator_id(self) -> str:
        return "analogy"

    @property
    def operator_name(self) -> str:
        return "Structure-Mapping Analogy Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.ANALOGY: 0.95,
            ProblemType.EXPLANATION: 0.3,
            ProblemType.GENERALIZATION: 0.5,
            ProblemType.CLASSIFICATION: 0.3,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        # Need at least some nodes with edges to compare structures
        view = context.graph_view
        if view.edge_count < 2:
            return 0.0

        # Boost if problem specifies focus nodes (source/target domains)
        if problem.focus_node_ids:
            base += 0.2

        return min(1.0, base)

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = context.graph_view.node_count
        e = context.graph_view.edge_count
        return ResourceCost(
            wall_clock_ms=max(2.0, n * e * 0.01),
            nodes_read=n,
            edges_read=e,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Find structural analogies between HCIR subgraphs."""
        start = time.time()
        view = context.graph_view

        # ── Identify source and target domains ───────────────────────
        source_ids, target_ids = self._identify_domains(view, problem)

        if not source_ids or not target_ids:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "Could not identify source and target domains"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Extract relational structure ─────────────────────────────
        source_structure = self._extract_structure(view, source_ids)
        target_structure = self._extract_structure(view, target_ids)

        if not source_structure or not target_structure:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "Insufficient relational structure"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Find best structural mapping ─────────────────────────────
        mapping = self._find_mapping(
            source_structure,
            target_structure,
            source_ids,
            target_ids,
        )

        if mapping is None or mapping.score < 0.1:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"mapping_score": mapping.score if mapping else 0},
                metadata={"reason": "No significant structural correspondence"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Transfer inferences ──────────────────────────────────────
        transfers = self._transfer_inferences(
            view,
            mapping,
            source_structure,
            target_ids,
        )
        mapping.transferred_inferences = transfers

        # ── Build result ─────────────────────────────────────────────
        proposed_ops: list[TransactionOperation] = []

        for transfer in transfers:
            edge = HCIREdge(
                edge_type=HCIREdgeType(transfer["edge_type"]),
                sources=[transfer["source"]],
                targets=[transfer["target"]],
                weight=transfer["confidence"],
                properties={"origin": "analogy", "source_analogy": transfer.get("source_edge", "")},
                provenance=Provenance(
                    created_by=self.operator_id,
                    source_type="inferred",
                    reason=f"Analogical transfer: {transfer['description']}",
                ),
            )
            proposed_ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE,
                    edge_data=edge.model_dump(),
                )
            )

        provenance = ProvenanceChain(
            conclusion=f"Structural analogy: {len(mapping.entity_map)} entity mappings",
            evidence_node_ids=list(source_ids | target_ids),
            operator_id=self.operator_id,
            reasoning_steps=[
                f"Source domain: {len(source_ids)} nodes",
                f"Target domain: {len(target_ids)} nodes",
                f"Mapping score: {mapping.score:.3f}",
                f"Entity mappings: {len(mapping.entity_map)}",
                f"Transferred inferences: {len(transfers)}",
            ],
            assumptions=[
                "Structural similarity implies functional similarity",
                "Relations in source domain transfer to target domain",
            ],
            confidence=mapping.score,
        )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "mapping_score": round(mapping.score, 4),
                "entity_mappings": mapping.entity_map,
                "transferred_inferences": len(transfers),
                "transfers": [
                    {"from": t["description"], "confidence": round(t["confidence"], 3)}
                    for t in transfers
                ],
            },
            confidence=mapping.score,
            evidence_refs=list(source_ids | target_ids),
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
    def _identify_domains(
        view: Any,
        problem: ReasoningProblem,
    ) -> tuple[set[str], set[str]]:
        """Identify source and target domains from the problem.

        Uses focus_node_ids: first half as source, second half as target.
        If only 1 focus node, use it as target and everything else as source.
        """
        focus = list(problem.focus_node_ids)
        all_ids = view.all_node_ids()

        if len(focus) >= 2:
            mid = len(focus) // 2
            source = set(focus[:mid])
            target = set(focus[mid:])
            return source, target

        if len(focus) == 1:
            target = {focus[0]}
            source = all_ids - target
            return source, target

        # No focus — try to split by category
        # Use first half of sorted IDs as source, second as target
        sorted_ids = sorted(all_ids)
        if len(sorted_ids) >= 4:
            mid = len(sorted_ids) // 2
            return set(sorted_ids[:mid]), set(sorted_ids[mid:])

        return set(), set()

    @staticmethod
    def _extract_structure(
        view: Any,
        node_ids: set[str],
    ) -> list[tuple[str, str, str]]:
        """Extract relational structure: list of (source, edge_type, target)."""
        structure: list[tuple[str, str, str]] = []
        for nid in node_ids:
            for edge in view.edges_from(nid):
                for target in edge.targets:
                    if target in node_ids:
                        structure.append((nid, edge.edge_type, target))
        return structure

    @staticmethod
    def _find_mapping(
        source_struct: list[tuple[str, str, str]],
        target_struct: list[tuple[str, str, str]],
        source_ids: set[str],
        target_ids: set[str],
    ) -> StructuralMapping | None:
        """Find the best structural mapping using greedy matching.

        Scores by matching edge-type patterns between domains.
        """
        # Build edge-type profiles for each node
        source_profiles: dict[str, dict[str, int]] = {}
        for src, etype, tgt in source_struct:
            source_profiles.setdefault(src, {})
            source_profiles[src][etype] = source_profiles[src].get(etype, 0) + 1

        target_profiles: dict[str, dict[str, int]] = {}
        for src, etype, tgt in target_struct:
            target_profiles.setdefault(src, {})
            target_profiles[src][etype] = target_profiles[src].get(etype, 0) + 1

        if not source_profiles or not target_profiles:
            return StructuralMapping(entity_map={}, relation_map={}, score=0.0)

        # Greedy matching: for each source node, find best target match
        entity_map: dict[str, str] = {}
        used_targets: set[str] = set()

        for s_id, s_profile in source_profiles.items():
            best_target: str | None = None
            best_similarity = 0.0

            for t_id, t_profile in target_profiles.items():
                if t_id in used_targets:
                    continue
                # Jaccard similarity on edge type sets
                s_types = set(s_profile.keys())
                t_types = set(t_profile.keys())
                if not s_types and not t_types:
                    continue
                intersection = s_types & t_types
                union = s_types | t_types
                similarity = len(intersection) / len(union) if union else 0.0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target = t_id

            if best_target is not None and best_similarity > 0:
                entity_map[s_id] = best_target
                used_targets.add(best_target)

        if not entity_map:
            return StructuralMapping(entity_map={}, relation_map={}, score=0.0)

        # Compute overall mapping score
        # Count how many source edges have a corresponding target edge
        matched_edges = 0
        total_source_edges = len(source_struct)

        for src, etype, tgt in source_struct:
            mapped_src = entity_map.get(src)
            mapped_tgt = entity_map.get(tgt)
            if mapped_src and mapped_tgt:
                # Check if this edge exists in target
                for t_src, t_etype, t_tgt in target_struct:
                    if t_src == mapped_src and t_tgt == mapped_tgt and t_etype == etype:
                        matched_edges += 1
                        break

        score = matched_edges / max(1, total_source_edges)

        return StructuralMapping(
            entity_map=entity_map,
            relation_map={},
            score=score,
        )

    @staticmethod
    def _transfer_inferences(
        view: Any,
        mapping: StructuralMapping,
        source_struct: list[tuple[str, str, str]],
        target_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Transfer source-domain relations to target domain.

        For each edge in source that doesn't exist in target,
        propose it as a transferred inference.
        """
        transfers: list[dict[str, Any]] = []

        for src, etype, tgt in source_struct:
            mapped_src = mapping.entity_map.get(src)
            mapped_tgt = mapping.entity_map.get(tgt)

            if not mapped_src or not mapped_tgt:
                continue

            # Check if this edge already exists in target
            existing = view.edges_from(mapped_src)
            already_exists = any(e.edge_type == etype and mapped_tgt in e.targets for e in existing)

            if not already_exists:
                try:
                    HCIREdgeType(etype)  # Validate edge type
                except ValueError:
                    continue

                transfers.append(
                    {
                        "source": mapped_src,
                        "target": mapped_tgt,
                        "edge_type": etype,
                        "confidence": mapping.score * 0.7,
                        "source_edge": f"{src} --{etype}--> {tgt}",
                        "description": (
                            f"{mapped_src} --{etype}--> {mapped_tgt} "
                            f"(by analogy with {src} --{etype}--> {tgt})"
                        ),
                    }
                )

        return transfers
