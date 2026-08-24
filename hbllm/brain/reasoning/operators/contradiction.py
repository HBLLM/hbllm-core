"""
Contradiction Operator — LLM-free graph-based contradiction detection.

The existing ``ContradictionDetector`` depends on an LLM for semantic
comparison.  This operator provides an LLM-free alternative that
detects contradictions structurally from HCIR edges and belief confidence.

Detection strategies (all LLM-free):
    1. **Edge-based**: CONTRADICTS edges already in the graph.
    2. **Confidence inversion**: Two beliefs about the same topic with
       inverted confidence (one high, one low).
    3. **Negation pattern**: Belief claims containing explicit negation
       of another belief's claim.
    4. **Prediction failure**: PredictionErrorNodes with high magnitude.

Independence Level: L1 (no LLM)
"""

from __future__ import annotations

import logging
import time
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
    BeliefNode,
    ContradictionNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    PredictionErrorNode,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)

# Negation markers for simple structural negation detection
_NEGATION_PAIRS = [
    ("is ", "is not "),
    ("can ", "cannot "),
    ("has ", "has no "),
    ("will ", "will not "),
    ("does ", "does not "),
    ("are ", "are not "),
    ("was ", "was not "),
]


class ContradictionOperator:
    """LLM-free contradiction detection over HCIR beliefs.

    Scans the frozen view for structural contradictions using
    edge analysis, negation patterns, and prediction errors.
    """

    @property
    def operator_id(self) -> str:
        return "contradiction"

    @property
    def operator_name(self) -> str:
        return "Graph-Based Contradiction Detector"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.CONTRADICTION: 0.95,
            ProblemType.EXPLANATION: 0.4,
            ProblemType.DIAGNOSIS: 0.5,
            ProblemType.CLASSIFICATION: 0.2,
        }
        base = type_scores.get(problem.problem_type, 0.1)

        view = context.graph_view
        n_beliefs = len(view.nodes_by_type(HCIRNodeType.BELIEF))
        if n_beliefs < 2:
            return 0.0

        return min(1.0, base + min(0.2, n_beliefs * 0.01))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = context.graph_view.node_count
        return ResourceCost(
            wall_clock_ms=max(1.0, n * n * 0.01),
            nodes_read=n,
            edges_read=context.graph_view.edge_count,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Detect contradictions in the frozen HCIR view."""
        start = time.time()
        view = context.graph_view

        contradictions: list[dict[str, Any]] = []

        # Strategy 1: Existing CONTRADICTS edges
        contradictions.extend(self._detect_edge_contradictions(view))

        # Strategy 2: Negation patterns in belief claims
        contradictions.extend(self._detect_negation_contradictions(view))

        # Strategy 3: High-magnitude prediction errors
        contradictions.extend(self._detect_prediction_failures(view))

        if not contradictions:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"contradictions_found": 0},
                metadata={"reason": "No contradictions detected"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                    edges_read=view.edge_count,
                ),
            )

        # ── Build result ─────────────────────────────────────────────
        proposed_ops: list[TransactionOperation] = []
        provenance_chains: list[ProvenanceChain] = []
        evidence_refs: list[str] = []

        for c in contradictions:
            # Propose a ContradictionNode
            cnode = ContradictionNode(
                description=c["description"],
                provenance=Provenance(
                    created_by=self.operator_id,
                    source_type="inferred",
                    reason=f"Detected via {c['strategy']}",
                ),
            )
            cnode.uncertainty.confidence = c["severity"]

            proposed_ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=cnode.model_dump(),
                )
            )

            # Propose CONTRADICTS edges between involved nodes
            if c.get("node_a") and c.get("node_b"):
                edge = HCIREdge(
                    edge_type=HCIREdgeType.CONTRADICTS,
                    sources=[c["node_a"]],
                    targets=[c["node_b"]],
                    weight=c["severity"],
                    provenance=Provenance(
                        created_by=self.operator_id,
                        source_type="inferred",
                    ),
                )
                proposed_ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_EDGE,
                        edge_data=edge.model_dump(),
                    )
                )
                evidence_refs.extend([c["node_a"], c["node_b"]])

            provenance_chains.append(
                ProvenanceChain(
                    conclusion=c["description"],
                    evidence_node_ids=[c.get("node_a", ""), c.get("node_b", "")],
                    operator_id=self.operator_id,
                    reasoning_steps=[f"Strategy: {c['strategy']}"],
                    confidence=c["severity"],
                )
            )

        evidence_refs = list(dict.fromkeys(r for r in evidence_refs if r))
        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "contradictions_found": len(contradictions),
                "contradictions": [
                    {"description": c["description"], "severity": c["severity"]}
                    for c in contradictions
                ],
            },
            confidence=max(c["severity"] for c in contradictions),
            evidence_refs=evidence_refs,
            proposed_transitions=proposed_ops,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Detection Strategies ─────────────────────────────────────────

    @staticmethod
    def _detect_edge_contradictions(view: Any) -> list[dict[str, Any]]:
        """Find existing CONTRADICTS edges in the view."""
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for node_id in view.all_node_ids():
            for edge in view.edges_from(node_id):
                if edge.edge_type == HCIREdgeType.CONTRADICTS:
                    for target in edge.targets:
                        pair = (min(node_id, target), max(node_id, target))
                        if pair not in seen:
                            seen.add(pair)
                            node_a = view.get_node(node_id)
                            node_b = view.get_node(target)
                            desc_a = getattr(node_a, "claim", node_id) if node_a else node_id
                            desc_b = getattr(node_b, "claim", target) if node_b else target
                            results.append(
                                {
                                    "description": f"Edge contradiction: '{desc_a}' vs '{desc_b}'",
                                    "severity": edge.weight,
                                    "strategy": "edge_contradiction",
                                    "node_a": node_id,
                                    "node_b": target,
                                }
                            )
        return results

    @staticmethod
    def _detect_negation_contradictions(view: Any) -> list[dict[str, Any]]:
        """Detect belief pairs with negation patterns."""
        results: list[dict[str, Any]] = []
        beliefs: list[tuple[str, str]] = []  # (node_id, claim_lower)

        for node in view.nodes_by_type(HCIRNodeType.BELIEF):
            if isinstance(node, BeliefNode) and node.claim.strip():
                beliefs.append((node.id, node.claim.strip().lower()))

        seen: set[tuple[str, str]] = set()

        for i, (id_a, claim_a) in enumerate(beliefs):
            for id_b, claim_b in beliefs[i + 1 :]:
                pair = (min(id_a, id_b), max(id_a, id_b))
                if pair in seen:
                    continue

                # Check for negation patterns
                for pos, neg in _NEGATION_PAIRS:
                    if pos in claim_a and neg in claim_b:
                        # Check if the rest matches
                        base_a = claim_a.replace(pos, "", 1).strip()
                        base_b = claim_b.replace(neg, "", 1).strip()
                        if (
                            base_a
                            and base_b
                            and (base_a == base_b or base_a in base_b or base_b in base_a)
                        ):
                            seen.add(pair)
                            results.append(
                                {
                                    "description": (
                                        f"Negation contradiction: '{claim_a}' vs '{claim_b}'"
                                    ),
                                    "severity": 0.8,
                                    "strategy": "negation_pattern",
                                    "node_a": id_a,
                                    "node_b": id_b,
                                }
                            )
                            break

                    elif neg in claim_a and pos in claim_b:
                        base_a = claim_a.replace(neg, "", 1).strip()
                        base_b = claim_b.replace(pos, "", 1).strip()
                        if (
                            base_a
                            and base_b
                            and (base_a == base_b or base_a in base_b or base_b in base_a)
                        ):
                            seen.add(pair)
                            results.append(
                                {
                                    "description": (
                                        f"Negation contradiction: '{claim_a}' vs '{claim_b}'"
                                    ),
                                    "severity": 0.8,
                                    "strategy": "negation_pattern",
                                    "node_a": id_a,
                                    "node_b": id_b,
                                }
                            )
                            break

        return results

    @staticmethod
    def _detect_prediction_failures(view: Any) -> list[dict[str, Any]]:
        """Detect high-magnitude prediction errors as implicit contradictions."""
        results: list[dict[str, Any]] = []

        for node in view.nodes_by_type(HCIRNodeType.PREDICTION_ERROR):
            if isinstance(node, PredictionErrorNode):
                if node.error_magnitude > 0.5:
                    results.append(
                        {
                            "description": (
                                f"Prediction failure: expected={node.predicted_value}, "
                                f"observed={node.observed_value}, "
                                f"magnitude={node.error_magnitude:.2f}"
                            ),
                            "severity": min(1.0, node.error_magnitude),
                            "strategy": "prediction_failure",
                            "node_a": node.prediction_id,
                            "node_b": node.id,
                        }
                    )

        return results
