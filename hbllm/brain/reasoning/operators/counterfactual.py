"""
Counterfactual Operator — wraps CounterfactualReasoner as a ReasoningOperator.

Performs 'What if...' reasoning: evaluates how beliefs would change
under hypothetical modifications to the evidence graph.

The wrapper adapts the existing async CounterfactualReasoner API to
the synchronous ReasoningOperator protocol using the frozen HCIR view.

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
    EvidenceNode,
    HCIREdgeType,
    HCIRNodeType,
    HypothesisNode,
)

logger = logging.getLogger(__name__)


class CounterfactualOperator:
    """'What if...' reasoning over immutable HCIR views.

    Evaluates the structural impact of hypothetically removing
    evidence or falsifying hypotheses — without an LLM.

    Uses pure graph traversal to compute confidence deltas.
    """

    @property
    def operator_id(self) -> str:
        return "counterfactual"

    @property
    def operator_name(self) -> str:
        return "Counterfactual Reasoning Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.COUNTERFACTUAL: 0.95,
            ProblemType.EXPLANATION: 0.4,
            ProblemType.DIAGNOSIS: 0.5,
            ProblemType.CAUSAL: 0.4,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        n_hyp = len(view.nodes_by_type(HCIRNodeType.HYPOTHESIS))
        n_evi = len(view.nodes_by_type(HCIRNodeType.EVIDENCE))
        n_beliefs = len(view.nodes_by_type(HCIRNodeType.BELIEF))

        if n_beliefs < 1 or (n_hyp + n_evi) < 1:
            return 0.0

        return min(1.0, base + min(0.3, (n_hyp + n_evi) * 0.03))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = context.graph_view.node_count
        return ResourceCost(
            wall_clock_ms=max(2.0, n * 0.5),
            nodes_read=n,
            edges_read=context.graph_view.edge_count,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Evaluate counterfactual scenarios over frozen HCIR."""
        start = time.time()
        view = context.graph_view

        scenarios: list[dict[str, Any]] = []

        # For each hypothesis in the view, ask: what if it's wrong?
        for node in view.nodes_by_type(HCIRNodeType.HYPOTHESIS):
            if isinstance(node, HypothesisNode):
                result = self._what_if_falsified(view, node.id)
                if result["affected_count"] > 0:
                    scenarios.append(result)

        # For each evidence node, ask: what if it were removed?
        for node in view.nodes_by_type(HCIRNodeType.EVIDENCE):
            if isinstance(node, EvidenceNode):
                result = self._what_if_removed(view, node.id)
                if result["affected_count"] > 0:
                    scenarios.append(result)

        if not scenarios:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"scenarios_evaluated": 0},
                metadata={"reason": "No impactful counterfactual scenarios found"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Sort by structural impact ────────────────────────────────
        scenarios.sort(key=lambda s: s["structural_impact"], reverse=True)

        provenance_chains: list[ProvenanceChain] = []
        for sc in scenarios[:10]:  # Cap at 10
            provenance_chains.append(
                ProvenanceChain(
                    conclusion=sc["scenario"],
                    evidence_node_ids=sc.get("affected_ids", []),
                    operator_id=self.operator_id,
                    reasoning_steps=sc.get("cascading", []),
                    confidence=1.0 - sc["structural_impact"],
                )
            )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "scenarios_evaluated": len(scenarios),
                "most_impactful": scenarios[0]["scenario"] if scenarios else "",
                "max_structural_impact": round(scenarios[0]["structural_impact"], 3)
                if scenarios
                else 0,
                "scenarios": [
                    {
                        "scenario": s["scenario"],
                        "affected_count": s["affected_count"],
                        "structural_impact": round(s["structural_impact"], 3),
                    }
                    for s in scenarios[:10]
                ],
            },
            confidence=0.8,
            evidence_refs=list({nid for s in scenarios for nid in s.get("affected_ids", [])}),
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
    def _what_if_falsified(view: Any, hypothesis_id: str) -> dict[str, Any]:
        """Evaluate the impact of falsifying a hypothesis."""
        node = view.get_node(hypothesis_id)
        claim = getattr(node, "claim", hypothesis_id) if node else hypothesis_id

        affected_ids: list[str] = []
        cascading: list[str] = []
        confidence_deltas: dict[str, float] = {}

        # Follow SUPPORTS/STRENGTHENS/DERIVED_FROM edges from hypothesis
        support_types = {
            HCIREdgeType.SUPPORTS,
            HCIREdgeType.STRENGTHENS,
            HCIREdgeType.DERIVED_FROM,
        }

        for edge in view.edges_from(hypothesis_id):
            if edge.edge_type in support_types:
                for target in edge.targets:
                    target_node = view.get_node(target)
                    if isinstance(target_node, BeliefNode):
                        affected_ids.append(target)
                        # Estimate impact based on edge weight and
                        # how many other supporters the belief has
                        other_support = len(
                            [
                                e
                                for e in view.edges_to(target)
                                if e.edge_type in support_types and hypothesis_id not in e.sources
                            ]
                        )
                        impact = edge.weight / max(1, other_support + 1)
                        confidence_deltas[target] = -impact
                        cascading.append(f"Belief '{target_node.claim[:50]}' loses {impact:.2f}")

            if edge.edge_type == HCIREdgeType.PREDICTS:
                for target in edge.targets:
                    cascading.append(f"Prediction {target} invalidated")

        total_beliefs = len(view.nodes_by_type(HCIRNodeType.BELIEF))
        structural_impact = len(affected_ids) / max(1, total_beliefs)

        return {
            "scenario": f"If hypothesis '{claim[:60]}' were falsified",
            "mutation_type": "falsify_hypothesis",
            "target_id": hypothesis_id,
            "affected_ids": affected_ids,
            "affected_count": len(affected_ids),
            "confidence_deltas": confidence_deltas,
            "cascading": cascading,
            "structural_impact": min(1.0, structural_impact),
        }

    @staticmethod
    def _what_if_removed(view: Any, evidence_id: str) -> dict[str, Any]:
        """Evaluate the impact of removing evidence."""
        node = view.get_node(evidence_id)
        desc = getattr(node, "claim", evidence_id) if node else evidence_id

        affected_ids: list[str] = []
        cascading: list[str] = []

        support_types = {
            HCIREdgeType.SUPPORTS,
            HCIREdgeType.STRENGTHENS,
        }

        # Find everything this evidence supports
        for edge in view.edges_from(evidence_id):
            if edge.edge_type in support_types:
                for target in edge.targets:
                    target_node = view.get_node(target)
                    if target_node is not None:
                        affected_ids.append(target)
                        label = getattr(target_node, "claim", target)
                        cascading.append(f"'{str(label)[:50]}' loses support")

        total_beliefs = len(view.nodes_by_type(HCIRNodeType.BELIEF))
        structural_impact = len(affected_ids) / max(1, total_beliefs)

        return {
            "scenario": f"If evidence '{desc[:60]}' were removed",
            "mutation_type": "remove_evidence",
            "target_id": evidence_id,
            "affected_ids": affected_ids,
            "affected_count": len(affected_ids),
            "cascading": cascading,
            "structural_impact": min(1.0, structural_impact),
        }
