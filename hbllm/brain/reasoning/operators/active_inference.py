"""
Active Inference Operator — wraps ActiveInferenceEngine as a ReasoningOperator.

Evaluates candidate actions by computing:
    Utility = w₁·Reward + w₂·InfoGain + w₃·FutureVal - w₄·Risk - w₅·Cost

Reads ActionNodes from the frozen HCIR view, evaluates them using the
Active Inference utility formula, and proposes the best action(s).

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
    ActionNode,
    HCIREdgeType,
    HCIRNodeType,
)

logger = logging.getLogger(__name__)


class ActiveInferenceOperator:
    """Active Inference action selection over HCIR ActionNodes.

    Wraps the Active Inference utility formula from the existing
    ``ActiveInferenceEngine`` but operates on the frozen view.
    """

    def __init__(
        self,
        w_reward: float = 0.35,
        w_info_gain: float = 0.25,
        w_future_val: float = 0.20,
        w_risk: float = 0.10,
        w_cost: float = 0.10,
    ) -> None:
        self._w_reward = w_reward
        self._w_info_gain = w_info_gain
        self._w_future_val = w_future_val
        self._w_risk = w_risk
        self._w_cost = w_cost

    @property
    def operator_id(self) -> str:
        return "active_inference"

    @property
    def operator_name(self) -> str:
        return "Active Inference Action Selection"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.PLANNING: 0.9,
            ProblemType.PREDICTION: 0.5,
            ProblemType.CONSTRAINT: 0.4,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        n_actions = len(view.nodes_by_type(HCIRNodeType.ACTION))
        if n_actions == 0:
            return 0.0

        return min(1.0, base + min(0.3, n_actions * 0.05))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = len(context.graph_view.nodes_by_type(HCIRNodeType.ACTION))
        return ResourceCost(
            wall_clock_ms=max(1.0, n * 0.5),
            nodes_read=n,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Evaluate and rank candidate actions by Active Inference utility."""
        start = time.time()
        view = context.graph_view

        actions = view.nodes_by_type(HCIRNodeType.ACTION)
        if not actions:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No ActionNodes in view"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Evaluate each action ─────────────────────────────────────
        evaluations: list[dict[str, Any]] = []

        for node in actions:
            if not isinstance(node, ActionNode):
                continue

            # Extract features from the ActionNode
            risk_factor = 0.1  # Default low risk
            cost_val = float(node.estimated_cost) if node.estimated_cost else 10.0

            # Compute information gain heuristic from HCIR structure
            # Actions connected to more goals → higher info gain
            goal_edges = [
                e
                for e in view.edges_to(node.id)
                if e.edge_type in (HCIREdgeType.DEPENDS_ON, HCIREdgeType.REQUIRES)
            ]
            info_gain = min(1.0, 0.3 + len(goal_edges) * 0.15)

            reward = 1.0 - (risk_factor * 0.5)
            future_val = 0.8 - (cost_val * 0.005)
            risk_penalty = risk_factor
            cost_norm = min(1.0, cost_val * 0.01)

            utility = (
                self._w_reward * reward
                + self._w_info_gain * info_gain
                + self._w_future_val * max(0, future_val)
                - self._w_risk * risk_penalty
                - self._w_cost * cost_norm
            )

            evaluations.append(
                {
                    "action_id": node.id,
                    "intent": node.intent,
                    "utility": utility,
                    "reward": reward,
                    "info_gain": info_gain,
                    "future_val": future_val,
                    "risk": risk_penalty,
                    "cost": cost_norm,
                }
            )

        # Sort by utility
        evaluations.sort(key=lambda e: e["utility"], reverse=True)

        # ── Build result ─────────────────────────────────────────────
        provenance_chains: list[ProvenanceChain] = []

        for rank, ev in enumerate(evaluations[:5]):
            provenance_chains.append(
                ProvenanceChain(
                    conclusion=f"Action '{ev['intent']}' ranked #{rank + 1}",
                    evidence_node_ids=[ev["action_id"]],
                    operator_id=self.operator_id,
                    reasoning_steps=[
                        f"Utility={ev['utility']:.3f}",
                        f"Reward={ev['reward']:.3f}, InfoGain={ev['info_gain']:.3f}",
                        f"FutureVal={ev['future_val']:.3f}",
                        f"Risk={ev['risk']:.3f}, Cost={ev['cost']:.3f}",
                    ],
                    confidence=max(0, min(1, ev["utility"])),
                )
            )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "actions_evaluated": len(evaluations),
                "best_action": evaluations[0]["intent"] if evaluations else "",
                "best_utility": round(evaluations[0]["utility"], 4) if evaluations else 0,
                "rankings": [
                    {
                        "action": e["intent"],
                        "utility": round(e["utility"], 4),
                    }
                    for e in evaluations[:5]
                ],
            },
            confidence=max(0, min(1, evaluations[0]["utility"])) if evaluations else 0,
            evidence_refs=[e["action_id"] for e in evaluations[:5]],
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=len(evaluations),
            ),
        )
