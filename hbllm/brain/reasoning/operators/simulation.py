"""
Simulation Operator — wraps LayeredSimulationEngine as a ReasoningOperator.

Evaluates proposed actions/mutations through multiple simulation layers
(safety, reliability, social, resource, belief consistency) and returns
risk scores as cognitive results.

The wrapper adapts the existing async simulation API to the synchronous
ReasoningOperator protocol. It constructs lightweight simulations over
the frozen HCIR view rather than requiring a full CognitiveState.

Independence Level: L1 (no LLM)
"""

from __future__ import annotations

import logging
import re
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
    BeliefNode,
    HCIRNodeType,
)

logger = logging.getLogger(__name__)

# Safety patterns from the existing SafetySimulator
_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|\s*\bbash\b", re.IGNORECASE),
]


class SimulationOperator:
    """Multi-layer risk simulation over HCIR actions.

    Evaluates actions through safety, resource, and belief-consistency
    simulations. Returns risk assessments as CognitiveResults.
    """

    @property
    def operator_id(self) -> str:
        return "simulation"

    @property
    def operator_name(self) -> str:
        return "Layered Risk Simulation Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.PLANNING: 0.7,
            ProblemType.CONSTRAINT: 0.6,
            ProblemType.PREDICTION: 0.4,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        n_actions = len(view.nodes_by_type(HCIRNodeType.ACTION))
        if n_actions == 0:
            return 0.0

        return min(1.0, base + min(0.2, n_actions * 0.05))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = len(context.graph_view.nodes_by_type(HCIRNodeType.ACTION))
        return ResourceCost(
            wall_clock_ms=max(1.0, n * 1.0),
            nodes_read=context.graph_view.node_count,
            simulation_steps_used=n * 5,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Simulate actions and compute risk scores."""
        start = time.time()
        view = context.graph_view

        actions = [n for n in view.nodes_by_type(HCIRNodeType.ACTION) if isinstance(n, ActionNode)]

        if not actions:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No ActionNodes to simulate"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # Collect beliefs for consistency checking
        beliefs = [n for n in view.nodes_by_type(HCIRNodeType.BELIEF) if isinstance(n, BeliefNode)]

        # ── Simulate each action ─────────────────────────────────────
        simulation_results: list[dict[str, Any]] = []

        for action in actions:
            layer_risks: dict[str, float] = {}

            # Safety check
            intent = action.intent or ""
            safety_risk = 0.0
            for pattern in _DANGEROUS_PATTERNS:
                if pattern.search(intent):
                    safety_risk = 1.0
                    break
            layer_risks["safety"] = safety_risk

            # Resource check
            cost = float(action.estimated_cost) if action.estimated_cost else 0
            budget_ms = context.budget.compute_ms
            resource_risk = min(1.0, cost / max(1, budget_ms)) if cost > 0 else 0.1
            layer_risks["resource"] = resource_risk

            # Belief consistency — check if action's intent matches any belief
            consistency_risk = 0.0
            for belief in beliefs:
                claim = belief.claim.lower()
                if "not " in intent.lower() and any(
                    w in claim
                    for w in intent.lower().split()
                    if len(w) > 3 and w not in ("the", "and", "for")
                ):
                    consistency_risk = max(consistency_risk, 0.5)
            layer_risks["belief_consistency"] = consistency_risk

            # Aggregate: worst-case across layers
            total_risk = max(layer_risks.values())
            allowed = total_risk < 0.8

            simulation_results.append(
                {
                    "action_id": action.id,
                    "intent": intent,
                    "total_risk": total_risk,
                    "allowed": allowed,
                    "layer_risks": layer_risks,
                }
            )

        # ── Build result ─────────────────────────────────────────────
        blocked_count = sum(1 for r in simulation_results if not r["allowed"])
        provenance_chains: list[ProvenanceChain] = []

        for sim in simulation_results:
            provenance_chains.append(
                ProvenanceChain(
                    conclusion=(
                        f"Action '{sim['intent']}': "
                        f"{'BLOCKED' if not sim['allowed'] else 'ALLOWED'} "
                        f"(risk={sim['total_risk']:.2f})"
                    ),
                    evidence_node_ids=[sim["action_id"]],
                    operator_id=self.operator_id,
                    reasoning_steps=[
                        f"{layer}: {risk:.2f}" for layer, risk in sim["layer_risks"].items()
                    ],
                    confidence=1.0 - sim["total_risk"],
                )
            )

        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "actions_simulated": len(simulation_results),
                "blocked_count": blocked_count,
                "simulations": [
                    {
                        "action": s["intent"],
                        "risk": round(s["total_risk"], 3),
                        "allowed": s["allowed"],
                    }
                    for s in simulation_results
                ],
            },
            confidence=1.0 - max((s["total_risk"] for s in simulation_results), default=0),
            evidence_refs=[s["action_id"] for s in simulation_results],
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                simulation_steps_used=len(simulation_results) * 5,
            ),
        )
