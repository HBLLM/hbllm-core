"""Mental Sandbox, Multi-Branch Search, and Counterfactual Reasoning Engine for A18.

Orchestrates ephemeral simulation branches, forward trajectory rollouts,
safety/risk gating, and goal evaluation without mutating canonical HCIR reality.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.simulation.branch import SimulationBranch
from hbllm.brain.simulation.operators import (
    ActionOperator,
    MoveOperator,
    OperatorExecutionResult,
    PushOperator,
    PutInOperator,
    StackOperator,
)
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


@dataclass
class PredictedWorldState:
    """A projected future world state derived from mental simulation."""

    branch_id: str
    depth: int
    state_hash: str
    confidence: float = 1.0
    risk: float = 0.0
    active_relations: list[tuple[str, str, str]] = field(default_factory=list)
    consequences: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """Result of evaluating a hypothetical action sequence or counterfactual query."""

    branch_id: str
    initial_revision: int
    actions: list[tuple[str, dict[str, Any]]]
    final_predicted_state: PredictedWorldState
    goal_achieved: bool = False
    risk_score: float = 0.0
    confidence: float = 1.0
    violations: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class ExecutionPlan:
    """A validated action sequence approved from mental simulation for physical execution."""

    plan_id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    source_branch_id: str = ""
    validated_actions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    predicted_confidence: float = 1.0
    predicted_risk: float = 0.0
    status: str = "READY_FOR_EXECUTION"


class MentalSandbox:
    """The central A18 embodied simulator and counterfactual reasoning engine."""

    def __init__(self) -> None:
        self._operators: dict[str, ActionOperator] = {
            "PUSH": PushOperator(),
            "STACK": StackOperator(),
            "PUT_IN": PutInOperator(),
            "MOVE": MoveOperator(),
        }

    def register_operator(self, operator: ActionOperator) -> None:
        self._operators[operator.name.upper()] = operator

    def fork_branch(
        self,
        base_graph: CognitiveGraph,
        branch_id: str | None = None,
        base_revision: int = 1,
    ) -> SimulationBranch:
        """Fork an isolated, ephemeral SimulationBranch from canonical HCIR."""
        bid = branch_id or f"sim_branch_{uuid.uuid4().hex[:6]}"
        return SimulationBranch(
            branch_id=bid,
            base_graph=base_graph,
            base_revision=base_revision,
            depth=0,
        )

    def simulate_action(
        self,
        branch: SimulationBranch,
        operator_name: str,
        params: dict[str, Any],
        step: int = 0,
    ) -> OperatorExecutionResult:
        """Apply a single action operator within a simulation branch."""
        op = self._operators.get(operator_name.upper())
        if op is None:
            pre_hash = branch.compute_current_state_hash()
            return OperatorExecutionResult(
                operator_name=operator_name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=["unknown_operator"],
                risk=0.8,
                reason=f"Operator {operator_name} not registered in MentalSandbox",
            )

        return op.execute(branch, params, step=step)

    def simulate_trajectory(
        self,
        base_graph: CognitiveGraph,
        action_sequence: list[tuple[str, dict[str, Any]]],
        branch_id: str | None = None,
        base_revision: int = 1,
    ) -> tuple[SimulationBranch, list[OperatorExecutionResult]]:
        """Roll a sequence of hypothetical actions forward in an isolated branch."""
        branch = self.fork_branch(base_graph, branch_id=branch_id, base_revision=base_revision)
        results: list[OperatorExecutionResult] = []

        for step, (op_name, params) in enumerate(action_sequence):
            res = self.simulate_action(branch, op_name, params, step=step + 1)
            results.append(res)
            branch.depth += 1
            branch.accumulated_risk = max(branch.accumulated_risk, res.risk)
            if res.violations:
                branch.violated_constraints.extend(res.violations)
            if not res.is_success:
                # Execution failed / constraint violated -> stop trajectory
                break

        return branch, results

    def evaluate_counterfactual(
        self,
        base_graph: CognitiveGraph,
        hypothetical_actions: list[tuple[str, dict[str, Any]]],
        goal_predicate: Callable[[SimulationBranch], bool] | None = None,
        branch_id: str | None = None,
    ) -> CounterfactualResult:
        """Evaluate 'What would happen if I did actions X?'."""
        branch, step_results = self.simulate_trajectory(base_graph, hypothetical_actions, branch_id=branch_id)

        all_consequences: list[str] = []
        all_violations: list[str] = list(branch.violated_constraints)
        for r in step_results:
            all_consequences.extend(r.consequences)
            all_violations.extend(r.violations)

        active_relations = [
            (src, e.edge_type.value, tgt)
            for e in branch.all_edges()
            for src in e.sources
            for tgt in e.targets
        ]

        pred_state = PredictedWorldState(
            branch_id=branch.branch_id,
            depth=branch.depth,
            state_hash=branch.compute_current_state_hash(),
            confidence=branch.confidence,
            risk=branch.accumulated_risk,
            active_relations=active_relations,
            consequences=all_consequences,
            violations=all_violations,
        )

        goal_achieved = goal_predicate(branch) if goal_predicate else (len(all_violations) == 0)

        explanation = (
            f"Simulation branch {branch.branch_id} reached state with confidence {branch.confidence:.2f} "
            f"and risk {branch.accumulated_risk:.2f}. "
        )
        if all_violations:
            explanation += f"Violated constraints: {', '.join(set(all_violations))}."
        elif goal_achieved:
            explanation += "Goal successfully achieved."

        return CounterfactualResult(
            branch_id=branch.branch_id,
            initial_revision=branch.base_revision,
            actions=hypothetical_actions,
            final_predicted_state=pred_state,
            goal_achieved=goal_achieved,
            risk_score=branch.accumulated_risk,
            confidence=branch.confidence,
            violations=list(set(all_violations)),
            explanation=explanation,
        )

    def multi_branch_search(
        self,
        base_graph: CognitiveGraph,
        candidate_trajectories: list[list[tuple[str, dict[str, Any]]]],
        goal_predicate: Callable[[SimulationBranch], bool],
    ) -> tuple[CounterfactualResult | None, list[CounterfactualResult]]:
        """Evaluate competing hypothetical trajectories in parallel and select the optimal plan.

        Returns:
            (winning_result, all_branch_results)
        """
        results: list[CounterfactualResult] = []
        for i, actions in enumerate(candidate_trajectories):
            res = self.evaluate_counterfactual(
                base_graph,
                actions,
                goal_predicate=goal_predicate,
                branch_id=f"plan_b{i+1}",
            )
            results.append(res)

        # Filter valid goal-achieving candidates with low risk
        valid_candidates = [r for r in results if r.goal_achieved and r.risk_score < 0.50]

        if not valid_candidates:
            # If none achieved goal safely, pick lowest risk
            sorted_all = sorted(results, key=lambda r: (r.risk_score, -r.confidence))
            return None, sorted_all

        # Rank valid by (lowest risk, highest confidence, shortest path)
        sorted_valid = sorted(valid_candidates, key=lambda r: (r.risk_score, -r.confidence, len(r.actions)))
        winner = sorted_valid[0]
        return winner, results

    def produce_execution_plan(self, winner_result: CounterfactualResult) -> ExecutionPlan:
        """Produce validated action sequence for physical actuator execution.

        Simulation facts remain outside reality; only the approved action plan is emitted.
        """
        return ExecutionPlan(
            source_branch_id=winner_result.branch_id,
            validated_actions=winner_result.actions,
            predicted_confidence=winner_result.confidence,
            predicted_risk=winner_result.risk_score,
            status="READY_FOR_EXECUTION",
        )
