"""Predictive Simulation Engine and Consequence Estimation."""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.autonomy.task_graph import Goal, TaskNode
from hbllm.brain.causality.causal_graph import CausalGraph
from hbllm.brain.simulation.models import CounterfactualScenario, PredictionOrigin
from hbllm.brain.simulation.projector import ProjectedState
from hbllm.brain.simulation.risk import RiskCategory, RiskEngine
from hbllm.brain.world_state import WorldStateEngine

logger = logging.getLogger(__name__)


class ConsequenceEstimator:
    """Estimates consequences of an action via rules, empirical data, or speculation."""

    def __init__(self, causal_graph: CausalGraph) -> None:
        self.causal_graph = causal_graph
        self.speculative_calls_this_hour = 0

    def estimate(
        self, task: TaskNode, projected_state: ProjectedState
    ) -> tuple[dict[str, dict[str, Any]], PredictionOrigin, float]:
        """Estimate state changes if this task executes.

        Returns:
            dict mapping entity_id to a dict of property mutations.
            origin of the prediction.
            confidence score.
        """
        # 1. Deterministic Mode (Hard rules)
        det_mutations = self._deterministic_rules(task, projected_state)
        if det_mutations:
            return det_mutations, PredictionOrigin.INFERRED, 1.0

        # 2. Empirical Mode (CausalGraph)
        emp_mutations, emp_conf = self._empirical_lookup(task)
        if emp_mutations:
            return emp_mutations, PredictionOrigin.HISTORICAL, emp_conf

        # 3. Speculative Mode (LLM)
        self.speculative_calls_this_hour += 1
        spec_mutations = self._speculative_hallucination(task)
        return spec_mutations, PredictionOrigin.SPECULATIVE, 0.5

    def _deterministic_rules(
        self, task: TaskNode, projected_state: ProjectedState
    ) -> dict[str, dict[str, Any]]:
        """Hardcoded logic for known system constraints."""
        if task.action_topic == "system.sleep":
            return {"system_state": {"status": "sleeping"}}
        return {}

    def _empirical_lookup(self, task: TaskNode) -> tuple[dict[str, dict[str, Any]], float]:
        """Query CausalGraph for past consequences of similar tasks."""
        # Stub implementation
        return {}, 0.0

    def _speculative_hallucination(self, task: TaskNode) -> dict[str, dict[str, Any]]:
        """LLM-backed consequence hallucination."""
        # Stub implementation
        return {"simulated_entity": {"outcome": f"simulated_{task.action_topic}"}}


class PredictiveSimulationEngine:
    """Core engine for tiered future simulation."""

    def __init__(
        self, world_state: WorldStateEngine, causal_graph: CausalGraph, risk_engine: RiskEngine
    ) -> None:
        self.world_state = world_state
        self.estimator = ConsequenceEstimator(causal_graph)
        self.risk_engine = risk_engine

    def simulate_plan(
        self, goal: Goal, tasks: list[TaskNode], tier: int = 1
    ) -> CounterfactualScenario:
        """Simulate a plan and return the scored scenario."""
        projected = ProjectedState(base_state=self.world_state)

        overall_origin = PredictionOrigin.INFERRED
        overall_confidence = 1.0

        # Determine simulation depth based on tier
        if tier == 0:
            # Heuristic only, no deep projection
            future_state = projected.finalize(origin=overall_origin)
            scenario = CounterfactualScenario(
                goal_id=goal.goal_id,
                proposed_tasks=tasks,
                predicted_state=future_state,
                utility_score=0.5,
            )
            self.risk_engine.evaluate_scenario(scenario)
            return scenario

        for task in tasks:
            mutations, origin, conf = self.estimator.estimate(task, projected)

            for entity_id, props in mutations.items():
                projected.update_entity(entity_id, props, confidence=conf)

            # Downgrade overall origin to worst case
            if origin == PredictionOrigin.SPECULATIVE:
                overall_origin = PredictionOrigin.SPECULATIVE
            elif (
                origin == PredictionOrigin.HISTORICAL
                and overall_origin != PredictionOrigin.SPECULATIVE
            ):
                overall_origin = PredictionOrigin.HISTORICAL

            overall_confidence *= conf

        future_state = projected.finalize(origin=overall_origin)
        future_state.predicted_confidence = overall_confidence

        # Basic heuristic risk formulation
        risks = {}
        if overall_origin == PredictionOrigin.SPECULATIVE:
            risks[RiskCategory.RELIABILITY.value] = 0.8

        scenario = CounterfactualScenario(
            goal_id=goal.goal_id,
            proposed_tasks=tasks,
            predicted_state=future_state,
            utility_score=0.8,
            risk_categories=risks,
        )

        self.risk_engine.evaluate_scenario(scenario)
        return scenario
