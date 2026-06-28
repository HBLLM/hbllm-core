"""Predictive Simulation Engine with Layered Simulators."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from hbllm.brain.autonomy.task_graph import Goal, TaskNode
from hbllm.brain.causality.causal_graph import CausalGraph
from hbllm.brain.cognitive_state import CognitiveState
from hbllm.brain.simulation.models import CounterfactualScenario, PredictionOrigin
from hbllm.brain.simulation.projector import ProjectedState
from hbllm.brain.simulation.risk import RiskCategory, RiskEngine
from hbllm.brain.world_state import WorldStateEngine

logger = logging.getLogger(__name__)


# ─── Simulation Layer Interface ───────────────────────────────────


class SimulationLayer(ABC):
    """Abstract base for a modular simulation layer."""

    @abstractmethod
    async def simulate(self, state: CognitiveState, proposed_mutation: dict[str, Any]) -> float:
        """Run the simulation layer.

        Returns:
            Risk score between 0.0 (no risk) and 1.0 (terminal threat).
        """
        pass


# ─── Domain-Specific Simulation Layers ────────────────────────────


class SafetySimulator(SimulationLayer):
    """Evaluates safety of actions, rejecting malicious patterns."""

    def __init__(self) -> None:
        self._dangerous_patterns = [
            re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
            re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
            re.compile(r"\bmkfs\b", re.IGNORECASE),
            re.compile(r"\bcurl\b.*\|\s*\bbash\b", re.IGNORECASE),
        ]

    async def simulate(self, state: CognitiveState, proposed_mutation: dict[str, Any]) -> float:
        mutation_type = proposed_mutation.get("type")
        if mutation_type == "action":
            action_cmd = str(proposed_mutation.get("action", ""))
            for pattern in self._dangerous_patterns:
                if pattern.search(action_cmd):
                    logger.warning("[Safety Simulator] Dangerous command blocked: %s", action_cmd)
                    return 1.0
        return 0.0


class ReliabilitySimulator(SimulationLayer):
    """Evaluates system reliability based on self-model confidence."""

    async def simulate(self, state: CognitiveState, proposed_mutation: dict[str, Any]) -> float:
        # Check overall policy confidence constraints
        policy = state.effective_policy
        if policy.simulation_depth > 1 and state.confidence < 0.4:
            # Low confidence triggers higher risk in reliability
            return 0.5
        return 0.1


class SocialSimulator(SimulationLayer):
    """Evaluates user interruption and cognitive loading impact."""

    async def simulate(self, state: CognitiveState, proposed_mutation: dict[str, Any]) -> float:
        mutation_type = proposed_mutation.get("type")
        if mutation_type == "action" and proposed_mutation.get("is_notification"):
            # Check attention limits
            attention = state.effective_policy.budget.attention_budget
            if attention < 0.5:
                # Low attention budget means high social cost for notifications
                return 0.8
        return 0.0


class ResourceSimulator(SimulationLayer):
    """Evaluates expected execution cost (latency, tokens)."""

    async def simulate(self, state: CognitiveState, proposed_mutation: dict[str, Any]) -> float:
        mutation_type = proposed_mutation.get("type")
        if mutation_type == "action":
            expected_latency = proposed_mutation.get("expected_latency_ms", 0.0)
            planning_budget = state.effective_policy.budget.planning_budget * 1000.0
            if expected_latency > planning_budget:
                # Exceeds allocated planning latency budget
                return 0.7
        return 0.1


class MemoryBeliefSimulator(SimulationLayer):
    """Evaluates database mutations and belief consistency updates."""

    async def simulate(self, state: CognitiveState, proposed_mutation: dict[str, Any]) -> float:
        mutation_type = proposed_mutation.get("type")
        if mutation_type == "belief_update":
            new_belief = proposed_mutation.get("belief", {})
            current_beliefs = state.beliefs
            for b in current_beliefs:
                if b.get("topic") == new_belief.get("topic") and b.get("value") != new_belief.get(
                    "value"
                ):
                    # Contradiction detected
                    logger.warning(
                        "[Memory Simulator] Direct belief contradiction: %s vs %s", b, new_belief
                    )
                    return 0.9
        return 0.0


# ─── Predictive Simulation Engine ─────────────────────────────────


class LayeredSimulationEngine:
    """Orchestrator for layered counterfactual simulations."""

    def __init__(self, layers: list[SimulationLayer] | None = None) -> None:
        self.layers = (
            layers
            if layers is not None
            else [
                SafetySimulator(),
                ReliabilitySimulator(),
                SocialSimulator(),
                ResourceSimulator(),
                MemoryBeliefSimulator(),
            ]
        )

    async def simulate_mutation(
        self, state: CognitiveState, proposed_mutation: dict[str, Any]
    ) -> dict[str, Any]:
        """Run all simulation layers, checking constraints against policy limits."""
        layer_risks = {}
        total_risk = 0.0

        for layer in self.layers:
            name = layer.__class__.__name__
            risk = await layer.simulate(state, proposed_mutation)
            layer_risks[name] = risk
            total_risk = max(total_risk, risk)  # Worst case scenario risk model

        # Default threshold of 0.8 to approve mutation
        allowed = total_risk < 0.8

        return {"allowed": allowed, "risk_score": total_risk, "layer_risks": layer_risks}

    # ─── Universal Interceptors ─────────────────────────────────────

    async def simulate_action(self, state: CognitiveState, command: str) -> bool:
        res = await self.simulate_mutation(state, {"type": "action", "action": command})
        return res["allowed"]

    async def simulate_memory_write(self, state: CognitiveState, key: str, value: Any) -> bool:
        res = await self.simulate_mutation(
            state, {"type": "memory_write", "key": key, "value": value}
        )
        return res["allowed"]

    async def simulate_belief_update(self, state: CognitiveState, belief: dict[str, Any]) -> bool:
        res = await self.simulate_mutation(state, {"type": "belief_update", "belief": belief})
        return res["allowed"]

    async def simulate_learning_commit(self, state: CognitiveState, learning_goal: str) -> bool:
        res = await self.simulate_mutation(
            state, {"type": "learning_commit", "goal": learning_goal}
        )
        return res["allowed"]


# ─── Legacy Backward Compatibility Implementations ──────────────────────


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
        return {}, 0.0

    def _speculative_hallucination(self, task: TaskNode) -> dict[str, dict[str, Any]]:
        """LLM-backed consequence hallucination."""
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

        if tier == 0:
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
