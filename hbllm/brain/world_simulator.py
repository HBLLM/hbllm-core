"""
World Simulator — predictive outcome simulation for planning.

Extends WorldModelNode with ability to simulate multiple action
paths before choosing the best one.

Flow:
1. PlannerNode generates candidate strategies
2. Simulator runs each strategy through outcome prediction
3. Scenarios are scored by expected reward
4. Best strategy is selected for execution
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A simulated scenario with predicted outcomes."""
    scenario_id: str
    strategy: str
    steps: list[str]
    predicted_outcome: str
    confidence: float  # 0-1
    expected_reward: float  # -1 to 1
    risks: list[str] = field(default_factory=list)
    resource_cost: float = 0.0  # estimated computational cost
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of simulating multiple scenarios."""
    best_scenario: Scenario
    all_scenarios: list[Scenario]
    simulation_time_ms: float
    consensus_confidence: float


class WorldSimulator:
    """
    Predictive world simulation for enhanced planning.

    Instead of just executing the first plan, the simulator:
    1. Generates multiple strategy variations
    2. Predicts outcomes for each
    3. Evaluates risks and rewards
    4. Selects the optimal path

    This dramatically improves complex reasoning and reduces errors.
    """

    def __init__(self, max_scenarios: int = 5, risk_weight: float = 0.3):
        self.max_scenarios = max_scenarios
        self.risk_weight = risk_weight
        self._simulations_run = 0

    async def simulate(
        self,
        goal: str,
        strategies: list[dict[str, Any]],
        predict_fn: Any = None,
    ) -> SimulationResult:
        """
        Simulate multiple strategies and select the best.

        Args:
            goal: What we're trying to achieve
            strategies: List of strategy dicts with 'name', 'steps'
            predict_fn: Optional async fn(strategy) -> outcome prediction
        """
        start = time.monotonic()
        self._simulations_run += 1

        scenarios: list[Scenario] = []
        for i, strategy in enumerate(strategies[:self.max_scenarios]):
            scenario = await self._simulate_strategy(
                f"scenario_{i}", goal, strategy, predict_fn,
            )
            scenarios.append(scenario)

        # Sort by composite score
        scenarios.sort(
            key=lambda s: s.expected_reward * s.confidence - len(s.risks) * self.risk_weight,
            reverse=True,
        )

        best = scenarios[0] if scenarios else Scenario(
            scenario_id="fallback", strategy="direct",
            steps=["Execute directly"], predicted_outcome="Unknown",
            confidence=0.3, expected_reward=0.0,
        )

        # Consensus confidence = how much do top strategies agree?
        if len(scenarios) >= 2:
            top_scores = [s.expected_reward for s in scenarios[:3]]
            spread = max(top_scores) - min(top_scores)
            consensus = max(0.0, 1.0 - spread)
        else:
            consensus = best.confidence

        elapsed = (time.monotonic() - start) * 1000

        return SimulationResult(
            best_scenario=best,
            all_scenarios=scenarios,
            simulation_time_ms=elapsed,
            consensus_confidence=round(consensus, 3),
        )

    async def _simulate_strategy(
        self,
        scenario_id: str,
        goal: str,
        strategy: dict[str, Any],
        predict_fn: Any = None,
    ) -> Scenario:
        """Simulate a single strategy."""
        name = strategy.get("name", "unnamed")
        steps = strategy.get("steps", [])
        tools = strategy.get("tools", [])

        # Predict outcome
        if predict_fn:
            prediction = await predict_fn(strategy)
            outcome = prediction.get("outcome", "Unknown")
            confidence = prediction.get("confidence", 0.5)
            reward = prediction.get("reward", 0.0)
            risks = prediction.get("risks", [])
        else:
            outcome, confidence, reward, risks = self._heuristic_predict(goal, steps, tools)

        return Scenario(
            scenario_id=scenario_id,
            strategy=name,
            steps=steps,
            predicted_outcome=outcome,
            confidence=confidence,
            expected_reward=reward,
            risks=risks,
            resource_cost=len(steps) * 0.1,  # simple cost estimate
        )

    def _heuristic_predict(
        self, goal: str, steps: list[str], tools: list[str],
    ) -> tuple[str, float, float, list[str]]:
        """Heuristic outcome prediction when no ML model is available."""
        risks: list[str] = []
        confidence = 0.6
        reward = 0.5

        # More steps = more risk of failure
        if len(steps) > 5:
            risks.append("Complex multi-step plan may have cascading failures")
            confidence -= 0.1
        if len(steps) > 10:
            risks.append("Very long plan — consider breaking into sub-goals")
            confidence -= 0.15

        # External tool dependencies = risk
        external_tools = [t for t in tools if t in {"api", "browser", "database"}]
        if external_tools:
            risks.append(f"Depends on external tools: {external_tools}")
            confidence -= 0.05

        # Simple plans are more likely to succeed
        if len(steps) <= 3:
            confidence += 0.1
            reward += 0.1

        confidence = max(0.1, min(1.0, confidence))
        reward = max(-1.0, min(1.0, reward))
        outcome = f"Execute {len(steps)} steps to achieve: {goal[:100]}"

        return outcome, confidence, reward, risks

    def stats(self) -> dict[str, int]:
        return {"simulations_run": self._simulations_run}
