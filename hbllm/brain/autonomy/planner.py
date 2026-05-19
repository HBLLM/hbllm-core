"""Anticipatory Planner for Goal Execution.

Integrates the PredictiveSimulationEngine to evaluate multiple candidate
strategies before committing them to the TaskGraphRuntime.
"""

from __future__ import annotations

import logging

from hbllm.brain.autonomy.task_graph import Goal, TaskGraphRuntime, TaskNode
from hbllm.brain.simulation.engine import PredictiveSimulationEngine
from hbllm.brain.simulation.models import CounterfactualScenario

logger = logging.getLogger(__name__)


class AnticipatoryPlanner:
    """Plans goals by forecasting consequences and arbitrating risk/utility."""

    def __init__(self, simulator: PredictiveSimulationEngine, runtime: TaskGraphRuntime) -> None:
        self.simulator = simulator
        self.runtime = runtime

    def plan_and_execute(self, goal: Goal, strategies: list[list[TaskNode]], tier: int = 1) -> bool:
        """Evaluate multiple strategies, select the safest/highest utility, and execute.

        Args:
            goal: The Goal to achieve.
            strategies: A list of candidate task sequences.
            tier: Simulation depth tier (0-3).

        Returns:
            True if a safe plan was found and submitted, False otherwise.
        """
        if not strategies:
            logger.warning("No strategies provided for goal %s", goal.name)
            return False

        scored_scenarios: list[CounterfactualScenario] = []

        # 1. Simulate and Score
        for i, task_sequence in enumerate(strategies):
            scenario = self.simulator.simulate_plan(goal, task_sequence, tier=tier)
            scenario.scenario_id = f"{goal.goal_id}_plan_{i}"
            scored_scenarios.append(scenario)

        # 2. Arbitrate / Rank
        # Sort by final score (highest is best)
        scored_scenarios.sort(
            key=lambda s: self.simulator.risk_engine.evaluate_scenario(s),
            reverse=True
        )

        best_scenario = scored_scenarios[0]
        final_score = self.simulator.risk_engine.evaluate_scenario(best_scenario)

        # 3. Safety Gate
        if final_score < 0.1: # Extremely low utility or unacceptable risk
            logger.error(
                "All strategies for goal %s exceeded risk thresholds or had negligible utility. Aborting.",
                goal.name
            )
            return False

        # 4. Commit to Execution
        logger.info(
            "Selected strategy %s for goal %s with projected utility %.2f",
            best_scenario.scenario_id, goal.name, final_score
        )

        # Submit to TaskGraphRuntime
        self.runtime.create_goal(goal, best_scenario.proposed_tasks)
        return True
