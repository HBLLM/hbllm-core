"""Metric continuous trajectory planner over ReferenceFrame coordinates (WP-C1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.planning.mental_simulation import MentalSimulationStep


class ContinuousKinematicPlanner:
    """Metric continuous trajectory planner over ReferenceFrame coordinates (WP-C1)."""

    @staticmethod
    def plan_continuous_trajectory(
        engine: AutonomousEpistemicEngine,
        available_actions: Sequence[Any],
        curr_grid: np.ndarray | None = None,
        goal_radius: float = 0.55,
        max_expansions: int = 512,
    ) -> list[MentalSimulationStep] | None:
        """Plan a multi-step kinematic trajectory in continuous metric space using grounded velocity vectors."""
        from hbllm.hcir.world.planning.mental_simulation import MentalSimulationStep

        raw_steps = GeodesicManifoldRouter.route_continuous_kinematic(
            engine=engine,
            available_actions=available_actions,
            curr_grid=curr_grid,
            goal_radius=goal_radius,
            max_expansions=max_expansions,
        )
        if not raw_steps:
            return None
        return [
            MentalSimulationStep(
                action=act,
                predicted_avatar_pos=lat_p,
                expected_mutation="continuous_kinematic_step",
            )
            for act, lat_p in raw_steps
        ]
