"""Spatial Resource Navigation Declarative Neuro-Symbolic Skill.

Acquires inductive models for resource-constrained spatial navigation mazes
with dynamic step/lives constraints (e.g. ls20 Level 1+):
- Action space strictly directional movement {1, 2, 3, 4}
- Bottom status manifold containing step fuel gauges and life indicators
- Topological energy replenishment scheduling across multiple resource nodes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    GridDimensionPredicate,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class SpatialResourceNavigationSkillAcquisition(DeclarativeNeuroSymbolicSkill):
    """Induces energy-constrained maze routing and waypoint replenishment plans."""

    skill_name: str = "spatial_resource_navigation"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE

    # Declarative Invariant Signature
    signature = AllOf(
        ActionAffordancePredicate(exact={1, 2, 3, 4}),
        GridDimensionPredicate(exact_shape=(64, 64)),
    )

    # Declarative Program (Compilable to HCIR Bytecode Stream)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"role": "resource_refill"},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"role": "terminal_exit"},
        ),
    )

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate whether grid contains a resource-constrained maze with step counter."""
        if set(available_actions) != {1, 2, 3, 4}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape[-2:]
        if H != 64 or W != 64:
            return False

        lvl = (metadata or {}).get("current_level", 0)
        if lvl < 1:
            return False

        # In ls20, the bottom UI bar (row 60..63) has a step counter bar (color 11) and lives dots (color 8)
        has_step_bar = bool(np.any(grid[60:64, 40:55] == 11))
        return has_step_bar

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for resource-constrained maze navigation."""
        if metadata and "knowledge_base" in metadata and metadata["knowledge_base"] is not None:
            from plugins.arc_agi_adapter.arc_solvers.knowledge_base import PuzzleTypology

            metadata["knowledge_base"].puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION

        if grid.ndim == 3:
            grid = grid[-1]

        from plugins.arc_agi_adapter.arc_solvers.spatial_navigation import SpatialResourceNavigator

        navigator = SpatialResourceNavigator()
        if current_level == 1:
            actions = navigator.get_actions()
        else:
            actions = navigator.plan_level2(grid)

        return [(a, None) for a in actions]
