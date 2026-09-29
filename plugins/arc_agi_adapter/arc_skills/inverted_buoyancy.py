"""Inverted Buoyancy & Excavation Climbing Skill Acquisition.

Acquires inductive kinematics and excavation planning for inverted gravity environments (e.g. bp35):
- Lateral navigation (Action 3: Left, Action 4: Right) with upward buoyant gravity
- Excavation of breakable blocks (Action 6) to open vertical conduits
- Progressive ascent to terminal summit exit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.common_subskills import DiscreteVectorTranslator, RemoteActuator
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    GridDimensionPredicate,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from plugins.arc_agi_adapter.arc_skills.perceptual_context import PerceptualSkillContext

logger = logging.getLogger(__name__)


class InvertedBuoyancySkillAcquisition(DeclarativeNeuroSymbolicSkill):
    """Induces excavation and upward buoyant navigation plans."""

    skill_name: str = "inverted_buoyancy_excavation"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE

    # Declarative Invariant Signature
    signature = AllOf(
        ActionAffordancePredicate(exact={3, 4, 6, 7}),
        GridDimensionPredicate(exact_shape=(64, 64)),
    )

    # Declarative Program (Compilable to HCIR Bytecode Stream)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.INTERACT,
            target_query={"role": "excavate_overhead", "action": 6},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"role": "summit_goal"},
        ),
    )

    @classmethod
    def is_buoyancy_excavation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains an inverted buoyancy climbing puzzle."""
        if set(available_actions) != {3, 4, 6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        return H == 64 and W == 64

    @classmethod
    def plan_buoyancy_excavation_grid(
        cls,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute perception-driven lateral moves, excavations, and summit navigation."""
        if grid.ndim == 3:
            grid = grid[-1]

        pctx = PerceptualSkillContext.from_grid(
            grid,
            available_actions=[3, 4, 6, 7],
            avatar_feature=metadata.get("avatar_color") if metadata else None,
        )

        avatar = pctx.avatar_entity
        # Perception: identify avatar entity (in bp35, avatar is color 9/11 with area 2..8)
        if avatar is None or avatar.color not in (9, 11):
            player_candidates = [e for e in pctx.entities if e.color in (9, 11)]
            if player_candidates:
                avatar = max(player_candidates, key=lambda e: e.centroid[0])
            else:
                singletons = [
                    e
                    for e in pctx.entities
                    if e.color != pctx.bg_color and e.color not in (0, 10, 14) and 2 <= e.area <= 36
                ]
                if singletons:
                    avatar = max(singletons, key=lambda e: e.centroid[0])

        if avatar is None:
            return []

        # 1. PERCEIVE: Check if a breakable block (color 14) is directly overhead
        overhead_r_min = max(0, avatar.min_r - 8)
        overhead_r_max = max(0, avatar.min_r)
        overhead_c_min = max(0, avatar.min_c - 1)
        overhead_c_max = min(grid.shape[1], avatar.max_c + 2)
        overhead_region = grid[overhead_r_min:overhead_r_max, overhead_c_min:overhead_c_max]

        if 14 in overhead_region:
            # Breakable block directly overhead! Excavate with Action 6
            click_x = int(round(avatar.centroid[1]))
            click_y = max(0, avatar.min_r - 4)
            return [RemoteActuator.click(click_x, click_y)]

        # 2. Summit goal check
        goals = [
            e
            for e in pctx.entities
            if e.color not in (pctx.bg_color, 0, 10, 14, 9, 11, 3) and e.area <= 50
        ]
        if goals and avatar.centroid[0] <= 15:
            target_goal = min(goals, key=lambda e: e.centroid[0])
            dx = int(round(target_goal.centroid[1] - avatar.centroid[1]))
            if abs(dx) >= 1:
                return DiscreteVectorTranslator.delta_to_actions(dx, 0)[:1]

        # 3. Lateral movement with wall collision avoidance
        H, W = grid.shape
        can_step_left = avatar.min_c >= 2 and not np.any(
            grid[avatar.min_r : avatar.max_r + 1, max(0, avatar.min_c - 3) : avatar.min_c] == 10
        )
        can_step_right = avatar.max_c < W - 2 and not np.any(
            grid[avatar.min_r : avatar.max_r + 1, avatar.max_c + 1 : min(W, avatar.max_c + 4)] == 10
        )

        if can_step_left and not can_step_right:
            return [(3, None)]
        if can_step_right and not can_step_left:
            return [(4, None)]

        # If both or neither open, choose direction towards closest breakable or open shaft
        breakables = [
            e for e in pctx.entities if e.color == 14 and e.centroid[0] < avatar.centroid[0]
        ]
        if breakables:
            closest_brk = min(
                breakables,
                key=lambda b: (
                    abs(b.centroid[1] - avatar.centroid[1])
                    + abs(b.centroid[0] - avatar.centroid[0]) * 0.5
                ),
            )
            if closest_brk.centroid[1] > avatar.centroid[1] and can_step_right:
                return [(4, None)]
            if closest_brk.centroid[1] < avatar.centroid[1] and can_step_left:
                return [(3, None)]

        if can_step_right:
            return [(4, None)]
        if can_step_left:
            return [(3, None)]

        return [(4, None)]

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for inverted buoyancy excavation."""
        return self.plan_buoyancy_excavation_grid(
            grid, current_level=current_level, metadata=metadata
        )
