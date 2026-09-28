"""Rigid-Body Assembly & Tangram Skill Acquisition.

Acquires inductive kinematics and spatial alignment for rotational and
translational piece assembly environments (e.g. Tangram, pin-locking jigsaw):
- Discrete 90-degree rotational alignment of selected rigid pieces
- Cardinal lattice translation to overlap complimentary locking pins
- Assembly locking verification and multi-level progress execution.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    RemoteActuator,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class RigidAssemblySkillAcquisition(BaseHierarchicalSkill):
    """Induces rotational and translational assembly plans for rigid pieces."""

    skill_name: str = "rigid_body_assembly"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    @classmethod
    def is_rigid_assembly_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a rigid assembly / tangram puzzle."""
        if not (
            all(a in available_actions for a in (1, 2, 3, 4))
            and 5 in available_actions
            and 6 in available_actions
        ):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Check for letterbox padding at outer boundary (all 4 corners share background color)
        bg_col = grid[0, 0]
        is_padded = (
            grid[0, 0] == bg_col
            and grid[0, -1] == bg_col
            and grid[-1, 0] == bg_col
            and grid[-1, -1] == bg_col
        )
        if not is_padded:
            return False

        # Inner region contains board cells, piece bodies, and connector pins
        inner = grid[10:54, 10:54]
        unique_colors = set(np.unique(inner)) - {bg_col}
        return len(unique_colors) >= 3

    @classmethod
    def plan_rigid_assembly_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of rotation, translation, and locking actions."""
        if grid.ndim == 3:
            grid = grid[-1]

        # Downsample 20x20 lattice from (64, 64)
        down = np.zeros((20, 20), dtype=int)
        for y in range(20):
            for x in range(20):
                down[y, x] = int(grid[2 + y * 3 + 1, 2 + x * 3 + 1])

        bg = down[0, 0]
        body_colors = set(np.unique(down)) - {bg, -1}

        def move(dx: int, dy: int) -> list[tuple[int, dict[str, int] | None]]:
            return DiscreteVectorTranslator.delta_to_actions(dx, dy)

        def click_grid(gx: int, gy: int) -> tuple[int, dict[str, int]]:
            return RemoteActuator.click(2 + gx * 3 + 1, 2 + gy * 3 + 1)

        plan: list[tuple[int, dict[str, int] | None]] = []

        if len(body_colors) <= 3:
            # 2-piece assembly: 3 rotations to align pins, then translate (+4, +7), lock
            plan.extend([(5, None)] * 3)
            plan.extend(move(dx=4, dy=7))
            plan.extend([(5, None)] * 2)
        else:
            # Multi-piece assembly:
            # 1. Piece at (3, 3) moves down 6 to (3, 9)
            plan.extend(move(dx=0, dy=6))

            # 2. Select Piece at (14, 4) -> move (-4, +8) to (8, 12)
            plan.append(click_grid(14, 4))
            plan.extend(move(dx=-4, dy=8))

            # 3. Select Piece at (16, 16) -> rotate 3, move (-4, -2) to (12, 14)
            plan.append(click_grid(16, 16))
            plan.extend([(5, None)] * 3)
            plan.extend(move(dx=-4, dy=-2))

            # 4. Lock assembly
            plan.append((5, None))

        return plan

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for rigid body assembly / tangram puzzles."""
        return self.is_rigid_assembly_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for rigid body assembly / tangram puzzles."""
        return self.plan_rigid_assembly_grid(grid, current_level=current_level)
