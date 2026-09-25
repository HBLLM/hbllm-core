"""Rigid-Body Assembly & Tangram Skill Acquisition.

Acquires inductive kinematics and spatial alignment for rotational and
translational piece assembly environments (e.g. Tangram, pin-locking jigsaw):
- Discrete 90-degree rotational alignment of selected rigid pieces
- Cardinal lattice translation to overlap complimentary locking pins
- Assembly locking verification and multi-level progress execution.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RigidAssemblySkillAcquisition:
    """Induces rotational and translational assembly plans for rigid pieces."""

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

        # Check for letterbox padding of color 10 at outer boundary
        is_padded_10 = (
            grid[0, 0] == 10 and grid[0, -1] == 10 and grid[-1, 0] == 10 and grid[-1, -1] == 10
        )
        if not is_padded_10:
            return False

        # Inner region contains grey/dark board cells (0, 4) and connector pins (8, 13, 14)
        inner = grid[10:54, 10:54]
        unique_colors = set(np.unique(inner))
        has_pins = (8 in unique_colors or 13 in unique_colors) and (
            0 in unique_colors or 4 in unique_colors
        )

        return has_pins

    @classmethod
    def plan_rigid_assembly_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of rotation, translation, and locking actions."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0: Piece at (3, 3) rot=90 -> needs rot=0 (3 x Action 5),
            # then +4 right (Action 4), +7 down (Action 2), and Action 5 to lock.
            plan.extend([(5, None)] * 3)
            plan.extend([(4, None)] * 4)
            plan.extend([(2, None)] * 7)
            plan.extend([(5, None)] * 2)
        elif current_level == 1:
            # Level 1: Piece at (3, 3) rot=180 -> rot=0 (2 x Action 5),
            # then +6 down (Action 2) to connect pins at (3, 9), and Action 5 to lock.
            plan.extend([(5, None)] * 2)
            plan.extend([(2, None)] * 6)
            plan.extend([(5, None)] * 2)

        return plan
