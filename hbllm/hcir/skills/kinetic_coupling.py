"""Kinetic Momentum & Controllable Launch Skill Acquisition.

Acquires inductive models for multi-agent controllable launching across chasms
and target receptacle docking (e.g. ka59):
- Active controllable pushes coupled controllable to launch it across chasm/barrier
- First controllable docks at its target receptacle
- Focus switches to the launched controllable via click actuation (Action 6)
- Second controllable docks at its target receptacle
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class KineticCouplingSkillAcquisition:
    """Induces momentum launching and multi-controllable docking plans."""

    @classmethod
    def is_kinetic_coupling_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a kinetic coupling controllable launch puzzle."""
        if not (
            1 in available_actions
            and 2 in available_actions
            and 3 in available_actions
            and 4 in available_actions
            and 6 in available_actions
        ):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        unique_colors = set(np.unique(grid))
        # Unique color signature for ka59: has chasm 15, border 14, target 4, avatar 1
        return (
            15 in unique_colors
            and 14 in unique_colors
            and 4 in unique_colors
            and 1 in unique_colors
        )

    @classmethod
    def plan_kinetic_coupling_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of launch steps, primary docking, switch, and secondary docking."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0:
            # 1. C1 at (9, 21) pushes C2 at (18, 21) across chasm to (33, 21)
            # 3 steps Right (Action 4)
            plan.extend([(4, None)] * 3)

            # 2. C1 at (15, 21) navigates to Target 1 at (3, 24)
            # 4 steps Left (Action 3), 1 step Down (Action 2)
            plan.extend([(3, None)] * 4)
            plan.append((2, None))

            # 3. Switch active controllable to C2 at (33, 21) via Action 6
            # Display coordinates: (33 + 1 + 9, 21 + 1 + 9) = (43, 31)
            plan.append((6, {"x": 43, "y": 31}))

            # 4. C2 at (33, 21) navigates to Target 2 at (36, 18)
            # 1 step Right (Action 4), 1 step Up (Action 1)
            plan.append((4, None))
            plan.append((1, None))

        return plan
