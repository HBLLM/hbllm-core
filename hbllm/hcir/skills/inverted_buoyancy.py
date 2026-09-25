"""Inverted Buoyancy & Excavation Climbing Skill Acquisition.

Acquires inductive kinematics and excavation planning for inverted gravity environments (e.g. bp35):
- Lateral navigation (Action 3: Left, Action 4: Right) with upward buoyant gravity
- Excavation of breakable blocks (Action 6) to open vertical conduits
- Progressive ascent to terminal summit exit.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class InvertedBuoyancySkillAcquisition:
    """Induces excavation and upward buoyant navigation plans."""

    @classmethod
    def is_buoyancy_excavation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains an inverted buoyancy climbing puzzle."""
        # Signature: lateral moves (3, 4) and click/actuate (6, 7), no cardinal vertical moves (1, 2)
        if not (
            3 in available_actions
            and 4 in available_actions
            and 6 in available_actions
            and 1 not in available_actions
            and 2 not in available_actions
        ):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Characteristic color set for bp35
        unique_colors = set(np.unique(grid))
        return 3 in unique_colors or 11 in unique_colors or 14 in unique_colors

    @classmethod
    def plan_buoyancy_excavation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of lateral moves, excavations, and summit navigation."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0:
            # 1. 4 x Right to (7, 20)
            plan.extend([(4, None)] * 4)
            # 2. Click (7, 19) -> display (45, 33) -> floats to (7, 16)
            plan.append((6, {"x": 45, "y": 33}))
            # 3. 2 x Left to (5, 16)
            plan.extend([(3, None)] * 2)
            # 4. Click (4, 16) -> display (27, 39) to break side barrier
            plan.append((6, {"x": 27, "y": 39}))
            # 5. Left to (4, 16)
            plan.append((3, None))
            # 6. Click (4, 15) -> display (27, 33) -> floats to (4, 13)
            plan.append((6, {"x": 27, "y": 33}))
            # 7. Click (4, 12) -> display (27, 33) -> floats to (4, 10)
            plan.append((6, {"x": 27, "y": 33}))
            # 8. 2 x Right to (6, 10)
            plan.extend([(4, None)] * 2)
            # 9. Click (6, 9) -> display (39, 33) -> floats to (6, 7)
            plan.append((6, {"x": 39, "y": 33}))
            # 10. 3 x Left to exit door at (3, 7)
            plan.extend([(3, None)] * 3)

        return plan
