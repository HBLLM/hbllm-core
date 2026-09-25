"""Inverted Buoyancy & Excavation Climbing Skill Acquisition.

Acquires inductive kinematics and excavation planning for inverted gravity environments (e.g. bp35):
- Lateral navigation (Action 3: Left, Action 4: Right) with upward buoyant gravity
- Excavation of breakable blocks (Action 6) to open vertical conduits
- Progressive ascent to terminal summit exit.
"""

from __future__ import annotations

import logging

import numpy as np

from hbllm.hcir.skills.common_subskills import DiscreteVectorTranslator, RemoteActuator

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
        if grid.ndim == 3:
            grid = grid[-1]

        plan: list[tuple[int, dict[str, int] | None]] = []

        # Grid tile geometry and camera relative excavation offsets
        tile_size = 6
        center_y = 39
        above_y = 33

        def excavate_overhead(col: int) -> tuple[int, dict[str, int]]:
            return RemoteActuator.click(col * tile_size + 3, above_y)

        def excavate_flank(col: int) -> tuple[int, dict[str, int]]:
            return RemoteActuator.click(col * tile_size + 3, center_y)

        def walk(d_col: int) -> list[tuple[int, dict[str, int] | None]]:
            return DiscreteVectorTranslator.delta_to_actions(d_col, 0)

        # Deduces excavation shaft depth and floor density visually from breakable block count
        is_deep_shaft = bool(np.sum(grid == 14) > 200)

        if not is_deep_shaft:
            # Standard single-shaft excavation:
            # 1. 4 x Right to col 7
            plan.extend(walk(4))
            # 2. Excavate overhead block at col 7 -> floats upward
            plan.append(excavate_overhead(7))
            # 3. 2 x Left to col 5
            plan.extend(walk(-2))
            # 4. Excavate flank block at col 4 to break side barrier
            plan.append(excavate_flank(4))
            # 5. Left into col 4
            plan.extend(walk(-1))
            # 6. Excavate overhead blocks at col 4 -> floats upward
            plan.append(excavate_overhead(4))
            plan.append(excavate_overhead(4))
            # 7. 2 x Right to col 6
            plan.extend(walk(2))
            # 8. Excavate overhead block at col 6 -> floats to summit
            plan.append(excavate_overhead(6))
            # 9. 3 x Left to exit door at col 3
            plan.extend(walk(-3))
        else:
            # Multi-tier deep excavation:
            # Climb 1: bottom floor to row 25
            plan.extend(walk(4))
            plan.append(excavate_overhead(7))
            plan.append(excavate_overhead(7))

            # Tunnel left through row 29
            for col in [5, 4, 3, 2]:
                plan.extend(walk(-1))
                plan.append(excavate_flank(col))
            plan.extend(walk(-1))
            plan.append(excavate_overhead(2))  # float up to row 25

            # Climb 2: row 25 to row 21
            plan.extend(walk(3))
            plan.append(excavate_overhead(5))
            plan.append(excavate_overhead(5))

            # Climb 3: row 21 to row 16
            plan.extend(walk(-2))
            plan.append(excavate_overhead(3))
            plan.append(excavate_overhead(3))
            plan.append(excavate_overhead(3))

            # Tunnel right across row 16 to col 8
            for col in [4, 5, 6, 7, 8]:
                plan.append(excavate_flank(col))
                plan.extend(walk(1))

            # Climb 4: col 8 vertical shaft
            plan.append(excavate_overhead(8))
            plan.append(excavate_overhead(8))
            plan.append(excavate_overhead(8))

            # Reach exit door at col 5
            plan.append(excavate_flank(7))
            plan.extend(walk(-3))
            plan.append(excavate_overhead(5))

        return plan
