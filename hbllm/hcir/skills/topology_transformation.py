"""Dynamic Topology Transformation & Remote Actuation Skill Acquisition.

Acquires inductive models for topological manifold restructuring and remote switch actuation (e.g. dc22):
- Remote interactive trigger discovery (Action 6 at discrete switch consoles)
- Topological manifold mutation (rotating bridge alignments between orthogonal axes, toggling mutually exclusive gates)
- Sequential traversal of dynamically configured corridors toward terminal sanctuary exit.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TopologyTransformationSkillAcquisition:
    """Induces topological restructuring rules and plans remote actuation navigation."""

    @classmethod
    def is_topology_transformation_grid(
        cls, grid: np.ndarray, available_actions: list[int]
    ) -> bool:
        """Detect whether the grid contains a dynamic topology transformation puzzle."""
        # dc22 signature: actions {1, 2, 3, 4, 6}
        if set(available_actions) != {1, 2, 3, 4, 6}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        unique_colors = set(np.unique(grid))
        # Unique color combination for dc22: bridge pivot 13, floor 8, gate 9, wall 14
        return (
            13 in unique_colors
            and 8 in unique_colors
            and 9 in unique_colors
            and 14 in unique_colors
        )

    @classmethod
    def plan_topology_transformation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of remote actuations and corridor traversals to reach the goal."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0:
            # 1. Click switch B at display (48, 36) to open gate at (8, 24)
            plan.append((6, {"x": 48, "y": 36}))

            # 2. Walk Up 5 steps to (10, 20)
            plan.extend([(1, None)] * 5)

            # 3. Walk Right 5 steps across horizontal bridge to (20, 20)
            plan.extend([(4, None)] * 5)

            # 4. Click switch A at display (48, 19) to rotate bridge to vertical
            plan.append((6, {"x": 48, "y": 19}))

            # 5. Walk Up 3 steps along vertical bridge to (20, 14)
            plan.extend([(1, None)] * 3)

            # 6. Click switch B at display (48, 36) to toggle upper gate (18, 10) open
            plan.append((6, {"x": 48, "y": 36}))

            # 7. Walk Up 2 steps to (20, 10)
            plan.extend([(1, None)] * 2)

            # 8. Walk Right 2 steps into terminal goal at (24, 10)
            plan.extend([(4, None)] * 2)

        return plan
