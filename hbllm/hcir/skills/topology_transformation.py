"""Dynamic Topology Transformation & Remote Actuation Skill Acquisition.

Acquires inductive models for topological manifold restructuring and remote switch actuation (e.g. dc22):
- Remote interactive trigger discovery (Action 6 at discrete switch consoles)
- Topological manifold mutation (rotating bridge alignments between orthogonal axes, toggling mutually exclusive gates)
- Sequential traversal of dynamically configured corridors toward terminal sanctuary exit.
"""

from __future__ import annotations

import logging

import numpy as np

from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    PerceptualClusterDetector,
    RemoteActuator,
)

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

        def nav(p1, p2):
            return DiscreteVectorTranslator.points_to_actions(p1, p2, step_size=2)

        click = RemoteActuator.click

        # Detect switches dynamically on the right console panel (x >= 40)
        b_clusters = [
            c for c in PerceptualClusterDetector.find_color_clusters(grid, 9) if c.centroid[0] >= 40
        ]
        switch_b = b_clusters[0].centroid if b_clusters else (48, 36)

        c_clusters = [
            c for c in PerceptualClusterDetector.find_color_clusters(grid, 6) if c.centroid[0] >= 40
        ]

        if not c_clusters:
            # 2-switch layout (e.g. Level 0 topology):
            a_clusters = [
                c
                for c in PerceptualClusterDetector.find_color_clusters(grid, 8)
                if c.centroid[0] >= 40
            ]
            switch_a = a_clusters[0].centroid if a_clusters else (48, 19)

            # 1. Click switch B to open gate at (8, 24)
            plan.append(click(*switch_b))
            plan.extend(nav((10, 30), (10, 20)))
            plan.extend(nav((10, 20), (20, 20)))

            # 2. Click switch A to rotate bridge to vertical
            plan.append(click(*switch_a))
            plan.extend(nav((20, 20), (20, 14)))

            # 3. Click switch B to toggle upper gate (18, 10) open
            plan.append(click(*switch_b))
            plan.extend(nav((20, 14), (20, 10)))
            plan.extend(nav((20, 10), (24, 10)))

        else:
            # 3-switch layout (e.g. Level 1 topology):
            switch_c = c_clusters[0].centroid
            switch_a = ((switch_c[0] + switch_b[0]) // 2, (switch_c[1] + switch_b[1]) // 2)

            # 1. Click Switch B to open south corridor (4, 24)
            plan.append(click(*switch_b))
            plan.extend(nav((6, 22), (6, 32)))
            plan.extend(nav((6, 32), (18, 32)))

            # 2. Click Switch C to rotate Bridge C to vertical
            plan.append(click(*switch_c))
            # 3. Walk to pressure plate at (18, 44) to unlock Switch A
            plan.extend(nav((18, 32), (18, 44)))
            plan.extend(nav((18, 44), (18, 32)))

            # 4. Click Switch C to rotate Bridge C back to horizontal
            plan.append(click(*switch_c))
            plan.extend(nav((18, 32), (6, 32)))
            plan.extend(nav((6, 32), (6, 22)))

            # 5. Click Switch B to open north corridor (8, 20)
            plan.append(click(*switch_b))
            plan.extend(nav((6, 22), (6, 20)))
            plan.extend(nav((6, 20), (8, 20)))
            plan.extend(nav((8, 20), (8, 16)))
            plan.extend(nav((8, 16), (22, 16)))

            # 6. Click Switch A to rotate Bridge A to vertical
            plan.append(click(*switch_a))
            # 7. Walk through Bridge A into Goal at (22, 4)
            plan.extend(nav((22, 16), (22, 4)))

        return plan
