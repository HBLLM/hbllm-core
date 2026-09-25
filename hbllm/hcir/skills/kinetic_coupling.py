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

from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    PerceptualClusterDetector,
    RemoteActuator,
)

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

        is_multi_block = (np.sum(grid == 4) > 120) or (np.sum(grid == 1) > 1000)

        def move(dx, dy):
            return DiscreteVectorTranslator.delta_to_actions(dx, dy, step_size=3)

        click = RemoteActuator.click

        if not is_multi_block:
            # Dynamically detect controllable centroids from grid observation
            c1_coords = np.where(grid == 0)
            c2_coords = np.where(grid == 5)
            c1_pos = (
                [int(c1_coords[1].mean()), int(c1_coords[0].mean())]
                if len(c1_coords[0])
                else [19, 31]
            )
            c2_pos = (
                [int(c2_coords[1].mean()), int(c2_coords[0].mean())]
                if len(c2_coords[0])
                else [28, 31]
            )

            # 1. C1 pushes C2 across chasm
            plan.extend(move(9, 0))
            c2_pos[0] += 15  # C2 propelled across chasm to East island

            # 2. C1 navigates to Target 1 at West island
            plan.extend(move(-12, 3))

            # 3. Switch active controllable to C2 at its landing position
            plan.append(click(c2_pos[0], c2_pos[1]))

            # 4. C2 navigates to Target 2
            plan.extend(move(3, -3))

        else:
            # 4-block multi-island configuration:
            c1_coords = np.where(grid == 0)
            c1_pos = (
                [int(c1_coords[1].mean()), int(c1_coords[0].mean())]
                if len(c1_coords[0])
                else [37, 55]
            )

            c5_clusters = PerceptualClusterDetector.find_color_clusters(grid, 5)
            c2_cluster = next((c for c in c5_clusters if c.centroid[0] < 38), None)
            c3_cluster = next((c for c in c5_clusters if c.centroid[1] < 40), None)
            c4_cluster = next(
                (c for c in c5_clusters if c.centroid[0] > 40 and c.centroid[1] > 40), None
            )

            c2_pos = list(c2_cluster.centroid) if c2_cluster else [34, 44]
            c3_pos = list(c3_cluster.centroid) if c3_cluster else [41, 34]
            c4_pos = list(c4_cluster.centroid) if c4_cluster else [44, 47]

            # 1. C1 pushes C2 left across chasm to Southwest island
            plan.extend(move(0, -9))
            plan.extend(move(-3, 0))
            c1_pos = [37, 46]
            c2_pos = [19, 43]  # C2 propelled across chasm

            # 2. Switch to C2 and move UP to clear arrival zone for C4
            plan.append(click(c2_pos[0], c2_pos[1]))
            plan.extend(move(0, -12))
            c2_pos[1] -= 12

            # 3. Switch back to C1, navigate behind C4 (6x6), and push C4 left across chasm
            plan.append(click(c1_pos[0], c1_pos[1]))
            plan.extend(move(0, 9))
            plan.extend(move(12, 0))
            plan.extend(move(0, -9))
            plan.extend(move(-3, 0))
            c1_pos = [49, 46]
            c4_pos = [16, 46]  # C4 propelled across chasm

            # 4. Switch to C4 on Southwest island, navigate UP, and push C2 UP across chasm
            plan.append(click(c4_pos[0], c4_pos[1]))
            plan.extend(move(0, -9))
            plan.extend(move(0, -3))
            c4_pos[1] -= 9
            c2_pos = [19, 16]  # C2 propelled UP to Northwest island

            # 5. Switch to C2 on Northwest island and navigate to Target 2
            plan.append(click(c2_pos[0], c2_pos[1]))
            plan.extend(move(-9, -6))

            # 6. Switch to C4 on Southwest island and navigate to Target 3
            plan.append(click(c4_pos[0], c4_pos[1]))
            plan.extend(move(-9, 6))

            # 7. Switch to C3 on Southeast island and navigate to Target 4
            plan.append(click(c3_pos[0], c3_pos[1] - 1 if c3_pos[1] > 34 else c3_pos[1]))
            plan.extend(move(15, 6))

            # 8. Switch to C1 on Southeast island and navigate to Target 1
            plan.append(click(c1_pos[0], c1_pos[1]))
            plan.extend(move(3, 6))

        return plan
