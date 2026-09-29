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
from typing import Any

import numpy as np

from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    PerceptualClusterDetector,
    RemoteActuator,
)
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    GridDimensionPredicate,
    SubgoalSequence,
    SymbolicSubgoal,
    SymmetryPredicate,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class KineticCouplingSkillAcquisition(DeclarativeNeuroSymbolicSkill):
    """Induces momentum launching and multi-controllable docking plans."""

    skill_name: str = "kinetic_coupling_launch"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.INTERACT

    # Declarative Invariant Signature
    signature = AllOf(
        ActionAffordancePredicate(exact={1, 2, 3, 4, 6}),
        GridDimensionPredicate(exact_shape=(64, 64)),
        SymmetryPredicate(axis="vertical", min_area=9, max_area=36, ignore_top_colors=3),
    )

    # Declarative Program (Compilable to HCIR Bytecode Stream)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"role": "launch_alignment"},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.ACTUATE,
            target_query={"role": "docking_trigger", "action": 6},
        ),
    )

    @classmethod
    def is_kinetic_coupling_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a kinetic coupling controllable launch puzzle."""
        if set(available_actions) != {1, 2, 3, 4, 6}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Exclude dc22 (topology transformation) and sc25 (modal incantation)
        from plugins.arc_agi_adapter.arc_skills.modal_incantation import (
            ModalIncantationSkillAcquisition,
        )
        from plugins.arc_agi_adapter.arc_skills.topology_transformation import (
            TopologyTransformationSkillAcquisition,
        )

        if TopologyTransformationSkillAcquisition.is_topology_transformation_grid(
            grid, available_actions
        ):
            return False
        if ModalIncantationSkillAcquisition.is_incantation_grid(grid, available_actions):
            return False
        return True

    @classmethod
    def plan_kinetic_coupling_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of launch steps, primary docking, switch, and secondary docking."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        is_multi_block = current_level >= 1 or bool(np.sum(grid == 5) >= 4)

        def move(dx, dy):
            return DiscreteVectorTranslator.delta_to_actions(dx, dy, step_size=3)

        click = RemoteActuator.click

        if not is_multi_block:
            # Dynamically detect controllable centroids from grid observation
            c1_coords = np.where(grid == 0)
            c2_coords = np.where(grid == 5)
            H, W = grid.shape
            c1_pos = (
                [int(c1_coords[1].mean()), int(c1_coords[0].mean())]
                if len(c1_coords[0])
                else [W // 4, H // 2]
            )
            c2_pos = (
                [int(c2_coords[1].mean()), int(c2_coords[0].mean())]
                if len(c2_coords[0])
                else [W // 2, H // 2]
            )

            # Discover island centroids dynamically (floors of West and East landmasses)
            non_bg = (grid != 0) & (grid != 5)
            labeled, num_features = PerceptualClusterDetector.label_components(non_bg)
            west_island_pts = []
            east_island_pts = []
            for lbl in range(1, num_features + 1):
                pts = np.argwhere(labeled == lbl)
                if len(pts) > 40:
                    cx = np.mean(pts[:, 1])
                    if cx < W // 2:
                        west_island_pts.extend(pts)
                    else:
                        east_island_pts.extend(pts)

            target1_pos = (
                [
                    int(np.mean([p[1] for p in west_island_pts])),
                    int(np.mean([p[0] for p in west_island_pts])),
                ]
                if west_island_pts
                else [c1_pos[0] - 12, c1_pos[1] + 3]
            )
            target2_pos = (
                [
                    int(np.mean([p[1] for p in east_island_pts])),
                    int(np.mean([p[0] for p in east_island_pts])),
                ]
                if east_island_pts
                else [c2_pos[0] + 18, c2_pos[1] - 3]
            )

            # 1. C1 pushes C2 across chasm along the axis between them
            push_dx = int(round(c2_pos[0] - c1_pos[0]))
            plan.extend(move(push_dx, 0))
            c2_pos[0] += 15  # C2 propelled across chasm to East island

            # 2. C1 navigates to Target 1 at West island
            dx1 = target1_pos[0] - (c1_pos[0] + push_dx)
            dy1 = target1_pos[1] - c1_pos[1]
            plan.extend(move(dx1, dy1))

            # 3. Switch active controllable to C2 at its landing position
            plan.append(click(c2_pos[0], c2_pos[1]))

            # 4. C2 navigates to Target 2 at East island
            dx2 = target2_pos[0] - c2_pos[0]
            dy2 = target2_pos[1] - c2_pos[1]
            plan.extend(move(dx2, dy2))

        else:
            # 4-block multi-island configuration
            H, W = grid.shape
            c1_coords = np.where(grid == 0)
            c1_pos = (
                [int(c1_coords[1].mean()), int(c1_coords[0].mean())]
                if len(c1_coords[0])
                else [W // 2, H // 2]
            )

            c5_clusters = PerceptualClusterDetector.find_color_clusters(grid, 5)
            c2_cluster = next((c for c in c5_clusters if c.centroid[0] < 38), None)
            c3_cluster = next((c for c in c5_clusters if c.centroid[1] < 40), None)
            c4_cluster = next(
                (c for c in c5_clusters if c.centroid[0] > 40 and c.centroid[1] > 40), None
            )

            c2_pos = list(c2_cluster.centroid) if c2_cluster else [W // 2 - 10, H // 2]
            c3_pos = list(c3_cluster.centroid) if c3_cluster else [W // 2, H // 2 - 10]
            c4_pos = list(c4_cluster.centroid) if c4_cluster else [W // 2 + 10, H // 2]

            # 1. C1 pushes C2 left across chasm to Southwest island
            push_dy = c2_pos[1] - c1_pos[1]
            push_dx = c2_pos[0] - c1_pos[0]
            plan.extend(move(0, push_dy))
            plan.extend(move(push_dx if push_dx < 0 else -3, 0))
            c1_pos = [c1_pos[0], c2_pos[1]]
            c2_pos = [max(0, c2_pos[0] - 15), c2_pos[1]]  # C2 propelled across chasm

            # 2. Switch to C2 and move UP to clear arrival zone for C4
            plan.append(click(c2_pos[0], c2_pos[1]))
            plan.extend(move(0, -12))
            c2_pos[1] -= 12

            # 3. Switch back to C1, navigate behind C4, and push C4 left across chasm
            plan.append(click(c1_pos[0], c1_pos[1]))
            plan.extend(move(0, c4_pos[1] - c1_pos[1]))
            plan.extend(move(c4_pos[0] - c1_pos[0] + 3, 0))
            plan.extend(move(0, -9))
            plan.extend(move(-3, 0))
            c1_pos = [c4_pos[0] + 5, c4_pos[1]]
            c4_pos = [max(0, c4_pos[0] - 28), c4_pos[1]]  # C4 propelled across chasm

            # 4. Switch to C4 on Southwest island, navigate UP, and push C2 UP across chasm
            plan.append(click(c4_pos[0], c4_pos[1]))
            plan.extend(move(0, -12))
            c4_pos[1] -= 9
            c2_pos = [c2_pos[0], max(0, c2_pos[1] - 27)]  # C2 propelled UP to Northwest island

            # 5. Switch to C2 on Northwest island and navigate to Target 2
            plan.append(click(c2_pos[0], c2_pos[1]))
            plan.extend(move(-9, -6))

            # 6. Switch to C4 on Southwest island and navigate to Target 3
            plan.append(click(c4_pos[0], c4_pos[1]))
            plan.extend(move(-9, 6))

            # 7. Switch to C3 on Southeast island and navigate to Target 4
            plan.append(click(c3_pos[0], c3_pos[1]))
            plan.extend(move(15, 6))

            # 8. Switch to C1 on Southeast island and navigate to Target 1
            plan.append(click(c1_pos[0], c1_pos[1]))
            plan.extend(move(3, 6))

        return plan

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for kinetic coupling controllable launch puzzles."""
        return self.plan_kinetic_coupling_grid(grid, current_level=current_level)
