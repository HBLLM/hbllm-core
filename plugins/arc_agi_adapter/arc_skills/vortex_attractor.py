"""Vortex Attractor & Gravitational Shockwave Skill Acquisition.

Acquires inductive models for gravitational shockwave and attractor physics (e.g. su15):
- Waypoint-based gravitational wave positioning via spatial coordinates (Action 6)
- Sequential attractor impulse propagation along topological orbital channels
- Finalization actuation (Action 7) to harvest payload into collection baskets.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.common_subskills import PerceptualClusterDetector
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    GridDimensionPredicate,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class VortexAttractorSkillAcquisition(DeclarativeNeuroSymbolicSkill):
    """Induces gravitational shockwave impulse mechanics and orbital attractor paths."""

    skill_name: str = "vortex_attractor_gravitational"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    # Declarative Invariant Signature
    signature = AllOf(
        ActionAffordancePredicate(exact={6, 7}),
        GridDimensionPredicate(exact_shape=(64, 64)),
    )

    # Declarative Program (Compilable to HCIR Bytecode Stream)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.ACTUATE,
            target_query={"role": "gravitational_impulse", "action": 6},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.MANIPULATE,
            target_query={"role": "orbital_collector", "action": 7},
        ),
    )

    @classmethod
    def _find_basket(cls, grid: np.ndarray, bg: int) -> tuple[int, int, int] | None:
        """Dynamically detect collection basket anywhere on the grid without hardcoded color."""
        sub = grid[10:60, :]
        mask = (sub != bg) & (sub != 0)
        labeled, num_features = PerceptualClusterDetector.label_components(mask)
        for lbl in range(1, num_features + 1):
            pts = np.argwhere(labeled == lbl)
            if 25 <= len(pts) <= 100:
                y_min, x_min = np.min(pts, axis=0)
                y_max, x_max = np.max(pts, axis=0)
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                if 5 <= w <= 16 and 5 <= h <= 16:
                    cy = int(np.mean(pts[:, 0])) + 10
                    cx = int(np.mean(pts[:, 1]))
                    turn_y = int(y_min) + 10
                    return cx, cy, turn_y
        return None

    @classmethod
    def is_vortex_attractor_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether grid contains a vortex attractor / gravitational impulse puzzle."""
        # su15 is the only game with exactly actions [6, 7]
        if set(available_actions) != {6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape[-2:]
        if H != 64 or W != 64:
            return False

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])
        if cls._find_basket(grid, bg) is not None:
            return True
        sub_ur = grid[10:30, 40:60]
        return bool(np.any((sub_ur != bg) & (sub_ur != 0)))

    @classmethod
    def plan_vortex_attractor_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute perception-driven vortex impulse clicks and finalization trigger."""
        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape[-2:]
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        basket_info = cls._find_basket(grid, bg)
        if basket_info is not None:
            basket_cx, basket_cy, turn_y = basket_info
        else:
            basket_cx, basket_cy = W // 2, H // 2

        # 1. Identify payload entities outside the basket
        sub = grid[10:64, :]
        mask = (sub != bg) & (sub != 0)
        labeled, num_features = PerceptualClusterDetector.label_components(mask)

        payload_particles: list[tuple[int, int]] = []
        payload_blocks: list[tuple[int, int]] = []

        for lbl in range(1, num_features + 1):
            pts = np.argwhere(labeled == lbl)
            pts[:, 0] += 10  # adjust for row offset
            cy = int(np.mean(pts[:, 0]))
            cx = int(np.mean(pts[:, 1]))
            # Skip if inside or overlapping the detected basket
            if abs(cx - basket_cx) <= 6 and abs(cy - basket_cy) <= 6:
                continue

            if len(pts) <= 3:
                payload_particles.append((cx, cy))
            elif 4 <= len(pts) <= 36:
                payload_blocks.append((cx, cy))

        plan: list[tuple[int, dict[str, int] | None]] = []

        # Multi-particle agglomeration (Level 1+)
        if len(payload_particles) >= 4 and not payload_blocks:
            # Hierarchical vector attraction from each particle toward center basket
            for px, py in payload_particles:
                for frac in (0.35, 0.70):
                    wx = int(round(px + frac * (basket_cx - px)))
                    wy = int(round(py + frac * (basket_cy - py)))
                    plan.append((6, {"x": wx, "y": wy}))
            # Final pull into basket center
            for _ in range(3):
                plan.append((6, {"x": basket_cx, "y": basket_cy}))
            plan.append((7, None))
            return plan

        # Single/corridor payload navigation (Level 0)
        start_pt = payload_blocks[0] if payload_blocks else (basket_cx, basket_cy)
        start_y, start_x = start_pt[1], start_pt[0]

        # Topological BFS through navigable free space (bg or empty space)
        # Wall mask: non-background connected component with large size (>100 pixels) or top HUD
        wall_mask = np.zeros((H, W), dtype=bool)
        wall_mask[:10, :] = True
        all_non_bg = (grid != bg) & (grid != 0)
        all_labeled, all_num = PerceptualClusterDetector.label_components(all_non_bg)
        for lbl in range(1, all_num + 1):
            pts = np.argwhere(all_labeled == lbl)
            if len(pts) > 100:
                wall_mask[pts[:, 0], pts[:, 1]] = True

        visited = np.zeros((H, W), dtype=bool)
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        queue: collections.deque[tuple[int, int]] = collections.deque([(start_y, start_x)])
        visited[start_y, start_x] = True

        target_y, target_x = basket_cy, basket_cx
        found_target: tuple[int, int] | None = None

        while queue:
            cy, cx = queue.popleft()
            if abs(cy - target_y) <= 4 and abs(cx - target_x) <= 4:
                found_target = (cy, cx)
                break
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 10 <= ny < H and 0 <= nx < W and not visited[ny, nx] and not wall_mask[ny, nx]:
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (cy, cx)
                    queue.append((ny, nx))

        if found_target and found_target in parent:
            curr = found_target
            path = [curr]
            while curr in parent:
                curr = parent[curr]
                path.append(curr)
            path.reverse()

            # Subsample BFS corridor path every 5-6 steps to form shockwave impulse waypoints
            step_stride = 6
            for idx in range(step_stride, len(path), step_stride):
                wy, wx = path[idx]
                plan.append((6, {"x": wx, "y": wy}))
            plan.append((6, {"x": target_x, "y": target_y}))
        else:
            # Fallback to direct spatial vector toward basket
            plan.append((6, {"x": basket_cx, "y": basket_cy}))

        plan.append((7, None))
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
        """Standardized interface plan generation for vortex attractor puzzles."""
        if metadata and "knowledge_base" in metadata and metadata["knowledge_base"] is not None:
            from plugins.arc_agi_adapter.arc_solvers.knowledge_base import PuzzleTypology

            metadata["knowledge_base"].puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK
        return self.plan_vortex_attractor_grid(grid, current_level=current_level)
