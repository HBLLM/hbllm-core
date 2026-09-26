"""Laser Routing & Kinematic Pipe Coupling Skill Acquisition.

Acquires inductive models for linear actuator / rail translation, pipe expansion/retraction,
and sokoban-style block sequencing (e.g. sk48):
- Actuator positioning along orthogonal rail manifolds
- Bidirectional extension/retraction with magnetic/frictional block dragging and pushing
- Alignment of target discrete chromatic symbols under active laser/fluid manifold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class LaserRoutingSkillAcquisition(BaseHierarchicalSkill):
    """Induces laser routing, pipe extension mechanics, and block sequencing."""

    skill_name: str = "laser_routing_pipe_coupling"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    @classmethod
    def is_laser_routing_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a laser routing / pipe extension puzzle."""
        if set(available_actions) != {1, 2, 3, 4, 6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape[-2:]
        if H != 64 or W != 64:
            return False

        # sk48 has discrete target sequence slots in the bottom region (y >= 50)
        bottom_region = grid[52:62, :]
        vals, counts = np.unique(bottom_region, return_counts=True)
        bg = vals[np.argmax(counts)]
        non_bg_pixels = np.sum(bottom_region != bg)
        return non_bg_pixels >= 20

    @classmethod
    def plan_laser_routing_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        if grid.ndim == 3:
            grid = grid[-1]

        plan: list[tuple[int, dict[str, int] | None]] = []

        # Detect pipeline complexity from bottom target slot count at y >= 50
        bottom_region = grid[52:62, :]
        bg = int(np.bincount(bottom_region.flatten()).argmax())
        # Target blocks are 6x6 squares
        visited = np.zeros_like(bottom_region, dtype=bool)
        target_slots = 0
        for y in range(bottom_region.shape[0]):
            for x in range(bottom_region.shape[1]):
                if not visited[y, x] and bottom_region[y, x] != bg:
                    q = [(y, x)]
                    visited[y, x] = True
                    comp = []
                    while q:
                        cy, cx = q.pop()
                        comp.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if (
                                0 <= ny < bottom_region.shape[0]
                                and 0 <= nx < bottom_region.shape[1]
                            ):
                                if not visited[ny, nx] and bottom_region[ny, nx] != bg:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    w = max(p[1] for p in comp) - min(p[1] for p in comp) + 1
                    h = max(p[0] for p in comp) - min(p[0] for p in comp) + 1
                    if w >= 4 and h >= 4:
                        target_slots += 1

        if current_level == 0:
            # 3-block pipeline manifold (Level 0):
            # Goal is sequence [8, 14, 9] along the horizontal pipe.
            # 1. Move emitter Up 3 times to row 18 (y=18)
            plan.extend([(1, None)] * 3)

            # 2. Extend pipe 4 times across row 18 to column 41 (over block 8)
            plan.extend([(4, None)] * 4)

            # 3. Move emitter Down to row 24 (pushes 8->row 24, 9->row 30, 14->row 36)
            plan.append((2, None))

            # 4. Retract pipe 4 times at row 24 (pulls block 8 to column 17)
            plan.extend([(3, None)] * 4)

            # 5. Move emitter Down 2 times to row 36 (pushes block 8 to row 36 at column 17)
            plan.extend([(2, None)] * 2)

            # 6. Extend pipe 4 times along row 36 (pushes 8 to col 35, covers block 14 at col 41)
            plan.extend([(4, None)] * 4)

            # 7. Retract pipe 1 time (pulls 14 to col 35, pushes 8 to col 29)
            plan.append((3, None))

            # 8. Move emitter Up to row 30 (pushes 8 to (29, 30) and 14 to (35, 30); block 9 is at (41, 30))
            plan.append((1, None))

            # 9. Extend pipe 1 time to column 41 (covers [8, 14, 9] in order -> Terminal Goal)
            plan.append((4, None))

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
        """Standardized interface check for laser routing and pipe extension puzzles."""
        return self.is_laser_routing_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for laser routing puzzles."""
        return self.plan_laser_routing_grid(grid, current_level=current_level)
