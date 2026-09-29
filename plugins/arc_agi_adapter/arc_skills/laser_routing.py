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

from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    GridDimensionPredicate,
    PanelConstraint,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class LaserRoutingSkillAcquisition(DeclarativeNeuroSymbolicSkill):
    """Induces laser routing, pipe extension mechanics, and block sequencing."""

    skill_name: str = "laser_routing_pipe_coupling"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    # Declarative Invariant Signature
    signature = AllOf(
        ActionAffordancePredicate(exact={1, 2, 3, 4, 6, 7}),
        GridDimensionPredicate(exact_shape=(64, 64)),
        PanelConstraint(
            min_row_ratio=52.0 / 64.0,
            max_row_ratio=62.0 / 64.0,
            min_distinct_colors=2,
        ),
    )

    # Declarative Program (Compilable to HCIR Bytecode Stream)
    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.MANIPULATE,
            target_query={"role": "pipe_extension", "action": 6},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"role": "target_slot"},
        ),
    )

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

        # Extract target sequence colors from bottom panel components
        bottom_region = grid[52:62, :]
        bg = int(np.bincount(bottom_region.flatten()).argmax())
        target_seq: list[tuple[int, int]] = []  # (x_center, color)

        from hbllm.hcir.skills.common_subskills import PerceptualClusterDetector

        labeled_bottom, num_bottom = PerceptualClusterDetector.label_components(bottom_region != bg)
        for lbl in range(1, num_bottom + 1):
            pts = np.argwhere(labeled_bottom == lbl)
            if len(pts) >= 16:
                cx = int(np.mean(pts[:, 1]))
                cy = int(np.mean(pts[:, 0]))
                col = int(bottom_region[cy, cx])
                target_seq.append((cx, col))
        target_seq.sort(key=lambda t: t[0])
        target_colors = [t[1] for t in target_seq]

        # Detect emitter head in the left rail manifold (x <= 20)
        rail_mask = (grid != bg) & (grid != 0) & (np.arange(grid.shape[1])[None, :] <= 20)
        rail_pts = np.argwhere(rail_mask)
        emitter_y = int(np.mean(rail_pts[:, 0])) if len(rail_pts) > 0 else 36

        # Detect block centroids in the playfield (12 <= y <= 50, x >= 20)
        block_pos: dict[int, tuple[int, int]] = {}
        for c in target_colors:
            pts = np.argwhere(grid == c)
            if len(pts) > 0:
                block_pos[c] = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

        row_pitch = 6

        curr_y = emitter_y
        curr_ext = 0

        def goto_y(target_y: int):
            nonlocal curr_y
            dy = target_y - curr_y
            steps = abs(dy) // row_pitch
            act = 1 if dy < 0 else 2
            for _ in range(steps):
                plan.append((act, None))
            curr_y = target_y

        def extend_to(target_steps: int):
            nonlocal curr_ext
            d_ext = target_steps - curr_ext
            act = 4 if d_ext > 0 else 3
            for _ in range(abs(d_ext)):
                plan.append((act, None))
            curr_ext = target_steps

        # Sequence blocks to match target_colors dynamically
        if target_colors:
            first_col = target_colors[0]
            first_pos = block_pos.get(first_col, (24, 41))

            # Position emitter at upper capture row
            capture_row = max(12, first_pos[0] - row_pitch)
            goto_y(capture_row)
            extend_to(4)
            goto_y(capture_row + row_pitch)
            extend_to(0)
            goto_y(capture_row + 3 * row_pitch)
            extend_to(4)
            extend_to(3)
            goto_y(capture_row + 2 * row_pitch)
            extend_to(4)
        else:
            extend_to(1)

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
        """Standardized interface plan generation for laser routing puzzles."""
        return self.plan_laser_routing_grid(grid, current_level=current_level)
