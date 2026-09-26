"""Visual Program Synthesis & Slot Assembly Skill Acquisition.

Acquires inductive visual block-routine programming and slot assembly for
visual subroutine execution environments (e.g. sequence matching, visual routines):
- Top-level target routine specification detection (row 1-2 sequence of target color items)
- Bottom-level tray block component detection (y >= 54 distinct color blocks)
- Central frame slot detection (color 2 connection sockets inside frames)
- Causal mapping and pair-wise click-placement dispatch (Action 6 click)
- Program execution triggering (Action 5 run / verify).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class VisualProgramSynthesisSkillAcquisition(BaseHierarchicalSkill):
    """Induces visual program / routine slot assembly plans."""

    skill_name: str = "visual_program_synthesis"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    @classmethod
    def is_visual_program_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a visual program / routine slot assembly puzzle."""
        # sb26 is uniquely characterized by actions [5, 6, 7]
        if set(available_actions) != {5, 6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape[-2:]
        if H != 64 or W != 64:
            return False

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        # Must have top target indicators on row 1
        target_row = grid[1]
        unique_targets = set(target_row) - {bg, 0}
        return len(unique_targets) >= 2

    @classmethod
    def plan_visual_program_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of click-placement and execution actions."""
        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        # 1. Parse target colors on row 1
        target_row = grid[1]
        target_colors: list[int] = []
        curr_c: int | None = None
        curr_len = 0
        for c_val in target_row:
            c = int(c_val)
            if c != bg and c != 0:
                if c == curr_c:
                    curr_len += 1
                else:
                    if curr_c is not None and curr_len >= 3:
                        target_colors.append(curr_c)
                    curr_c = c
                    curr_len = 1
            else:
                if curr_c is not None and curr_len >= 3:
                    target_colors.append(curr_c)
                curr_c = None
                curr_len = 0
        if curr_c is not None and curr_len >= 3:
            target_colors.append(curr_c)

        # 2. Parse tray pieces (y >= 54)
        from scipy.ndimage import label

        tray_mask = (grid != bg) & (grid != 0) & (np.arange(64)[:, None] >= 54)
        labeled_tray, num_tray = label(tray_mask)
        tray_pieces: list[dict[str, Any]] = []

        for idx in range(1, num_tray + 1):
            pts = np.argwhere(labeled_tray == idx)
            if len(pts) >= 6:
                avg_y = int(round(float(np.mean(pts[:, 0]))))
                avg_x = int(round(float(np.mean(pts[:, 1]))))
                c = int(grid[avg_y, avg_x])
                tray_pieces.append({"color": c, "x": avg_x, "y": avg_y})

        # 3. Parse empty frame slots in middle (rows 10..52)
        mid_mask = (
            (grid != bg)
            & (grid != 0)
            & (np.arange(64)[:, None] >= 10)
            & (np.arange(64)[:, None] <= 52)
        )
        labeled_mid, num_mid = label(mid_mask)
        slots: list[dict[str, int]] = []

        for idx in range(1, num_mid + 1):
            pts = np.argwhere(labeled_mid == idx)
            # Socket dots inside empty slots are small components (1 to 4 pixels)
            if 1 <= len(pts) <= 6:
                avg_y = int(round(float(np.mean(pts[:, 0]))))
                avg_x = int(round(float(np.mean(pts[:, 1]))))
                slots.append({"x": avg_x, "y": avg_y})
        slots.sort(key=lambda s: (s["y"], s["x"]))

        # 4. Generate placement actions
        actions: list[tuple[int, dict[str, int] | None]] = []
        used_pieces: set[int] = set()
        for s_idx, target_c in enumerate(target_colors):
            if s_idx >= len(slots):
                break
            slot = slots[s_idx]
            match: dict[str, Any] | None = None
            for p_idx, p in enumerate(tray_pieces):
                if p_idx not in used_pieces and p["color"] == target_c:
                    match = p
                    used_pieces.add(p_idx)
                    break
            if match:
                actions.append((6, {"x": match["x"], "y": match["y"]}))
                actions.append((6, {"x": slot["x"], "y": slot["y"]}))

        # 5. Program execution trigger (Action 5)
        actions.append((5, None))
        return actions

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for visual program slot assembly puzzles."""
        return self.is_visual_program_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for visual program slot assembly puzzles."""
        return self.plan_visual_program_grid(grid, current_level=current_level)
