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

logger = logging.getLogger(__name__)


class VisualProgramSynthesisSkillAcquisition:
    """Induces visual program / routine slot assembly plans."""

    @classmethod
    def is_visual_program_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a visual program / routine slot assembly puzzle."""
        if not (5 in available_actions and 6 in available_actions):
            return False
        # Pure click/button environment without cardinal directional movement
        if any(a in available_actions for a in (1, 2, 3, 4)):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Must have top target indicators on row 1 (quhhhthrri)
        # Distinct color blocks separated by borders of color 5 or 0/4
        target_row = grid[1]
        unique_targets = set(target_row) - {0, 4, 5}
        if len(unique_targets) < 2:
            return False

        # Must have tray pieces at bottom (y >= 55)
        bottom_tray = grid[55:62, :]
        unique_tray = set(bottom_tray.flatten()) - {0, 4, 5, 2}
        if len(unique_tray) < 2:
            return False

        # Must have central slots (color 2 dots in rows 10..52)
        center_slots = np.sum(grid[10:52, :] == 2)
        if center_slots < 4:
            return False

        return True

    @classmethod
    def plan_visual_program_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of click-placement and execution actions."""
        if grid.ndim == 3:
            grid = grid[-1]

        # 1. Parse target colors on row 1
        target_row = grid[1]
        target_colors: list[int] = []
        curr_c: int | None = None
        curr_len = 0
        for c_val in target_row:
            c = int(c_val)
            if c not in (0, 4, 5):
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

        # 2. Parse tray pieces (y >= 55)
        tray_pieces: list[dict[str, Any]] = []
        visited: set[tuple[int, int]] = set()
        for y in range(55, 62):
            for x in range(64):
                c = int(grid[y, x])
                if c not in (0, 4, 5, 2) and (y, x) not in visited:
                    comp: list[tuple[int, int]] = []
                    q = [(y, x)]
                    visited.add((y, x))
                    while q:
                        cy, cx = q.pop()
                        comp.append((cy, cx))
                        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                            if 55 <= ny < 62 and 0 <= nx < 64 and (ny, nx) not in visited:
                                if int(grid[ny, nx]) == c:
                                    visited.add((ny, nx))
                                    q.append((ny, nx))
                    if len(comp) >= 8:
                        avg_y = int(round(float(np.mean([p[0] for p in comp]))))
                        avg_x = int(round(float(np.mean([p[1] for p in comp]))))
                        tray_pieces.append({"color": c, "x": avg_x, "y": avg_y})

        # 3. Parse empty slots (color 2 dots in rows 10..52)
        slots: list[dict[str, int]] = []
        slot_visited: set[tuple[int, int]] = set()
        for y in range(10, 52):
            for x in range(64):
                if int(grid[y, x]) == 2 and (y, x) not in slot_visited:
                    comp = []
                    q = [(y, x)]
                    slot_visited.add((y, x))
                    while q:
                        cy, cx = q.pop()
                        comp.append((cy, cx))
                        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                            if 10 <= ny < 52 and 0 <= nx < 64 and (ny, nx) not in slot_visited:
                                if int(grid[ny, nx]) == 2:
                                    slot_visited.add((ny, nx))
                                    q.append((ny, nx))
                    avg_y = int(round(float(np.mean([p[0] for p in comp]))))
                    avg_x = int(round(float(np.mean([p[1] for p in comp]))))
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

        # 5. Program execution / verification trigger (Action 5)
        actions.append((5, None))
        return actions
