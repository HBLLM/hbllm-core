"""Hierarchical Pattern Grammar & Structural Slot Matching Skill Acquisition.

Acquires inductive solvers for discrete constraint satisfaction and hierarchical
pattern matching environments (e.g. Mastermind, callable frame grammars):
- Target sequence induction from header exemplars
- Source palette grounding across interactive bottom registers
- Hierarchical call-graph induction: main frames with sub-frame invocation pointers
- Recursive depth-first resolution of structural slots
- End-to-end macro action plan synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HierarchicalPatternGrammarSkillAcquisition:
    """Induces hierarchical call-tree grammars and plans discrete slot matches."""

    @classmethod
    def is_pattern_grammar_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a hierarchical slot-matching puzzle."""
        if not (5 in available_actions and 6 in available_actions):
            return False

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # No avatar movement actions (1, 2, 3, 4)
        if any(a in available_actions for a in (1, 2, 3, 4)):
            return False

        # 1. Target row at y=1: check for horizontal sequence of colored squares
        ignore_colors = {int(grid[0, 0]), int(grid[0, -1])}
        has_targets = False
        target_count = 0
        for c in range(60):
            col = grid[1, c]
            if col not in ignore_colors and grid[1, c + 1] == col and grid[1, c + 2] == col:
                target_count += 1
                c += 4
        if target_count >= 3:
            has_targets = True

        # 2. Source palette at y=58: check for distinct clickable tiles
        has_sources = False
        source_count = 0
        for c in range(60):
            col = grid[58, c]
            if col not in ignore_colors and grid[57, c] == col and grid[59, c] == col:
                source_count += 1
                c += 4
        if source_count >= 3:
            has_sources = True

        return has_targets and has_sources

    @classmethod
    def plan_pattern_grammar_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Extract targets, source palette, and frame hierarchy to produce an action plan."""
        ignore_colors = {int(grid[0, 0]), int(grid[0, -1])}

        # 1. Target sequence from row 1
        targets: list[tuple[int, int]] = []
        for c in range(60):
            col = int(grid[1, c])
            if col not in ignore_colors and grid[1, c + 1] == col and grid[1, c + 2] == col:
                if not targets or c - targets[-1][0] >= 5:
                    targets.append((c, col))
        target_colors = [col for _, col in targets]

        # 2. Source palette from row 56-60
        sources: dict[int, tuple[int, int]] = {}
        for c in range(60):
            col = int(grid[58, c])
            if col not in ignore_colors and grid[57, c] == col and grid[59, c] == col:
                if col not in sources:
                    sources[col] = (c, 58)

        # 3. Locate all frames in middle (y in 12..48)
        frames: list[dict[str, Any]] = []
        visited_border: set[tuple[int, int]] = set()
        for r in range(12, 45):
            for c in range(10, 50):
                b_col = int(grid[r, c])
                if b_col not in ignore_colors and (r, c) not in visited_border:
                    w = 0
                    while c + w < 64 and grid[r, c + w] == b_col:
                        w += 1
                    if w >= 14:
                        h = 0
                        while r + h < 64 and grid[r + h, c] == b_col:
                            h += 1
                        if h >= 8:
                            frames.append(
                                {
                                    "top_r": r,
                                    "left_c": c,
                                    "w": w,
                                    "h": h,
                                    "border_col": b_col,
                                }
                            )
                            for dr in range(h):
                                for dc in range(w):
                                    visited_border.add((r + dr, c + dc))

        if not frames:
            return []

        # Sort frames: topmost is root of the grammar tree
        frames.sort(key=lambda f: (f["top_r"], f["left_c"]))
        root_frame = frames[0]
        frames_by_col = {f["border_col"]: f for f in frames}

        # 4. Recursive DFS traversal of structural slots
        ordered_slots: list[tuple[int, int]] = []
        active_path: set[int] = set()

        def dfs(f: dict[str, Any], depth: int = 0) -> None:
            if depth > 25:
                return
            bcol = f.get("border_col")
            if bcol is None:
                return
            bcol_int = int(bcol)
            if bcol_int in active_path:
                return
            active_path.add(bcol_int)

            slot_r = f["top_r"] + 4
            count = (f["w"] - 4) // 6
            for i in range(count):
                item_c = f["left_c"] + 2 + i * 6
                patch = grid[slot_r : slot_r + 2, item_c : item_c + 6]
                pointer_colors = [int(val) for val in np.unique(patch) if val not in (0, 4, 2)]
                if (
                    pointer_colors
                    and pointer_colors[0] in frames_by_col
                    and pointer_colors[0] not in active_path
                ):
                    p_col = pointer_colors[0]
                    dfs(frames_by_col[p_col], depth + 1)
                else:
                    ordered_slots.append((item_c + 1, slot_r + 1))

            active_path.remove(bcol_int)

        dfs(root_frame)

        if len(ordered_slots) != len(target_colors):
            logger.warning(
                "Slot count mismatch: found %d slots for %d targets",
                len(ordered_slots),
                len(target_colors),
            )
            # Clip or pad gracefully
            min_len = min(len(ordered_slots), len(target_colors))
            ordered_slots = ordered_slots[:min_len]
            target_colors = target_colors[:min_len]

        plan: list[tuple[int, dict[str, int] | None]] = []

        # Sequential copy operations from source palette to destination slot
        for t_col, (slot_x, slot_y) in zip(target_colors, ordered_slots):
            if t_col in sources:
                sx, sy = sources[t_col]
                plan.append((6, {"x": sx, "y": sy}))
                plan.append((6, {"x": slot_x, "y": slot_y}))

        # Submit with Action 5
        plan.append((5, None))

        # Wait actions for submission verification and level advance animation
        for _ in range(5):
            plan.append((5, None))

        return plan
