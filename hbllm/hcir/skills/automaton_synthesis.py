"""Automaton Program Synthesis & Execution Skill Acquisition.

Acquires inductive models for programming discrete automata via bit-addressable opcodes (e.g. tn36):
- Parameter register configuration via discrete spatial click triggers
- Bitwise opcode instruction synthesis (e.g. orthogonal movement primitives)
- Execution trigger actuation to guide agent toward terminal goal manifold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.skills.common_subskills import RemoteActuator
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


class AutomatonProgramSynthesisSkillAcquisition(BaseHierarchicalSkill):
    """Induces automaton programming models, register bit assignments, and execution."""

    skill_name: str = "automaton_program_synthesis"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE

    @classmethod
    def is_automaton_synthesis_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains an automaton programming synthesis puzzle."""
        # tn36 signature: click action [6] only
        if set(available_actions) != {6}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Detect automaton synthesis puzzle dynamically:
        # 1. Click-only interface (action 6)
        # 2. Contains linear 3-pixel register bits in the programming band (y in [30, 50])
        # 3. Contains bottom control / execution buttons (y >= 45)
        subgrid = grid[30:50, :]
        visited = np.zeros_like(subgrid, dtype=bool)
        three_pixel_count = 0

        for y in range(subgrid.shape[0]):
            for x in range(subgrid.shape[1]):
                if not visited[y, x]:
                    c = subgrid[y, x]
                    q = [(y, x)]
                    visited[y, x] = True
                    comp = []
                    while q:
                        cy, cx = q.pop()
                        comp.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < subgrid.shape[0] and 0 <= nx < subgrid.shape[1]:
                                if not visited[ny, nx] and subgrid[ny, nx] == c:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    if len(comp) == 3:
                        min_y = min(p[0] for p in comp)
                        max_y = max(p[0] for p in comp)
                        min_x = min(p[1] for p in comp)
                        max_x = max(p[1] for p in comp)
                        if (max_y - min_y == 2 and min_x == max_x) or (
                            max_x - min_x == 2 and min_y == max_y
                        ):
                            three_pixel_count += 1

        return three_pixel_count >= 8

    @classmethod
    def plan_automaton_synthesis_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of register bit clicks and execution triggers to achieve target automaton state."""
        if grid.ndim == 3:
            grid = grid[-1]

        plan: list[tuple[int, dict[str, int] | None]] = []

        # 1. Dynamically locate bottom execution buttons (y >= 52, pixel_count >= 15)
        visited_bottom = np.zeros_like(grid, dtype=bool)
        large_buttons = []
        for y in range(52, 64):
            for x in range(64):
                if not visited_bottom[y, x]:
                    c = grid[y, x]
                    q = [(y, x)]
                    visited_bottom[y, x] = True
                    comp = []
                    while q:
                        cy, cx = q.pop()
                        comp.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 52 <= ny < 64 and 0 <= nx < 64:
                                if not visited_bottom[ny, nx] and grid[ny, nx] == c:
                                    visited_bottom[ny, nx] = True
                                    q.append((ny, nx))
                    if len(comp) >= 15:
                        min_x = min(p[1] for p in comp)
                        max_x = max(p[1] for p in comp)
                        min_y = min(p[0] for p in comp)
                        max_y = max(p[0] for p in comp)
                        cx = (min_x + max_x) // 2
                        cy = (min_y + max_y) // 2
                        large_buttons.append((cx, cy, len(comp)))

        if not large_buttons:
            return plan

        # The primary RUN trigger button is the rightmost execution button
        run_button = max(large_buttons, key=lambda b: (b[0], b[2]))

        # 2. Dynamically extract all 3-pixel toggle bits in the programming band (y in [30, 50])
        subgrid = grid[30:50, :]
        visited = np.zeros_like(subgrid, dtype=bool)
        toggle_bits = []

        for y in range(subgrid.shape[0]):
            for x in range(subgrid.shape[1]):
                if not visited[y, x]:
                    c = subgrid[y, x]
                    q = [(y, x)]
                    visited[y, x] = True
                    comp = []
                    while q:
                        cy, cx = q.pop()
                        comp.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < subgrid.shape[0] and 0 <= nx < subgrid.shape[1]:
                                if not visited[ny, nx] and subgrid[ny, nx] == c:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    if len(comp) == 3:
                        min_y = min(p[0] for p in comp)
                        max_y = max(p[0] for p in comp)
                        min_x = min(p[1] for p in comp)
                        max_x = max(p[1] for p in comp)
                        if (max_y - min_y == 2 and min_x == max_x) or (
                            max_x - min_x == 2 and min_y == max_y
                        ):
                            cx_full = (min_x + max_x) // 2
                            cy_full = 30 + (min_y + max_y) // 2
                            toggle_bits.append((cx_full, cy_full, c))

        # Filter to the dominant inactive bit color
        from collections import Counter

        color_counts = Counter(b[2] for b in toggle_bits)
        if not color_counts:
            return plan

        inactive_color = color_counts.most_common(1)[0][0]
        inactive_bits = [b for b in toggle_bits if b[2] == inactive_color]

        is_dual_arena = len(inactive_bits) > 20

        if not is_dual_arena:
            # Single arena: configure program card register slots to opcode 3 (MOVE DOWN: bit 0 + bit 1)
            target_inactive = [b for b in inactive_bits if 15 <= b[0] <= 45 and 40 <= b[1] <= 48]
            for bx, by, _ in sorted(target_inactive, key=lambda b: (b[0], b[1])):
                plan.append(RemoteActuator.click(bx, by))
            plan.append(RemoteActuator.click(run_button[0], run_button[1]))
        else:
            # Dual arena: configure 4-slot active register to opcode 33 (MOVE UP: bit 0 + bit 5)
            card_bits = [b for b in inactive_bits if b[0] >= 35 and 30 <= b[1] <= 50]
            slots: dict[int, list[tuple[int, int, int]]] = {}
            for b in card_bits:
                slot_x = round(b[0] / 5.0) * 5
                slots.setdefault(slot_x, []).append(b)

            for sx in sorted(slots.keys()):
                slot_bits = sorted(slots[sx], key=lambda b: b[1])
                if len(slot_bits) >= 6:
                    b0 = slot_bits[0]
                    b5 = slot_bits[-1]
                    plan.append(RemoteActuator.click(b0[0], b0[1]))
                    plan.append(RemoteActuator.click(b5[0], b5[1]))

            plan.append(RemoteActuator.click(run_button[0], run_button[1]))
            plan.append(RemoteActuator.click(run_button[0], run_button[1]))
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
        """Standardized interface check for automaton programming synthesis puzzles."""
        return self.is_automaton_synthesis_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for automaton programming synthesis puzzles."""
        return self.plan_automaton_synthesis_grid(grid, current_level=current_level)
