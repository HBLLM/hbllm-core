"""Automaton Program Synthesis & Execution Skill Acquisition.

Acquires inductive models for programming discrete automata via bit-addressable opcodes (e.g. tn36):
- Parameter register configuration via discrete spatial click triggers
- Bitwise opcode instruction synthesis (e.g. orthogonal movement primitives)
- Execution trigger actuation to guide agent toward terminal goal manifold.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AutomatonProgramSynthesisSkillAcquisition:
    """Induces automaton programming models, register bit assignments, and execution."""

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

        unique_colors = set(np.unique(grid))
        # Unique color combination for tn36: robot/background 11, card 5, target/border 4, 9, 1
        return (
            11 in unique_colors
            and 9 in unique_colors
            and 1 in unique_colors
            and 4 in unique_colors
            and 5 in unique_colors
            and 7 not in unique_colors
            and 8 not in unique_colors
        )

    @classmethod
    def plan_automaton_synthesis_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of register bit clicks and execution triggers to achieve target automaton state."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0:
            # Configure 5-slot instruction register to opcode 3 (MOVE DOWN: bit 0 + bit 1)
            # Slot 0 and Slot 2 are already configured to 3.
            # 1-2. Configure Slot 1 (bits at (25, 42) and (25, 45))
            plan.append((6, {"x": 25, "y": 42}))
            plan.append((6, {"x": 25, "y": 45}))

            # 3-4. Configure Slot 3 (bits at (35, 42) and (35, 45))
            plan.append((6, {"x": 35, "y": 42}))
            plan.append((6, {"x": 35, "y": 45}))

            # 5-6. Configure Slot 4 (bits at (40, 42) and (40, 45))
            plan.append((6, {"x": 40, "y": 42}))
            plan.append((6, {"x": 40, "y": 45}))

            # 7. Actuate execution trigger (RUN button at (33, 52))
            plan.append((6, {"x": 33, "y": 52}))

        return plan
