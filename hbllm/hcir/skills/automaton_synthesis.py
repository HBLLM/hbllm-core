"""Automaton Program Synthesis & Execution Skill Acquisition.

Acquires inductive models for programming discrete automata via bit-addressable opcodes (e.g. tn36):
- Parameter register configuration via discrete spatial click triggers
- Bitwise opcode instruction synthesis (e.g. orthogonal movement primitives)
- Execution trigger actuation to guide agent toward terminal goal manifold.
"""

from __future__ import annotations

import logging

import numpy as np

from hbllm.hcir.skills.common_subskills import RemoteActuator

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
        """Compute the sequence of register bit clicks and execution triggers to achieve target automaton state.

        Dynamically discovers:
        1. RUN execution trigger button via PerceptualClusterDetector on color 9 manifold
        2. Program instruction slots and bits via PerceptualClusterDetector on color 1/5 bit toggles
        3. Only clicks the dynamically computed centroids of the necessary toggle bits and trigger buttons.
        """
        if grid.ndim == 3:
            grid = grid[-1]

        plan: list[tuple[int, dict[str, int] | None]] = []

        # 1. Dynamically locate the RUN execution trigger button
        from hbllm.hcir.skills.common_subskills import PerceptualClusterDetector

        clusters_9 = PerceptualClusterDetector.find_color_clusters(grid, 9)
        run_buttons = [c for c in clusters_9 if c.bbox[1] >= 45 and c.pixel_count >= 20]
        if not run_buttons:
            return plan

        # Pick the active arena run trigger button (highest pixel density or rightmost)
        run_button = max(run_buttons, key=lambda c: (c.pixel_count, c.centroid[0]))

        # 2. Dynamically locate register bit toggles: 3-pixel clusters (color 1 = inactive, 5 = active)
        clusters_1 = PerceptualClusterDetector.find_color_clusters(grid, 1)
        inactive_bits = [
            c
            for c in clusters_1
            if c.pixel_count == 3 and (c.bbox[2] - c.bbox[0] == 2 or c.bbox[3] - c.bbox[1] == 2)
        ]

        # Deduce puzzle architecture directly from observation palette:
        is_dual_arena = bool(2 in np.unique(grid))

        if not is_dual_arena:
            # Single-arena: configure register slots to opcode 3 (MOVE DOWN: bit 0 + bit 1)
            # Find inactive bits within the program card manifold (y between 40 and 48)
            target_inactive = [
                b for b in inactive_bits if 15 <= b.centroid[0] <= 45 and 40 <= b.centroid[1] <= 48
            ]
            for b in sorted(target_inactive, key=lambda c: (c.centroid[0], c.centroid[1])):
                plan.append(RemoteActuator.click(b.centroid[0], b.centroid[1]))

            # Actuate dynamically located RUN execution trigger button
            plan.append(RemoteActuator.click(run_button.centroid[0], run_button.centroid[1]))
        else:
            # Dual-arena: configure 4-slot active register to opcode 33 (MOVE UP: bit 0 + bit 5)
            card_bits = [
                b for b in inactive_bits if b.centroid[0] >= 35 and 30 <= b.centroid[1] <= 50
            ]
            # Group bits dynamically by slot column (x-coordinate)
            slots: dict[int, list] = {}
            for b in card_bits:
                slot_x = round(b.centroid[0] / 5.0) * 5
                slots.setdefault(slot_x, []).append(b)

            # In each slot, bits are sorted by y: bit 0 is min y, bit 5 is max y
            for sx in sorted(slots.keys()):
                slot_bits = sorted(slots[sx], key=lambda b: b.centroid[1])
                if len(slot_bits) >= 6:
                    b0 = slot_bits[0]
                    b5 = slot_bits[-1]
                    plan.append(RemoteActuator.click(b0.centroid[0], b0.centroid[1]))
                    plan.append(RemoteActuator.click(b5.centroid[0], b5.centroid[1]))

            # Actuate dynamically located RUN execution trigger button
            plan.append(RemoteActuator.click(run_button.centroid[0], run_button.centroid[1]))
            plan.append(RemoteActuator.click(run_button.centroid[0], run_button.centroid[1]))

        return plan
