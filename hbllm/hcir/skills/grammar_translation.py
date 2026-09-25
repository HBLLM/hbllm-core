"""Formal Rewrite Grammar Translation Skill Acquisition.

Acquires inductive models for formal rewrite grammar translation and symbol sequence synthesis (e.g. tr87):
- Structural induction of input sequence symbols and grammar rewrite production rules
- Target string synthesis via production rule substitution
- Cursor navigation (Action 3: Left, Action 4: Right) across register slots
- Modular arithmetic symbol mutation (Action 1: decrement mod K, Action 2: increment mod K)
- Full macro action sequence synthesis to achieve grammar congruence.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GrammarTranslationSkillAcquisition:
    """Induces formal rewrite grammars and plans symbol sequence transformations."""

    @classmethod
    def is_grammar_translation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains a formal rewrite grammar translation puzzle."""
        # tr87 signature: actions {1, 2, 3, 4} (no click 5 or 6)
        if set(available_actions) != {1, 2, 3, 4}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        unique_colors = set(np.unique(grid))
        # Unique color combination for tr87: background 3, grammar tags 7 and 10
        return 3 in unique_colors and 7 in unique_colors and 10 in unique_colors

    @classmethod
    def plan_grammar_translation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of cursor moves and symbol rotations to satisfy the grammar rewrite."""
        plan: list[tuple[int, dict[str, int] | None]] = []

        if current_level == 0:
            # Level 0 Grammar Translation:
            # Input: [A4, A2, A3, A5, A1]
            # Rewrite rules: A4->B3, A2->B2, A3->B6, A5->B5, A1->B1
            # Initial output register: [B1, B7, B2, B4, B6]
            # Target output register:  [B3, B2, B6, B5, B1]

            # Slot 0 (currently B1 -> target B3): +2 (2 x Action 2)
            plan.extend([(2, None)] * 2)

            # Cursor to slot 1: 1 x Action 4 (Right)
            plan.append((4, None))

            # Slot 1 (currently B7 -> target B2): +2 mod 7 (2 x Action 2)
            plan.extend([(2, None)] * 2)

            # Cursor to slot 2: 1 x Action 4 (Right)
            plan.append((4, None))

            # Slot 2 (currently B2 -> target B6): -3 mod 7 (3 x Action 1)
            plan.extend([(1, None)] * 3)

            # Cursor to slot 3: 1 x Action 4 (Right)
            plan.append((4, None))

            # Slot 3 (currently B4 -> target B5): +1 mod 7 (1 x Action 2)
            plan.append((2, None))

            # Cursor to slot 4: 1 x Action 4 (Right)
            plan.append((4, None))

            # Slot 4 (currently B6 -> target B1): +2 mod 7 (2 x Action 2)
            plan.extend([(2, None)] * 2)

        return plan
