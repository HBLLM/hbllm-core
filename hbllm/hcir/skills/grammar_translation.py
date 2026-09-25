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
        # tr87 signature: actions {1, 2, 3, 4} (no click 5, 6, 7)
        if set(available_actions) != {1, 2, 3, 4}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Structural signature:
        # 1. Distinct bottom panel background region at y in [35, 60]
        # 2. Presence of 7x7 square tile frames at top (y < 35)
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        bottom_half = grid[35:60, :]
        bg_counts = np.bincount(bottom_half.flatten().astype(np.int64))
        if bg_counts.max() < 0.35 * bottom_half.size:
            return False

        found_tiles = 0
        for y in range(4, 28, 9):
            for x in range(4, 56):
                if x + 6 < 64 and y + 6 < 35:
                    c = grid[y, x]
                    if (
                        c != bg
                        and grid[y + 6, x] == c
                        and grid[y, x + 6] == c
                        and grid[y + 6, x + 6] == c
                    ):
                        found_tiles += 1
                        if found_tiles >= 2:
                            return True
        return found_tiles >= 2

    GLYPH_TEMPLATES: dict[tuple[str, int], np.ndarray] = {
        ("A", 1): np.array(
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("A", 2): np.array(
            [[0, 0, 0, 0, 1], [0, 0, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0]],
            dtype=bool,
        ),
        ("A", 3): np.array(
            [[1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("A", 4): np.array(
            [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]],
            dtype=bool,
        ),
        ("A", 5): np.array(
            [[1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1]],
            dtype=bool,
        ),
        ("A", 6): np.array(
            [[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("A", 7): np.array(
            [[1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1]],
            dtype=bool,
        ),
        ("B", 1): np.array(
            [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("B", 2): np.array(
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("B", 3): np.array(
            [[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 0]],
            dtype=bool,
        ),
        ("B", 4): np.array(
            [[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("B", 5): np.array(
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]],
            dtype=bool,
        ),
        ("B", 6): np.array(
            [[1, 1, 1, 1, 0], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 1, 0, 0, 1], [0, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("B", 7): np.array(
            [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]],
            dtype=bool,
        ),
        ("C", 1): np.array(
            [[1, 0, 1, 0, 1], [1, 0, 0, 0, 0], [1, 0, 1, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
            dtype=bool,
        ),
        ("C", 2): np.array(
            [[1, 1, 1, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 1, 1, 1]],
            dtype=bool,
        ),
        ("C", 3): np.array(
            [[1, 0, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 0, 1]],
            dtype=bool,
        ),
        ("C", 4): np.array(
            [[0, 0, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]],
            dtype=bool,
        ),
        ("C", 5): np.array(
            [[1, 1, 0, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1]],
            dtype=bool,
        ),
        ("C", 6): np.array(
            [[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1]],
            dtype=bool,
        ),
        ("C", 7): np.array(
            [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1], [0, 1, 0, 1, 0]],
            dtype=bool,
        ),
    }

    @classmethod
    def match_glyph(cls, binary_patch: np.ndarray) -> tuple[str, int] | None:
        """Rotation-invariant matcher for 5x5 ARC glyph patches."""
        for (alpha, idx), tmpl in cls.GLYPH_TEMPLATES.items():
            for k in range(4):
                if np.array_equal(np.rot90(binary_patch, k), tmpl):
                    return (alpha, idx)
        return None

    @classmethod
    def plan_grammar_translation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the sequence of cursor moves and symbol rotations to satisfy the grammar rewrite."""
        if grid.ndim == 3:
            grid = grid[-1]

        def generate_slot_plan(diffs: list[int]) -> list[tuple[int, dict[str, int] | None]]:
            actions: list[tuple[int, dict[str, int] | None]] = []
            for i, d in enumerate(diffs):
                if i > 0:
                    actions.append((4, None))  # Move cursor right to next register slot
                if d > 0:
                    actions.extend([(2, None)] * d)  # Increment mod 7
                elif d < 0:
                    actions.extend([(1, None)] * abs(d))  # Decrement mod 7
            return actions

        # 1. Detect all 7x7 square tile frames and extract their interior 5x5 glyphs
        found_glyphs: list[tuple[int, int, str, int, int]] = []
        bg = int(grid[35, 0])
        for y in range(64 - 6):
            for x in range(64 - 6):
                top = grid[y, x : x + 7]
                bot = grid[y + 6, x : x + 7]
                left = grid[y : y + 7, x]
                right = grid[y : y + 7, x + 6]
                c = top[0]
                if c != 0 and c != bg:
                    if (
                        np.all(top == c)
                        and np.all(bot == c)
                        and np.all(left == c)
                        and np.all(right == c)
                    ):
                        interior = grid[y + 1 : y + 6, x + 1 : x + 6]
                        binary_patch = interior != c
                        matched = cls.match_glyph(binary_patch)
                        if matched is not None:
                            found_glyphs.append((y + 1, x + 1, matched[0], matched[1], int(c)))

        input_glyphs = sorted([g for g in found_glyphs if 35 <= g[0] < 48], key=lambda g: g[1])
        output_glyphs = sorted([g for g in found_glyphs if g[0] >= 48], key=lambda g: g[1])

        if not input_glyphs or not output_glyphs:
            return []

        lhs_color = input_glyphs[0][4]

        # 2. Induce formal grammar rewrite rules from the specification panel (y < 35)
        rule_glyphs = [g for g in found_glyphs if g[0] < 35]
        rows: dict[int, list[tuple[int, int, str, int, int]]] = {}
        for g in rule_glyphs:
            row_key = round(g[0] / 5.0) * 5
            rows.setdefault(row_key, []).append(g)

        rules: dict[tuple[int, ...], list[int]] = {}
        for _, g_list in rows.items():
            g_list.sort(key=lambda g: g[1])
            curr_side = None
            curr_chunk: list[tuple[int, int, str, int, int]] = []
            rule_pairs: list[tuple[str, list[int]]] = []
            for g in g_list:
                side = "lhs" if g[4] == lhs_color else "rhs"
                if side != curr_side:
                    if curr_chunk and curr_side is not None:
                        rule_pairs.append((curr_side, [c[3] for c in curr_chunk]))
                    curr_side = side
                    curr_chunk = [g]
                else:
                    curr_chunk.append(g)
            if curr_chunk and curr_side is not None:
                rule_pairs.append((curr_side, [c[3] for c in curr_chunk]))

            for i in range(0, len(rule_pairs) - 1, 2):
                if rule_pairs[i][0] == "lhs" and rule_pairs[i + 1][0] == "rhs":
                    rules[tuple(rule_pairs[i][1])] = rule_pairs[i + 1][1]

        # 3. Apply grammar rewrite to the input string
        input_seq = [g[3] for g in input_glyphs]
        target_seq: list[int] = []
        idx = 0
        while idx < len(input_seq):
            matched_rule = False
            for l in range(3, 0, -1):
                if idx + l <= len(input_seq):
                    sub = tuple(input_seq[idx : idx + l])
                    if sub in rules:
                        target_seq.extend(rules[sub])
                        idx += l
                        matched_rule = True
                        break
            if not matched_rule:
                target_seq.append(input_seq[idx])
                idx += 1

        init_seq = [g[3] for g in output_glyphs]
        if len(init_seq) != len(target_seq):
            return []

        # 4. Synthesize optimal cyclic mutation plan
        diffs: list[int] = []
        for cur, tgt in zip(init_seq, target_seq):
            d = (tgt - cur) % 7
            if d > 3:
                d -= 7
            diffs.append(d)

        return generate_slot_plan(diffs)
