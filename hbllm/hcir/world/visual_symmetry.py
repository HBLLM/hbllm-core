"""Visual Symmetry Analysis for HCIR World Kernel.

Domain-agnostic geometric symmetry detection and pattern completion, migrated
from the ARC-AGI plugin to the core HCIR platform for universal reuse across
domains (robotics, browser, game environments).

Provides:
  - Multi-axis symmetry scoring (horizontal, vertical, diagonal, rotational)
  - Dominant symmetry identification
  - Asymmetric pattern completion via reflection
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VisualSymmetryAnalyzer:
    """Analyzes geometric symmetries (reflectional, rotational, diagonal) across visual grids."""

    @staticmethod
    def compute_symmetry_scores(grid: np.ndarray, background_color: int = 0) -> dict[str, float]:
        """Calculates matching ratio (0.0 to 1.0) of foreground patterns for symmetries."""
        H, W = grid.shape
        scores: dict[str, float] = {}

        def _calc_sym(t_grid: np.ndarray) -> float:
            fg_mask = (grid != background_color) | (t_grid != background_color)
            total_fg = int(np.sum(fg_mask))
            if total_fg == 0:
                return 0.0
            matching_fg = int(np.sum((grid == t_grid) & fg_mask))
            return float(matching_fg / total_fg)

        # Horizontal symmetry (reflection across horizontal midline)
        h_flipped = np.flipud(grid)
        scores["horizontal"] = _calc_sym(h_flipped)

        # Vertical symmetry (reflection across vertical midline)
        v_flipped = np.fliplr(grid)
        scores["vertical"] = _calc_sym(v_flipped)

        # Diagonal and rotational symmetries (square grids)
        if H == W:
            scores["main_diagonal"] = _calc_sym(grid.T)
            scores["anti_diagonal"] = _calc_sym(np.flipud(np.fliplr(grid.T)))
            scores["rotational_90"] = _calc_sym(np.rot90(grid, 1))
            scores["rotational_180"] = _calc_sym(np.rot90(grid, 2))
        else:
            scores["main_diagonal"] = 0.0
            scores["anti_diagonal"] = 0.0
            scores["rotational_90"] = 0.0
            scores["rotational_180"] = _calc_sym(np.flipud(np.fliplr(grid)))

        return scores

    @staticmethod
    def find_dominant_symmetry(grid: np.ndarray, background_color: int = 0) -> tuple[str, float]:
        """Identifies the symmetry axis with the highest matching score on foreground."""
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(
            grid, background_color=background_color
        )
        return max(scores.items(), key=lambda item: item[1])

    @staticmethod
    def predict_symmetric_completion(
        grid: np.ndarray,
        symmetry_type: str = "vertical",
        background_color: int = 0,
    ) -> np.ndarray:
        """Completes an incomplete or asymmetric pattern by reflecting the non-empty half."""
        completed = grid.copy()
        H, W = grid.shape

        if symmetry_type == "vertical":
            mid = W // 2
            left_half = grid[:, :mid]
            right_half = grid[:, mid + (1 if W % 2 != 0 else 0) :]
            left_density = int(np.sum(left_half != background_color))
            right_density = int(np.sum(right_half != background_color))

            if left_density >= right_density:
                mirrored = np.fliplr(left_half)
                completed[:, W - mid :] = mirrored
            else:
                mirrored = np.fliplr(right_half)
                completed[:, :mid] = mirrored

        elif symmetry_type == "horizontal":
            mid = H // 2
            top_half = grid[:mid, :]
            bottom_half = grid[mid + (1 if H % 2 != 0 else 0) :, :]
            top_density = int(np.sum(top_half != background_color))
            bottom_density = int(np.sum(bottom_half != background_color))

            if top_density >= bottom_density:
                mirrored = np.flipud(top_half)
                completed[H - mid :, :] = mirrored
            else:
                mirrored = np.flipud(bottom_half)
                completed[:mid, :] = mirrored

        elif symmetry_type == "main_diagonal" and H == W:
            # Complete by reflecting upper triangle to lower (or vice versa)
            upper_density = int(np.sum(np.triu(grid != background_color, k=1)))
            lower_density = int(np.sum(np.tril(grid != background_color, k=-1)))
            if upper_density >= lower_density:
                for r in range(H):
                    for c in range(r):
                        completed[r, c] = completed[c, r]
            else:
                for r in range(H):
                    for c in range(r + 1, W):
                        completed[r, c] = completed[c, r]

        elif symmetry_type == "rotational_180":
            # Complete by rotating 180° from the denser half
            mid_r = H // 2
            top_density = int(np.sum(grid[:mid_r] != background_color))
            bottom_density = int(np.sum(grid[mid_r:] != background_color))
            rotated = np.rot90(grid, 2)
            if top_density >= bottom_density:
                completed[mid_r:] = rotated[mid_r:]
            else:
                completed[:mid_r] = rotated[:mid_r]

        return completed

    @staticmethod
    def is_near_symmetric(
        grid: np.ndarray,
        threshold: float = 0.85,
        background_color: int = 0,
    ) -> tuple[bool, str | None]:
        """Check if the grid is near-symmetric along any axis.

        Returns:
            Tuple of (is_symmetric, best_axis_name) or (False, None).
        """
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(
            grid, background_color=background_color
        )
        best_axis, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= threshold:
            return True, best_axis
        return False, None

    @staticmethod
    def extract_discrepancy_targets(
        grid: np.ndarray,
        background_color: int = 0,
        threshold: float = 0.55,
    ) -> list[tuple[int, int, int, float]]:
        """Extract spatial coordinates and expected features where the grid violates dominant symmetry.

        Returns:
            List of (r, c, expected_feature, confidence) tuples for coordinates that should be completed.
        """
        sym_type, sym_score = VisualSymmetryAnalyzer.find_dominant_symmetry(
            grid, background_color=background_color
        )
        if sym_score < threshold or sym_score >= 0.999:
            return []

        completed = VisualSymmetryAnalyzer.predict_symmetric_completion(
            grid, symmetry_type=sym_type, background_color=background_color
        )
        diff_mask = (grid != completed) & (completed != background_color)
        pts = np.argwhere(diff_mask)
        targets: list[tuple[int, int, int, float]] = []
        for p in pts:
            r, c = int(p[0]), int(p[1])
            expected_feat = int(completed[r, c])
            targets.append((r, c, expected_feat, sym_score))
        return targets

    @staticmethod
    def detect_periodicity(
        grid: np.ndarray,
        axis: int = 0,
        min_period: int = 1,
    ) -> tuple[int, float]:
        """W105: Detects spatial repetition periodicity along rows (axis=0) or columns (axis=1).

        Returns:
            (best_period, confidence)
        """
        if grid.ndim != 2:
            return 0, 0.0
        size = grid.shape[axis]
        if size <= 1:
            return 1, 1.0

        best_p = 0
        best_match = 0.0

        for p in range(min_period, (size // 2) + 1):
            if size % p == 0:
                reps = size // p
                matches = 0
                total_comparisons = (reps - 1) * (grid.shape[1 - axis] * p)
                if total_comparisons == 0:
                    continue
                for r in range(1, reps):
                    if axis == 0:
                        matches += int(np.sum(grid[:p, :] == grid[r * p : (r + 1) * p, :]))
                    else:
                        matches += int(np.sum(grid[:, :p] == grid[:, r * p : (r + 1) * p]))
                score = matches / total_comparisons
                if score > best_match:
                    best_match = score
                    best_p = p

        return best_p, float(best_match)

    @staticmethod
    def detect_tiling(
        grid: np.ndarray,
    ) -> tuple[tuple[int, int], tuple[int, int], float]:
        """W106: Discovers fundamental tile unit (h, w) and repetition counts (reps_r, reps_c).

        Returns:
            ((tile_h, tile_w), (reps_r, reps_c), confidence)
        """
        H, W = grid.shape
        pr, conf_r = VisualSymmetryAnalyzer.detect_periodicity(grid, axis=0)
        pc, conf_c = VisualSymmetryAnalyzer.detect_periodicity(grid, axis=1)

        tile_h = pr if pr > 0 else H
        tile_w = pc if pc > 0 else W
        reps_r = H // tile_h if tile_h > 0 else 1
        reps_c = W // tile_w if tile_w > 0 else 1

        overall_conf = (
            float((conf_r + conf_c) / 2.0) if (pr > 0 and pc > 0) else float(max(conf_r, conf_c))
        )
        return (tile_h, tile_w), (reps_r, reps_c), overall_conf

    @staticmethod
    def detect_counting_relation(
        counts: list[int],
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """W107: Identifies arithmetic/algebraic counting relations (constant difference, scaling, conservation)."""
        if len(counts) < 2:
            return False, "insufficient_samples", 0.0, {}

        # 1. Conservation (all equal)
        if all(c == counts[0] for c in counts):
            return True, "conserved", 1.0, {"constant_value": counts[0]}

        # 2. Arithmetic progression
        diffs = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
        if all(d == diffs[0] for d in diffs):
            return True, "arithmetic_step", 1.0, {"step": diffs[0]}

        # 3. Geometric scaling ratio
        if all(counts[i] != 0 and counts[i + 1] % counts[i] == 0 for i in range(len(counts) - 1)):
            ratios = [counts[i + 1] // counts[i] for i in range(len(counts) - 1)]
            if all(r == ratios[0] for r in ratios):
                return True, "geometric_ratio", 1.0, {"ratio": ratios[0]}

        return False, "irregular", 0.0, {}

    @staticmethod
    def detect_sequence_progression(
        series: list[float | int],
    ) -> tuple[bool, str, float]:
        """W108: Detects monotonic, arithmetic, or alternating progression in a numerical or spatial sequence."""
        if len(series) < 2:
            return False, "insufficient_length", 0.0

        is_increasing = all(series[i + 1] > series[i] for i in range(len(series) - 1))
        is_decreasing = all(series[i + 1] < series[i] for i in range(len(series) - 1))

        if is_increasing:
            return True, "strictly_increasing", 1.0
        if is_decreasing:
            return True, "strictly_decreasing", 1.0

        # Monotonic non-decreasing
        if all(series[i + 1] >= series[i] for i in range(len(series) - 1)):
            return True, "monotonic_non_decreasing", 0.9
        if all(series[i + 1] <= series[i] for i in range(len(series) - 1)):
            return True, "monotonic_non_increasing", 0.9

        return False, "non_monotonic", 0.0

    @staticmethod
    def detect_permutation_order(
        seq_a: list[int],
        seq_b: list[int],
    ) -> tuple[bool, str]:
        """W109: Detects permutation relationships between two entity or color sequences."""
        if len(seq_a) != len(seq_b) or sorted(seq_a) != sorted(seq_b):
            return False, "not_a_permutation"

        if seq_a == seq_b:
            return True, "identity"

        if seq_a == list(reversed(seq_b)):
            return True, "reversed"

        # Cyclic shift
        n = len(seq_a)
        doubled = seq_a + seq_a
        for shift in range(1, n):
            if doubled[shift : shift + n] == seq_b:
                return True, f"cyclic_shift_{shift}"

        return True, "arbitrary_permutation"

    @staticmethod
    def is_structurally_equivalent(
        patch_a: np.ndarray,
        patch_b: np.ndarray,
        group: str = "D4",
    ) -> tuple[bool, str | None]:
        """W110: Tests whether patch_b is isomorphic to patch_a under dihedral group D4 transformations."""
        if patch_a.shape != patch_b.shape and patch_a.shape != (patch_b.shape[1], patch_b.shape[0]):
            return False, None

        transforms: list[tuple[str, np.ndarray]] = [
            ("identity", patch_a.copy()),
            ("rot_90", np.rot90(patch_a, 1)),
            ("rot_180", np.rot90(patch_a, 2)),
            ("rot_270", np.rot90(patch_a, 3)),
            ("flip_ud", np.flipud(patch_a)),
            ("flip_lr", np.fliplr(patch_a)),
            ("transpose", patch_a.T),
            ("anti_transpose", np.rot90(np.fliplr(patch_a), 1)),
        ]

        for op_name, transformed in transforms:
            if transformed.shape == patch_b.shape and np.array_equal(transformed, patch_b):
                return True, op_name

        return False, None
