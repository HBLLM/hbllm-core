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

import numpy as np

logger = logging.getLogger(__name__)


class VisualSymmetryAnalyzer:
    """Analyzes geometric symmetries (reflectional, rotational, diagonal) across visual grids."""

    @staticmethod
    def compute_symmetry_scores(grid: np.ndarray) -> dict[str, float]:
        """Calculates matching ratio (0.0 to 1.0) for horizontal, vertical, diagonal, and rotational symmetries."""
        H, W = grid.shape
        scores: dict[str, float] = {}

        # Horizontal symmetry (reflection across horizontal midline)
        h_flipped = np.flipud(grid)
        scores["horizontal"] = float(np.mean(grid == h_flipped))

        # Vertical symmetry (reflection across vertical midline)
        v_flipped = np.fliplr(grid)
        scores["vertical"] = float(np.mean(grid == v_flipped))

        # Diagonal and rotational symmetries (square grids)
        if H == W:
            scores["main_diagonal"] = float(np.mean(grid == grid.T))
            scores["anti_diagonal"] = float(np.mean(grid == np.flipud(np.fliplr(grid.T))))
            scores["rotational_90"] = float(np.mean(grid == np.rot90(grid, 1)))
            scores["rotational_180"] = float(np.mean(grid == np.rot90(grid, 2)))
        else:
            scores["main_diagonal"] = 0.0
            scores["anti_diagonal"] = 0.0
            scores["rotational_90"] = 0.0
            scores["rotational_180"] = float(np.mean(grid == np.flipud(np.fliplr(grid))))

        return scores

    @staticmethod
    def find_dominant_symmetry(grid: np.ndarray) -> tuple[str, float]:
        """Identifies the symmetry axis with the highest matching score."""
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
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
    ) -> tuple[bool, str | None]:
        """Check if the grid is near-symmetric along any axis.

        Returns:
            Tuple of (is_symmetric, best_axis_name) or (False, None).
        """
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
        best_axis, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= threshold:
            return True, best_axis
        return False, None
