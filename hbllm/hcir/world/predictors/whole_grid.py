"""Whole-Grid Outcome Prediction & Raster Rollout Engine (W083).

Provides the complete execution and rasterization path for 2D discrete grid world modeling:
1. Output resolution & dimension inference across training demonstrations.
2. Background canvas allocation and multi-layer rendering.
3. Raster transformation execution and boundary clipping.
4. Exact-match verification and pixel-level divergence metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


class DimensionMode(StrEnum):
    """Modes for inferring target output canvas resolution."""

    SAME = "SAME"  # Invariant dimensions: H_out = H_in, W_out = W_in
    FIXED = "FIXED"  # Constant output dimensions across all demonstrations
    SCALED = "SCALED"  # Proportional scaling by rational factor (s_r, s_c)
    CROP = "CROP"  # Dynamic shape based on extracted bounding box
    UNKNOWN = "UNKNOWN"  # Cannot deduce dimension rule


@dataclass
class DimensionRule:
    """Inferred dimensional relationship between input and output grids."""

    mode: DimensionMode
    fixed_shape: tuple[int, int] | None = None
    scale_factors: tuple[float, float] | None = None
    confidence: float = 1.0

    def compute_output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int] | None:
        """Calculate predicted output shape for a given input shape."""
        H_in, W_in = input_shape
        if self.mode == DimensionMode.SAME:
            return (H_in, W_in)
        elif self.mode == DimensionMode.FIXED:
            return self.fixed_shape
        elif self.mode == DimensionMode.SCALED and self.scale_factors is not None:
            sr, sc = self.scale_factors
            return (int(round(H_in * sr)), int(round(W_in * sc)))
        elif self.mode == DimensionMode.CROP:
            # Crop depends on grid contents, cannot determine from shape alone
            return None
        return None


class ExecutableRule(Protocol):
    """Protocol for any transformation rule executable on a 2D numpy array."""

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray: ...


class WholeGridPredictor:
    """Whole-grid outcome prediction and raster synthesis engine (W083)."""

    name: str = "whole_grid"

    def predict_state(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
    ) -> tuple[dict[str, Any], float]:
        """Compute whole-grid raster outcome prediction."""
        predicted = dict(snapshot.variables)
        grid = predicted.get("grid")
        if grid is None:
            grid = predicted.get("grid_2d")
        if grid is None and hasattr(snapshot, "metadata"):
            grid = snapshot.metadata.get("grid")

        if grid is not None and isinstance(grid, np.ndarray) and grid.ndim == 2:
            predicted["grid"] = np.copy(grid)
            return predicted, 0.90
        return predicted, 0.50

    @staticmethod
    def estimate_background_color(grid: np.ndarray) -> int:
        """Estimate background color using edge frequency and modal analysis.

        In discrete grid environments, color 0 is the conventional background. When non-zero,
        the background is typically the dominant border/perimeter color.
        """
        if grid.size == 0:
            return 0
        H, W = grid.shape
        if H <= 2 or W <= 2:
            vals, counts = np.unique(grid, return_counts=True)
            return int(vals[np.argmax(counts)])

        # Sample border pixels (top, bottom, left, right)
        border_pixels = np.concatenate(
            [
                grid[0, :],
                grid[-1, :],
                grid[1:-1, 0],
                grid[1:-1, -1],
            ]
        )
        b_vals, b_counts = np.unique(border_pixels, return_counts=True)
        top_border_color = int(b_vals[np.argmax(b_counts)])

        # If 0 is present in the grid and constitutes at least 30% of perimeter, prefer 0
        if 0 in b_vals:
            zero_idx = np.where(b_vals == 0)[0][0]
            if b_counts[zero_idx] / len(border_pixels) >= 0.30:
                return 0

        return top_border_color

    @staticmethod
    def infer_dimension_rule(
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> DimensionRule:
        """Infer output canvas dimension rule from input-output training pairs."""
        if not train_pairs:
            return DimensionRule(mode=DimensionMode.UNKNOWN, confidence=0.0)

        # 1. Check SAME (Invariant) dimension rule
        all_same = all(x.shape == y.shape for x, y in train_pairs)
        if all_same:
            return DimensionRule(mode=DimensionMode.SAME, confidence=1.0)

        # 2. Check FIXED dimension rule (all output grids have identical shape)
        first_y_shape: tuple[int, int] = (
            int(train_pairs[0][1].shape[0]),
            int(train_pairs[0][1].shape[1]),
        )
        all_fixed = all(y.shape == first_y_shape for _, y in train_pairs)
        inputs_differ = any(x.shape != train_pairs[0][0].shape for x, _ in train_pairs)
        if all_fixed and inputs_differ:
            return DimensionRule(
                mode=DimensionMode.FIXED,
                fixed_shape=first_y_shape,
                confidence=0.95,
            )

        # 3. Check SCALED dimension rule (rational scale factors sr, sc)
        scale_ratios: list[tuple[float, float]] = []
        is_consistent_scale = True
        for x, y in train_pairs:
            if x.shape[0] == 0 or x.shape[1] == 0:
                is_consistent_scale = False
                break
            sr = y.shape[0] / x.shape[0]
            sc = y.shape[1] / x.shape[1]
            scale_ratios.append((sr, sc))

        if is_consistent_scale and scale_ratios:
            first_sr, first_sc = scale_ratios[0]
            if all(
                abs(sr - first_sr) < 1e-5 and abs(sc - first_sc) < 1e-5 for sr, sc in scale_ratios
            ):
                return DimensionRule(
                    mode=DimensionMode.SCALED,
                    scale_factors=(first_sr, first_sc),
                    confidence=0.90,
                )

        # 4. If all fixed and inputs also fixed, FIXED is a strong hypothesis
        if all_fixed:
            return DimensionRule(
                mode=DimensionMode.FIXED,
                fixed_shape=first_y_shape,
                confidence=0.80,
            )

        # 5. Check if all outputs are smaller than inputs (consistent crop)
        all_smaller = all(
            y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1] for x, y in train_pairs
        )
        if all_smaller:
            return DimensionRule(mode=DimensionMode.CROP, confidence=0.70)

        return DimensionRule(mode=DimensionMode.UNKNOWN, confidence=0.0)

    @staticmethod
    def render_prediction(
        rule: ExecutableRule,
        input_grid: np.ndarray,
        target_shape: tuple[int, int] | None = None,
        default_bg: int = 0,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Execute a rule and render the resulting full-canvas raster grid.

        Handles target canvas allocation, centering/padding, and shape verification.
        """
        raw_output = rule.execute(input_grid, context)
        if not isinstance(raw_output, np.ndarray):
            raw_output = np.asarray(raw_output, dtype=int)

        if raw_output.ndim != 2:
            raise ValueError(f"Rule returned array with ndim={raw_output.ndim}, expected 2D grid")

        if target_shape is None or raw_output.shape == target_shape:
            return raw_output.astype(int)

        # Dimension adaptation: allocate canvas and place raw_output
        target_H, target_W = target_shape
        canvas = np.full((target_H, target_W), default_bg, dtype=int)

        out_H, out_W = raw_output.shape
        place_H = min(target_H, out_H)
        place_W = min(target_W, out_W)

        canvas[:place_H, :place_W] = raw_output[:place_H, :place_W]
        return canvas

    @staticmethod
    def compute_grid_metrics(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
    ) -> tuple[bool, float, int]:
        """Compute exact match, pixel accuracy, and mismatch count.

        Returns:
            (is_exact_match, pixel_accuracy, mismatch_count)
        """
        if predicted.shape != ground_truth.shape:
            total_elements = max(predicted.size, ground_truth.size, 1)
            return False, 0.0, total_elements

        diff_mask = predicted != ground_truth
        mismatch_count = int(np.sum(diff_mask))
        total_pixels = ground_truth.size
        if total_pixels == 0:
            return True, 1.0, 0

        pixel_accuracy = float(1.0 - (mismatch_count / total_pixels))
        is_exact = mismatch_count == 0
        return is_exact, pixel_accuracy, mismatch_count
