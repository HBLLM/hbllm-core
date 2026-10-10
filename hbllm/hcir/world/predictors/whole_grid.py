"""Whole-Grid Outcome Prediction & Raster Rollout Engine (W083).

Provides the complete execution and rasterization path for 2D discrete grid world modeling:
1. Output resolution & dimension inference across training demonstrations.
2. Background canvas allocation and multi-layer rendering.
3. Raster transformation execution and boundary clipping.
4. Exact-match verification and pixel-level divergence metrics.
"""

from __future__ import annotations

import logging
from collections import deque
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

    @classmethod
    def predict_next_state(
        cls,
        grid: np.ndarray,
        rule: ExecutableRule,
        context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """W081: Deterministic or parameterized next-state outcome prediction."""
        return cls.render_prediction(rule, grid, context=context)

    @classmethod
    def predict_object_outcomes(
        cls,
        objects: list[dict[str, Any]],
        rule: ExecutableRule,
        canvas_shape: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """W082: Predicts object-level outcomes by transforming isolated object masks."""
        outcomes: list[dict[str, Any]] = []
        for obj in objects:
            coords = obj.get("coords", [])
            color = int(obj.get("color", 1))
            obj_grid = np.zeros(canvas_shape, dtype=int)
            for r, c in coords:
                if 0 <= r < canvas_shape[0] and 0 <= c < canvas_shape[1]:
                    obj_grid[r, c] = color

            transformed = cls.render_prediction(rule, obj_grid)
            raw_pts = np.argwhere(transformed != 0)
            new_coords = [tuple(int(x) for x in p) for p in raw_pts]
            new_color = (
                int(transformed[new_coords[0][0], new_coords[0][1]]) if new_coords else color
            )

            outcomes.append(
                {
                    "object_id": obj.get("id", obj.get("object_id", "obj")),
                    "original_coords": coords,
                    "predicted_coords": new_coords,
                    "predicted_color": new_color,
                    "survived": len(new_coords) > 0,
                }
            )
        return outcomes

    @classmethod
    def simulate_rollout(
        cls,
        initial_grid: np.ndarray,
        rules: list[ExecutableRule] | ExecutableRule,
        criteria: SimulationStoppingCriteria | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        """W084: Multi-step forward simulation rolling out sequential transformation rules."""
        if criteria is None:
            criteria = SimulationStoppingCriteria()

        rule_seq: list[ExecutableRule] = rules if isinstance(rules, list) else [rules]
        trajectory: list[np.ndarray] = [initial_grid.copy()]
        curr = initial_grid.copy()

        for step in range(1, criteria.max_steps + 1):
            rule = rule_seq[(step - 1) % len(rule_seq)]
            next_state = cls.render_prediction(rule, curr, context=context)

            stop, _ = cls.should_stop_simulation(step, next_state, trajectory, criteria)
            trajectory.append(next_state)
            curr = next_state
            if stop:
                break

        return trajectory

    @classmethod
    def simulate_alternative_outcomes(
        cls,
        grid: np.ndarray,
        candidate_rules: list[ExecutableRule],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """W085: Alternative-outcome simulation exploring branching candidate hypotheses."""
        outcomes: list[dict[str, Any]] = []
        for rule in candidate_rules:
            pred = cls.render_prediction(rule, grid, context=context)
            rule_id = getattr(rule, "rule_id", getattr(rule, "name", str(type(rule).__name__)))
            complexity = float(getattr(rule, "complexity", 1.0))
            outcomes.append(
                {
                    "rule_id": rule_id,
                    "rule": rule,
                    "predicted_grid": pred,
                    "complexity": complexity,
                }
            )
        return outcomes

    @staticmethod
    def compute_prediction_error(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """W086: Normalized pixel discrepancy error in [0.0, 1.0]."""
        if predicted.shape != ground_truth.shape:
            return 1.0
        total = ground_truth.size
        if total == 0:
            return 0.0
        diff = np.sum(predicted != ground_truth)
        return float(diff / total)

    @staticmethod
    def localize_prediction_error(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """W087: Localizes spatial discrepancy error mask and bounding boxes of error clusters."""
        if predicted.shape != ground_truth.shape:
            H = max(predicted.shape[0], ground_truth.shape[0])
            W = max(predicted.shape[1], ground_truth.shape[1])
            mask = np.ones((H, W), dtype=bool)
            return mask, [(0, 0, H - 1, W - 1)]

        mask = predicted != ground_truth
        if not np.any(mask):
            return mask, []

        visited = np.zeros_like(mask, dtype=bool)
        bboxes: list[tuple[int, int, int, int]] = []
        H, W = mask.shape

        for r in range(H):
            for c in range(W):
                if mask[r, c] and not visited[r, c]:
                    min_r, max_r = r, r
                    min_c, max_c = c, c
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        min_r, max_r = min(min_r, curr_r), max(max_r, curr_r)
                        min_c, max_c = min(min_c, curr_c), max(max_c, curr_c)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    bboxes.append((min_r, min_c, max_r, max_c))

        return mask, bboxes

    @classmethod
    def evaluate_rollout(
        cls,
        rollout_states: list[np.ndarray],
        target_grid: np.ndarray | None = None,
    ) -> float:
        """W088: Evaluates utility/validity of simulated rollout trajectory."""
        if not rollout_states:
            return 0.0
        final_state = rollout_states[-1]
        if target_grid is not None:
            err = cls.compute_prediction_error(final_state, target_grid)
            return float(max(0.0, 1.0 - err))

        if len(rollout_states) > 1:
            diffs = [
                cls.compute_prediction_error(rollout_states[i], rollout_states[i - 1])
                for i in range(1, len(rollout_states))
            ]
            stability = 1.0 - (sum(diffs) / len(diffs))
            return float(max(0.0, min(1.0, stability)))
        return 1.0

    @staticmethod
    def should_stop_simulation(
        step: int,
        current_state: np.ndarray,
        previous_states: list[np.ndarray],
        criteria: SimulationStoppingCriteria,
    ) -> tuple[bool, str]:
        """W089: Evaluates stopping conditions across step limits, convergence fixpoints, cycles, and goals."""
        if step >= criteria.max_steps:
            return True, "max_steps_reached"

        if criteria.target_grid is not None and np.array_equal(current_state, criteria.target_grid):
            return True, "target_goal_reached"

        if previous_states:
            last_state = previous_states[-1]
            if current_state.shape == last_state.shape:
                diff_count = int(np.sum(current_state != last_state))
                total = current_state.size or 1
                delta_ratio = diff_count / total
                if delta_ratio <= criteria.convergence_delta_threshold:
                    return True, "fixpoint_converged"

            if criteria.detect_cycles:
                for past_idx, past_state in enumerate(previous_states[:-1]):
                    if current_state.shape == past_state.shape and np.array_equal(
                        current_state, past_state
                    ):
                        return True, f"cycle_detected_period_{step - past_idx}"

        return False, "continue"


@dataclass
class SimulationStoppingCriteria:
    """W089: Configurable termination conditions for multi-step forward simulation rollouts."""

    max_steps: int = 20
    convergence_delta_threshold: float = 0.0
    detect_cycles: bool = True
    target_grid: np.ndarray | None = None
