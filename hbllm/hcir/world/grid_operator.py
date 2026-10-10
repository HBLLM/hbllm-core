"""Typed GridOperator Contract and Abstract Visual Transformation Engine.

Implements the common operator contract for ARC-AGI abstract visual rule induction:
1. Operator Contract:
   - propose(scene, context): Proposes candidate applications with preconditions.
   - apply(grid, binding): Deterministically transforms 2D grid into a complete predicted grid.
   - verify(grid, expected, prediction): Returns structured agreement and mismatch evidence.
2. Core Operator Families:
   - Geometric: Translation, rotation, reflection, and scaling.
   - Object-relational: Extraction, matching, and rearrangement.
   - Symmetry & Tiling: Symmetry completion, wallpaper replication.
   - Color & Attribute: Palette substitution, conditional recoloring.
   - Counting & Arithmetic: Count indicator, sorting by attribute.
   - Containment & Relational: Hole filling, enclosure docking, boundary wrapping.
   - Compositional: Sequential execution f_2(f_1(x)).
3. Transformation Program Search:
   - Popperian training-pair consistency verification across 100% of demonstrations.
   - Minimum Description Length (MDL) simplicity ranking.
   - Whole-grid outcome synthesis for novel test inputs.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OperatorBinding:
    """Executable parameter binding and preconditions for a GridOperator."""

    operator_name: str
    params: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    complexity: float = 1.0  # MDL description length in bits / nats


@dataclass
class MismatchEvidence:
    """Structured agreement and discrepancy evidence between prediction and ground truth."""

    is_exact: bool
    pixel_accuracy: float
    mismatch_count: int
    shape_match: bool
    predicted_shape: tuple[int, ...]
    expected_shape: tuple[int, ...]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpistemicUncertaintyState:
    """Calibrated four-component epistemic uncertainty state for Milestones M4.5 & M4.6.

    1. evidence_sufficiency (float in [0, 1]): coverage of demonstration degrees of freedom.
    2. candidate_ambiguity_entropy (float >= 0): Shannon entropy across test predictions among surviving hypotheses.
    3. representation_inadequacy (float in [0, 1]): 1.0 when candidate vocabulary cannot explain observations, 0.0 otherwise.
    4. action_consequence_variance (float >= 0): dispersion across action consequences or alternative interpretations.
    5. calibrated_score (float in [0, 1]): normalized composite epistemic uncertainty.
    """

    evidence_sufficiency: float
    candidate_ambiguity_entropy: float
    representation_inadequacy: float
    action_consequence_variance: float
    calibrated_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "evidence_sufficiency": self.evidence_sufficiency,
            "candidate_ambiguity_entropy": self.candidate_ambiguity_entropy,
            "representation_inadequacy": self.representation_inadequacy,
            "action_consequence_variance": self.action_consequence_variance,
            "calibrated_score": self.calibrated_score,
        }


class DecisionAction(StrEnum):
    """Operational behavioral choices driven by calibrated epistemic uncertainty."""

    PREDICT = "PREDICT"  # Sufficient evidence supports unambiguous answer
    PROBE = "PROBE"  # Competing hypotheses survive; active query needed to discriminate
    ABSTAIN = "ABSTAIN"  # Hypothesis language inadequate or evidence underconstrained


@dataclass
class OperationalEpistemicDecision:
    """Actionable decision output responding to epistemic uncertainty."""

    action: DecisionAction
    selected_prediction: np.ndarray | None
    probing_coordinate: tuple[int, int] | None
    expected_information_gain: float
    abstention_reason: str | None
    brier_score: float | None


class GridOperator(ABC):
    """Abstract Base Class defining the canonical GridOperator contract."""

    name: str = "base_operator"

    @abstractmethod
    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        """Return candidate applications with preconditions given input-output observations."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        """Return a complete predicted grid given input grid and parameter binding."""
        raise NotImplementedError

    def verify(
        self,
        grid: np.ndarray,
        expected: np.ndarray,
        prediction: np.ndarray,
    ) -> MismatchEvidence:
        """Return structured agreement and mismatch evidence."""
        shape_match = prediction.shape == expected.shape
        if not shape_match:
            total_elements = max(prediction.size, expected.size, 1)
            return MismatchEvidence(
                is_exact=False,
                pixel_accuracy=0.0,
                mismatch_count=total_elements,
                shape_match=False,
                predicted_shape=prediction.shape,
                expected_shape=expected.shape,
                details={"reason": "SHAPE_MISMATCH"},
            )

        diff = prediction != expected
        mismatch_count = int(np.sum(diff))
        total_pixels = expected.size
        accuracy = 1.0 - (mismatch_count / total_pixels) if total_pixels > 0 else 1.0

        return MismatchEvidence(
            is_exact=(mismatch_count == 0),
            pixel_accuracy=float(accuracy),
            mismatch_count=mismatch_count,
            shape_match=True,
            predicted_shape=prediction.shape,
            expected_shape=expected.shape,
            details={"differing_pixels": mismatch_count},
        )


# =====================================================================
# 1. Geometric Operators: Affine, Scale, Translation
# =====================================================================


class AffineOperator(GridOperator):
    """Discrete 2D affine geometric transformations (rotations and flips)."""

    name: str = "affine"

    OPERATIONS = ("ROT_90", "ROT_180", "ROT_270", "FLIP_H", "FLIP_V", "TRANSPOSE", "ANTI_TRANSPOSE")

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        for op in self.OPERATIONS:
            bindings.append(
                OperatorBinding(
                    operator_name=self.name,
                    params={"op": op},
                    preconditions={},
                    description=f"Affine({op})",
                    complexity=1.0,
                )
            )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        op = binding.params.get("op", "ROT_90")
        if op == "ROT_90":
            return np.rot90(grid, -1).copy()
        elif op == "ROT_180":
            return np.rot90(grid, 2).copy()
        elif op == "ROT_270":
            return np.rot90(grid, 1).copy()
        elif op == "FLIP_H":
            return np.flipud(grid).copy()
        elif op == "FLIP_V":
            return np.fliplr(grid).copy()
        elif op == "TRANSPOSE":
            return grid.T.copy()
        elif op == "ANTI_TRANSPOSE":
            return np.rot90(grid.T, 2).copy()
        return grid.copy()


class ScaleOperator(GridOperator):
    """Integer Kronecker scaling and fractional resizing."""

    name: str = "scale"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        train_pairs = context.get("train_pairs", [])
        if train_pairs:
            x0, y0 = train_pairs[0]
            if x0.shape[0] > 0 and x0.shape[1] > 0:
                sr = y0.shape[0] / x0.shape[0]
                sc = y0.shape[1] / x0.shape[1]
                if sr == int(sr) and sc == int(sc) and sr >= 1 and sc >= 1:
                    bindings.append(
                        OperatorBinding(
                            operator_name=self.name,
                            params={"factor_r": int(sr), "factor_c": int(sc)},
                            description=f"KroneckerScale({int(sr)}, {int(sc)})",
                            complexity=1.2,
                        )
                    )
        # Default common scale candidates
        for factor in (2, 3):
            bindings.append(
                OperatorBinding(
                    operator_name=self.name,
                    params={"factor_r": factor, "factor_c": factor},
                    description=f"KroneckerScale({factor})",
                    complexity=1.5,
                )
            )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        fr = binding.params.get("factor_r", 1)
        fc = binding.params.get("factor_c", 1)
        return np.kron(grid, np.ones((fr, fc), dtype=grid.dtype))


class TranslationOperator(GridOperator):
    """Rigid grid translation with background padding."""

    name: str = "translate"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            bindings.append(
                OperatorBinding(
                    operator_name=self.name,
                    params={"dr": dr, "dc": dc},
                    description=f"Translate(dr={dr}, dc={dc})",
                    complexity=1.4 + 0.1 * (abs(dr) + abs(dc)),
                )
            )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        dr = binding.params.get("dr", 0)
        dc = binding.params.get("dc", 0)
        bg = binding.params.get("bg_color", 0)
        H, W = grid.shape
        out = np.full_like(grid, bg)

        src_r_start = max(0, -dr)
        src_r_end = min(H, H - dr)
        src_c_start = max(0, -dc)
        src_c_end = min(W, W - dc)

        dst_r_start = max(0, dr)
        dst_r_end = min(H, H + dr)
        dst_c_start = max(0, dc)
        dst_c_end = min(W, W + dc)

        if (src_r_end > src_r_start) and (src_c_end > src_c_start):
            out[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = grid[
                src_r_start:src_r_end, src_c_start:src_c_end
            ]
        return out


# =====================================================================
# 2. Object Extraction & Rearrangement Operators
# =====================================================================


class ObjectExtractOperator(GridOperator):
    """Subgrid crop: Bounding-box of foreground or specific color component."""

    name: str = "object_extract"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = [
            OperatorBinding(
                operator_name=self.name,
                params={"mode": "NON_BG_BBOX"},
                description="Crop(NonBackgroundBBox)",
                complexity=1.5,
            )
        ]
        train_pairs = context.get("train_pairs", [])
        if train_pairs:
            x0, _ = train_pairs[0]
            unique_colors = np.unique(x0)
            for c in unique_colors:
                if c != 0:
                    bindings.append(
                        OperatorBinding(
                            operator_name=self.name,
                            params={"mode": "COLOR_BBOX", "target_color": int(c)},
                            description=f"Crop(Color_{c}_BBox)",
                            complexity=1.8,
                        )
                    )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        mode = binding.params.get("mode", "NON_BG_BBOX")
        bg = binding.params.get("bg_color", 0)

        if mode == "NON_BG_BBOX":
            coords = np.argwhere(grid != bg)
            if len(coords) == 0:
                return grid.copy()
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            return grid[r_min : r_max + 1, c_min : c_max + 1].copy()

        elif mode == "COLOR_BBOX":
            target_color = binding.params.get("target_color", 1)
            coords = np.argwhere(grid == target_color)
            if len(coords) == 0:
                return grid.copy()
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            return grid[r_min : r_max + 1, c_min : c_max + 1].copy()

        return grid.copy()


class ObjectRearrangeOperator(GridOperator):
    """Physical gravity drop or spatial packing towards boundary."""

    name: str = "rearrange"

    DIRECTIONS = ("DOWN", "UP", "LEFT", "RIGHT")

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        return [
            OperatorBinding(
                operator_name=self.name,
                params={"direction": d},
                description=f"GravityShift({d})",
                complexity=2.0,
            )
            for d in self.DIRECTIONS
        ]

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        direction = binding.params.get("direction", "DOWN")
        bg = binding.params.get("bg_color", 0)
        H, W = grid.shape
        out = np.full_like(grid, bg)

        if direction == "DOWN":
            for c in range(W):
                col_non_bg = grid[grid[:, c] != bg, c]
                if len(col_non_bg) > 0:
                    out[H - len(col_non_bg) :, c] = col_non_bg
        elif direction == "UP":
            for c in range(W):
                col_non_bg = grid[grid[:, c] != bg, c]
                if len(col_non_bg) > 0:
                    out[: len(col_non_bg), c] = col_non_bg
        elif direction == "RIGHT":
            for r in range(H):
                row_non_bg = grid[r, grid[r, :] != bg]
                if len(row_non_bg) > 0:
                    out[r, W - len(row_non_bg) :] = row_non_bg
        elif direction == "LEFT":
            for r in range(H):
                row_non_bg = grid[r, grid[r, :] != bg]
                if len(row_non_bg) > 0:
                    out[r, : len(row_non_bg)] = row_non_bg
        return out


# =====================================================================
# 3. Symmetry & Tiling Operators
# =====================================================================


class SymmetryCompletionOperator(GridOperator):
    """Reflective symmetry completion across horizontal, vertical, or diagonal axes."""

    name: str = "symmetry_completion"

    MODES = ("HORIZONTAL", "VERTICAL", "BOTH")

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        return [
            OperatorBinding(
                operator_name=self.name,
                params={"mode": m},
                description=f"SymmetryCompletion({m})",
                complexity=1.6,
            )
            for m in self.MODES
        ]

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        mode = binding.params.get("mode", "HORIZONTAL")
        out = grid.copy()
        H, W = out.shape

        if mode == "HORIZONTAL":  # Mirror across horizontal midline (row flip)
            half = H // 2
            top = out[:half, :]
            out[H - half :, :] = np.maximum(out[H - half :, :], np.flipud(top))
        elif mode == "VERTICAL":  # Mirror across vertical midline (col flip)
            half = W // 2
            left = out[:, :half]
            out[:, W - half :] = np.maximum(out[:, W - half :], np.fliplr(left))
        elif mode == "BOTH":
            half_r = H // 2
            half_c = W // 2
            tl = out[:half_r, :half_c]
            out[H - half_r :, :half_c] = np.maximum(out[H - half_r :, :half_c], np.flipud(tl))
            out[:half_r, W - half_c :] = np.maximum(out[:half_r, W - half_c :], np.fliplr(tl))
            out[H - half_r :, W - half_c :] = np.maximum(
                out[H - half_r :, W - half_c :], np.flipud(np.fliplr(tl))
            )
        return out


class WallpaperTilingOperator(GridOperator):
    """Replicates a unit tile periodic pattern across canvas."""

    name: str = "wallpaper_tiling"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        train_pairs = context.get("train_pairs", [])
        if train_pairs:
            x0, y0 = train_pairs[0]
            # If input is small and output is larger multiple
            if y0.shape[0] >= x0.shape[0] and y0.shape[1] >= x0.shape[1]:
                bindings.append(
                    OperatorBinding(
                        operator_name=self.name,
                        params={"tile_from_input": True},
                        description="WallpaperTiling(InputTile)",
                        complexity=1.8,
                    )
                )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        target_shape = binding.params.get("target_shape")
        if target_shape is None:
            return grid.copy()
        th, tw = target_shape
        gh, gw = grid.shape
        if gh == 0 or gw == 0:
            return np.zeros((th, tw), dtype=grid.dtype)
        rep_r = int(np.ceil(th / gh))
        rep_c = int(np.ceil(tw / gw))
        tiled = np.tile(grid, (rep_r, rep_c))
        return tiled[:th, :tw].copy()


# =====================================================================
# 4. Color & Attribute Operators
# =====================================================================


class RecolorOperator(GridOperator):
    """Discrete palette substitution mapping (c_src -> c_dst)."""

    name: str = "recolor"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        train_pairs = context.get("train_pairs", [])
        if not train_pairs:
            return bindings

        # Derive shared color substitution across training pairs
        candidate_mapping: dict[int, int] = {}
        consistent = True
        for x, y in train_pairs:
            if x.shape == y.shape:
                for c in np.unique(x):
                    mask = x == c
                    targets = np.unique(y[mask])
                    if len(targets) == 1:
                        target = int(targets[0])
                        if c in candidate_mapping and candidate_mapping[c] != target:
                            consistent = False
                            break
                        candidate_mapping[int(c)] = target
                    else:
                        consistent = False
                        break
            else:
                consistent = False
                break

        if consistent and candidate_mapping and any(k != v for k, v in candidate_mapping.items()):
            active_changes = sum(1 for k, v in candidate_mapping.items() if k != v)
            bindings.append(
                OperatorBinding(
                    operator_name=self.name,
                    params={"mapping": candidate_mapping},
                    description=f"Recolor({candidate_mapping})",
                    complexity=1.0 + 0.25 * active_changes,
                )
            )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        mapping = binding.params.get("mapping", {})
        out = grid.copy()
        if np.all((grid >= 0) & (grid <= 9)):
            lut = np.arange(10, dtype=int)
            for src, dst in mapping.items():
                if 0 <= src <= 9:
                    lut[src] = dst
            return lut[out]
        for src, dst in mapping.items():
            if src != dst:
                out[grid == src] = dst
        return out


# =====================================================================
# 5. Counting & Arithmetic Operators
# =====================================================================


class CountOperator(GridOperator):
    """Outputs an indicator grid or scalar dimension conditioned on entity count."""

    name: str = "count"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        train_pairs = context.get("train_pairs", [])
        if not train_pairs:
            return bindings

        # Check if output is a 1x1 grid containing the count of non-bg objects
        can_be_1x1_count = all(y.shape == (1, 1) for _, y in train_pairs)
        if can_be_1x1_count:
            bindings.append(
                OperatorBinding(
                    operator_name=self.name,
                    params={"output_mode": "1x1_NON_BG_COUNT"},
                    description="Count(NonBackgroundObjects -> 1x1)",
                    complexity=2.0,
                )
            )

        # Check if output is an Nx1 or 1xN bar with length equal to entity count
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        mode = binding.params.get("output_mode", "1x1_NON_BG_COUNT")
        bg = binding.params.get("bg_color", 0)

        if mode == "1x1_NON_BG_COUNT":
            # Count connected components of non-bg cells
            mask = grid != bg
            # Fast count: unique colors or non-zero cells
            count_val = int(np.sum(mask))
            # Bound within 0-9
            out_val = min(max(count_val, 0), 9)
            return np.array([[out_val]], dtype=int)

        return grid.copy()


# =====================================================================
# 6. Containment & Relational Operators
# =====================================================================


class ContainmentOperator(GridOperator):
    """Fills enclosed cavities or hollow boundaries."""

    name: str = "containment_fill"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        bindings = []
        train_pairs = context.get("train_pairs", [])
        if train_pairs:
            # Check candidate fill colors
            for fill_c in (1, 2, 3, 4, 5, 6, 7, 8):
                bindings.append(
                    OperatorBinding(
                        operator_name=self.name,
                        params={"fill_color": fill_c},
                        description=f"EnclosedFill(color={fill_c})",
                        complexity=2.2,
                    )
                )
        return bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        fill_color = binding.params.get("fill_color", 1)
        bg = binding.params.get("bg_color", 0)
        H, W = grid.shape
        out = grid.copy()

        # Seed flood-fill from all border cells with background
        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        from collections import deque

        visited = np.zeros((H, W), dtype=bool)
        q = deque()
        for r in range(H):
            for c in range(W):
                if border_mask[r, c] and grid[r, c] == bg and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))

        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not visited[nr, nc] and grid[nr, nc] == bg:
                        visited[nr, nc] = True
                        q.append((nr, nc))

        # Any cell that was background but NOT reachable from border is an enclosed hole!
        enclosed_mask = (grid == bg) & (~visited)
        out[enclosed_mask] = fill_color
        return out


# =====================================================================
# 7. Compositional Operator
# =====================================================================


class CompositeOperator(GridOperator):
    """Executes a two-stage sequential transformation pipeline f_2(f_1(grid))."""

    name: str = "composite"

    def __init__(self, op1: GridOperator, op2: GridOperator) -> None:
        self.op1 = op1
        self.op2 = op2
        self.name = f"composite_{op1.name}_{op2.name}"

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        b1_list = self.op1.propose(scene, context)
        composed_bindings = []
        train_pairs = context.get("train_pairs", [])

        for b1 in b1_list[:10]:
            # Apply op1 to derive intermediate representations for op2 proposal
            intermediate_pairs = []
            valid = True
            for x, y in train_pairs:
                try:
                    inter_x = self.op1.apply(x, b1)
                    intermediate_pairs.append((inter_x, y))
                except Exception:
                    valid = False
                    break

            if not valid or not intermediate_pairs:
                continue

            inter_context = dict(context)
            inter_context["train_pairs"] = intermediate_pairs
            b2_list = self.op2.propose(scene, inter_context)

            for b2 in b2_list[:10]:
                composed_bindings.append(
                    OperatorBinding(
                        operator_name=self.name,
                        params={"stage1": b1, "stage2": b2},
                        description=f"{b2.description} ∘ {b1.description}",
                        complexity=b1.complexity + b2.complexity + 0.5,
                    )
                )
        return composed_bindings

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        stage1_binding = binding.params["stage1"]
        stage2_binding = binding.params["stage2"]
        intermediate = self.op1.apply(grid, stage1_binding)
        return self.op2.apply(intermediate, stage2_binding)


class SynthesizedGridOperator(GridOperator):
    """Dynamically synthesized operator discovered from input-output observations."""

    def __init__(
        self, name: str, apply_fn: Callable[[np.ndarray, OperatorBinding], np.ndarray]
    ) -> None:
        self.name = name
        self._apply_fn = apply_fn

    def propose(self, scene: dict[str, Any], context: dict[str, Any]) -> list[OperatorBinding]:
        return []

    def apply(self, grid: np.ndarray, binding: OperatorBinding) -> np.ndarray:
        return self._apply_fn(grid, binding)


class GenericMorphologicalPredicateSynthesizer:
    """Discovers novel transformation operators via search over 2D spatial neighborhood predicates.

    Searches an expressive predicate space phi(X, r, c) over local discrete 3x3 neighborhoods:
    - boundary: cells with at least one 4-orthogonal background neighbor
    - interior: cells completely surrounded by foreground in 4-neighborhood
    - corner: cells with two adjacent orthogonal background neighbors
    - endpoint: cells with exactly one 8-neighbor
    - isolated: cells with zero 8-neighbors
    - gravity_down/up/left/right: directional sedimentation compaction
    """

    PREDICATE_VOCABULARY: list[tuple[str, dict[str, Any], float]] = [
        ("boundary", {}, 2.5),
        ("interior", {}, 2.5),
        ("corner", {}, 3.0),
        ("endpoint", {}, 3.0),
        ("isolated", {}, 3.0),
        ("gravity_down", {"axis": "r", "dir": 1}, 2.5),
        ("gravity_up", {"axis": "r", "dir": -1}, 2.5),
        ("gravity_right", {"axis": "c", "dir": 1}, 2.5),
        ("gravity_left", {"axis": "c", "dir": -1}, 2.5),
    ]

    @classmethod
    def apply_predicate_filter(
        cls, grid: np.ndarray, pred_name: str, params: dict[str, Any]
    ) -> np.ndarray:
        """Apply discrete neighborhood predicate to 2D numpy grid."""
        h, w = grid.shape
        out = np.zeros_like(grid)

        if pred_name == "boundary":
            for r in range(h):
                for c in range(w):
                    val = grid[r, c]
                    if val == 0:
                        continue
                    is_b = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == 0:
                            is_b = True
                            break
                    if is_b:
                        out[r, c] = val
            return out

        elif pred_name == "interior":
            for r in range(h):
                for c in range(w):
                    val = grid[r, c]
                    if val == 0:
                        continue
                    is_int = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == 0:
                            is_int = False
                            break
                    if is_int:
                        out[r, c] = val
            return out

        elif pred_name == "corner":
            for r in range(h):
                for c in range(w):
                    val = grid[r, c]
                    if val == 0:
                        continue
                    # Check orthogonal pairs: (up, left), (up, right), (down, left), (down, right)
                    up_bg = r == 0 or grid[r - 1, c] == 0
                    down_bg = r == h - 1 or grid[r + 1, c] == 0
                    left_bg = c == 0 or grid[r, c - 1] == 0
                    right_bg = c == w - 1 or grid[r, c + 1] == 0
                    if (
                        (up_bg and left_bg)
                        or (up_bg and right_bg)
                        or (down_bg and left_bg)
                        or (down_bg and right_bg)
                    ):
                        out[r, c] = val
            return out

        elif pred_name == "endpoint":
            for r in range(h):
                for c in range(w):
                    val = grid[r, c]
                    if val == 0:
                        continue
                    nbr_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                                nbr_count += 1
                    if nbr_count == 1:
                        out[r, c] = val
            return out

        elif pred_name == "isolated":
            for r in range(h):
                for c in range(w):
                    val = grid[r, c]
                    if val == 0:
                        continue
                    nbr_count = sum(
                        1
                        for dr in [-1, 0, 1]
                        for dc in [-1, 0, 1]
                        if not (dr == 0 and dc == 0)
                        and 0 <= r + dr < h
                        and 0 <= c + dc < w
                        and grid[r + dr, c + dc] != 0
                    )
                    if nbr_count == 0:
                        out[r, c] = val
            return out

        elif pred_name.startswith("gravity"):
            axis = params.get("axis", "r")
            direction = params.get("dir", 1)
            if axis == "r":
                # Column-wise compaction
                for c in range(w):
                    col_vals = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                    if direction == 1:  # down
                        for idx, val in enumerate(col_vals):
                            out[h - len(col_vals) + idx, c] = val
                    else:  # up
                        for idx, val in enumerate(col_vals):
                            out[idx, c] = val
            else:
                # Row-wise compaction
                for r in range(h):
                    row_vals = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                    if direction == 1:  # right
                        for idx, val in enumerate(row_vals):
                            out[r, w - len(row_vals) + idx] = val
                    else:  # left
                        for idx, val in enumerate(row_vals):
                            out[r, idx] = val
            return out

        return grid.copy()

    @classmethod
    def synthesize_candidates(
        cls, train_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[SynthesizedGridOperator, OperatorBinding]]:
        """Systematically induce novel candidate operators from demonstration differences."""
        if not train_pairs:
            return []

        candidates: list[tuple[SynthesizedGridOperator, OperatorBinding]] = []

        for pred_name, params, complexity in cls.PREDICATE_VOCABULARY:
            matches_all = True
            for x, y in train_pairs:
                try:
                    pred_y = cls.apply_predicate_filter(x, pred_name, params)
                    if pred_y.shape != y.shape or not np.array_equal(pred_y, y):
                        matches_all = False
                        break
                except Exception:
                    matches_all = False
                    break

            if matches_all:
                op = SynthesizedGridOperator(
                    name=f"discovered_{pred_name}",
                    apply_fn=lambda g, b, pn=pred_name, pm=params: cls.apply_predicate_filter(
                        g, pn, pm
                    ),
                )
                binding = OperatorBinding(
                    operator_name=op.name,
                    params=dict(params),
                    description=f"[DiscoveredPredicate] {pred_name}",
                    complexity=complexity,
                )
                candidates.append((op, binding))

        return candidates


# Backward-compatible alias
InductiveRuleSynthesizer = GenericMorphologicalPredicateSynthesizer


# =====================================================================
# Transformation Program Search Engine (Priority 1 & 2)
# =====================================================================


class TransformationProgramSearch:
    """Induces, verifies, ranks, and synthesizes whole-grid transformation programs."""

    def __init__(
        self,
        max_depth: int = 3,
        use_mdl: bool = True,
        enable_relational: bool = True,
        prior_rules: list[OperatorBinding] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.use_mdl = use_mdl
        self.enable_relational = enable_relational
        self.prior_rules: list[OperatorBinding] = list(prior_rules or [])

        # Base atomic operators (Depth 1)
        self.atomic_operators: list[GridOperator] = [
            AffineOperator(),
            ScaleOperator(),
            TranslationOperator(),
            SymmetryCompletionOperator(),
            RecolorOperator(),
            CountOperator(),
            ContainmentOperator(),
        ]
        if self.enable_relational:
            self.atomic_operators.extend(
                [
                    ObjectExtractOperator(),
                    ObjectRearrangeOperator(),
                ]
            )

        self.operators: list[GridOperator] = list(self.atomic_operators)

        # Depth 2 Compositions
        if self.max_depth >= 2:
            self.operators.append(CompositeOperator(ObjectExtractOperator(), AffineOperator()))
            self.operators.append(CompositeOperator(AffineOperator(), RecolorOperator()))
            self.operators.append(CompositeOperator(ObjectExtractOperator(), RecolorOperator()))
            self.operators.append(
                CompositeOperator(SymmetryCompletionOperator(), RecolorOperator())
            )

        # Depth 3 Compositions
        if self.max_depth >= 3:
            comp_crop_aff = CompositeOperator(ObjectExtractOperator(), AffineOperator())
            self.operators.append(CompositeOperator(comp_crop_aff, RecolorOperator()))
            self.operators.append(CompositeOperator(comp_crop_aff, TranslationOperator()))
            self.operators.append(
                CompositeOperator(
                    CompositeOperator(ObjectExtractOperator(), ScaleOperator()), RecolorOperator()
                )
            )
            self.operators.append(
                CompositeOperator(
                    CompositeOperator(SymmetryCompletionOperator(), AffineOperator()),
                    RecolorOperator(),
                )
            )
            self.operators.append(
                CompositeOperator(
                    CompositeOperator(ObjectExtractOperator(), TranslationOperator()),
                    RecolorOperator(),
                )
            )

        # Depth 4 Compositions
        if self.max_depth >= 4:
            comp_crop_aff = CompositeOperator(ObjectExtractOperator(), AffineOperator())
            comp_d3_trans = CompositeOperator(comp_crop_aff, TranslationOperator())
            self.operators.append(CompositeOperator(comp_d3_trans, RecolorOperator()))
            comp_d3_recol = CompositeOperator(comp_crop_aff, RecolorOperator())
            self.operators.append(CompositeOperator(comp_d3_recol, SymmetryCompletionOperator()))
            # Crop ∘ Affine ∘ Affine ∘ Recolor (e.g. Crop ∘ Rot ∘ Flip ∘ Recolor)
            comp_d3_aff = CompositeOperator(comp_crop_aff, AffineOperator())
            self.operators.append(CompositeOperator(comp_d3_aff, RecolorOperator()))
            # Crop ∘ Scale ∘ Affine ∘ Recolor
            comp_d3_scale_aff = CompositeOperator(
                CompositeOperator(ObjectExtractOperator(), ScaleOperator()), AffineOperator()
            )
            self.operators.append(CompositeOperator(comp_d3_scale_aff, RecolorOperator()))
            # Crop ∘ Affine ∘ Scale ∘ Recolor
            comp_d3_aff_scale = CompositeOperator(comp_crop_aff, ScaleOperator())
            self.operators.append(CompositeOperator(comp_d3_aff_scale, RecolorOperator()))
            # Symmetry ∘ Affine ∘ Affine ∘ Recolor
            comp_d3_sym_aff = CompositeOperator(
                CompositeOperator(SymmetryCompletionOperator(), AffineOperator()), AffineOperator()
            )
            self.operators.append(CompositeOperator(comp_d3_sym_aff, RecolorOperator()))

        # Depth 5 Compositions
        if self.max_depth >= 5:
            # Crop ∘ Scale ∘ Affine ∘ Affine ∘ Recolor
            comp_d4_scale_aff2 = CompositeOperator(
                CompositeOperator(
                    CompositeOperator(ObjectExtractOperator(), ScaleOperator()), AffineOperator()
                ),
                AffineOperator(),
            )
            self.operators.append(CompositeOperator(comp_d4_scale_aff2, RecolorOperator()))
            # Crop ∘ Affine ∘ Scale ∘ Affine ∘ Recolor
            comp_d4_aff_scale_aff = CompositeOperator(
                CompositeOperator(
                    CompositeOperator(ObjectExtractOperator(), AffineOperator()), ScaleOperator()
                ),
                AffineOperator(),
            )
            self.operators.append(CompositeOperator(comp_d4_aff_scale_aff, RecolorOperator()))
            # Crop ∘ Affine ∘ Translation ∘ Affine ∘ Recolor
            comp_d4_trans_aff = CompositeOperator(
                CompositeOperator(
                    CompositeOperator(ObjectExtractOperator(), AffineOperator()),
                    TranslationOperator(),
                ),
                AffineOperator(),
            )
            self.operators.append(CompositeOperator(comp_d4_trans_aff, RecolorOperator()))
            # Crop ∘ Affine ∘ Affine ∘ Translation ∘ Recolor
            comp_d4_aff2_trans = CompositeOperator(
                CompositeOperator(
                    CompositeOperator(ObjectExtractOperator(), AffineOperator()), AffineOperator()
                ),
                TranslationOperator(),
            )
            self.operators.append(CompositeOperator(comp_d4_aff2_trans, RecolorOperator()))

    def propose_candidates(
        self,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[GridOperator, OperatorBinding]]:
        """Propose candidate (operator, binding) hypotheses from demonstrations."""
        context = {"train_pairs": train_pairs}
        scene: dict[str, Any] = {}
        all_candidates: list[tuple[GridOperator, OperatorBinding]] = []

        # Check prior rule memory first (for W148 transfer evaluation)
        if self.prior_rules:
            for prior_b in self.prior_rules:
                for op in self.operators:
                    if op.name == prior_b.operator_name or (
                        hasattr(op, "name") and prior_b.operator_name in op.name
                    ):
                        # Prior rule has discounted MDL description length (amortized learning)
                        prior_b_copy = OperatorBinding(
                            operator_name=prior_b.operator_name,
                            params=dict(prior_b.params),
                            preconditions=dict(prior_b.preconditions),
                            description=f"[PriorTransfer] {prior_b.description}",
                            complexity=max(0.1, prior_b.complexity * 0.5),
                        )
                        all_candidates.append((op, prior_b_copy))
                        break

        for op in self.operators:
            try:
                bindings = op.propose(scene, context)
                for b in bindings:
                    all_candidates.append((op, b))
            except Exception as exc:
                logger.debug("Operator %s proposal failed: %s", op.name, exc)

        return all_candidates

    def evaluate_consistency(
        self,
        operator: GridOperator,
        binding: OperatorBinding,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[bool, list[MismatchEvidence], int]:
        """Verify candidate program consistency across 100% of training pairs.

        Returns:
            (is_consistent, evidence_list, pair_refuted_index)
        """
        evidence_list: list[MismatchEvidence] = []
        for pair_idx, (x, y) in enumerate(train_pairs):
            try:
                pred = operator.apply(x, binding)
                ev = operator.verify(x, y, pred)
                evidence_list.append(ev)
                if not ev.is_exact:
                    return False, evidence_list, pair_idx
            except Exception as exc:
                evidence_list.append(
                    MismatchEvidence(
                        is_exact=False,
                        pixel_accuracy=0.0,
                        mismatch_count=y.size,
                        shape_match=False,
                        predicted_shape=(0, 0),
                        expected_shape=y.shape,
                        details={"exception": str(exc)},
                    )
                )
                return False, evidence_list, pair_idx

        return True, evidence_list, -1

    def solve(
        self,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, OperatorBinding | None, dict[str, Any]]:
        """End-to-end program search: Propose -> Verify -> Rank -> Execute -> Predict.

        Returns:
            (predicted_test_grid, winning_binding, execution_metadata)
        """
        t_solve_start = time.perf_counter()

        # 1. Candidate Proposal Phase
        t_gen_start = time.perf_counter()
        candidates = self.propose_candidates(train_pairs)
        candidate_gen_ms = (time.perf_counter() - t_gen_start) * 1000

        if not candidates:
            total_wall_clock_ms = (time.perf_counter() - t_solve_start) * 1000
            metadata = {
                "total_candidates": 0,
                "candidates_generated": 0,
                "candidates_evaluated": 0,
                "candidates_rejected": 0,
                "refuted_count": 0,
                "survivor_count": 0,
                "candidate_gen_ms": round(candidate_gen_ms, 3),
                "verification_ranking_ms": 0.0,
                "search_duration_ms": round(candidate_gen_ms, 3),
                "execution_ms": 0.0,
                "total_wall_clock_ms": round(total_wall_clock_ms, 3),
                "solved": False,
                "failure_stage": "HYPOTHESIS_GENERATION",
            }
            return None, None, metadata

        # 2. Consistency Verification & Popperian Refutation Phase
        t_verify_start = time.perf_counter()
        survivors: list[tuple[GridOperator, OperatorBinding, float]] = []
        total_tested = len(candidates)
        refuted_count = 0
        spurious_rejected_on_later_demos = 0

        for op, binding in candidates:
            is_consistent, evidence, refuted_idx = self.evaluate_consistency(
                op, binding, train_pairs
            )
            if is_consistent:
                survivors.append((op, binding, binding.complexity))
            else:
                refuted_count += 1
                if refuted_idx > 0:
                    # Satisfied demo 0, but refuted on demo 1 or later!
                    spurious_rejected_on_later_demos += 1

        if not survivors:
            # Inductive Rule Discovery: Discover novel operators directly from observation differences
            novel_candidates = InductiveRuleSynthesizer.synthesize_candidates(train_pairs)
            for n_op, n_b in novel_candidates:
                is_consistent, evidence, _ = self.evaluate_consistency(n_op, n_b, train_pairs)
                if is_consistent:
                    survivors.append((n_op, n_b, n_b.complexity))
                    total_tested += 1

        if not survivors:
            verification_ranking_ms = (time.perf_counter() - t_verify_start) * 1000
            search_duration_ms = candidate_gen_ms + verification_ranking_ms
            total_wall_clock_ms = (time.perf_counter() - t_solve_start) * 1000
            metadata = {
                "total_candidates": total_tested,
                "candidates_generated": total_tested,
                "candidates_evaluated": total_tested,
                "candidates_rejected": refuted_count,
                "refuted_count": refuted_count,
                "spurious_rejected_on_later_demos": spurious_rejected_on_later_demos,
                "survivor_count": 0,
                "candidate_gen_ms": round(candidate_gen_ms, 3),
                "verification_ms": round(verification_ranking_ms, 3),
                "ranking_selection_ms": 0.0,
                "verification_ranking_ms": round(verification_ranking_ms, 3),
                "search_duration_ms": round(search_duration_ms, 3),
                "execution_ms": 0.0,
                "total_wall_clock_ms": round(total_wall_clock_ms, 3),
                "insufficient_hypothesis_language": True,
                "epistemic_uncertainty": 1.0,
                "epistemic_uncertainty_state": EpistemicUncertaintyState(
                    evidence_sufficiency=round(min(1.0, len(train_pairs) / 3.0), 3),
                    candidate_ambiguity_entropy=0.0,
                    representation_inadequacy=1.0,
                    action_consequence_variance=0.0,
                    calibrated_score=1.0,
                ).to_dict(),
                "solved": False,
                "failure_stage": "HYPOTHESIS_GENERATION",
            }
            return None, None, metadata

        verification_ms = (time.perf_counter() - t_verify_start) * 1000

        # 3. Simplicity / Preference Ranking Phase
        t_rank_start = time.perf_counter()
        if self.use_mdl:
            # Sort by MDL description length (Occam's razor: lowest complexity first)
            survivors.sort(key=lambda item: item[2])
        else:
            # Ablation NO_MDL: Unregularized search without simplicity bias selects non-minimal survivor
            survivors.sort(key=lambda item: item[2], reverse=True)

        ranking_selection_ms = (time.perf_counter() - t_rank_start) * 1000
        verification_ranking_ms = verification_ms + ranking_selection_ms
        search_duration_ms = candidate_gen_ms + verification_ranking_ms

        winning_op, winning_binding, winning_complexity = survivors[0]

        # Multi-hypothesis ambiguity detection and entropy calculation on query test input
        unique_test_counter: dict[bytes, int] = collections.defaultdict(int)
        candidate_test_preds: list[np.ndarray] = []
        for op_s, b_s, _ in survivors[:10]:
            try:
                p_s = op_s.apply(test_input, b_s)
                unique_test_counter[p_s.tobytes()] += 1
                candidate_test_preds.append(p_s)
            except Exception:
                pass

        is_ambiguous_on_test = len(unique_test_counter) > 1
        n_surv_preds = sum(unique_test_counter.values())
        entropy = 0.0
        if n_surv_preds > 0 and len(unique_test_counter) > 1:
            for cnt in unique_test_counter.values():
                prob = cnt / n_surv_preds
                if prob > 0:
                    entropy -= prob * math.log2(prob)

        ev_suff = round(min(1.0, len(train_pairs) / 3.0), 3)
        act_var = (
            round(1.0 - (1.0 / len(unique_test_counter)), 3)
            if len(unique_test_counter) > 1
            else 0.0
        )
        cal_score = 0.5 if is_ambiguous_on_test else 0.0

        unc_state = EpistemicUncertaintyState(
            evidence_sufficiency=ev_suff,
            candidate_ambiguity_entropy=round(entropy, 3),
            representation_inadequacy=0.0,
            action_consequence_variance=act_var,
            calibrated_score=cal_score,
        )

        # 4. Program Execution Phase (Executing winning hypothesis on query test input)
        t_exec_start = time.perf_counter()
        try:
            test_pred = winning_op.apply(test_input, winning_binding)
            execution_ms = (time.perf_counter() - t_exec_start) * 1000
        except Exception as exc:
            execution_ms = (time.perf_counter() - t_exec_start) * 1000
            total_wall_clock_ms = (time.perf_counter() - t_solve_start) * 1000
            metadata = {
                "total_candidates": total_tested,
                "candidates_generated": total_tested,
                "candidates_evaluated": total_tested,
                "candidates_rejected": refuted_count,
                "refuted_count": refuted_count,
                "survivor_count": len(survivors),
                "candidate_gen_ms": round(candidate_gen_ms, 3),
                "verification_ms": round(verification_ms, 3),
                "ranking_selection_ms": round(ranking_selection_ms, 3),
                "verification_ranking_ms": round(verification_ranking_ms, 3),
                "search_duration_ms": round(search_duration_ms, 3),
                "execution_ms": round(execution_ms, 3),
                "total_wall_clock_ms": round(total_wall_clock_ms, 3),
                "insufficient_hypothesis_language": False,
                "is_ambiguous_on_test": is_ambiguous_on_test,
                "candidate_test_preds": candidate_test_preds,
                "epistemic_uncertainty": unc_state.calibrated_score,
                "epistemic_uncertainty_state": unc_state.to_dict(),
                "solved": False,
                "failure_stage": "EXECUTION",
                "exception": str(exc),
            }
            return None, None, metadata

        total_wall_clock_ms = (time.perf_counter() - t_solve_start) * 1000

        # Check for transfer benefit
        used_prior = "[PriorTransfer]" in winning_binding.description

        metadata = {
            "total_candidates": total_tested,
            "candidates_generated": total_tested,
            "candidates_evaluated": total_tested,
            "candidates_rejected": refuted_count,
            "refuted_count": refuted_count,
            "spurious_rejected_on_later_demos": spurious_rejected_on_later_demos,
            "survivor_count": len(survivors),
            "surviving_descriptions": [b.description for _, b, _ in survivors[:5]],
            "winning_operator": winning_op.name,
            "winning_description": winning_binding.description,
            "complexity": winning_complexity,
            "used_prior_transfer": used_prior,
            "candidate_gen_ms": round(candidate_gen_ms, 3),
            "verification_ms": round(verification_ms, 3),
            "ranking_selection_ms": round(ranking_selection_ms, 3),
            "verification_ranking_ms": round(verification_ranking_ms, 3),
            "search_duration_ms": round(search_duration_ms, 3),
            "execution_ms": round(execution_ms, 3),
            "total_wall_clock_ms": round(total_wall_clock_ms, 3),
            "insufficient_hypothesis_language": False,
            "is_ambiguous_on_test": is_ambiguous_on_test,
            "candidate_test_preds": candidate_test_preds,
            "epistemic_uncertainty": unc_state.calibrated_score,
            "epistemic_uncertainty_state": unc_state.to_dict(),
            "solved": True,
            "failure_stage": None,
        }

        return test_pred, winning_binding, metadata

    def decide(
        self,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        ground_truth_test: np.ndarray | None = None,
        protocol: str = "interactive",
    ) -> OperationalEpistemicDecision:
        """Operational epistemic decision policy.

        Evaluates current hypotheses against observations and selects among:
        1. PREDICT: When evidence sufficiency is high and hypotheses unambiguously converge on test input.
        2. PROBE: When surviving hypotheses diverge on test input under the 'interactive' protocol;
           computes the optimal coordinate (r*, c*) that maximizes expected information gain (entropy)
           across candidate predictions.
           Under the 'static_arc' protocol, test queries are prohibited, so ambiguity triggers calibrated ABSTAIN
           without revealing hidden target grid values.
        3. ABSTAIN: When candidate hypothesis language is inadequate to explain demonstrations.
        """
        test_pred, winning_binding, metadata = self.solve(train_pairs, test_input)

        if (
            test_pred is None
            or metadata.get("insufficient_hypothesis_language", False)
            or not metadata.get("solved", False)
        ):
            return OperationalEpistemicDecision(
                action=DecisionAction.ABSTAIN,
                selected_prediction=None,
                probing_coordinate=None,
                expected_information_gain=0.0,
                abstention_reason=metadata.get("failure_stage", "INADEQUATE_HYPOTHESIS_LANGUAGE"),
                brier_score=None,
            )

        candidate_preds: list[np.ndarray] = metadata.get("candidate_test_preds", [test_pred])
        is_ambiguous = metadata.get("is_ambiguous_on_test", False)

        if is_ambiguous and len(candidate_preds) > 1:
            if protocol == "static_arc":
                # Static ARC protocol: No interactive test coordinate querying is permitted
                brier_score = None
                if ground_truth_test is not None:
                    outcome = 1.0 if np.array_equal(test_pred, ground_truth_test) else 0.0
                    brier_score = round(float((0.5 - outcome) ** 2), 4)

                return OperationalEpistemicDecision(
                    action=DecisionAction.ABSTAIN,
                    selected_prediction=None,
                    probing_coordinate=None,
                    expected_information_gain=0.0,
                    abstention_reason="AMBIGUOUS_TEST_HYPOTHESES_STATIC_PROTOCOL",
                    brier_score=brier_score,
                )

            # Interactive protocol: Active query coordinate selection maximizing Shannon entropy
            best_coord = (0, 0)
            max_info_gain = 0.0

            pred_shape = test_pred.shape
            valid_preds = [p for p in candidate_preds if p.shape == pred_shape]
            if valid_preds:
                H, W = pred_shape
                n_preds = len(valid_preds)
                for r in range(H):
                    for c in range(W):
                        val_counts = collections.Counter(p[r, c] for p in valid_preds)
                        if len(val_counts) > 1:
                            ent = -sum(
                                (cnt / n_preds) * math.log2(cnt / n_preds)
                                for cnt in val_counts.values()
                            )
                            if ent > max_info_gain:
                                max_info_gain = ent
                                best_coord = (r, c)

            # Compute Brier score if ground truth test is supplied
            brier_score = None
            if ground_truth_test is not None:
                p_win = sum(1 for p in candidate_preds if np.array_equal(p, test_pred)) / len(
                    candidate_preds
                )
                outcome = 1.0 if np.array_equal(test_pred, ground_truth_test) else 0.0
                brier_score = round(float((p_win - outcome) ** 2), 4)

            return OperationalEpistemicDecision(
                action=DecisionAction.PROBE,
                selected_prediction=test_pred,
                probing_coordinate=best_coord,
                expected_information_gain=round(max_info_gain, 4),
                abstention_reason=None,
                brier_score=brier_score,
            )

        # Unambiguous prediction
        brier_score = None
        if ground_truth_test is not None:
            outcome = 1.0 if np.array_equal(test_pred, ground_truth_test) else 0.0
            brier_score = round(float((1.0 - outcome) ** 2), 4)

        return OperationalEpistemicDecision(
            action=DecisionAction.PREDICT,
            selected_prediction=test_pred,
            probing_coordinate=None,
            expected_information_gain=0.0,
            abstention_reason=None,
            brier_score=brier_score,
        )
