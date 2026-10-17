"""Visual Symmetry Analysis for HCIR World Kernel.

Domain-agnostic geometric symmetry detection and pattern completion, migrated
from the ARC-AGI plugin to the core HCIR platform for universal reuse across
domains (robotics, browser, game environments).

Provides:
  - Multi-axis symmetry scoring (horizontal, vertical, diagonal, rotational)
  - Dominant symmetry identification
  - Asymmetric pattern completion via reflection
  - Dihedral D4 group structural equivalence
  - Arithmetic sequence & spatial permutation tracking
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class SymmetryAxis(StrEnum):
    """Geometric reflectional and rotational symmetry axes across 2D visual frames."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    MAIN_DIAGONAL = "main_diagonal"
    ANTI_DIAGONAL = "anti_diagonal"
    ROTATIONAL_90 = "rotational_90"
    ROTATIONAL_180 = "rotational_180"


class CountingRelationType(StrEnum):
    """Algebraic progression types for spatial object feature cardinalities."""

    CONSERVED = "conserved"
    ARITHMETIC_STEP = "arithmetic_step"
    GEOMETRIC_RATIO = "geometric_ratio"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    IRREGULAR = "irregular"


class SequenceProgressionType(StrEnum):
    """Monotonicity and ordering dynamics across numerical or spatial sequences."""

    STRICTLY_INCREASING = "strictly_increasing"
    STRICTLY_DECREASING = "strictly_decreasing"
    MONOTONIC_NON_DECREASING = "monotonic_non_decreasing"
    MONOTONIC_NON_INCREASING = "monotonic_non_increasing"
    INSUFFICIENT_LENGTH = "insufficient_length"
    NON_MONOTONIC = "non_monotonic"


class DihedralTransform(StrEnum):
    """D4 Dihedral group operations for 2D visual lattices."""

    IDENTITY = "identity"
    ROT_90 = "rot_90"
    ROT_180 = "rot_180"
    ROT_270 = "rot_270"
    FLIP_UD = "flip_ud"
    FLIP_LR = "flip_lr"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti_transpose"


@dataclass(frozen=True)
class DiscrepancyTarget:
    """A spatial coordinate and expected feature where a canvas violates dominant symmetry."""

    r: int
    c: int
    expected_feature: int
    confidence: float

    def to_tuple(self) -> tuple[int, int, int, float]:
        """Convert to legacy (r, c, feature, confidence) tuple."""
        return (self.r, self.c, self.expected_feature, self.confidence)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export discrepancy target as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"symmetry_discrepancy:{self.r}_{self.c}",
            operator="==",
            value=self.expected_feature,
            relation=f"conf:{self.confidence:.2f}",
            target_entity=f"({self.r},{self.c})",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording symmetry discrepancy."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"discrepancy_{self.r}_{self.c}",
                node_data={
                    "r": self.r,
                    "c": self.c,
                    "expected_feature": self.expected_feature,
                    "confidence": self.confidence,
                },
            )
        ]


@dataclass(frozen=True)
class SymmetryAnalysisResult:
    """Structured result of multi-axis visual symmetry analysis."""

    dominant_axis: SymmetryAxis
    dominant_score: float
    scores: dict[str, float]
    is_symmetric: bool

    def to_symbolic_predicate(self) -> CausalPredicate:
        """Export symmetry pattern as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="visual_symmetry",
            operator="==",
            value=self.dominant_axis.value,
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export symmetry analysis as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"symmetry_axis:{self.dominant_axis.value}",
            operator="==",
            value=self.dominant_score,
            relation=f"is_sym:{self.is_symmetric}",
            target_entity=self.dominant_axis.value,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording symmetry analysis."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"symmetry_{self.dominant_axis.value}",
                node_data={
                    "dominant_axis": self.dominant_axis.value,
                    "dominant_score": self.dominant_score,
                    "is_symmetric": self.is_symmetric,
                    "scores": self.scores,
                },
            )
        ]


@dataclass(frozen=True)
class PeriodicTilingResult:
    """Discovered fundamental 2D tile unit and spatial repetition schedule."""

    tile_shape: tuple[int, int]
    repetition_counts: tuple[int, int]
    confidence: float

    def to_symbolic_predicate(self) -> CausalPredicate:
        """Export periodic tiling structure as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="periodic_tiling",
            operator="==",
            value=self.tile_shape,
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export periodic tiling structure as a declarative causal predicate."""
        return CausalPredicate(
            variable="periodic_tiling_unit",
            operator="==",
            value=f"{self.tile_shape[0]}x{self.tile_shape[1]}",
            relation=f"reps:{self.repetition_counts[0]}x{self.repetition_counts[1]}",
            target_entity=f"conf:{self.confidence:.2f}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording periodic tiling."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"tiling_{self.tile_shape[0]}_{self.tile_shape[1]}",
                node_data={
                    "tile_shape": list(self.tile_shape),
                    "repetition_counts": list(self.repetition_counts),
                    "confidence": self.confidence,
                },
            )
        ]


@dataclass(frozen=True)
class CountingRelationResult:
    """Empirical algebraic relation across feature cardinalities."""

    is_relation: bool
    relation_type: CountingRelationType
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export counting relation as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"counting_relation:{self.relation_type.value}",
            operator="==",
            value=self.is_relation,
            relation=f"conf:{self.confidence:.2f}",
            target_entity=str(self.parameters),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording counting relation."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"counting_rel_{self.relation_type.value}",
                node_data={
                    "is_relation": self.is_relation,
                    "relation_type": self.relation_type.value,
                    "confidence": self.confidence,
                    "parameters": self.parameters,
                },
            )
        ]


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
        scores[SymmetryAxis.HORIZONTAL.value] = _calc_sym(h_flipped)

        # Vertical symmetry (reflection across vertical midline)
        v_flipped = np.fliplr(grid)
        scores[SymmetryAxis.VERTICAL.value] = _calc_sym(v_flipped)

        # Diagonal and rotational symmetries (square grids)
        if H == W:
            scores[SymmetryAxis.MAIN_DIAGONAL.value] = _calc_sym(grid.T)
            scores[SymmetryAxis.ANTI_DIAGONAL.value] = _calc_sym(np.flipud(np.fliplr(grid.T)))
            scores[SymmetryAxis.ROTATIONAL_90.value] = _calc_sym(np.rot90(grid, 1))
            scores[SymmetryAxis.ROTATIONAL_180.value] = _calc_sym(np.rot90(grid, 2))
        else:
            scores[SymmetryAxis.MAIN_DIAGONAL.value] = 0.0
            scores[SymmetryAxis.ANTI_DIAGONAL.value] = 0.0
            scores[SymmetryAxis.ROTATIONAL_90.value] = 0.0
            scores[SymmetryAxis.ROTATIONAL_180.value] = _calc_sym(np.flipud(np.fliplr(grid)))

        return scores

    @staticmethod
    def find_dominant_symmetry(grid: np.ndarray, background_color: int = 0) -> tuple[str, float]:
        """Identifies the symmetry axis with the highest matching score on foreground."""
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(
            grid, background_color=background_color
        )
        return max(scores.items(), key=lambda item: item[1])

    @classmethod
    def analyze_symmetry(
        cls,
        grid: np.ndarray,
        background_color: int = 0,
        threshold: float = 0.85,
    ) -> SymmetryAnalysisResult:
        """Perform comprehensive, typed symmetry evaluation."""
        scores = cls.compute_symmetry_scores(grid, background_color=background_color)
        best_axis_str, best_score = max(scores.items(), key=lambda item: item[1])
        axis_enum = SymmetryAxis(best_axis_str)
        return SymmetryAnalysisResult(
            dominant_axis=axis_enum,
            dominant_score=round(best_score, 4),
            scores=scores,
            is_symmetric=best_score >= threshold,
        )

    @staticmethod
    def predict_symmetric_completion(
        grid: np.ndarray,
        symmetry_type: str = "vertical",
        background_color: int = 0,
    ) -> np.ndarray:
        """Completes an incomplete or asymmetric pattern by reflecting the non-empty half."""
        completed = grid.copy()
        H, W = grid.shape

        if symmetry_type in (SymmetryAxis.VERTICAL.value, "vertical"):
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

        elif symmetry_type in (SymmetryAxis.HORIZONTAL.value, "horizontal"):
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

        elif symmetry_type in (SymmetryAxis.MAIN_DIAGONAL.value, "main_diagonal") and H == W:
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

        elif symmetry_type in (SymmetryAxis.ROTATIONAL_180.value, "rotational_180"):
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
        """Check if the grid is near-symmetric along any axis."""
        scores = VisualSymmetryAnalyzer.compute_symmetry_scores(
            grid, background_color=background_color
        )
        best_axis, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= threshold:
            return True, best_axis
        return False, None

    @staticmethod
    def extract_typed_discrepancy_targets(
        grid: np.ndarray,
        background_color: int = 0,
        threshold: float = 0.55,
    ) -> list[DiscrepancyTarget]:
        """Extract spatial coordinates and expected features where the grid violates dominant symmetry."""
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
        targets: list[DiscrepancyTarget] = []
        for p in pts:
            r, c = int(p[0]), int(p[1])
            expected_feat = int(completed[r, c])
            targets.append(
                DiscrepancyTarget(
                    r=r,
                    c=c,
                    expected_feature=expected_feat,
                    confidence=sym_score,
                )
            )
        return targets

    @staticmethod
    def extract_discrepancy_targets(
        grid: np.ndarray,
        background_color: int = 0,
        threshold: float = 0.55,
    ) -> list[tuple[int, int, int, float]]:
        """Legacy tuple compatibility interface for discrepancy targets."""
        targets = VisualSymmetryAnalyzer.extract_typed_discrepancy_targets(
            grid, background_color=background_color, threshold=threshold
        )
        return [t.to_tuple() for t in targets]

    @staticmethod
    def detect_periodicity(
        grid: np.ndarray,
        axis: int = 0,
        min_period: int = 1,
    ) -> tuple[int, float]:
        """W105: Detects spatial repetition periodicity along rows (axis=0) or columns (axis=1)."""
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

    @classmethod
    def analyze_tiling(
        cls,
        grid: np.ndarray,
    ) -> PeriodicTilingResult:
        """Discovers fundamental tile unit (h, w) and repetition schedule."""
        H, W = grid.shape
        pr, conf_r = cls.detect_periodicity(grid, axis=0)
        pc, conf_c = cls.detect_periodicity(grid, axis=1)

        tile_h = pr if pr > 0 else H
        tile_w = pc if pc > 0 else W
        reps_r = H // tile_h if tile_h > 0 else 1
        reps_c = W // tile_w if tile_w > 0 else 1

        overall_conf = (
            float((conf_r + conf_c) / 2.0) if (pr > 0 and pc > 0) else float(max(conf_r, conf_c))
        )
        return PeriodicTilingResult(
            tile_shape=(tile_h, tile_w),
            repetition_counts=(reps_r, reps_c),
            confidence=round(overall_conf, 4),
        )

    @staticmethod
    def detect_tiling(
        grid: np.ndarray,
    ) -> tuple[tuple[int, int], tuple[int, int], float]:
        """Legacy tuple compatibility interface for tiling discovery."""
        res = VisualSymmetryAnalyzer.analyze_tiling(grid)
        return res.tile_shape, res.repetition_counts, res.confidence

    @staticmethod
    def detect_counting_relation(
        counts: Sequence[int],
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """W107: Identifies arithmetic/algebraic counting relations (constant difference, scaling, conservation)."""
        if len(counts) < 2:
            return False, CountingRelationType.INSUFFICIENT_SAMPLES.value, 0.0, {}

        # 1. Conservation (all equal)
        if all(c == counts[0] for c in counts):
            return True, CountingRelationType.CONSERVED.value, 1.0, {"constant_value": counts[0]}

        # 2. Arithmetic progression
        diffs = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
        if all(d == diffs[0] for d in diffs):
            return True, CountingRelationType.ARITHMETIC_STEP.value, 1.0, {"step": diffs[0]}

        # 3. Geometric scaling ratio
        if all(counts[i] != 0 and counts[i + 1] % counts[i] == 0 for i in range(len(counts) - 1)):
            ratios = [counts[i + 1] // counts[i] for i in range(len(counts) - 1)]
            if all(r == ratios[0] for r in ratios):
                return True, CountingRelationType.GEOMETRIC_RATIO.value, 1.0, {"ratio": ratios[0]}

        return False, CountingRelationType.IRREGULAR.value, 0.0, {}

    @staticmethod
    def detect_sequence_progression(
        series: Sequence[float | int],
    ) -> tuple[bool, str, float]:
        """W108: Detects monotonic, arithmetic, or alternating progression in a numerical or spatial sequence."""
        if len(series) < 2:
            return False, SequenceProgressionType.INSUFFICIENT_LENGTH.value, 0.0

        is_increasing = all(series[i + 1] > series[i] for i in range(len(series) - 1))
        is_decreasing = all(series[i + 1] < series[i] for i in range(len(series) - 1))

        if is_increasing:
            return True, SequenceProgressionType.STRICTLY_INCREASING.value, 1.0
        if is_decreasing:
            return True, SequenceProgressionType.STRICTLY_DECREASING.value, 1.0

        # Monotonic non-decreasing
        if all(series[i + 1] >= series[i] for i in range(len(series) - 1)):
            return True, SequenceProgressionType.MONOTONIC_NON_DECREASING.value, 0.9
        if all(series[i + 1] <= series[i] for i in range(len(series) - 1)):
            return True, SequenceProgressionType.MONOTONIC_NON_INCREASING.value, 0.9

        return False, SequenceProgressionType.NON_MONOTONIC.value, 0.0

    @staticmethod
    def detect_permutation_order(
        seq_a: Sequence[int],
        seq_b: Sequence[int],
    ) -> tuple[bool, str]:
        """W109: Detects permutation relationships between two entity or color sequences."""
        list_a = list(seq_a)
        list_b = list(seq_b)
        if len(list_a) != len(list_b) or sorted(list_a) != sorted(list_b):
            return False, "not_a_permutation"

        if list_a == list_b:
            return True, "identity"

        if list_a == list(reversed(list_b)):
            return True, "reversed"

        # Cyclic shift
        n = len(list_a)
        doubled = list_a + list_a
        for shift in range(1, n):
            if doubled[shift : shift + n] == list_b:
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

        transforms: list[tuple[DihedralTransform, np.ndarray]] = [
            (DihedralTransform.IDENTITY, patch_a.copy()),
            (DihedralTransform.ROT_90, np.rot90(patch_a, 1)),
            (DihedralTransform.ROT_180, np.rot90(patch_a, 2)),
            (DihedralTransform.ROT_270, np.rot90(patch_a, 3)),
            (DihedralTransform.FLIP_UD, np.flipud(patch_a)),
            (DihedralTransform.FLIP_LR, np.fliplr(patch_a)),
            (DihedralTransform.TRANSPOSE, patch_a.T),
            (DihedralTransform.ANTI_TRANSPOSE, np.rot90(np.fliplr(patch_a), 1)),
        ]

        for op_enum, transformed in transforms:
            if transformed.shape == patch_b.shape and np.array_equal(transformed, patch_b):
                return True, op_enum.value

        return False, None

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export visual symmetry analyzer as a declarative causal predicate."""
        return CausalPredicate(
            variable="visual_symmetry_analyzer",
            operator="==",
            value="ACTIVE",
            relation="dihedral_d4_analysis",
            target_entity="visual_symmetry",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering visual symmetry analyzer."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="visual_symmetry_analyzer",
                node_data={
                    "faculty": "visual_symmetry",
                    "status": "ACTIVE",
                },
            )
        ]
