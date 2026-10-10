"""Frozen 2D Discrete Transformation Task Manifest for Generalization & Ablation Evaluation.

Provides two strictly separated, immutable task splits:
1. Development Manifest (12 tasks): Smoke test and hyperparameter sanity checking.
2. Independent Evaluation Manifest (30 tasks): Stratified evaluation set across:
   - Geometric: Orthogonal rotations (90, 180, 270), flips (H, V), transpositions, translations, scaling.
   - Color: Palette permutations, binary inversions, multi-color substitution cycles.
   - Morphology: Bounding box non-bg crops, color-specific extractions, gravity shifts.
   - Symmetry: Horizontal, vertical, bilateral mirror completion, interior cavity fill.
   - Compositional: Depth-2 (Crop ∘ Rotate, Crop ∘ Recolor, Sym ∘ Recolor) and Depth-3 (Crop ∘ Rotate ∘ Recolor).
   - Transfer (W148): Source tasks, analogous target tasks with novel palettes, and negative-transfer distractor tasks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ManifestTask:
    """Frozen, immutable task specification for reproducible benchmarking."""

    task_id: str
    family: (
        str  # GEOMETRIC | COLOR | MORPHOLOGY | SYMMETRY | COMPOSITIONAL | DISAMBIGUATION | TRANSFER
    )
    depth: int
    train_pairs: tuple[tuple[np.ndarray, np.ndarray], ...]
    test_input: np.ndarray
    test_output: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute deterministic 16-character SHA-256 fingerprint of the task specification."""
        h = hashlib.sha256()
        h.update(self.task_id.encode())
        h.update(self.family.encode())
        h.update(str(self.depth).encode())
        for x, y in self.train_pairs:
            h.update(x.tobytes())
            h.update(y.tobytes())
        h.update(self.test_input.tobytes())
        h.update(self.test_output.tobytes())
        return h.hexdigest()[:16]


def get_development_manifest() -> list[ManifestTask]:
    """Return the 12-task development split for smoke testing and debugging."""
    manifest: list[ManifestTask] = []

    # 1. Depth 1: Pure Geometric Rotation 180
    t1_p1_x = np.array([[1, 2], [3, 4]])
    t1_p1_y = np.rot90(t1_p1_x, 2)
    t1_p2_x = np.array([[5, 6, 7], [8, 0, 1]])
    t1_p2_y = np.rot90(t1_p2_x, 2)
    t1_test_x = np.array([[2, 0], [1, 9]])
    t1_test_y = np.rot90(t1_test_x, 2)
    manifest.append(
        ManifestTask(
            task_id="dev_depth1_geo_rot180",
            family="GEOMETRIC",
            depth=1,
            train_pairs=((t1_p1_x, t1_p1_y), (t1_p2_x, t1_p2_y)),
            test_input=t1_test_x,
            test_output=t1_test_y,
            metadata={"expected_op": "ROT_180"},
        )
    )

    # 2. Depth 1: Pure Geometric Flip Vertical (Horizontal axis)
    t2_p1_x = np.array([[1, 2, 3], [0, 0, 0]])
    t2_p1_y = np.flipud(t2_p1_x)
    t2_p2_x = np.array([[4, 5], [6, 7], [8, 9]])
    t2_p2_y = np.flipud(t2_p2_x)
    t2_test_x = np.array([[9, 8], [7, 6]])
    t2_test_y = np.flipud(t2_test_x)
    manifest.append(
        ManifestTask(
            task_id="dev_depth1_geo_flip_v",
            family="GEOMETRIC",
            depth=1,
            train_pairs=((t2_p1_x, t2_p1_y), (t2_p2_x, t2_p2_y)),
            test_input=t2_test_x,
            test_output=t2_test_y,
            metadata={"expected_op": "FLIP_H"},
        )
    )

    # 3. Depth 1: Palette Substitution (Recoloring)
    t3_p1_x = np.array([[2, 3], [3, 2]])
    t3_p1_y = np.array([[7, 8], [8, 7]])
    t3_p2_x = np.array([[2, 0, 3], [0, 2, 0]])
    t3_p2_y = np.array([[7, 0, 8], [0, 7, 0]])
    t3_test_x = np.array([[3, 2, 3], [2, 0, 2]])
    t3_test_y = np.array([[8, 7, 8], [7, 0, 7]])
    manifest.append(
        ManifestTask(
            task_id="dev_depth1_color_permute",
            family="COLOR",
            depth=1,
            train_pairs=((t3_p1_x, t3_p1_y), (t3_p2_x, t3_p2_y)),
            test_input=t3_test_x,
            test_output=t3_test_y,
            metadata={"expected_mapping": {2: 7, 3: 8}},
        )
    )

    # 4. Depth 1: Bounding Box Foreground Extraction
    t4_p1_x = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    t4_p1_y = np.array([[5]])
    t4_p2_x = np.array([[0, 0, 0, 0], [0, 4, 4, 0], [0, 4, 4, 0], [0, 0, 0, 0]])
    t4_p2_y = np.array([[4, 4], [4, 4]])
    t4_test_x = np.array([[0, 0, 0, 0, 0], [0, 6, 6, 6, 0], [0, 6, 0, 6, 0], [0, 0, 0, 0, 0]])
    t4_test_y = np.array([[6, 6, 6], [6, 0, 6]])
    manifest.append(
        ManifestTask(
            task_id="dev_depth1_obj_crop",
            family="MORPHOLOGY",
            depth=1,
            train_pairs=((t4_p1_x, t4_p1_y), (t4_p2_x, t4_p2_y)),
            test_input=t4_test_x,
            test_output=t4_test_y,
            metadata={"expected_mode": "NON_BG_BBOX"},
        )
    )

    # 5. Depth 1: Symmetry Completion (Mirror Reflection)
    t5_p1_x = np.array([[1, 2, 0], [0, 0, 0]])
    t5_p1_y = np.array([[1, 2, 0], [1, 2, 0]])
    t5_p2_x = np.array([[3, 4, 5], [0, 0, 0]])
    t5_p2_y = np.array([[3, 4, 5], [3, 4, 5]])
    t5_test_x = np.array([[8, 9, 7], [0, 0, 0]])
    t5_test_y = np.array([[8, 9, 7], [8, 9, 7]])
    manifest.append(
        ManifestTask(
            task_id="dev_depth1_sym_mirror",
            family="SYMMETRY",
            depth=1,
            train_pairs=((t5_p1_x, t5_p1_y), (t5_p2_x, t5_p2_y)),
            test_input=t5_test_x,
            test_output=t5_test_y,
            metadata={"expected_mode": "HORIZONTAL"},
        )
    )

    # 6. Disambiguation
    t6_p1_x = np.array([[1, 0], [0, 1]])
    t6_p1_y = np.array([[1, 0], [0, 1]])
    t6_p2_x = np.array([[1, 2], [3, 4]])
    t6_p2_y = np.rot90(t6_p2_x, 2)
    t6_test_x = np.array([[5, 6, 7], [8, 9, 0]])
    t6_test_y = np.rot90(t6_test_x, 2)
    manifest.append(
        ManifestTask(
            task_id="dev_disambiguation_spurious_rejection",
            family="DISAMBIGUATION",
            depth=1,
            train_pairs=((t6_p1_x, t6_p1_y), (t6_p2_x, t6_p2_y)),
            test_input=t6_test_x,
            test_output=t6_test_y,
            metadata={"spurious_on_demo1": ["FLIP_H", "FLIP_V"], "true_rule": "ROT_180"},
        )
    )

    # 7. Depth 2: Crop -> Rotate 90
    t7_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    t7_p1_y = np.rot90(np.array([[1, 2], [3, 4]]), -1)
    t7_p2_x = np.array([[0, 0, 0], [0, 5, 6], [0, 7, 8]])
    t7_p2_y = np.rot90(np.array([[5, 6], [7, 8]]), -1)
    t7_test_x = np.array([[0, 0, 0, 0], [0, 2, 4, 0], [0, 6, 8, 0], [0, 0, 0, 0]])
    t7_test_y = np.rot90(np.array([[2, 4], [6, 8]]), -1)
    manifest.append(
        ManifestTask(
            task_id="dev_depth2_comp_crop_rot",
            family="COMPOSITIONAL",
            depth=2,
            train_pairs=((t7_p1_x, t7_p1_y), (t7_p2_x, t7_p2_y)),
            test_input=t7_test_x,
            test_output=t7_test_y,
            metadata={"stages": ["object_extract", "affine"]},
        )
    )

    # 8. Depth 2: Crop -> Recolor
    t8_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 1, 0], [0, 0, 0, 0]])
    t8_p1_y = np.array([[7, 8], [8, 7]])
    t8_p2_x = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]])
    t8_p2_y = np.array([[7, 7], [8, 8]])
    t8_test_x = np.array([[0, 0, 0, 0], [0, 2, 2, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    t8_test_y = np.array([[8, 8], [7, 7]])
    manifest.append(
        ManifestTask(
            task_id="dev_depth2_comp_crop_recolor",
            family="COMPOSITIONAL",
            depth=2,
            train_pairs=((t8_p1_x, t8_p1_y), (t8_p2_x, t8_p2_y)),
            test_input=t8_test_x,
            test_output=t8_test_y,
            metadata={"stages": ["object_extract", "recolor"]},
        )
    )

    # 9. Depth 3: Crop -> Rotate 90 -> Recolor
    t9_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    t9_p1_y = np.array([[7, 5], [8, 6]])
    t9_p2_x = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 0], [0, 0, 3, 4, 0], [0, 0, 0, 0, 0]])
    t9_p2_y = np.array([[7, 5], [8, 6]])
    t9_test_x = np.array([[0, 0, 0], [1, 2, 0], [3, 4, 0]])
    t9_test_y = np.array([[7, 5], [8, 6]])
    manifest.append(
        ManifestTask(
            task_id="dev_depth3_comp_crop_rot_recolor",
            family="COMPOSITIONAL",
            depth=3,
            train_pairs=((t9_p1_x, t9_p1_y), (t9_p2_x, t9_p2_y)),
            test_input=t9_test_x,
            test_output=t9_test_y,
            metadata={"stages": ["object_extract", "affine", "recolor"]},
        )
    )

    # 10. Transfer Source
    manifest.append(
        ManifestTask(
            task_id="dev_transfer_source_crop_rot",
            family="TRANSFER",
            depth=2,
            train_pairs=((t7_p1_x, t7_p1_y), (t7_p2_x, t7_p2_y)),
            test_input=t7_test_x,
            test_output=t7_test_y,
            metadata={"role": "SOURCE"},
        )
    )

    # 11. Transfer Target
    t11_p1_x = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 8, 9, 0],
            [0, 0, 7, 6, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    t11_p1_y = np.rot90(np.array([[8, 9], [7, 6]]), -1)
    t11_test_x = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 9, 8, 7, 0, 0],
            [0, 6, 5, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    t11_test_y = np.rot90(np.array([[9, 8, 7], [6, 5, 4]]), -1)
    manifest.append(
        ManifestTask(
            task_id="dev_transfer_target_novel_palette",
            family="TRANSFER",
            depth=2,
            train_pairs=((t11_p1_x, t11_p1_y),),
            test_input=t11_test_x,
            test_output=t11_test_y,
            metadata={"role": "TARGET_ANALOGOUS"},
        )
    )

    # 12. Negative Transfer
    t12_p1_x = np.array([[1, 2], [3, 4]])
    t12_p1_y = np.fliplr(t12_p1_x)
    t12_test_x = np.array([[5, 6], [7, 8]])
    t12_test_y = np.fliplr(t12_test_x)
    manifest.append(
        ManifestTask(
            task_id="dev_transfer_negative_conflict",
            family="TRANSFER",
            depth=1,
            train_pairs=((t12_p1_x, t12_p1_y),),
            test_input=t12_test_x,
            test_output=t12_test_y,
            metadata={"role": "NEGATIVE_CONFLICT"},
        )
    )

    return manifest


def get_evaluation_manifest() -> list[ManifestTask]:
    """Return 30 independently selected, stratified evaluation tasks across families and depths."""
    eval_tasks: list[ManifestTask] = []

    # ── GEOMETRIC FAMILY (8 tasks) ──────────────────────────────────────────

    # Task 1: Rot90 on 5x3 asymmetric matrix
    e1_p1_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 0], [2, 0, 3]])
    e1_p1_y = np.rot90(e1_p1_x, -1)
    e1_p2_x = np.array([[4, 0], [1, 2], [3, 5], [9, 8], [7, 6]])
    e1_p2_y = np.rot90(e1_p2_x, -1)
    e1_test_x = np.array([[9, 1, 2], [0, 3, 4], [5, 6, 0], [7, 0, 8], [1, 2, 3]])
    e1_test_y = np.rot90(e1_test_x, -1)
    eval_tasks.append(
        ManifestTask(
            "eval_geo_rot90_asym",
            "GEOMETRIC",
            1,
            ((e1_p1_x, e1_p1_y), (e1_p2_x, e1_p2_y)),
            e1_test_x,
            e1_test_y,
        )
    )

    # Task 2: Rot270 on 4x6 matrix
    e2_p1_x = np.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2], [3, 4, 5, 6, 7, 8], [9, 0, 1, 2, 3, 4]]
    )
    e2_p1_y = np.rot90(e2_p1_x, 1)
    e2_test_x = np.array(
        [[5, 4, 3, 2, 1, 0], [0, 9, 8, 7, 6, 5], [1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]]
    )
    e2_test_y = np.rot90(e2_test_x, 1)
    eval_tasks.append(
        ManifestTask(
            "eval_geo_rot270_rect", "GEOMETRIC", 1, ((e2_p1_x, e2_p1_y),), e2_test_x, e2_test_y
        )
    )

    # Task 3: Transpose (main diagonal) on asymmetric 4x4 matrix
    e3_p1_x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 2, 3], [4, 5, 6, 1]])
    e3_p1_y = e3_p1_x.T
    e3_test_x = np.array([[2, 4, 6, 8], [1, 3, 5, 7], [0, 2, 4, 6], [9, 7, 5, 3]])
    e3_test_y = e3_test_x.T
    eval_tasks.append(
        ManifestTask(
            "eval_geo_transpose_diag", "GEOMETRIC", 1, ((e3_p1_x, e3_p1_y),), e3_test_x, e3_test_y
        )
    )

    # Task 4: Anti-transpose on 4x4 matrix
    e4_p1_x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]])
    e4_p1_y = np.rot90(e4_p1_x.T, 2)
    e4_test_x = np.array([[8, 7, 6, 5], [4, 3, 2, 1], [0, 9, 8, 7], [6, 5, 4, 3]])
    e4_test_y = np.rot90(e4_test_x.T, 2)
    eval_tasks.append(
        ManifestTask(
            "eval_geo_anti_transpose", "GEOMETRIC", 1, ((e4_p1_x, e4_p1_y),), e4_test_x, e4_test_y
        )
    )

    # Task 5: Flip Horizontal across vertical axis (left-right) on 4x5
    e5_p1_x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    e5_p1_y = np.fliplr(e5_p1_x)
    e5_test_x = np.array([[9, 8, 7, 6, 5], [4, 3, 2, 1, 0], [1, 3, 5, 7, 9], [2, 4, 6, 8, 0]])
    e5_test_y = np.fliplr(e5_test_x)
    eval_tasks.append(
        ManifestTask(
            "eval_geo_flip_vertical_axis",
            "GEOMETRIC",
            1,
            ((e5_p1_x, e5_p1_y),),
            e5_test_x,
            e5_test_y,
        )
    )

    # Task 6: Rigid translation dr=+1, dc=0
    e6_p1_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e6_p1_y = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
    e6_test_x = np.array([[2, 4, 6], [1, 3, 5], [7, 8, 9]])
    e6_test_y = np.array([[0, 0, 0], [2, 4, 6], [1, 3, 5]])
    eval_tasks.append(
        ManifestTask(
            "eval_geo_translate_down", "GEOMETRIC", 1, ((e6_p1_x, e6_p1_y),), e6_test_x, e6_test_y
        )
    )

    # Task 7: Rigid translation dr=0, dc=+1
    e7_p1_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e7_p1_y = np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]])
    e7_test_x = np.array([[5, 6, 7], [8, 9, 1], [2, 3, 4]])
    e7_test_y = np.array([[0, 5, 6], [0, 8, 9], [0, 2, 3]])
    eval_tasks.append(
        ManifestTask(
            "eval_geo_translate_right", "GEOMETRIC", 1, ((e7_p1_x, e7_p1_y),), e7_test_x, e7_test_y
        )
    )

    # Task 8: Kronecker integer 2x scale
    e8_p1_x = np.array([[1, 2], [3, 4]])
    e8_p1_y = np.kron(e8_p1_x, np.ones((2, 2), dtype=int))
    e8_test_x = np.array([[5, 6], [7, 8]])
    e8_test_y = np.kron(e8_test_x, np.ones((2, 2), dtype=int))
    eval_tasks.append(
        ManifestTask(
            "eval_geo_scale_2x", "GEOMETRIC", 1, ((e8_p1_x, e8_p1_y),), e8_test_x, e8_test_y
        )
    )

    # ── COLOR FAMILY (5 tasks) ──────────────────────────────────────────────

    # Task 9: Color swap 1<->2 (asymmetric demonstration preventing accidental affine match)
    e9_p1_x = np.array([[1, 1, 2], [2, 1, 2]])
    e9_p1_y = np.array([[2, 2, 1], [1, 2, 1]])
    e9_test_x = np.array([[1, 1, 2], [2, 0, 1]])
    e9_test_y = np.array([[2, 2, 1], [1, 0, 2]])
    eval_tasks.append(
        ManifestTask(
            "eval_col_binary_swap", "COLOR", 1, ((e9_p1_x, e9_p1_y),), e9_test_x, e9_test_y
        )
    )

    # Task 10: 3-way color permutation 1->4, 2->5, 3->6
    e10_p1_x = np.array([[1, 2, 3], [3, 2, 1]])
    e10_p1_y = np.array([[4, 5, 6], [6, 5, 4]])
    e10_test_x = np.array([[2, 1], [3, 2], [1, 3]])
    e10_test_y = np.array([[5, 4], [6, 5], [4, 6]])
    eval_tasks.append(
        ManifestTask(
            "eval_col_3way_permute", "COLOR", 1, ((e10_p1_x, e10_p1_y),), e10_test_x, e10_test_y
        )
    )

    # Task 11: Background constant recoloring (single color mapping 7->9)
    e11_p1_x = np.array([[7, 0, 7], [0, 7, 0]])
    e11_p1_y = np.array([[9, 0, 9], [0, 9, 0]])
    e11_test_x = np.array([[0, 7, 7, 0], [7, 0, 0, 7]])
    e11_test_y = np.array([[0, 9, 9, 0], [9, 0, 0, 9]])
    eval_tasks.append(
        ManifestTask(
            "eval_col_single_target", "COLOR", 1, ((e11_p1_x, e11_p1_y),), e11_test_x, e11_test_y
        )
    )

    # Task 12: 4-color mapping
    e12_p1_x = np.array([[1, 2], [3, 4]])
    e12_p1_y = np.array([[6, 7], [8, 9]])
    e12_test_x = np.array([[4, 3, 2, 1], [1, 2, 3, 4]])
    e12_test_y = np.array([[9, 8, 7, 6], [6, 7, 8, 9]])
    eval_tasks.append(
        ManifestTask(
            "eval_col_4way_map", "COLOR", 1, ((e12_p1_x, e12_p1_y),), e12_test_x, e12_test_y
        )
    )

    # Task 13: Invert colors 8<->0 with other colors preserved
    e13_p1_x = np.array([[8, 1, 8], [1, 8, 1]])
    e13_p1_y = np.array([[3, 1, 3], [1, 3, 1]])
    e13_test_x = np.array([[1, 8, 1], [8, 1, 8]])
    e13_test_y = np.array([[1, 3, 1], [3, 1, 3]])
    eval_tasks.append(
        ManifestTask(
            "eval_col_selective_recolor",
            "COLOR",
            1,
            ((e13_p1_x, e13_p1_y),),
            e13_test_x,
            e13_test_y,
        )
    )

    # ── MORPHOLOGY & EXTRACTION (5 tasks) ───────────────────────────────────

    # Task 14: Center foreground non-bg crop from 7x7 to 3x3
    e14_p1_x = np.zeros((7, 7), dtype=int)
    e14_p1_x[2:5, 2:5] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e14_p1_y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e14_test_x = np.zeros((8, 8), dtype=int)
    e14_test_x[3:6, 2:5] = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    e14_test_y = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    eval_tasks.append(
        ManifestTask(
            "eval_morph_crop_center_3x3",
            "MORPHOLOGY",
            1,
            ((e14_p1_x, e14_p1_y),),
            e14_test_x,
            e14_test_y,
        )
    )

    # Task 15: Color-specific crop (extract color 4 bounding box from mixed grid)
    e15_p1_x = np.zeros((6, 6), dtype=int)
    e15_p1_x[1:3, 2:5] = 4
    e15_p1_y = np.array([[4, 4, 4], [4, 4, 4]])
    e15_test_x = np.zeros((7, 7), dtype=int)
    e15_test_x[3:5, 1:4] = 4
    e15_test_y = np.array([[4, 4, 4], [4, 4, 4]])
    eval_tasks.append(
        ManifestTask(
            "eval_morph_crop_color_4",
            "MORPHOLOGY",
            1,
            ((e15_p1_x, e15_p1_y),),
            e15_test_x,
            e15_test_y,
        )
    )

    # Task 16: Asymmetric corner crop from 6x6 down to 2x3
    e16_p1_x = np.zeros((6, 6), dtype=int)
    e16_p1_x[4:6, 3:6] = np.array([[2, 3, 4], [5, 6, 7]])
    e16_p1_y = np.array([[2, 3, 4], [5, 6, 7]])
    e16_test_x = np.zeros((8, 8), dtype=int)
    e16_test_x[5:7, 4:7] = np.array([[7, 8, 9], [1, 2, 3]])
    e16_test_y = np.array([[7, 8, 9], [1, 2, 3]])
    eval_tasks.append(
        ManifestTask(
            "eval_morph_crop_corner",
            "MORPHOLOGY",
            1,
            ((e16_p1_x, e16_p1_y),),
            e16_test_x,
            e16_test_y,
        )
    )

    # Task 17: Gravity shift DOWN
    e17_p1_x = np.array([[1, 0, 2], [0, 0, 0], [0, 3, 0]])
    e17_p1_y = np.array([[0, 0, 0], [0, 0, 0], [1, 3, 2]])
    e17_test_x = np.array([[0, 5, 0], [4, 0, 6], [0, 0, 0]])
    e17_test_y = np.array([[0, 0, 0], [0, 0, 0], [4, 5, 6]])
    eval_tasks.append(
        ManifestTask(
            "eval_morph_gravity_down",
            "MORPHOLOGY",
            1,
            ((e17_p1_x, e17_p1_y),),
            e17_test_x,
            e17_test_y,
        )
    )

    # Task 18: Gravity shift RIGHT
    e18_p1_x = np.array([[1, 0, 0], [0, 2, 0], [3, 0, 0]])
    e18_p1_y = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]])
    e18_test_x = np.array([[5, 0, 0], [0, 6, 0], [7, 0, 0]])
    e18_test_y = np.array([[0, 0, 5], [0, 0, 6], [0, 0, 7]])
    eval_tasks.append(
        ManifestTask(
            "eval_morph_gravity_right",
            "MORPHOLOGY",
            1,
            ((e18_p1_x, e18_p1_y),),
            e18_test_x,
            e18_test_y,
        )
    )

    # ── SYMMETRY & CONTAINMENT (4 tasks) ───────────────────────────────────

    # Task 19: Vertical mirror symmetry completion (right reflects left)
    e19_p1_x = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [5, 6, 0, 0]])
    e19_p1_y = np.array([[1, 2, 2, 1], [3, 4, 4, 3], [5, 6, 6, 5]])
    e19_test_x = np.array([[7, 8, 0, 0], [9, 1, 0, 0], [2, 3, 0, 0]])
    e19_test_y = np.array([[7, 8, 8, 7], [9, 1, 1, 9], [2, 3, 3, 2]])
    eval_tasks.append(
        ManifestTask(
            "eval_sym_mirror_vertical",
            "SYMMETRY",
            1,
            ((e19_p1_x, e19_p1_y),),
            e19_test_x,
            e19_test_y,
        )
    )

    # Task 20: Horizontal mirror symmetry completion (bottom reflects top)
    e20_p1_x = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
    e20_p1_y = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [1, 2, 3]])
    e20_test_x = np.array([[9, 8, 7], [6, 5, 4], [0, 0, 0], [0, 0, 0]])
    e20_test_y = np.array([[9, 8, 7], [6, 5, 4], [6, 5, 4], [9, 8, 7]])
    eval_tasks.append(
        ManifestTask(
            "eval_sym_mirror_horizontal",
            "SYMMETRY",
            1,
            ((e20_p1_x, e20_p1_y),),
            e20_test_x,
            e20_test_y,
        )
    )

    # Task 21: Bilateral 4-way symmetry completion
    e21_p1_x = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    e21_p1_y = np.array([[1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3], [1, 2, 2, 1]])
    e21_test_x = np.array([[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    e21_test_y = np.array([[5, 6, 6, 5], [7, 8, 8, 7], [7, 8, 8, 7], [5, 6, 6, 5]])
    eval_tasks.append(
        ManifestTask(
            "eval_sym_bilateral_4way",
            "SYMMETRY",
            1,
            ((e21_p1_x, e21_p1_y),),
            e21_test_x,
            e21_test_y,
        )
    )

    # Task 22: Enclosed interior hole fill (color 4)
    e22_p1_x = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    e22_p1_y = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 4, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    e22_test_x = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 2, 0, 0, 2, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    e22_test_y = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 2, 4, 4, 2, 0],
            [0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    eval_tasks.append(
        ManifestTask(
            "eval_sym_enclosed_cavity_fill",
            "SYMMETRY",
            1,
            ((e22_p1_x, e22_p1_y),),
            e22_test_x,
            e22_test_y,
        )
    )

    # ── COMPOSITIONAL FAMILY (5 tasks) ──────────────────────────────────────

    # Task 23: Depth 2: Crop -> Flip H
    e23_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    e23_p1_y = np.flipud(np.array([[1, 2], [3, 4]]))
    e23_test_x = np.array([[0, 0, 0, 0, 0], [0, 5, 6, 7, 0], [0, 8, 9, 1, 0], [0, 0, 0, 0, 0]])
    e23_test_y = np.flipud(np.array([[5, 6, 7], [8, 9, 1]]))
    eval_tasks.append(
        ManifestTask(
            "eval_comp_d2_crop_flip",
            "COMPOSITIONAL",
            2,
            ((e23_p1_x, e23_p1_y),),
            e23_test_x,
            e23_test_y,
        )
    )

    # Task 24: Depth 2: Crop -> Rotate 270
    e24_p1_x = np.array([[0, 0, 0, 0], [0, 2, 3, 0], [0, 4, 5, 0], [0, 0, 0, 0]])
    e24_p1_y = np.rot90(np.array([[2, 3], [4, 5]]), 1)
    e24_test_x = np.array([[0, 0, 0], [0, 7, 8], [0, 9, 1], [0, 0, 0]])
    e24_test_y = np.rot90(np.array([[7, 8], [9, 1]]), 1)
    eval_tasks.append(
        ManifestTask(
            "eval_comp_d2_crop_rot270",
            "COMPOSITIONAL",
            2,
            ((e24_p1_x, e24_p1_y),),
            e24_test_x,
            e24_test_y,
        )
    )

    # Task 25: Depth 2: Crop -> Recolor (3->7, 4->8)
    e25_p1_x = np.array([[0, 0, 0, 0], [0, 3, 4, 0], [0, 4, 3, 0], [0, 0, 0, 0]])
    e25_p1_y = np.array([[7, 8], [8, 7]])
    e25_test_x = np.array([[0, 0, 0], [3, 3, 0], [4, 4, 0]])
    e25_test_y = np.array([[7, 7], [8, 8]])
    eval_tasks.append(
        ManifestTask(
            "eval_comp_d2_crop_recolor",
            "COMPOSITIONAL",
            2,
            ((e25_p1_x, e25_p1_y),),
            e25_test_x,
            e25_test_y,
        )
    )

    # Task 26: Depth 2: Symmetry complete -> Recolor (1->5, 2->6)
    e26_p1_x = np.array([[1, 2, 0, 0], [2, 1, 0, 0]])
    e26_p1_y = np.array([[5, 6, 6, 5], [6, 5, 5, 6]])
    e26_test_x = np.array([[2, 2, 0, 0], [1, 1, 0, 0]])
    e26_test_y = np.array([[6, 6, 6, 6], [5, 5, 5, 5]])
    eval_tasks.append(
        ManifestTask(
            "eval_comp_d2_sym_recolor",
            "COMPOSITIONAL",
            2,
            ((e26_p1_x, e26_p1_y),),
            e26_test_x,
            e26_test_y,
        )
    )

    # Task 27: Depth 3: Crop -> Rotate 90 -> Recolor (1->7, 2->8, 3->9)
    # cropped: [[1, 2], [3, 1]] -> rot90: [[3, 1], [1, 2]] -> recolor: [[9, 7], [7, 8]]
    e27_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 1, 0], [0, 0, 0, 0]])
    e27_p1_y = np.array([[9, 7], [7, 8]])
    e27_test_x = np.array([[0, 0, 0, 0, 0], [0, 2, 3, 0, 0], [0, 1, 2, 0, 0], [0, 0, 0, 0, 0]])
    # cropped: [[2, 3], [1, 2]] -> rot90: [[1, 2], [2, 3]] -> recolor: [[7, 8], [8, 9]]
    e27_test_y = np.array([[7, 8], [8, 9]])
    eval_tasks.append(
        ManifestTask(
            "eval_comp_d3_crop_rot_recolor",
            "COMPOSITIONAL",
            3,
            ((e27_p1_x, e27_p1_y),),
            e27_test_x,
            e27_test_y,
        )
    )

    # ── TRANSFER & ANALOGY (3 tasks) ────────────────────────────────────────

    # Task 28: Source schema task (Crop -> Flip H)
    eval_tasks.append(
        ManifestTask(
            "eval_transfer_source_crop_flip",
            "TRANSFER",
            2,
            ((e23_p1_x, e23_p1_y),),
            e23_test_x,
            e23_test_y,
            {"role": "SOURCE"},
        )
    )

    # Task 29: Target analogous task with novel palette (shares Crop -> Flip H)
    e29_p1_x = np.array([[0, 0, 0, 0, 0], [0, 8, 9, 1, 0], [0, 7, 6, 2, 0], [0, 0, 0, 0, 0]])
    e29_p1_y = np.flipud(np.array([[8, 9, 1], [7, 6, 2]]))
    e29_test_x = np.array([[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 2, 0], [0, 0, 0, 0]])
    e29_test_y = np.flipud(np.array([[4, 5], [3, 2]]))
    eval_tasks.append(
        ManifestTask(
            "eval_transfer_target_novel",
            "TRANSFER",
            2,
            ((e29_p1_x, e29_p1_y),),
            e29_test_x,
            e29_test_y,
            {"role": "TARGET_ANALOGOUS"},
        )
    )

    # Task 30: Conflicting negative transfer task (Requires Rot180, conflicting with Crop->Flip prior)
    e30_p1_x = np.array([[1, 2], [3, 4]])
    e30_p1_y = np.rot90(e30_p1_x, 2)
    e30_test_x = np.array([[5, 6], [7, 8]])
    e30_test_y = np.rot90(e30_test_x, 2)
    eval_tasks.append(
        ManifestTask(
            "eval_transfer_negative_refute",
            "TRANSFER",
            1,
            ((e30_p1_x, e30_p1_y),),
            e30_test_x,
            e30_test_y,
            {"role": "NEGATIVE_CONFLICT"},
        )
    )

    return eval_tasks


def get_frozen_manifest() -> list[ManifestTask]:
    """Compatibility alias returning the development manifest split."""
    return get_development_manifest()


# =====================================================================
# Cryptographic Hashes & Provenance Ledger
# =====================================================================

FROZEN_DEVELOPMENT_HASHES: dict[str, str] = {
    "dev_depth1_geo_rot180": "7b20fc000866dbf8",
    "dev_depth1_geo_flip_v": "86b5fe7f156774ab",
    "dev_depth1_color_permute": "aebcc80074850293",
    "dev_depth1_obj_crop": "66dd0519e36c70d1",
    "dev_depth1_sym_mirror": "4fd7d1ed30629ca9",
    "dev_disambiguation_spurious_rejection": "96d494c1f9873e8c",
    "dev_depth2_comp_crop_rot": "f087cf84130982b5",
    "dev_depth2_comp_crop_recolor": "e528d52ca8d496b0",
    "dev_depth3_comp_crop_rot_recolor": "fe38a01387555d5a",
    "dev_transfer_source_crop_rot": "91cdc61de8fef2bd",
    "dev_transfer_target_novel_palette": "dd762de9c3a0520c",
    "dev_transfer_negative_conflict": "8b274ba2d7f73b4e",
}

FROZEN_EVALUATION_HASHES: dict[str, str] = {
    "eval_geo_rot90_asym": "33d9c43282bbee4d",
    "eval_geo_rot270_rect": "6afedb1d7bd9e76f",
    "eval_geo_transpose_diag": "6451d9a3172642ba",
    "eval_geo_anti_transpose": "6a1cbcd4b39f691d",
    "eval_geo_flip_vertical_axis": "3f429b826bab47c6",
    "eval_geo_translate_down": "04154896c29b9ade",
    "eval_geo_translate_right": "a00fbd70098b3614",
    "eval_geo_scale_2x": "be0e8bdd3f23bb9b",
    "eval_col_binary_swap": "160540902d475134",
    "eval_col_3way_permute": "1af98f695e6c7cc8",
    "eval_col_single_target": "a02bf8a3b096ecee",
    "eval_col_4way_map": "1d2194fb485be38b",
    "eval_col_selective_recolor": "0deccbc8a0133edc",
    "eval_morph_crop_center_3x3": "2dbe614105b9252b",
    "eval_morph_crop_color_4": "0c0e2995030fd044",
    "eval_morph_crop_corner": "940b0f538469cf5e",
    "eval_morph_gravity_down": "3554471997a5d4af",
    "eval_morph_gravity_right": "816db7d2ac55a01b",
    "eval_sym_mirror_vertical": "c20b6a5d307ce468",
    "eval_sym_mirror_horizontal": "5baac11401adec37",
    "eval_sym_bilateral_4way": "f6cbc7988972435f",
    "eval_sym_enclosed_cavity_fill": "aeb36572e3e3e48d",
    "eval_comp_d2_crop_flip": "c9c1a2055964dfb8",
    "eval_comp_d2_crop_rot270": "3ab317a8c3d92ea7",
    "eval_comp_d2_crop_recolor": "fa3232bfe91d442f",
    "eval_comp_d2_sym_recolor": "e58143bacd065b8b",
    "eval_comp_d3_crop_rot_recolor": "2229d617f8b274bb",
    "eval_transfer_source_crop_flip": "240c73465bf0ef88",
    "eval_transfer_target_novel": "d85fec36bb11688d",
    "eval_transfer_negative_refute": "470792085196bcd3",
}


def verify_manifest_integrity() -> tuple[bool, list[str]]:
    """Cryptographically verify that development and evaluation manifests match frozen SHA-256 hashes.

    Returns:
        (is_valid, list_of_mismatch_messages)
    """
    mismatches: list[str] = []

    dev_tasks = get_development_manifest()
    for t in dev_tasks:
        h = t.compute_hash()
        expected = FROZEN_DEVELOPMENT_HASHES.get(t.task_id)
        if expected is None:
            mismatches.append(f"Unregistered dev task: {t.task_id}")
        elif h != expected:
            mismatches.append(
                f"Hash drift in dev task {t.task_id}: computed {h} != expected {expected}"
            )

    eval_tasks = get_evaluation_manifest()
    for t in eval_tasks:
        h = t.compute_hash()
        expected = FROZEN_EVALUATION_HASHES.get(t.task_id)
        if expected is None:
            mismatches.append(f"Unregistered eval task: {t.task_id}")
        elif h != expected:
            mismatches.append(
                f"Hash drift in eval task {t.task_id}: computed {h} != expected {expected}"
            )

    return len(mismatches) == 0, mismatches
