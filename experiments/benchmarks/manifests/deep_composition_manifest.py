"""Deep Compositional Generalization and Exploratory Stress Test Manifest.

Separate, held-out evaluation manifests for high-order transformation pipelines:
1. Depth 3 Extended Evaluation Split (5 tasks):
   - comp_d3_crop_flip_recolor: Crop ∘ Flip H ∘ Recolor
   - comp_d3_crop_rot_trans: Crop ∘ Rotate 90 ∘ Translation
   - comp_d3_crop_scale_recolor: Crop ∘ Scale 2x ∘ Recolor
   - comp_d3_sym_rot_recolor: SymmetryCompletion ∘ Rotate 180 ∘ Recolor
   - comp_d3_crop_trans_recolor: Crop ∘ Translation Right ∘ Recolor

2. Depth 4 Exploratory Stress Test Split (2 tasks):
   - stress_d4_crop_rot_trans_recolor: Crop ∘ Rotate 90 ∘ Translate ∘ Recolor
   - stress_d4_crop_rot_sym_recolor: Crop ∘ Rotate 90 ∘ SymmetryCompletion ∘ Recolor
"""

from __future__ import annotations

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask


def get_depth3_extended_manifest() -> list[ManifestTask]:
    """Return 5 independently constructed depth-3 compositional tasks."""
    tasks: list[ManifestTask] = []

    # 1. Crop -> Flip H -> Recolor
    t1_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    t1_p1_y = np.array([[7, 6], [9, 8]])
    t1_test_x = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 0], [0, 0, 3, 4, 0], [0, 0, 0, 0, 0]])
    t1_test_y = np.array([[7, 6], [9, 8]])
    tasks.append(
        ManifestTask(
            "comp_d3_crop_flip_recolor",
            "COMPOSITIONAL",
            3,
            ((t1_p1_x, t1_p1_y),),
            t1_test_x,
            t1_test_y,
            {"stages": ["crop", "flip_h", "recolor"]},
        )
    )

    # 2. Crop -> Rotate 90 -> Translate Down
    t2_p1_x = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]]
    )
    t2_p1_y = np.array([[0, 0, 0], [7, 4, 1], [8, 5, 2]])
    t2_test_x = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0],
            [0, 0, 4, 5, 6, 0],
            [0, 0, 7, 8, 9, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    t2_test_y = np.array([[0, 0, 0], [7, 4, 1], [8, 5, 2]])
    tasks.append(
        ManifestTask(
            "comp_d3_crop_rot_trans",
            "COMPOSITIONAL",
            3,
            ((t2_p1_x, t2_p1_y),),
            t2_test_x,
            t2_test_y,
            {"stages": ["crop", "rot_90", "translate_down"]},
        )
    )

    # 3. Crop -> Scale 2x -> Recolor
    t3_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    t3_scaled = np.kron(np.array([[1, 2], [3, 4]]), np.ones((2, 2), dtype=int))
    lut3 = {1: 5, 2: 6, 3: 7, 4: 8}
    t3_p1_y = np.vectorize(lambda c: lut3.get(c, c))(t3_scaled)
    t3_test_x = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 0, 0], [0, 3, 4, 0, 0], [0, 0, 0, 0, 0]])
    t3_test_y = t3_p1_y.copy()
    tasks.append(
        ManifestTask(
            "comp_d3_crop_scale_recolor",
            "COMPOSITIONAL",
            3,
            ((t3_p1_x, t3_p1_y),),
            t3_test_x,
            t3_test_y,
            {"stages": ["crop", "scale_2x", "recolor"]},
        )
    )

    # 4. Symmetry Horizontal -> Rotate 180 -> Recolor (with 2 demos for Popperian refutation of spurious affine shortcuts)
    t4_p1_x = np.array([[1, 2], [0, 0]])
    t4_p1_y = np.array([[8, 7], [8, 7]])
    t4_p2_x = np.array([[3, 4], [0, 0]])
    t4_p2_y = np.array([[6, 5], [6, 5]])
    t4_test_x = np.array([[2, 1], [0, 0]])
    t4_test_y = np.array([[7, 8], [7, 8]])
    tasks.append(
        ManifestTask(
            "comp_d3_sym_rot_recolor",
            "COMPOSITIONAL",
            3,
            ((t4_p1_x, t4_p1_y), (t4_p2_x, t4_p2_y)),
            t4_test_x,
            t4_test_y,
            {"stages": ["sym_h", "rot_180", "recolor"]},
        )
    )

    # 5. Crop -> Translate Right -> Recolor
    t5_p1_x = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]]
    )
    t5_trans = np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]])
    lut5 = {0: 0, 1: 2, 2: 3, 4: 5, 5: 6, 7: 8, 8: 9}
    t5_p1_y = np.vectorize(lambda c: lut5.get(c, c))(t5_trans)
    t5_test_x = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0, 0],
            [0, 4, 5, 6, 0, 0],
            [0, 7, 8, 9, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    t5_test_y = t5_p1_y.copy()
    tasks.append(
        ManifestTask(
            "comp_d3_crop_trans_recolor",
            "COMPOSITIONAL",
            3,
            ((t5_p1_x, t5_p1_y),),
            t5_test_x,
            t5_test_y,
            {"stages": ["crop", "translate_right", "recolor"]},
        )
    )

    return tasks


def get_depth4_exploratory_manifest() -> list[ManifestTask]:
    """Return 2 exploratory depth-4 compositional stress-test tasks."""
    tasks: list[ManifestTask] = []

    # 1. Crop -> Rotate 90 -> Translate Down -> Recolor
    t1_p1_x = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]]
    )
    t1_trans = np.array([[0, 0, 0], [7, 4, 1], [8, 5, 2]])
    lut1 = {0: 0, 7: 8, 4: 5, 1: 2, 8: 9, 5: 6, 2: 3}
    t1_p1_y = np.vectorize(lambda c: lut1.get(c, c))(t1_trans)
    t1_test_x = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0],
            [0, 0, 4, 5, 6, 0],
            [0, 0, 7, 8, 9, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    t1_test_y = t1_p1_y.copy()
    tasks.append(
        ManifestTask(
            "stress_d4_crop_rot_trans_recolor",
            "COMPOSITIONAL",
            4,
            ((t1_p1_x, t1_p1_y),),
            t1_test_x,
            t1_test_y,
            {"stages": ["crop", "rot_90", "translate_down", "recolor"]},
        )
    )

    # 2. Crop -> Rotate 90 -> Symmetry Horizontal -> Recolor
    t2_p1_x = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    t2_p1_y = np.array([[5, 6], [5, 6]])
    t2_test_x = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 0, 0], [0, 3, 4, 0, 0], [0, 0, 0, 0, 0]])
    t2_test_y = t2_p1_y.copy()
    tasks.append(
        ManifestTask(
            "stress_d4_crop_rot_sym_recolor",
            "COMPOSITIONAL",
            4,
            ((t2_p1_x, t2_p1_y),),
            t2_test_x,
            t2_test_y,
            {"stages": ["crop", "rot_90", "sym_h", "recolor"]},
        )
    )

    return tasks
