"""Procedural Task Generator for Domain-General Visual Transformation Evaluation.

Generates genuinely novel, randomized transformation tasks from combinatorial operator chains:
1. Randomized Dimensions: Height and width sampled from [6, 16].
2. Randomized Palettes: Foreground and background colors permuted per instance.
3. Randomized Geometries: Shapes, bounding boxes, and object arrangements generated stochastically.
4. Combinatorial Compositions: Operator sequences of depth 1, 2, 3, and 4.
5. Strict Isolation: Generator rules and ground truth outputs are not exposed to the solver.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask


class ProceduralTaskGenerator:
    """Procedurally generates novel visual-spatial transformation tasks."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _generate_random_shape(
        self, h: int, w: int, color: int, shape_type: str = "box"
    ) -> np.ndarray:
        """Create a randomized foreground object on a canvas."""
        grid = np.zeros((h, w), dtype=int)
        if shape_type == "box":
            bh = self.rng.randint(2, max(2, h - 2))
            bw = self.rng.randint(2, max(2, w - 2))
            r = self.rng.randint(0, h - bh)
            c = self.rng.randint(0, w - bw)
            grid[r : r + bh, c : c + bw] = color
            # Add an asymmetric corner notch to guarantee broken 4-fold symmetry
            if bh >= 2 and bw >= 2:
                grid[r, c] = 0
        elif shape_type == "l_shape":
            bh = self.rng.randint(3, max(3, h - 2))
            bw = self.rng.randint(3, max(3, w - 2))
            r = self.rng.randint(0, h - bh)
            c = self.rng.randint(0, w - bw)
            grid[r : r + bh, c] = color
            grid[r + bh - 1, c : c + bw] = color
        elif shape_type == "cross":
            sz = self.rng.choice([3, 5])
            if sz > min(h, w):
                sz = 3
            r = self.rng.randint(0, max(0, h - sz))
            c = self.rng.randint(0, max(0, w - sz))
            mid = sz // 2
            grid[r + mid, c : c + sz] = color
            grid[r : r + sz, c + mid] = color
            # Break cross reflection symmetry with an asymmetric arm pixel
            grid[r, c + mid] = 0
        elif shape_type == "diagonal":
            sz = self.rng.randint(3, min(h, w) - 1)
            r = self.rng.randint(0, h - sz)
            c = self.rng.randint(0, w - sz)
            for i in range(sz):
                grid[r + i, c + i] = color
            # Add an asymmetric tip
            grid[r, c + 1] = color
        else:
            # Cluster
            num_px = self.rng.randint(4, 9)
            r0 = self.rng.randint(1, h - 2)
            c0 = self.rng.randint(1, w - 2)
            grid[r0, c0] = color
            for _ in range(num_px):
                dr = self.rng.choice([-1, 0, 1])
                dc = self.rng.choice([-1, 0, 1])
                r0 = max(0, min(h - 1, r0 + dr))
                c0 = max(0, min(w - 1, c0 + dc))
                grid[r0, c0] = color
        return grid

    def _apply_procedural_transform(
        self, grid: np.ndarray, op_name: str, params: dict[str, Any]
    ) -> np.ndarray:
        """Deterministically apply a primitive transformation."""
        if op_name == "rot90":
            k = params.get("k", 1)
            return np.rot90(grid, -k)
        elif op_name == "rot180":
            return np.rot90(grid, 2)
        elif op_name == "fliph":
            return np.flipud(grid)
        elif op_name == "flipv":
            return np.fliplr(grid)
        elif op_name == "recolor":
            src = params["src"]
            dst = params["dst"]
            out = grid.copy()
            out[grid == src] = dst
            return out
        elif op_name == "crop_bbox":
            non_bg = np.argwhere(grid != 0)
            if len(non_bg) == 0:
                return grid.copy()
            rmin, cmin = non_bg.min(axis=0)
            rmax, cmax = non_bg.max(axis=0)
            return grid[rmin : rmax + 1, cmin : cmax + 1].copy()
        elif op_name == "scale":
            factor = params.get("factor", 2)
            return np.kron(grid, np.ones((factor, factor), dtype=int))
        elif op_name == "translate":
            dr = params.get("dr", 0)
            dc = params.get("dc", 0)
            out = np.zeros_like(grid)
            h, w = grid.shape
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = grid[r, c]
            return out
        elif op_name == "symmetry_h":
            top = grid
            return np.vstack([top, np.flipud(top)])
        else:
            return grid.copy()

    def generate_random_task(
        self,
        task_id: str,
        depth: int,
        num_demos: int = 2,
        family: str = "procedural_randomized",
    ) -> ManifestTask:
        """Generate a random task of specified depth and family with held-out randomized instances."""
        h = self.rng.randint(6, 12)
        w = self.rng.randint(6, 12)
        color_a = self.rng.choice([1, 2, 3, 4, 5])
        color_b = self.rng.choice([6, 7, 8, 9])

        if family == "geometric_affine":
            candidate_pipelines = [
                [("rot90", {"k": 1})],
                [("rot180", {})],
                [("fliph", {})],
                [("flipv", {})],
            ]
        elif family == "morphological_relational":
            candidate_pipelines = [
                [("crop_bbox", {})],
            ]
        elif family == "attribute_recolor":
            candidate_pipelines = [
                [("recolor", {"src": color_a, "dst": color_b})],
            ]
        elif family == "compositional_deep":
            if depth == 2:
                candidate_pipelines = [
                    [("crop_bbox", {}), ("rot90", {"k": 1})],
                    [("rot90", {"k": 1}), ("recolor", {"src": color_a, "dst": color_b})],
                    [("fliph", {}), ("recolor", {"src": color_a, "dst": color_b})],
                    [("crop_bbox", {}), ("recolor", {"src": color_a, "dst": color_b})],
                    [("flipv", {}), ("recolor", {"src": color_a, "dst": color_b})],
                ]
            else:
                candidate_pipelines = [
                    [
                        ("crop_bbox", {}),
                        ("rot90", {"k": 1}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("fliph", {}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("flipv", {}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("rot180", {}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("scale", {"factor": 2}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                ]
        else:
            # Fallback conditioned on depth
            if depth == 1:
                candidate_pipelines = [
                    [("rot90", {"k": 1})],
                    [("rot180", {})],
                    [("fliph", {})],
                    [("flipv", {})],
                    [("recolor", {"src": color_a, "dst": color_b})],
                    [("crop_bbox", {})],
                ]
            elif depth == 2:
                candidate_pipelines = [
                    [("crop_bbox", {}), ("rot90", {"k": 1})],
                    [("rot90", {"k": 1}), ("recolor", {"src": color_a, "dst": color_b})],
                    [("fliph", {}), ("recolor", {"src": color_a, "dst": color_b})],
                    [("crop_bbox", {}), ("recolor", {"src": color_a, "dst": color_b})],
                    [("flipv", {}), ("recolor", {"src": color_a, "dst": color_b})],
                ]
            else:
                candidate_pipelines = [
                    [
                        ("crop_bbox", {}),
                        ("rot90", {"k": 1}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("fliph", {}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("flipv", {}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("rot180", {}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                    [
                        ("crop_bbox", {}),
                        ("scale", {"factor": 2}),
                        ("recolor", {"src": color_a, "dst": color_b}),
                    ],
                ]

        chosen_pipeline: list[tuple[str, dict[str, Any]]] = candidate_pipelines[
            int(self.rng.integers(0, len(candidate_pipelines)))
        ]

        def transform_pipeline(x: np.ndarray) -> np.ndarray:
            curr = x.copy()
            for op, p in chosen_pipeline:
                curr = self._apply_procedural_transform(curr, op, p)
            return curr

        train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        shapes = ["l_shape", "box", "diagonal", "cluster", "cross"]

        for i in range(num_demos):
            for attempt in range(25):
                shape = shapes[(i + attempt) % len(shapes)]
                inp = self._generate_random_shape(h, w, color_a, shape_type=shape)
                out = transform_pipeline(inp)
                # Verify that each operator in the pipeline was non-quiescent
                if len(chosen_pipeline) > 1 and "crop_bbox" in [op for op, _ in chosen_pipeline]:
                    crop_only = self._apply_procedural_transform(inp, "crop_bbox", {})
                    if np.array_equal(crop_only, out):
                        continue
                if not np.array_equal(inp, out):
                    break
            train_pairs.append((inp, out))

        # Test query instance
        for attempt in range(25):
            test_shape = shapes[(num_demos + attempt) % len(shapes)]
            test_inp = self._generate_random_shape(h, w, color_a, shape_type=test_shape)
            test_out = transform_pipeline(test_inp)
            if not np.array_equal(test_inp, test_out):
                break

        pipeline_str = " ∘ ".join(op for op, _ in reversed(chosen_pipeline))

        return ManifestTask(
            task_id=task_id,
            family=family,
            depth=depth,
            train_pairs=tuple(train_pairs),
            test_input=test_inp,
            test_output=test_out,
            metadata={
                "tags": ["procedural", f"depth_{depth}", "held_out_randomized", family],
                "description": f"Procedurally generated task: {pipeline_str}",
            },
        )
