"""Task Generator for Milestone 3 Phase 1: Global Modular Invariant Challenge.

Generates rigorous inductive mechanism induction tasks across 4 invariant families:
1. parity_color_inversion_shapes: Object-count parity with diverse geometric shapes.
2. parity_foreground_recolor: Pixel-count parity invariant.
3. modular_ternary_cycle: Modulo-3 ternary state cycle.
4. alternating_parity_reflection: Count-conditioned spatial reflection.

AST Independence: Zero imports from HCIR solver or GridOperator internals.
"""

from __future__ import annotations

import random
from typing import ClassVar

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask


class GlobalInvariantTaskGenerator:
    """Generates procedural challenge tasks for global modular invariant induction."""

    FAMILIES: ClassVar[list[str]] = [
        "parity_color_inversion_shapes",
        "parity_foreground_recolor",
        "modular_ternary_cycle",
        "alternating_parity_reflection",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def _draw_random_shape(
        self, grid: np.ndarray, top_r: int, left_c: int, color: int, shape_type: str
    ) -> None:
        H, W = grid.shape
        if shape_type == "single":
            if 0 <= top_r < H and 0 <= left_c < W:
                grid[top_r, left_c] = color
        elif shape_type == "cross":
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]:
                r, c = top_r + dr, left_c + dc
                if 0 <= r < H and 0 <= c < W:
                    grid[r, c] = color
        elif shape_type == "block":
            for dr in range(2):
                for dc in range(2):
                    r, c = top_r + dr, left_c + dc
                    if 0 <= r < H and 0 <= c < W:
                        grid[r, c] = color
        elif shape_type == "l_shape":
            for dr, dc in [(0, 0), (1, 0), (2, 0), (2, 1)]:
                r, c = top_r + dr, left_c + dc
                if 0 <= r < H and 0 <= c < W:
                    grid[r, c] = color

    def _generate_shapes_grid(self, H: int, W: int, count: int, color: int) -> np.ndarray:
        grid = np.zeros((H, W), dtype=int)
        shapes = ["single", "cross", "block", "l_shape"]
        placed = 0
        attempts = 0

        while placed < count and attempts < 100:
            attempts += 1
            shape = self.rng.choice(shapes)
            r = self.rng.randint(0, max(0, H - 4))
            c = self.rng.randint(0, max(0, W - 4))

            # Check overlap
            test_grid = np.copy(grid)
            self._draw_random_shape(test_grid, r, c, color, shape)
            if np.sum(test_grid != 0) > np.sum(grid != 0):
                grid = test_grid
                placed += 1

        # If placement fell short, add single pixels
        while placed < count:
            r = self.rng.randint(0, H - 1)
            c = self.rng.randint(0, W - 1)
            if grid[r, c] == 0:
                grid[r, c] = color
                placed += 1

        return grid

    def generate_task(self, task_id: str, family: str) -> ManifestTask:
        """Generate single task instance with 3 train pairs and 1 test pair."""
        H = self.rng.randint(10, 16)
        W = self.rng.randint(10, 16)

        c1 = self.rng.randint(1, 4)
        c2 = self.rng.randint(5, 8)
        c3 = 9

        train_pairs: list[tuple[np.ndarray, np.ndarray]] = []

        if family == "parity_color_inversion_shapes":
            # Rule: If count % 2 == 1 -> recolor to c2; if count % 2 == 0 -> identity (c1)
            # Ensure training demonstrations cover BOTH residues
            counts = [
                self.rng.choice([1, 3, 5]),  # Odd
                self.rng.choice([2, 4]),  # Even
                self.rng.choice([1, 2, 3, 4, 5]),
            ]
            self.rng.shuffle(counts)

            for cnt in counts:
                x_grid = self._generate_shapes_grid(H, W, cnt, c1)
                y_grid = np.copy(x_grid)
                if cnt % 2 == 1:
                    y_grid[x_grid == c1] = c2
                train_pairs.append((x_grid, y_grid))

            # Test query
            test_cnt = self.rng.choice([1, 2, 3, 4, 5])
            test_in = self._generate_shapes_grid(H, W, test_cnt, c1)
            test_out = np.copy(test_in)
            if test_cnt % 2 == 1:
                test_out[test_in == c1] = c2

        elif family == "parity_foreground_recolor":
            # Rule: If pixel count % 2 == 1 -> c2; if pixel count % 2 == 0 -> c1
            pixel_counts = [
                self.rng.choice([3, 5, 7]),  # Odd
                self.rng.choice([4, 6, 8]),  # Even
                self.rng.choice([3, 4, 5, 6, 7, 8]),
            ]
            self.rng.shuffle(pixel_counts)

            for p_cnt in pixel_counts:
                x_grid = np.zeros((H, W), dtype=int)
                placed = 0
                while placed < p_cnt:
                    r = self.rng.randint(0, H - 1)
                    c = self.rng.randint(0, W - 1)
                    if x_grid[r, c] == 0:
                        x_grid[r, c] = c1
                        placed += 1

                y_grid = np.copy(x_grid)
                target_col = c2 if (p_cnt % 2 == 1) else c1
                y_grid[x_grid != 0] = target_col
                train_pairs.append((x_grid, y_grid))

            test_p = self.rng.choice([3, 4, 5, 6, 7, 8])
            test_in = np.zeros((H, W), dtype=int)
            placed = 0
            while placed < test_p:
                r = self.rng.randint(0, H - 1)
                c = self.rng.randint(0, W - 1)
                if test_in[r, c] == 0:
                    test_in[r, c] = c1
                    placed += 1
            test_out = np.copy(test_in)
            target_col = c2 if (test_p % 2 == 1) else c1
            test_out[test_in != 0] = target_col

        elif family == "modular_ternary_cycle":
            # Rule: count % 3 == 0 -> c1 (identity); % 3 == 1 -> c2; % 3 == 2 -> c3
            counts_mod = [0, 1, 2]
            self.rng.shuffle(counts_mod)
            counts_3 = [c_mod + 3 * self.rng.choice([1, 2]) for c_mod in counts_mod]

            for cnt in counts_3:
                x_grid = self._generate_shapes_grid(H, W, cnt, c1)
                y_grid = np.copy(x_grid)
                res = cnt % 3
                if res == 1:
                    y_grid[x_grid == c1] = c2
                elif res == 2:
                    y_grid[x_grid == c1] = c3
                train_pairs.append((x_grid, y_grid))

            test_cnt = self.rng.randint(3, 9)
            test_in = self._generate_shapes_grid(H, W, test_cnt, c1)
            test_out = np.copy(test_in)
            res = test_cnt % 3
            if res == 1:
                test_out[test_in == c1] = c2
            elif res == 2:
                test_out[test_in == c1] = c3

        elif family == "alternating_parity_reflection":
            # Rule: count % 2 == 1 -> fliplr(x); count % 2 == 0 -> identity
            counts_refl = [
                self.rng.choice([1, 3, 5]),
                self.rng.choice([2, 4]),
                self.rng.choice([1, 2, 3, 4, 5]),
            ]
            self.rng.shuffle(counts_refl)

            for cnt in counts_refl:
                x_grid = self._generate_shapes_grid(H, W, cnt, c1)
                y_grid = np.fliplr(x_grid).copy() if (cnt % 2 == 1) else np.copy(x_grid)
                train_pairs.append((x_grid, y_grid))

            test_cnt = self.rng.choice([1, 2, 3, 4, 5])
            test_in = self._generate_shapes_grid(H, W, test_cnt, c1)
            test_out = np.fliplr(test_in).copy() if (test_cnt % 2 == 1) else np.copy(test_in)

        else:
            raise ValueError(f"Unknown family: {family}")

        return ManifestTask(
            task_id=task_id,
            family=family,
            depth=2,
            train_pairs=tuple(train_pairs),
            test_input=test_in,
            test_output=test_out,
            metadata={
                "grid_height": H,
                "grid_width": W,
                "num_demonstrations": len(train_pairs),
            },
        )

    @classmethod
    def generate_benchmark_suite(cls, seed: int = 42) -> list[ManifestTask]:
        """Generate 20 challenge tasks (5 tasks per family) for Milestone 3 Phase 1."""
        gen = cls(seed=seed)
        suite: list[ManifestTask] = []
        for fam in cls.FAMILIES:
            for idx in range(1, 6):
                tid = f"m3_p1_{fam}_{idx:02d}"
                suite.append(gen.generate_task(tid, fam))
        return suite
