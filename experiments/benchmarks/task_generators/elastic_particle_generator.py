"""Task Generator for Milestone 3 Phase 3: Elastic Particle Dynamics & Angle-of-Incidence Deflection.

Generates rigorous inductive mechanism induction tasks across 4 kinematic families:
1. single_deflection_billiard: Single collision with barrier causing elastic reflection.
2. multi_rebound_box_reflection: Multiple rebounds off interior barriers and borders.
3. variable_barrier_elastic_bounce: Diverse grid sizes (8x8 to 14x14) with randomized barrier layouts.
4. absorbing_target_termination: Ray bounces off barriers until hitting an absorbing target marker.

AST Independence: Zero imports from HCIR solver or GridOperator internals.
"""

from __future__ import annotations

import random
from typing import ClassVar

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask


class ElasticParticleTaskGenerator:
    """Generates procedural challenge tasks for elastic particle ray kinematics induction."""

    FAMILIES: ClassVar[list[str]] = [
        "single_deflection_billiard",
        "multi_rebound_box_reflection",
        "variable_barrier_elastic_bounce",
        "absorbing_target_termination",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def _simulate_ray(
        grid: np.ndarray,
        emitter_coord: tuple[int, int],
        velocity: tuple[int, int],
        barrier_col: int,
        beam_col: int,
        absorbing_col: int | None = None,
        max_steps: int = 150,
    ) -> np.ndarray:
        """Ground-truth ray kinematics simulator."""
        out = grid.copy()
        H, W = out.shape
        r, c = emitter_coord
        dr, dc = velocity

        for _ in range(max_steps):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                break

            cell_val = int(out[nr, nc])
            if absorbing_col is not None and cell_val == absorbing_col:
                # Terminate before overwriting absorbing target
                break

            if cell_val == barrier_col:
                # 90-degree elastic deflection: reverse horizontal component
                dc = -dc
                refl_r, refl_c = r + dr, c + dc
                if 0 <= refl_r < H and 0 <= refl_c < W and out[refl_r, refl_c] != barrier_col:
                    r, c = refl_r, refl_c
                    if absorbing_col is not None and int(out[r, c]) == absorbing_col:
                        break
                    out[r, c] = beam_col
                else:
                    break
            else:
                r, c = nr, nc
                out[r, c] = beam_col

        return out

    def _generate_single_task_instance(
        self,
        H: int,
        W: int,
        emitter_col: int,
        barrier_col: int,
        beam_col: int,
        family: str,
        absorbing_col: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a single (input, output) pair with guaranteed non-empty ray."""
        for _ in range(100):
            grid = np.zeros((H, W), dtype=int)
            start_r = self.rng.randint(0, 2)
            start_c = self.rng.randint(0, max(1, W // 3))
            grid[start_r, start_c] = emitter_col

            vr, vc = 1, 1

            if family == "single_deflection_billiard":
                bar_r = self.rng.randint(3, H - 3)
                bar_c = start_c + (bar_r - start_r)
                if 0 <= bar_c < W:
                    grid[bar_r, bar_c] = barrier_col
                    out = self._simulate_ray(
                        grid, (start_r, start_c), (vr, vc), barrier_col, beam_col
                    )
                    if np.sum(out == beam_col) >= 4:
                        return grid, out

            elif family == "multi_rebound_box_reflection":
                # Place 2 barriers along the path
                bar1_r = self.rng.randint(2, 4)
                bar1_c = start_c + (bar1_r - start_r)
                if 0 <= bar1_c < W:
                    grid[bar1_r, bar1_c] = barrier_col
                    # After first bounce, vc is -1. Place second barrier along reflected path
                    bar2_r = bar1_r + self.rng.randint(2, 3)
                    bar2_c = bar1_c - (bar2_r - bar1_r)
                    if 0 <= bar2_r < H and 0 <= bar2_c < W:
                        grid[bar2_r, bar2_c] = barrier_col
                        out = self._simulate_ray(
                            grid, (start_r, start_c), (vr, vc), barrier_col, beam_col
                        )
                        if np.sum(out == beam_col) >= 6:
                            return grid, out

            elif family == "variable_barrier_elastic_bounce":
                bar_r = self.rng.randint(3, H - 3)
                bar_c = start_c + (bar_r - start_r)
                if 0 <= bar_c < W:
                    grid[bar_r, bar_c] = barrier_col
                    # Add an adjacent barrier pixel to form a 2-pixel surface
                    if bar_c + 1 < W:
                        grid[bar_r, bar_c + 1] = barrier_col
                    out = self._simulate_ray(
                        grid, (start_r, start_c), (vr, vc), barrier_col, beam_col
                    )
                    if np.sum(out == beam_col) >= 4:
                        return grid, out

            elif family == "absorbing_target_termination":
                bar_r = self.rng.randint(2, H - 4)
                bar_c = start_c + (bar_r - start_r)
                if 0 <= bar_c < W and absorbing_col is not None:
                    grid[bar_r, bar_c] = barrier_col
                    # Target located along reflected path
                    steps_after = self.rng.randint(2, 3)
                    tgt_r = bar_r + steps_after
                    tgt_c = bar_c - steps_after
                    if 0 <= tgt_r < H and 0 <= tgt_c < W:
                        grid[tgt_r, tgt_c] = absorbing_col
                        out = self._simulate_ray(
                            grid,
                            (start_r, start_c),
                            (vr, vc),
                            barrier_col,
                            beam_col,
                            absorbing_col=absorbing_col,
                        )
                        if np.sum(out == beam_col) >= 3:
                            return grid, out

        # Deterministic fallback
        grid = np.zeros((H, W), dtype=int)
        grid[0, 1] = emitter_col
        grid[4, 5] = barrier_col
        out = self._simulate_ray(grid, (0, 1), (1, 1), barrier_col, beam_col)
        return grid, out

    def generate_task(self, task_id: str, family: str) -> ManifestTask:
        """Generate single task instance with 3 train pairs and 1 test query."""
        H = self.rng.randint(8, 12)
        W = self.rng.randint(8, 12)

        palette = self.rng.sample([1, 2, 3, 5, 6, 7, 8], 4)
        emitter_col, barrier_col, beam_col, absorbing_col = palette

        train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(3):
            xi, yi = self._generate_single_task_instance(
                H, W, emitter_col, barrier_col, beam_col, family, absorbing_col=absorbing_col
            )
            train_pairs.append((xi, yi))

        if family == "variable_barrier_elastic_bounce":
            test_in, test_out = self._generate_single_task_instance(
                H + 2,
                W + 2,
                emitter_col,
                barrier_col,
                beam_col,
                family,
                absorbing_col=absorbing_col,
            )
        else:
            test_in, test_out = self._generate_single_task_instance(
                H, W, emitter_col, barrier_col, beam_col, family, absorbing_col=absorbing_col
            )

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
                "emitter_color": emitter_col,
                "barrier_color": barrier_col,
                "beam_color": beam_col,
                "absorbing_color": absorbing_col
                if family == "absorbing_target_termination"
                else None,
            },
        )

    @classmethod
    def generate_benchmark_suite(cls, seed: int = 42) -> list[ManifestTask]:
        """Generate 20 challenge tasks (5 tasks per family) for Milestone 3 Phase 3."""
        gen = cls(seed=seed)
        suite: list[ManifestTask] = []
        for fam in cls.FAMILIES:
            for idx in range(1, 6):
                tid = f"m3_p3_{fam}_{idx:02d}"
                suite.append(gen.generate_task(tid, fam))
        return suite
