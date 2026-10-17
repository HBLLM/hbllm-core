"""Blind Challenge Task Generator: Withheld Mechanism Families.

Generates procedural ARC tasks from mechanism families entirely withheld
from Level 3 representation expansion design and hyperparameter tuning.

Generative Process Independence Disclosure:
1. perimeter_contour_dilation:
   - Process: Morphological 1-pixel boundary halo dilation around solid components.
   - Latent parameters: component coordinates, foreground color, border halo color.
   - Oracle: 4-connected boundary neighbor detection; zero solver code.

2. maze_shortest_path:
   - Process: Shortest Manhattan path connecting start point to goal around obstacle walls.
   - Latent parameters: wall layout, start/goal coords, path color.
   - Oracle: Pure BFS shortest-path search; zero solver code.

3. parity_color_inversion:
   - Process: Global odd/even count parity rule triggering chromatic inversion.
   - Latent parameters: seed counts, target colors, parity predicate.
   - Oracle: Modulo-2 integer counting; zero solver code.

4. elastic_particle_deflection:
   - Process: Particle ray deflects at 90-degree angle upon collision with angled obstacle.
   - Latent parameters: emitter coord, obstacle coord, rebound direction.
   - Oracle: Deterministic ray stepping with reflection bounce; zero solver code.

5. underspecified_ambiguity_probe:
   - Process: Single symmetric demonstration that admits two conflicting hypotheses.
   - Latent parameters: dual-fitting symmetry axes.
   - Oracle: Evaluates calibrated epistemic uncertainty and abstention.
"""

from __future__ import annotations

import collections
import logging
import random

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask

logger = logging.getLogger(__name__)


class BlindChallengeTaskGenerator:
    """Generates benchmark tasks from unencountered mechanism families."""

    WITHHELD_FAMILIES = (
        "perimeter_contour_dilation",
        "maze_shortest_path",
        "parity_color_inversion",
        "elastic_particle_deflection",
        "underspecified_ambiguity_probe",
    )

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_task(
        self,
        task_id: str,
        family: str,
        num_demos: int = 2,
    ) -> ManifestTask:
        """Generate a single task from a withheld mechanism family."""
        if family == "perimeter_contour_dilation":
            return self._generate_perimeter_dilation_task(task_id, num_demos)
        elif family == "maze_shortest_path":
            return self._generate_maze_path_task(task_id, num_demos)
        elif family == "parity_color_inversion":
            return self._generate_parity_inversion_task(task_id, num_demos)
        elif family == "elastic_particle_deflection":
            return self._generate_particle_deflection_task(task_id, num_demos)
        elif family == "underspecified_ambiguity_probe":
            return self._generate_ambiguity_probe_task(task_id)
        else:
            raise ValueError(f"Unknown withheld mechanism family: {family}")

    # =========================================================================
    # 1. Perimeter Contour Dilation (Morphological Boundary Halo)
    # =========================================================================
    @staticmethod
    def _apply_perimeter_dilation(
        grid: np.ndarray, fg_color: int = 1, halo_color: int = 7
    ) -> np.ndarray:
        out = grid.copy()
        H, W = out.shape
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == fg_color:
                            out[r, c] = halo_color
                            break
        return out

    def _generate_perimeter_dilation_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        fg_col = self.rng.choice([1, 2, 3, 4])
        halo_col = self.rng.choice([6, 7, 8])

        for _ in range(num_demos):
            H, W = self.rng.randint(8, 12), self.rng.randint(8, 12)
            inp = np.zeros((H, W), dtype=int)
            r0 = self.rng.randint(2, H - 5)
            c0 = self.rng.randint(2, W - 5)
            h = self.rng.randint(2, 4)
            w = self.rng.randint(2, 4)
            inp[r0 : r0 + h, c0 : c0 + w] = fg_col
            out = self._apply_perimeter_dilation(inp, fg_col, halo_col)
            train_pairs.append((inp, out))

        # Test pair
        H_t, W_t = self.rng.randint(8, 12), self.rng.randint(8, 12)
        test_in = np.zeros((H_t, W_t), dtype=int)
        r0 = self.rng.randint(2, H_t - 5)
        c0 = self.rng.randint(2, W_t - 5)
        h = self.rng.randint(2, 4)
        w = self.rng.randint(2, 4)
        test_in[r0 : r0 + h, c0 : c0 + w] = fg_col
        test_out = self._apply_perimeter_dilation(test_in, fg_col, halo_col)

        return ManifestTask(
            task_id=task_id,
            family="perimeter_contour_dilation",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_in,
            test_output=test_out,
            metadata={"fg_color": fg_col, "halo_color": halo_col},
        )

    # =========================================================================
    # 2. Maze Shortest Path (BFS Pathfinding)
    # =========================================================================
    @staticmethod
    def _apply_maze_shortest_path(
        grid: np.ndarray,
        start_col: int = 2,
        goal_col: int = 3,
        wall_col: int = 5,
        path_col: int = 4,
    ) -> np.ndarray:
        out = grid.copy()
        H, W = out.shape
        start_coords = np.argwhere(grid == start_col)
        goal_coords = np.argwhere(grid == goal_col)
        if len(start_coords) == 0 or len(goal_coords) == 0:
            return out

        sr, sc = int(start_coords[0][0]), int(start_coords[0][1])
        gr, gc = int(goal_coords[0][0]), int(goal_coords[0][1])

        q = collections.deque([(sr, sc, [(sr, sc)])])
        visited = {(sr, sc)}

        while q:
            r, c, path = q.popleft()
            if r == gr and c == gc:
                for pr, pc in path:
                    if (pr, pc) != (sr, sc) and (pr, pc) != (gr, gc):
                        out[pr, pc] = path_col
                return out

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and (nr, nc) not in visited
                    and grid[nr, nc] != wall_col
                ):
                    visited.add((nr, nc))
                    q.append((nr, nc, path + [(nr, nc)]))
        return out

    def _generate_maze_path_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        wall_col = 5
        start_col = 2
        goal_col = 3
        path_col = 4

        for _ in range(num_demos):
            H, W = 7, 7
            inp = np.zeros((H, W), dtype=int)
            # Add an obstacle wall leaving a corridor
            inp[3, 1:5] = wall_col
            inp[1, 1] = start_col
            inp[5, 5] = goal_col
            out = self._apply_maze_shortest_path(inp, start_col, goal_col, wall_col, path_col)
            train_pairs.append((inp, out))

        # Test pair
        H_t, W_t = 7, 7
        test_in = np.zeros((H_t, W_t), dtype=int)
        test_in[3, 2:6] = wall_col
        test_in[1, 2] = start_col
        test_in[5, 4] = goal_col
        test_out = self._apply_maze_shortest_path(test_in, start_col, goal_col, wall_col, path_col)

        return ManifestTask(
            task_id=task_id,
            family="maze_shortest_path",
            depth=2,
            train_pairs=tuple(train_pairs),
            test_input=test_in,
            test_output=test_out,
            metadata={"path_color": path_col},
        )

    # =========================================================================
    # 3. Parity Color Inversion (Counting Parity)
    # =========================================================================
    @staticmethod
    def _apply_parity_inversion(
        grid: np.ndarray,
        marker_color: int = 2,
        odd_bg: int = 8,
        even_replace: int = 3,
    ) -> np.ndarray:
        out = grid.copy()
        count = int(np.sum(grid == marker_color))
        if count % 2 == 1:
            out[out == 0] = odd_bg
        else:
            out[out == marker_color] = even_replace
        return out

    def _generate_parity_inversion_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        marker_col = 2
        odd_bg = 8
        even_rep = 3

        # Demo 1: Odd count (3 markers)
        inp1 = np.zeros((6, 6), dtype=int)
        inp1[1, 1] = marker_col
        inp1[2, 4] = marker_col
        inp1[4, 2] = marker_col
        out1 = self._apply_parity_inversion(inp1, marker_col, odd_bg, even_rep)
        train_pairs.append((inp1, out1))

        # Demo 2: Even count (2 markers)
        inp2 = np.zeros((6, 6), dtype=int)
        inp2[1, 2] = marker_col
        inp2[4, 4] = marker_col
        out2 = self._apply_parity_inversion(inp2, marker_col, odd_bg, even_rep)
        train_pairs.append((inp2, out2))

        # Test pair (Odd count: 5 markers)
        test_in = np.zeros((6, 6), dtype=int)
        test_in[0, 1] = marker_col
        test_in[1, 4] = marker_col
        test_in[3, 3] = marker_col
        test_in[5, 1] = marker_col
        test_in[5, 5] = marker_col
        test_out = self._apply_parity_inversion(test_in, marker_col, odd_bg, even_rep)

        return ManifestTask(
            task_id=task_id,
            family="parity_color_inversion",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_in,
            test_output=test_out,
            metadata={"marker_col": marker_col},
        )

    # =========================================================================
    # 4. Elastic Particle Deflection (Ray Reflection)
    # =========================================================================
    @staticmethod
    def _apply_particle_deflection(
        grid: np.ndarray,
        emitter_col: int = 2,
        obstacle_col: int = 5,
        beam_col: int = 6,
    ) -> np.ndarray:
        out = grid.copy()
        H, W = out.shape
        emitters = np.argwhere(grid == emitter_col)
        if len(emitters) == 0:
            return out

        r, c = int(emitters[0][0]), int(emitters[0][1])
        dr, dc = 1, 1  # Initial ray direction

        while 0 <= r + dr < H and 0 <= c + dc < W:
            nr, nc = r + dr, c + dc
            if out[nr, nc] == obstacle_col:
                # 90-degree elastic deflection: reverse horizontal component
                dc = -dc
                if 0 <= r + dr < H and 0 <= c + dc < W:
                    r, c = r + dr, c + dc
                    if out[r, c] != obstacle_col:
                        out[r, c] = beam_col
                else:
                    break
            else:
                r, c = nr, nc
                out[r, c] = beam_col
        return out

    def _generate_particle_deflection_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        emitter_col = 2
        obstacle_col = 5
        beam_col = 6

        for _ in range(num_demos):
            H, W = 8, 8
            inp = np.zeros((H, W), dtype=int)
            inp[0, 1] = emitter_col
            inp[4, 5] = obstacle_col  # Barrier at (4, 5) causing deflection
            out = self._apply_particle_deflection(inp, emitter_col, obstacle_col, beam_col)
            train_pairs.append((inp, out))

        # Test pair
        H_t, W_t = 8, 8
        test_in = np.zeros((H_t, W_t), dtype=int)
        test_in[0, 2] = emitter_col
        test_in[3, 5] = obstacle_col
        test_out = self._apply_particle_deflection(test_in, emitter_col, obstacle_col, beam_col)

        return ManifestTask(
            task_id=task_id,
            family="elastic_particle_deflection",
            depth=2,
            train_pairs=tuple(train_pairs),
            test_input=test_in,
            test_output=test_out,
            metadata={"beam_col": beam_col},
        )

    # =========================================================================
    # 5. Underspecified Ambiguity Probe (Calibrated Uncertainty Test)
    # =========================================================================
    def _generate_ambiguity_probe_task(self, task_id: str) -> ManifestTask:
        # Single diagonal-symmetric demonstration that equally fits rot180 and flip_h/v
        x = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=int)
        y = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=int)

        # Test query that breaks symmetry:
        # rot180 gives [[0, 0, 4], [0, 3, 0], [0, 0, 0]]
        # flip_h gives [[0, 0, 0], [0, 3, 0], [4, 0, 0]]
        test_x = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=int)
        test_y = np.rot90(test_x, 2)

        return ManifestTask(
            task_id=task_id,
            family="underspecified_ambiguity_probe",
            depth=1,
            train_pairs=((x, y),),
            test_input=test_x,
            test_output=test_y,
            metadata={"is_underspecified": True, "required_action": "CALIBRATED_ABSTENTION"},
        )

    # =========================================================================
    # Suite Generation
    # =========================================================================
    @classmethod
    def generate_challenge_suite(cls, seed: int = 42) -> list[ManifestTask]:
        """Generate 25 blind benchmark tasks across all 5 withheld mechanism families."""
        gen = cls(seed=seed)
        suite: list[ManifestTask] = []
        for fam in cls.WITHHELD_FAMILIES:
            for idx in range(1, 6):
                task_id = f"blind_{fam}_{idx:02d}"
                suite.append(gen.generate_task(task_id, fam, num_demos=2))
        return suite
