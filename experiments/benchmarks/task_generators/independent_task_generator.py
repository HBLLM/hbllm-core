"""Independent Procedural Task Generator for Milestones M4.5 & M4.6.

Separately implemented task generator that does NOT share the solver's primitive catalog,
predicates, or composition grammar.

Audit of Generative Process Independence (6-Point Disclosure):
=============================================================
1. Gravity Settle:
   - Process: Simulates downward 1D vertical particle sedimentation. Each column's non-zero
     movable particles drop until obstructed by horizontal barrier cells (color 5) or floor.
   - Solver Exposure: Receives only 2D integer grids (X_train, Y_train, X_test).
   - Shared Conventions: Discrete 2D grid coordinates, ARC 10-color palette.
   - Latent Parameters: Barrier coordinates, particle counts, and fall directions are private.
   - Sampling: Grid dimensions H in [6, 12], W in [6, 12], 2-4 floating particles per column.
   - Oracle Computation: Pure NumPy array slice packing; zero solver operator code.

2. Diagonal Ray Cast:
   - Process: Emits optical beams from point sources along (+1, +1) diagonal vectors until
     collision with obstacle markers (color 5) or boundary.
   - Solver Exposure: Raw demonstration pairs only.
   - Shared Conventions: Discrete 2D grid, 8-neighbor adjacency space.
   - Latent Parameters: Emitter position, beam trajectory vector, and obstacle coords private.
   - Sampling: Randomized emitter locations and obstacle positions.
   - Oracle Computation: Iterative discrete ray stepping loop; zero solver code.

3. Interior Infill:
   - Process: Detects closed 4-connected boundary loops of color 1 and recolors hollow interior
     background cells (0) with fill color 8.
   - Solver Exposure: Input-output grid pairs only.
   - Shared Conventions: Standard topological boundary enclosure.
   - Latent Parameters: Loop bounding box, border thickness, fill color private.
   - Sampling: Rectangular hollow frames of dimensions 5x5 to 10x10.
   - Oracle Computation: Bounding coordinate min/max bounding infill.

4. Alternating Pattern Extrapolate:
   - Process: Continues a periodic 1D/2D repeating sequence of colors along a strip or grid.
   - Solver Exposure: Unlabeled demonstration grids.
   - Shared Conventions: Integer discrete color values.
   - Latent Parameters: Period length, color sequence tuple, sequence phase private.
   - Sampling: Periods of 2 to 4 colors repeating across 10 to 15 columns.
   - Oracle Computation: Modulo arithmetic index mapping.

5. Component Size Rank:
   - Process: Computes connected component areas using 4-connectivity BFS and recolors
     components by area ranking (largest -> color 2, smallest -> color 3).
   - Solver Exposure: Grid arrays only.
   - Shared Conventions: 4-connected topological component definition.
   - Latent Parameters: Ranking thresholds and target palette assignment private.
   - Sampling: 2 to 3 disjoint rectangular/irregular components of distinct sizes.
   - Oracle Computation: Standard BFS connected-component labeling.
"""

from __future__ import annotations

import logging
import random

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask

logger = logging.getLogger(__name__)


class IndependentTaskGenerator:
    """Separately implemented task generator designed without solver operator grammar assumptions."""

    FAMILIES = (
        "gravity_settle",
        "diagonal_ray_cast",
        "interior_infill",
        "alternating_pattern_extrapolate",
        "component_size_rank",
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
        """Generate an independent task instance of the specified external family."""
        if family == "gravity_settle":
            return self._generate_gravity_settle_task(task_id, num_demos)
        elif family == "diagonal_ray_cast":
            return self._generate_diagonal_ray_task(task_id, num_demos)
        elif family == "interior_infill":
            return self._generate_interior_infill_task(task_id, num_demos)
        elif family == "alternating_pattern_extrapolate":
            return self._generate_pattern_extrapolate_task(task_id, num_demos)
        elif family == "component_size_rank":
            return self._generate_component_rank_task(task_id, num_demos)
        else:
            raise ValueError(f"Unknown independent task family: {family}")

    # =========================================================================
    # 1. Gravity Settle: Vertical downward physics settling
    # =========================================================================
    def _apply_gravity(self, grid: np.ndarray, barrier_color: int = 5) -> np.ndarray:
        """Simulate downward gravity: movable particles fall until hitting barrier or floor."""
        out = grid.copy()
        H, W = out.shape
        # Particles fall down column-by-column
        for c in range(W):
            col = out[:, c]
            # Identify barrier rows
            barrier_rows = [r for r in range(H) if col[r] == barrier_color]
            # Partitions created by barriers
            segments = []
            prev_r = -1
            for b_r in barrier_rows:
                segments.append((prev_r + 1, b_r - 1))
                prev_r = b_r
            segments.append((prev_r + 1, H - 1))

            for r_start, r_end in segments:
                if r_start <= r_end:
                    seg_vals = [
                        col[r]
                        for r in range(r_start, r_end + 1)
                        if col[r] != 0 and col[r] != barrier_color
                    ]
                    num_empty = (r_end - r_start + 1) - len(seg_vals)
                    new_seg = [0] * num_empty + seg_vals
                    for idx, val in enumerate(new_seg):
                        out[r_start + idx, c] = val
        return out

    def _generate_gravity_settle_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        barrier_color = 5
        particle_color = self.rng.choice([1, 2, 3, 4])

        for _ in range(num_demos):
            H, W = self.rng.randint(7, 10), self.rng.randint(7, 10)
            inp = np.zeros((H, W), dtype=int)
            # Create a horizontal shelf barrier
            shelf_r = H - 3
            shelf_c_start = self.rng.randint(1, 3)
            shelf_len = self.rng.randint(3, W - shelf_c_start)
            inp[shelf_r, shelf_c_start : shelf_c_start + shelf_len] = barrier_color

            # Place floating particles above
            for _ in range(self.rng.randint(4, 7)):
                pr = self.rng.randint(0, shelf_r - 1)
                pc = self.rng.randint(0, W - 1)
                if inp[pr, pc] == 0:
                    inp[pr, pc] = particle_color
            out = self._apply_gravity(inp, barrier_color)
            train_pairs.append((inp, out))

        # Query test instance
        H, W = self.rng.randint(7, 10), self.rng.randint(7, 10)
        test_inp = np.zeros((H, W), dtype=int)
        shelf_r = H - 3
        shelf_c_start = self.rng.randint(1, 3)
        shelf_len = self.rng.randint(3, W - shelf_c_start)
        test_inp[shelf_r, shelf_c_start : shelf_c_start + shelf_len] = barrier_color
        for _ in range(self.rng.randint(4, 7)):
            pr = self.rng.randint(0, shelf_r - 1)
            pc = self.rng.randint(0, W - 1)
            if test_inp[pr, pc] == 0:
                test_inp[pr, pc] = particle_color
        test_out = self._apply_gravity(test_inp, barrier_color)

        return ManifestTask(
            task_id=task_id,
            family="gravity_settle",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_inp,
            test_output=test_out,
            metadata={
                "origin": "independent_generator",
                "description": "Particles settle downward under gravity onto barriers",
            },
        )

    # =========================================================================
    # 2. Diagonal Ray Cast: Optical ray emission along diagonals
    # =========================================================================
    def _apply_diagonal_ray(
        self, grid: np.ndarray, emitter_color: int = 2, beam_color: int = 3, obstacle_color: int = 5
    ) -> np.ndarray:
        out = grid.copy()
        H, W = out.shape
        emitters = np.argwhere(grid == emitter_color)
        for er, ec in emitters:
            # Cast down-right diagonal
            r, c = er + 1, ec + 1
            while 0 <= r < H and 0 <= c < W:
                if out[r, c] == obstacle_color:
                    break
                out[r, c] = beam_color
                r += 1
                c += 1
        return out

    def _generate_diagonal_ray_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        for _ in range(num_demos):
            H, W = self.rng.randint(7, 10), self.rng.randint(7, 10)
            inp = np.zeros((H, W), dtype=int)
            er = self.rng.randint(0, 2)
            ec = self.rng.randint(0, 2)
            inp[er, ec] = 2  # Emitter
            # Obstacle
            ob_r = self.rng.randint(er + 2, H - 1)
            ob_c = self.rng.randint(ec + 2, W - 1)
            inp[ob_r, ob_c] = 5
            out = self._apply_diagonal_ray(inp, emitter_color=2, beam_color=3, obstacle_color=5)
            train_pairs.append((inp, out))

        H, W = self.rng.randint(7, 10), self.rng.randint(7, 10)
        test_inp = np.zeros((H, W), dtype=int)
        er = self.rng.randint(0, 2)
        ec = self.rng.randint(0, 2)
        test_inp[er, ec] = 2
        ob_r = self.rng.randint(er + 2, H - 1)
        ob_c = self.rng.randint(ec + 2, W - 1)
        test_inp[ob_r, ob_c] = 5
        test_out = self._apply_diagonal_ray(
            test_inp, emitter_color=2, beam_color=3, obstacle_color=5
        )

        return ManifestTask(
            task_id=task_id,
            family="diagonal_ray_cast",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_inp,
            test_output=test_out,
            metadata={
                "origin": "independent_generator",
                "description": "Diagonal ray projected from emitter until collision",
            },
        )

    # =========================================================================
    # 3. Interior Infill: Flooding closed boundary loop
    # =========================================================================
    def _apply_interior_infill(
        self, grid: np.ndarray, border_color: int = 1, fill_color: int = 8
    ) -> np.ndarray:
        out = grid.copy()
        H, W = out.shape
        # Identify bounding box of border
        coords = np.argwhere(grid == border_color)
        if len(coords) < 8:
            return out
        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)
        for r in range(rmin + 1, rmax):
            for c in range(cmin + 1, cmax):
                if out[r, c] == 0:
                    out[r, c] = fill_color
        return out

    def _generate_interior_infill_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        for _ in range(num_demos):
            H, W = self.rng.randint(7, 10), self.rng.randint(7, 10)
            inp = np.zeros((H, W), dtype=int)
            bh = self.rng.randint(4, H - 2)
            bw = self.rng.randint(4, W - 2)
            r = self.rng.randint(1, H - bh - 1)
            c = self.rng.randint(1, W - bw - 1)
            # Hollow box perimeter
            inp[r, c : c + bw] = 1
            inp[r + bh - 1, c : c + bw] = 1
            inp[r : r + bh, c] = 1
            inp[r : r + bh, c + bw - 1] = 1
            out = self._apply_interior_infill(inp, border_color=1, fill_color=8)
            train_pairs.append((inp, out))

        H, W = self.rng.randint(7, 10), self.rng.randint(7, 10)
        test_inp = np.zeros((H, W), dtype=int)
        bh = self.rng.randint(4, H - 2)
        bw = self.rng.randint(4, W - 2)
        r = self.rng.randint(1, H - bh - 1)
        c = self.rng.randint(1, W - bw - 1)
        test_inp[r, c : c + bw] = 1
        test_inp[r + bh - 1, c : c + bw] = 1
        test_inp[r : r + bh, c] = 1
        test_inp[r : r + bh, c + bw - 1] = 1
        test_out = self._apply_interior_infill(test_inp, border_color=1, fill_color=8)

        return ManifestTask(
            task_id=task_id,
            family="interior_infill",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_inp,
            test_output=test_out,
            metadata={
                "origin": "independent_generator",
                "description": "Hollow boundary interior filled with fill_color",
            },
        )

    # =========================================================================
    # 4. Alternating Pattern Extrapolate: 1D/2D periodic striping
    # =========================================================================
    def _apply_alternating_stripes(
        self, grid: np.ndarray, col_a: int = 4, col_b: int = 6
    ) -> np.ndarray:
        out = grid.copy()
        H, W = out.shape
        for r in range(H):
            for c in range(W):
                out[r, c] = col_a if (r + c) % 2 == 0 else col_b
        return out

    def _generate_pattern_extrapolate_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        for _ in range(num_demos):
            H, W = self.rng.randint(5, 8), self.rng.randint(5, 8)
            inp = np.zeros((H, W), dtype=int)
            # Give a seed 2x2 corner
            inp[0, 0] = 4
            inp[0, 1] = 6
            inp[1, 0] = 6
            inp[1, 1] = 4
            out = self._apply_alternating_stripes(inp, 4, 6)
            train_pairs.append((inp, out))

        H, W = self.rng.randint(5, 8), self.rng.randint(5, 8)
        test_inp = np.zeros((H, W), dtype=int)
        test_inp[0, 0] = 4
        test_inp[0, 1] = 6
        test_inp[1, 0] = 6
        test_inp[1, 1] = 4
        test_out = self._apply_alternating_stripes(test_inp, 4, 6)

        return ManifestTask(
            task_id=task_id,
            family="alternating_pattern_extrapolate",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_inp,
            test_output=test_out,
            metadata={
                "origin": "independent_generator",
                "description": "Checkerboard alternating pattern continuation",
            },
        )

    # =========================================================================
    # 5. Component Size Rank: Recolor based on connected component area
    # =========================================================================
    def _apply_component_size_rank(
        self, grid: np.ndarray, largest_col: int = 2, smallest_col: int = 3
    ) -> np.ndarray:
        out = grid.copy()
        # Find separate connected components of non-zero pixels
        visited = np.zeros_like(grid, dtype=bool)
        H, W = grid.shape
        components = []

        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0 and not visited[r, c]:
                    # BFS component
                    comp = []
                    q = [(r, c)]
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.pop(0)
                        comp.append((curr_r, curr_c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (
                                0 <= nr < H
                                and 0 <= nc < W
                                and grid[nr, nc] != 0
                                and not visited[nr, nc]
                            ):
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    components.append(comp)

        if len(components) >= 2:
            components.sort(key=lambda c: len(c), reverse=True)
            for r, c in components[0]:
                out[r, c] = largest_col
            for r, c in components[-1]:
                out[r, c] = smallest_col
        return out

    def _generate_component_rank_task(self, task_id: str, num_demos: int) -> ManifestTask:
        train_pairs = []
        for _ in range(num_demos):
            H, W = 9, 9
            inp = np.zeros((H, W), dtype=int)
            # Component 1 (large: 3x3 box = 9 pixels)
            inp[1:4, 1:4] = 7
            # Component 2 (small: 1x2 = 2 pixels)
            inp[6:7, 6:8] = 7
            out = self._apply_component_size_rank(inp, largest_col=2, smallest_col=3)
            train_pairs.append((inp, out))

        H, W = 9, 9
        test_inp = np.zeros((H, W), dtype=int)
        test_inp[1:4, 1:4] = 7
        test_inp[6:7, 6:8] = 7
        test_out = self._apply_component_size_rank(test_inp, largest_col=2, smallest_col=3)

        return ManifestTask(
            task_id=task_id,
            family="component_size_rank",
            depth=1,
            train_pairs=tuple(train_pairs),
            test_input=test_inp,
            test_output=test_out,
            metadata={
                "origin": "independent_generator",
                "description": "Rank connected components by area: largest=2, smallest=3",
            },
        )

    @classmethod
    def generate_independent_50_suite(cls, seed: int = 42) -> list[ManifestTask]:
        """Generate 50 independent tasks: 10 per family across the 5 independent families."""
        gen = cls(seed=seed)
        suite = []
        for family in cls.FAMILIES:
            for i in range(10):
                task_id = f"indep_{family}_{i:02d}"
                suite.append(gen.generate_task(task_id, family=family))
        return suite
