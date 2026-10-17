"""Task Generator for Milestone 3 Phase 2: Topological Graph & Geodesic Maze Challenge.

Generates rigorous inductive mechanism induction tasks across 4 topological families:
1. geodesic_shortest_path_maze: Classic maze with corridors, turns, and dead ends.
2. variable_corridor_obstacle_avoidance: Diverse grid sizes (7x7 to 13x13) with obstacle barriers.
3. same_color_landmark_connector: Two landmark markers of the exact same color separated by walls.
4. ambiguous_multi_path_abstention: Symmetric topologies with multiple equidistant shortest paths.

AST Independence: Zero imports from HCIR solver or GridOperator internals.
"""

from __future__ import annotations

import collections
import random
from typing import ClassVar

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask


class TopologicalMazeTaskGenerator:
    """Generates procedural challenge tasks for topological graph and geodesic maze induction."""

    FAMILIES: ClassVar[list[str]] = [
        "geodesic_shortest_path_maze",
        "variable_corridor_obstacle_avoidance",
        "same_color_landmark_connector",
        "ambiguous_multi_path_abstention",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def _find_all_shortest_paths(
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        traversable_colors: set[int],
    ) -> list[list[tuple[int, int]]]:
        """Compute all minimum-hop paths using BFS."""
        H, W = grid.shape
        sr, sc = start
        gr, gc = goal

        if (sr, sc) == (gr, gc):
            return [[(sr, sc)]]

        dist: dict[tuple[int, int], int] = {(sr, sc): 0}
        parents: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
        queue = collections.deque([(sr, sc)])
        found_goal = False
        goal_dist = float("inf")

        while queue:
            curr_r, curr_c = queue.popleft()
            d = dist[(curr_r, curr_c)]
            if d >= goal_dist:
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell_val = int(grid[nr, nc])
                    if (nr, nc) == (gr, gc) or cell_val in traversable_colors:
                        new_d = d + 1
                        if (nr, nc) not in dist:
                            dist[(nr, nc)] = new_d
                            parents[(nr, nc)].append((curr_r, curr_c))
                            queue.append((nr, nc))
                            if (nr, nc) == (gr, gc):
                                found_goal = True
                                goal_dist = new_d
                        elif dist[(nr, nc)] == new_d:
                            parents[(nr, nc)].append((curr_r, curr_c))

        if not found_goal:
            return []

        paths: list[list[tuple[int, int]]] = []

        def backtrack(curr: tuple[int, int], current_path: list[tuple[int, int]]) -> None:
            if curr == (sr, sc):
                paths.append(list(reversed(current_path)))
                return
            for p in parents[curr]:
                backtrack(p, current_path + [p])

        backtrack((gr, gc), [(gr, gc)])
        return paths

    def _generate_single_maze_instance(
        self,
        H: int,
        W: int,
        wall_col: int,
        start_col: int,
        goal_col: int,
        path_col: int,
        ensure_unique: bool = True,
        force_ambiguous: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate an input grid and corresponding ground-truth output grid."""
        for _ in range(500):
            grid = np.zeros((H, W), dtype=int)

            if force_ambiguous:
                # Construct symmetric obstacle probe
                sr, sc = H // 2, 1
                gr, gc = H // 2, W - 2
                grid[sr, sc] = start_col
                grid[gr, gc] = goal_col

                # Center barrier causing equal top/bottom path splits
                mid_c = W // 2
                grid[H // 2, mid_c] = wall_col
                if H >= 7:
                    grid[H // 2, mid_c - 1] = wall_col
                    grid[H // 2, mid_c + 1] = wall_col

                all_paths = self._find_all_shortest_paths(
                    grid, (sr, sc), (gr, gc), traversable_colors={0}
                )
                if len(all_paths) >= 2:
                    # Ground truth for ambiguous task: input unchanged or identity
                    out_grid = np.copy(grid)
                    return grid, out_grid
                continue

            # Non-ambiguous maze generation
            sr, sc = self.rng.randint(0, H - 1), self.rng.randint(0, 1)
            gr, gc = self.rng.randint(0, H - 1), self.rng.randint(W - 2, W - 1)

            # Avoid placing start and goal at same position
            if (sr, sc) == (gr, gc):
                continue

            grid[sr, sc] = start_col
            grid[gr, gc] = goal_col

            # Fill random obstacle walls
            num_walls = int(H * W * 0.25)
            for _w in range(num_walls):
                wr = self.rng.randint(0, H - 1)
                wc = self.rng.randint(0, W - 1)
                if (wr, wc) not in {(sr, sc), (gr, gc)}:
                    grid[wr, wc] = wall_col

            all_paths = self._find_all_shortest_paths(
                grid, (sr, sc), (gr, gc), traversable_colors={0}
            )

            if ensure_unique and len(all_paths) == 1:
                path = all_paths[0]
                if len(path) >= 4:  # Non-trivial length
                    out_grid = np.copy(grid)
                    for pr, pc in path:
                        if (pr, pc) != (sr, sc) and (pr, pc) != (gr, gc):
                            out_grid[pr, pc] = path_col
                    return grid, out_grid

        # Fallback deterministic corridor if random attempts exhausted
        grid = np.zeros((H, W), dtype=int)
        grid[1, 1] = start_col
        grid[H - 2, W - 2] = goal_col
        # Wall off middle except a single serpentine path
        for r in range(H):
            for c in range(W):
                if (r, c) not in {(1, 1), (H - 2, W - 2)}:
                    if r % 2 == 1 and c % 2 == 1:
                        pass
                    else:
                        grid[r, c] = wall_col

        # Clear a deterministic corridor
        curr_r, curr_c = 1, 1
        path_coords = [(curr_r, curr_c)]
        while curr_c < W - 2:
            curr_c += 1
            grid[curr_r, curr_c] = 0
            path_coords.append((curr_r, curr_c))
        while curr_r < H - 2:
            curr_r += 1
            grid[curr_r, curr_c] = 0
            path_coords.append((curr_r, curr_c))

        out_grid = np.copy(grid)
        for pr, pc in path_coords:
            if (pr, pc) != (1, 1) and (pr, pc) != (H - 2, W - 2):
                out_grid[pr, pc] = path_col
        return grid, out_grid

    def generate_task(self, task_id: str, family: str) -> ManifestTask:
        """Generate single task instance with 3 train pairs and 1 test query."""
        H = self.rng.randint(7, 11)
        W = self.rng.randint(7, 11)

        palette = self.rng.sample([1, 2, 3, 5, 6, 7, 8], 4)
        wall_col, c1, c2, path_col = palette

        if family == "geodesic_shortest_path_maze":
            start_col, goal_col = c1, c2
            train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for _ in range(3):
                xi, yi = self._generate_single_maze_instance(
                    H, W, wall_col, start_col, goal_col, path_col, ensure_unique=True
                )
                train_pairs.append((xi, yi))
            test_in, test_out = self._generate_single_maze_instance(
                H, W, wall_col, start_col, goal_col, path_col, ensure_unique=True
            )

        elif family == "variable_corridor_obstacle_avoidance":
            # Variable grid sizes across demonstrations
            start_col, goal_col = c1, c2
            train_pairs = []
            for dim_offset in [-1, 0, 1]:
                h_i = max(7, H + dim_offset)
                w_i = max(7, W + dim_offset)
                xi, yi = self._generate_single_maze_instance(
                    h_i, w_i, wall_col, start_col, goal_col, path_col, ensure_unique=True
                )
                train_pairs.append((xi, yi))
            test_in, test_out = self._generate_single_maze_instance(
                H + 2, W + 2, wall_col, start_col, goal_col, path_col, ensure_unique=True
            )

        elif family == "same_color_landmark_connector":
            # Start and goal share the identical landmark color
            landmark_col = c1
            train_pairs = []
            for _ in range(3):
                xi, yi = self._generate_single_maze_instance(
                    H, W, wall_col, landmark_col, landmark_col, path_col, ensure_unique=True
                )
                train_pairs.append((xi, yi))
            test_in, test_out = self._generate_single_maze_instance(
                H, W, wall_col, landmark_col, landmark_col, path_col, ensure_unique=True
            )

        elif family == "ambiguous_multi_path_abstention":
            # Training demonstrations have unambiguous shortest paths
            start_col, goal_col = c1, c2
            train_pairs = []
            for _ in range(3):
                xi, yi = self._generate_single_maze_instance(
                    H,
                    W,
                    wall_col,
                    start_col,
                    goal_col,
                    path_col,
                    ensure_unique=True,
                    force_ambiguous=False,
                )
                train_pairs.append((xi, yi))
            # Test query has symmetric ambiguous multi-paths
            test_in, test_out = self._generate_single_maze_instance(
                H,
                W,
                wall_col,
                start_col,
                goal_col,
                path_col,
                ensure_unique=False,
                force_ambiguous=True,
            )

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
                "is_ambiguous_probe": (family == "ambiguous_multi_path_abstention"),
            },
        )

    @classmethod
    def generate_benchmark_suite(cls, seed: int = 42) -> list[ManifestTask]:
        """Generate 20 challenge tasks (5 tasks per family) for Milestone 3 Phase 2."""
        gen = cls(seed=seed)
        suite: list[ManifestTask] = []
        for fam in cls.FAMILIES:
            for idx in range(1, 6):
                tid = f"m3_p2_{fam}_{idx:02d}"
                suite.append(gen.generate_task(tid, fam))
        return suite
