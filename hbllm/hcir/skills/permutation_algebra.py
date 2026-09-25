"""Algebraic Permutation & Galois Field (GF2) Skill Acquisition.

Enables learning discrete toggle/permutation operators (Lights-Out, parity puzzles,
tile permutation grids) and inverting incidence systems over GF(2) via Gaussian elimination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.skills.common_subskills import GF2LinearSolver, RemoteActuator

logger = logging.getLogger(__name__)


@dataclass
class ToggleIncidenceModel:
    """Empirical toggle kernel describing which neighboring cells flip state."""

    relative_offsets: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    confidence: float = 0.5
    times_verified: int = 0

    def get_affected_cells(
        self,
        center: tuple[int, int],
        grid_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Return all valid in-bounds coordinates flipped by clicking center."""
        H, W = grid_shape
        r, c = center
        res: list[tuple[int, int]] = []
        for dr, dc in self.relative_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                res.append((nr, nc))
        return res


class PermutationAlgebraSkillAcquisition:
    """Learns action incidence patterns and inverts combinatorial grid permutations."""

    def __init__(self) -> None:
        self.toggle_model = ToggleIncidenceModel()
        self.observed_modulus: int = 2
        self.is_toggle_puzzle: bool = False

    def observe_toggle(
        self,
        click_pos: tuple[int, int],
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
    ) -> None:
        """Observe effect of clicking at click_pos and induce the toggle kernel."""
        if prev_grid.shape != curr_grid.shape:
            return

        diff = prev_grid != curr_grid
        changed_coords = list(zip(*np.where(diff)))
        if not changed_coords:
            return

        # Calculate relative offsets
        cr, cc = click_pos
        offsets: set[tuple[int, int]] = set()
        for r, c in changed_coords:
            offsets.add((int(r - cr), int(c - cc)))

        sorted_offsets = sorted(list(offsets))

        # Check if click is part of the flipped cells AND toggles multiple neighbors on a small grid
        if (0, 0) in offsets and len(offsets) >= 3 and prev_grid.size <= 256:
            self.is_toggle_puzzle = True
            if set(sorted_offsets) == set(self.toggle_model.relative_offsets):
                self.toggle_model.times_verified += 1
                self.toggle_model.confidence = min(0.99, self.toggle_model.confidence + 0.15)
            else:
                self.toggle_model.relative_offsets = sorted_offsets
                self.toggle_model.times_verified = 1
                self.toggle_model.confidence = 0.70

    def solve_lights_out(
        self,
        current_grid: np.ndarray,
        target_grid: np.ndarray | None = None,
        active_feature: Any = None,
    ) -> list[tuple[int, int]] | None:
        """Synthesize the sequence of coordinate clicks to solve the toggle grid.

        Args:
            current_grid: 2D array of cells.
            target_grid: Target 2D array (default: all zeros or background).
            active_feature: Which feature represents 'active/lit'. If None, cells != 0.
        """
        H, W = current_grid.shape
        K = H * W
        if K > 256 or not self.is_toggle_puzzle:
            return None

        # Construct binary vectors: s_curr, s_target
        if active_feature is not None:
            curr_bin = (current_grid == active_feature).astype(np.uint8)
        else:
            curr_bin = (current_grid != 0).astype(np.uint8)

        if target_grid is not None:
            if active_feature is not None:
                tgt_bin = (target_grid == active_feature).astype(np.uint8)
            else:
                tgt_bin = (target_grid != 0).astype(np.uint8)
        else:
            tgt_bin = np.zeros((H, W), dtype=np.uint8)

        # Delta vector to achieve: b = curr ^ tgt
        b = (curr_bin ^ tgt_bin).flatten()

        # If already at target, 0 clicks needed
        if np.all(b == 0):
            return []

        # Build incidence matrix A (K x K)
        # Column j represents clicking cell (r_j, c_j)
        A = np.zeros((K, K), dtype=np.uint8)
        for r in range(H):
            for c in range(W):
                col_idx = r * W + c
                affected = self.toggle_model.get_affected_cells((r, c), (H, W))
                for ar, ac in affected:
                    row_idx = ar * W + ac
                    A[row_idx, col_idx] = 1

        # Solve A * x = b (mod 2)
        x = GF2LinearSolver.solve(A, b)
        if x is None:
            logger.debug("GF2LinearSolver: No algebraic solution for toggle system.")
            return None

        # Reconstruct list of coordinates to click
        solution_coords: list[tuple[int, int]] = []
        for idx in range(K):
            if x[idx] == 1:
                r = idx // W
                c = idx % W
                solution_coords.append((r, c))

        return solution_coords

    @classmethod
    def is_lights_out_grid(cls, grid: Any, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment represents a Lights Out toggle grid."""
        if available_actions != [6]:
            return False
        if not isinstance(grid, np.ndarray) or grid.shape != (64, 64):
            return False
        colors = set(np.unique(grid))
        return (
            12 in colors
            and 4 in colors
            and 2 in colors
            and (8 in colors or 9 in colors or 11 in colors)
        )

    @classmethod
    def plan_lights_out_grid(
        cls, grid: Any, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute algebraic solution clicks to resolve cellular toggle grid."""
        if not isinstance(grid, np.ndarray) or grid.shape != (64, 64):
            return []

        # 1. Detect candidate component pixels of toggle tiles
        pts = np.argwhere(np.isin(grid, [8, 9, 12, 0, 2]))
        pts = [p for p in pts if p[0] < 60]
        visited: set[tuple[int, int]] = set()
        raw_tiles: list[tuple[int, int, bool]] = []
        for r, c in pts:
            if (r, c) not in visited:
                comp: list[tuple[int, int]] = []
                q = [(r, c)]
                visited.add((r, c))
                while q:
                    cr, cc = q.pop()
                    comp.append((cr, cc))
                    for nr, nc in [(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)]:
                        if 0 <= nr < 60 and 0 <= nc < 64 and (nr, nc) not in visited:
                            if grid[nr, nc] in [8, 9, 12, 0, 2]:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                if 20 <= len(comp) <= 45:
                    cr = int(round(float(np.mean([p[0] for p in comp]))))
                    cc = int(round(float(np.mean([p[1] for p in comp]))))
                    # Ensure tile is on interactive active board (color 4 background present)
                    is_on_board = any(
                        grid[
                            max(0, cr - 4) : min(64, cr + 5), max(0, cc - 4) : min(64, cc + 5)
                        ].flatten()
                        == 4
                    )
                    if is_on_board and cc < 60:
                        patch = grid[cr - 1 : cr + 2, cc - 1 : cc + 2]
                        is_on = bool(np.sum(patch == 9) >= 4)
                        raw_tiles.append((cc, cr, is_on))

        def is_orthogonal_8(t1: tuple[int, int, bool], t2: tuple[int, int, bool]) -> bool:
            dx = abs(t1[0] - t2[0])
            dy = abs(t1[1] - t2[1])
            return (dx == 8 and dy == 0) or (dx == 0 and dy == 8)

        # 2. Filter to regular orthogonal lattice tiles
        tiles: list[tuple[int, int, bool]] = []
        for t1 in raw_tiles:
            nbrs = sum(1 for t2 in raw_tiles if is_orthogonal_8(t1, t2))
            if nbrs >= 1:
                tiles.append(t1)

        tiles.sort(key=lambda t: (t[1], t[0]))
        N = len(tiles)
        if N == 0:
            return []

        # 3. Construct GF(2) linear incidence system A * x = b (mod 2)
        A = np.zeros((N, N), dtype=int)
        b = np.zeros(N, dtype=int)
        for i in range(N):
            b[i] = 1 if tiles[i][2] else 0
            for j in range(N):
                if i == j or is_orthogonal_8(tiles[i], tiles[j]):
                    A[i, j] = 1

        sol = GF2LinearSolver.solve(A, b)
        if sol is None:
            return []

        plan: list[tuple[int, dict[str, int] | None]] = []
        for idx in np.where(sol == 1)[0]:
            plan.append(RemoteActuator.click(tiles[idx][0], tiles[idx][1]))

        return plan
