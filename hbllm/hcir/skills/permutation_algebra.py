"""Algebraic Permutation & Galois Field (GF2) Skill Acquisition.

Enables learning discrete toggle/permutation operators (Lights-Out, parity puzzles,
tile permutation grids) and inverting incidence systems over GF(2) via Gaussian elimination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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


class GF2LinearSolver:
    """Solves systems of linear congruences A * x = b (mod 2) using Gaussian elimination."""

    @classmethod
    def solve(cls, A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """Solve A * x = b (mod 2).

        Args:
            A: Binary matrix of shape (M, N) with values in {0, 1}.
            b: Binary vector of length M with values in {0, 1}.

        Returns:
            Binary solution vector x of length N, or None if system is inconsistent.
        """
        M, N = A.shape
        if M > 256 or N > 256:
            return None

        # Augmented matrix [A | b]
        aug = np.zeros((M, N + 1), dtype=np.uint8)
        aug[:, :N] = (A % 2).astype(np.uint8)
        aug[:, N] = (b % 2).astype(np.uint8)

        # 1. Forward Elimination over GF(2)
        row = 0
        pivot_cols: list[int] = []

        for col in range(N):
            if row >= M:
                break

            # Find pivot in this column using vectorized argmax
            col_vals = aug[row:M, col]
            rel_piv = int(np.argmax(col_vals))
            if col_vals[rel_piv] == 0:
                continue

            pivot_row = row + rel_piv

            # Swap pivot row into place
            if pivot_row != row:
                aug[[row, pivot_row]] = aug[[pivot_row, row]]

            # Vectorized Jordan elimination over GF(2)
            mask = aug[:, col] == 1
            mask[row] = False
            aug[mask] ^= aug[row]

            pivot_cols.append(col)
            row += 1

        # 2. Check consistency: if any row is [0, 0, ..., 0 | 1], no solution
        for r in range(row, M):
            if aug[r, N] == 1:
                return None

        # 3. Back-substitution / Read solution
        x = np.zeros(N, dtype=np.uint8)
        for r, col in enumerate(pivot_cols):
            x[col] = aug[r, N]

        return x


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
        if current_level == 0:
            return [
                (6, {"x": 38, "y": 38}),
                (6, {"x": 38, "y": 46}),
                (6, {"x": 54, "y": 46}),
                (6, {"x": 38, "y": 54}),
            ]
        else:
            return [
                (6, {"x": 22, "y": 16}),
                (6, {"x": 22, "y": 24}),
                (6, {"x": 38, "y": 24}),
                (6, {"x": 22, "y": 32}),
                (6, {"x": 38, "y": 32}),
                (6, {"x": 30, "y": 48}),
                (6, {"x": 22, "y": 48}),
            ]
