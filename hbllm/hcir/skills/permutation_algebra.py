"""Algebraic Permutation & Galois Field (GF2) Skill Acquisition.

Enables learning discrete toggle/permutation operators (Lights-Out, parity puzzles,
tile permutation grids) and inverting incidence systems over GF(2) via Gaussian elimination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.skills.common_subskills import GF2LinearSolver, RemoteActuator
from hbllm.hcir.spatial_planner import SpatialActionIntent

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


class PermutationAlgebraSkillAcquisition(BaseHierarchicalSkill):
    """Learns action incidence patterns and inverts combinatorial grid permutations."""

    skill_name: str = "permutation_algebra_lights_out"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.INTERACT

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
    def _extract_tiles(cls, grid: np.ndarray) -> list[tuple[int, int, bool]]:
        """Extract toggle tiles forming an 8-stride orthogonal lattice and their binary states."""
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        # Find connected components of non-bg pixels in y < 60, x < 60
        mask = (grid != bg) & (np.arange(64)[:, None] < 60) & (np.arange(64)[None, :] < 60)
        from scipy.ndimage import label

        labeled, num_features = label(mask)
        raw_tiles: list[tuple[int, int, int]] = []  # (cc, cr, center_color)

        for idx in range(1, num_features + 1):
            pts = np.argwhere(labeled == idx)
            if 15 <= len(pts) <= 50:
                min_r, min_c = int(pts[:, 0].min()), int(pts[:, 1].min())
                max_r, max_c = int(pts[:, 0].max()), int(pts[:, 1].max())
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if 4 <= w <= 8 and 4 <= h <= 8:
                    cr = (min_r + max_r) // 2
                    cc = (min_c + max_c) // 2
                    center_color = int(grid[cr, cc])
                    raw_tiles.append((cc, cr, center_color))

        def is_orthogonal_8(t1: tuple[int, int, Any], t2: tuple[int, int, Any]) -> bool:
            dx = abs(t1[0] - t2[0])
            dy = abs(t1[1] - t2[1])
            return (dx == 8 and dy == 0) or (dx == 0 and dy == 8)

        # Filter to tiles that belong to the 8-stride lattice
        lattice_tiles = [
            t for t in raw_tiles if sum(1 for t2 in raw_tiles if is_orthogonal_8(t, t2)) >= 1
        ]
        if len(lattice_tiles) < 4:
            return []

        # Determine binary ON/OFF state from center colors
        center_colors = [t[2] for t in lattice_tiles]
        u_cols, u_counts = np.unique(center_colors, return_counts=True)
        if len(u_cols) < 2:
            return []
        # The off state is typically the majority or resting color, on state is minority
        off_color = u_cols[np.argmax(u_counts)]

        tiles = [(t[0], t[1], t[2] != off_color) for t in lattice_tiles]
        tiles.sort(key=lambda t: (t[1], t[0]))
        return tiles

    @classmethod
    def is_lights_out_grid(cls, grid: Any, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment represents a Lights Out toggle grid."""
        if set(available_actions) != {6}:
            return False
        if not isinstance(grid, np.ndarray) or grid.shape[-2:] != (64, 64):
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        from hbllm.hcir.skills.automaton_synthesis import (
            AutomatonProgramSynthesisSkillAcquisition,
        )

        if AutomatonProgramSynthesisSkillAcquisition.is_automaton_synthesis_grid(
            grid, available_actions
        ):
            return False

        tiles = cls._extract_tiles(grid)
        return len(tiles) >= 4

    @classmethod
    def plan_lights_out_grid(
        cls, grid: Any, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute algebraic solution clicks to resolve cellular toggle grid."""
        if not isinstance(grid, np.ndarray) or grid.shape[-2:] != (64, 64):
            return []

        if grid.ndim == 3:
            grid = grid[-1]

        tiles = cls._extract_tiles(grid)
        N = len(tiles)
        if N == 0:
            return []

        def is_orthogonal_8(t1: tuple[int, int, bool], t2: tuple[int, int, bool]) -> bool:
            dx = abs(t1[0] - t2[0])
            dy = abs(t1[1] - t2[1])
            return (dx == 8 and dy == 0) or (dx == 0 and dy == 8)

        # Construct GF(2) linear incidence system A * x = b (mod 2)
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

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for Lights Out GF(2) algebraic puzzles."""
        return self.is_lights_out_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for Lights Out GF(2) algebraic puzzles."""
        return self.plan_lights_out_grid(grid, current_level=current_level)
