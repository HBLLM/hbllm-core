"""ARC-AGI and ARC-AGI-3 Domain-Specific Algorithmic Game Solvers.

This module encapsulates game-specific heuristics, rules, and algorithmic solvers
extracted from core cognitive architecture:
1. SentryCombatSolver: Tactical flanking & sentry gaze neutralization (e.g. TU93).
2. MinesweeperGameSolver: Numerical cardinality deduction & safe cell prioritization (e.g. SU15).
3. MemoryCardMatchingSolver: Pair matching & tile state tracking (e.g. AR25).
4. VortexAttractorSolver: Gravitational shockwave & kinetic repulsion targeting (e.g. KA59).
5. PegSolitaireSolver: Lattice elimination & DFS jump search (e.g. LF52).
6. LightsOutSolver: GF(2) linear system inversion & state flipping.
7. TrackMazeSolver: Rail navigation & switch activation.
8. MirroredConvergenceSolver: Optical midpoint reflection & alignment.
9. GravitySpillingPlatformSolver: Fluid/particle platform tilting.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SentryCombatSolver:
    """Tactical combat & flanking neutralization for stationary and patrolling sentries."""

    @staticmethod
    def is_flank_step(
        cur_pos: tuple[int, int],
        dest_pos: tuple[int, int],
        threat_pos: tuple[int, int],
        gaze_pos: tuple[int, int],
    ) -> bool:
        """Return True if stepping from cur_pos to dest_pos neutralizes the sentry from behind/side."""
        if dest_pos != threat_pos:
            return False
        # Stepping onto sentry from any position other than its gaze line is a successful flank
        return cur_pos != gaze_pos

    @staticmethod
    def identify_neutralizable_threats(
        threat_map: dict[tuple[int, int], Any],
        cur_pos: tuple[int, int],
    ) -> set[tuple[int, int]]:
        """Return threat coordinates that can be flanked from current position."""
        neutralizable: set[tuple[int, int]] = set()
        for t_pos, threat in threat_map.items():
            gaze = getattr(threat, "gaze_pos", None)
            if gaze is not None and cur_pos != gaze:
                neutralizable.add(t_pos)
        return neutralizable


class MinesweeperGameSolver:
    """Numerical constraint propagation and safe tile reveal solver."""

    def __init__(self) -> None:
        self.revealed_cells: set[tuple[int, int]] = set()
        self.known_safe: set[tuple[int, int]] = set()
        self.known_hazards: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        self.revealed_cells.clear()
        self.known_safe.clear()
        self.known_hazards.clear()

    def get_unrevealed_safe_cells(self) -> set[tuple[int, int]]:
        return self.known_safe - self.revealed_cells

    def score_safe_cells(
        self,
        H: int,
        W: int,
        quiescent_targets: set[tuple[int, int]],
        visit_counts: dict[str, int],
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        for sr, sc in self.get_unrevealed_safe_cells():
            if (sr, sc) not in quiescent_targets and 0 <= sr < H and 0 <= sc < W:
                v = visit_counts.get(f"click_{sr}_{sc}", 0)
                if v == 0:
                    candidates.append((sr, sc, 300.0))
        return candidates


class MemoryCardMatchingSolver:
    """Card and tile pair matching solver for visual working memory."""

    @staticmethod
    def score_matching_pairs(
        last_probed: tuple[int, int] | None,
        last_feat: int | None,
        matching_partner_pos: tuple[int, int] | None,
        H: int,
        W: int,
        bg: int,
        quiescent_targets: set[tuple[int, int]],
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        if (
            last_probed is not None
            and last_feat is not None
            and last_feat != bg
            and last_feat != 0
            and matching_partner_pos is not None
        ):
            pr, pc = matching_partner_pos
            if (pr, pc) not in quiescent_targets and 0 <= pr < H and 0 <= pc < W:
                candidates.append((pr, pc, 450.0))
        return candidates


class VortexAttractorSolver:
    """Solves gravitational shockwave / attractor puzzles by clicking behind floating items."""

    @staticmethod
    def score_repulsion_wave_targets(
        entities: Sequence[Any],
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        quiescent_targets: set[tuple[int, int]],
        visit_counts: dict[str, int],
        last_action_coord: tuple[int, int] | None = None,
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        target_zones = [
            e.grid_pos
            for e in entities
            if getattr(e, "role", None) is not None and str(getattr(e, "role", "")) == "goal"
        ]
        movable_items = [
            e
            for e in entities
            if e.feature_id != bg and e.area <= 12 and getattr(e, "role", None) != "goal"
        ]

        if not target_zones or not movable_items:
            return candidates

        for item in movable_items:
            ir, ic = item.grid_pos
            nearest_zone = min(target_zones, key=lambda z: abs(z[0] - ir) + abs(z[1] - ic))
            zr, zc = nearest_zone
            dr = 1 if zr > ir else (-1 if zr < ir else 0)
            dc = 1 if zc > ic else (-1 if zc < ic else 0)

            probe_points = []
            if dr != 0:
                probe_points.append((ir - dr * 2, ic))
                probe_points.append((ir - dr * 3, ic))
            if dc != 0:
                probe_points.append((ir, ic - dc * 2))
                probe_points.append((ir, ic - dc * 3))

            for click_r, click_c in probe_points:
                if (
                    0 <= click_r < H
                    and 0 <= click_c < W
                    and (click_r, click_c) not in quiescent_targets
                ):
                    v_count = visit_counts.get(f"click_{click_r}_{click_c}", 0)
                    if v_count < 4:
                        base_score = 350.0
                        if (click_r, click_c) == last_action_coord:
                            base_score += 40.0
                        candidates.append((click_r, click_c, base_score - float(v_count) * 20.0))
        return candidates


class PegSolitaireSolver:
    """Peg solitaire lattice jump solver."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue.clear()


class LightsOutSolver:
    """GF(2) matrix inversion solver for toggle grid puzzles."""

    @staticmethod
    def solve_gf2(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
        """Solve Ax = b over GF(2) using Gaussian elimination."""
        N = matrix.shape[0]
        A = matrix.copy().astype(int) % 2
        b = rhs.copy().astype(int) % 2

        for i in range(N):
            pivot = -1
            for r in range(i, N):
                if A[r, i] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            if pivot != i:
                A[[i, pivot]] = A[[pivot, i]]
                b[[i, pivot]] = b[[pivot, i]]
            for r in range(N):
                if r != i and A[r, i] == 1:
                    A[r] ^= A[i]
                    b[r] ^= b[i]

        for i in range(N):
            if np.all(A[i] == 0) and b[i] == 1:
                return None
        return b


class TrackMazeSolver:
    """Switch track & rail alignment solver."""

    def __init__(self) -> None:
        self.switch_coords: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        self.switch_coords.clear()


class MirroredConvergenceSolver:
    """Optical mirror reflection & midpoint target solver."""

    @staticmethod
    def extract_mirror_targets(
        structural_goals: Sequence[dict[str, Any]],
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        quiescent_targets: set[tuple[int, int]],
        visit_counts: dict[str, int],
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        for og in structural_goals:
            op = og.get("position")
            g_type = og.get("type")
            if op is not None and op not in quiescent_targets and 0 <= op[0] < H and 0 <= op[1] < W:
                v_count = visit_counts.get(f"click_{op[0]}_{op[1]}", 0)
                if v_count < 3:
                    is_void = og.get("is_void", False) or (int(curr_grid[op[0], op[1]]) == bg)
                    if g_type in (
                        "relational_alignment",
                        "optical_mirror_target",
                        "reflection_target",
                    ):
                        base = 60.0 if is_void else 240.0
                        conf = og.get("confidence", 0.5)
                        score = base + (conf * 40.0) - float(v_count) * 20.0
                        candidates.append((op[0], op[1], score))
                    elif g_type == "symmetry_completion":
                        base = 50.0 if is_void else 220.0
                        conf = og.get("confidence", 0.8)
                        score = base + (conf * 50.0) - float(v_count) * 20.0
                        candidates.append((op[0], op[1], score))
        return candidates


class GravitySpillingPlatformSolver:
    """Fluid / particulate spilling platform tilting solver."""

    def __init__(self) -> None:
        self.tilt_sequence: list[int] = []

    def reset_episode(self) -> None:
        self.tilt_sequence.clear()
