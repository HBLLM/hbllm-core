"""Intuitive Physics & Ballistics Engine for HCIR World Kernel.

Modeled on mammalian cerebellar internal models and parietal VIP/LIP neural circuits:
1. Environmental Gravity & Passive Drift: Detects uncommanded downward acceleration
   and environmental force fields (g = [1, 0]).
2. Ground Support & Surface Affordance: Differentiates solid landing platforms from
   empty air / chasms.
3. Ballistic Arc & Projectile Simulation: Projects forward parabolic trajectories for
   jumping, leaping, falling, and projectile launches.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BallisticTrajectory:
    """A projected trajectory under physical forces (gravity, momentum)."""

    path: list[tuple[int, int]]
    landing_pos: tuple[int, int]
    is_grounded: bool
    is_terminal_hazard: bool = False


class IntuitivePhysicsEngine:
    """Domain-agnostic intuitive physics, momentum, and ballistics engine."""

    def __init__(self, history_len: int = 16) -> None:
        self.history_len = history_len
        self.observed_transitions: deque[tuple[tuple[int, int], tuple[int, int], bool]] = deque(
            maxlen=history_len
        )
        self.gravity_evidence: int = 0
        self.has_gravity: bool = False
        self.gravity_vector: tuple[int, int] = (0, 0)
        self.known_platforms: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        """Reset transient episode observations while preserving calibrated physics laws."""
        self.observed_transitions.clear()
        # Keep has_gravity if confirmed across multiple episodes, otherwise retain
        self.known_platforms.clear()

    def record_transition(
        self,
        prev_pos: tuple[int, int] | None,
        curr_pos: tuple[int, int] | None,
        is_displacement_action: bool,
        commanded_delta: tuple[int, int] = (0, 0),
    ) -> None:
        """Observe sensorimotor transition to detect uncommanded environmental forces."""
        if prev_pos is None or curr_pos is None:
            return

        actual_dr = curr_pos[0] - prev_pos[0]

        # Passive downward drift detection:
        # If the commanded action was NOT downward (commanded_delta[0] <= 0)
        # but the actual displacement moved downward (actual_dr > 0), this is gravity.
        if commanded_delta[0] <= 0 and actual_dr > 0:
            self.gravity_evidence += 1
            if self.gravity_evidence >= 2:
                self.has_gravity = True
                self.gravity_vector = (1, 0)
                logger.info(
                    "IntuitivePhysicsEngine: Confirmed environmental gravity field g=%s (evidence=%d).",
                    self.gravity_vector,
                    self.gravity_evidence,
                )
        elif commanded_delta[0] > 0 and actual_dr > 0:
            # Commanded downward move, neutral evidence
            pass
        elif actual_dr == 0 and not is_displacement_action:
            # Stationary on a surface -> decay false gravity evidence if not confirmed
            if not self.has_gravity and self.gravity_evidence > 0:
                self.gravity_evidence -= 1

    def is_supported(
        self,
        r: int,
        c: int,
        grid: np.ndarray,
        barrier_features: set[int],
        static_barriers: set[tuple[int, int]] | None = None,
    ) -> bool:
        """Check if an entity at (r, c) rests on a solid surface / floor."""
        H, W = grid.shape
        below_r = r + 1

        # Grid bottom boundary acts as ground support
        if below_r >= H:
            return True

        if static_barriers and (below_r, c) in static_barriers:
            return True

        val = int(grid[below_r, c])
        return val in barrier_features

    def project_fall(
        self,
        start_r: int,
        start_c: int,
        grid: np.ndarray,
        barrier_features: set[int],
        static_barriers: set[tuple[int, int]] | None = None,
        lethal_features: set[int] | None = None,
        lethal_coords: set[tuple[int, int]] | None = None,
        max_fall: int = 16,
    ) -> tuple[tuple[int, int], list[tuple[int, int]], bool]:
        """Project the vertical falling trajectory until hitting ground support or hazard.

        Returns:
            (landing_pos, trajectory_path, is_lethal)
        """
        H, W = grid.shape
        cur_r = start_r
        cur_c = start_c
        trajectory: list[tuple[int, int]] = []

        for _ in range(max_fall):
            if self.is_supported(cur_r, cur_c, grid, barrier_features, static_barriers):
                return (cur_r, cur_c), trajectory, False

            next_r = cur_r + 1
            if next_r >= H:
                return (cur_r, cur_c), trajectory, False

            if static_barriers and (next_r, cur_c) in static_barriers:
                return (cur_r, cur_c), trajectory, False

            cur_r = next_r
            trajectory.append((cur_r, cur_c))

            if lethal_features and int(grid[cur_r, cur_c]) in lethal_features:
                return (cur_r, cur_c), trajectory, True

            if lethal_coords and (cur_r, cur_c) in lethal_coords:
                return (cur_r, cur_c), trajectory, True

        return (cur_r, cur_c), trajectory, False

    def simulate_ballistic_jump(
        self,
        start_pos: tuple[int, int],
        horizontal_dir: int,
        jump_height: int,
        grid: np.ndarray,
        barrier_features: set[int],
        static_barriers: set[tuple[int, int]] | None = None,
    ) -> BallisticTrajectory:
        """Simulate a parabolic jump arc: ascent -> apex -> descent -> landing."""
        H, W = grid.shape
        r, c = start_pos
        path: list[tuple[int, int]] = []

        # 1. Ascent phase (upward impulse)
        for _ in range(jump_height):
            nr = r - 1
            nc = c + horizontal_dir
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if static_barriers and (nr, nc) in static_barriers:
                break
            r, c = nr, nc
            path.append((r, c))

        # 2. Descent phase under gravity
        landing_pos, fall_path, is_lethal = self.project_fall(
            r, c, grid, barrier_features, static_barriers
        )
        path.extend(fall_path)

        return BallisticTrajectory(
            path=path,
            landing_pos=landing_pos,
            is_grounded=True,
            is_terminal_hazard=is_lethal,
        )
