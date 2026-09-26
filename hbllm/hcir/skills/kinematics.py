"""Kinematic Momentum, Inertia & Contact Physics Skill Acquisition.

Enables learning friction coefficients, sliding momentum (ice mazes),
and simulating raycast slide trajectories with collision stopping.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent

logger = logging.getLogger(__name__)


@dataclass
class KinematicModel:
    """Empirical physical parameterization for environment kinematics."""

    friction: float = 1.0  # 1.0 = single step; 0.0 = frictionless slide
    gravity: tuple[int, int] = (0, 0)  # (dr, dc)
    is_sliding_environment: bool = False
    confidence: float = 0.5
    observations_tested: int = 0


class KinematicMomentumSkillAcquisition(BaseHierarchicalSkill):
    """Discovers sliding/momentum mechanics and plans multi-step inertial routes."""

    skill_name: str = "kinematic_momentum_sliding"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE

    def __init__(self) -> None:
        self.model = KinematicModel()

    def observe_displacement(
        self,
        intended_delta: tuple[int, int],
        start_pos: tuple[int, int],
        end_pos: tuple[int, int],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> None:
        """Analyze actual displacement vs intended single-step delta."""
        self.model.observations_tested += 1
        dr, dc = intended_delta
        if dr == 0 and dc == 0:
            return

        actual_dr = end_pos[0] - start_pos[0]
        actual_dc = end_pos[1] - start_pos[1]
        dist = abs(actual_dr) + abs(actual_dc)

        # If displacement along intended axis is > 1 cell, sliding momentum is detected!
        if dist > 1:
            # Check if direction matched intended axis
            sgn_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
            sgn_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
            act_sgn_r = 1 if actual_dr > 0 else (-1 if actual_dr < 0 else 0)
            act_sgn_c = 1 if actual_dc > 0 else (-1 if actual_dc < 0 else 0)

            if (sgn_r, sgn_c) == (act_sgn_r, act_sgn_c):
                # Verify that it stopped at an obstacle or wall
                next_r = end_pos[0] + sgn_r
                next_c = end_pos[1] + sgn_c
                H, W = grid_shape
                at_boundary = next_r < 0 or next_r >= H or next_c < 0 or next_c >= W
                at_barrier = (next_r, next_c) in barriers

                if at_boundary or at_barrier:
                    self.model.is_sliding_environment = True
                    self.model.friction = 0.0
                    self.model.confidence = min(0.99, self.model.confidence + 0.25)
        elif dist == 1 and not self.model.is_sliding_environment:
            self.model.friction = 1.0

    @classmethod
    def simulate_slide(
        cls,
        start_pos: tuple[int, int],
        direction: tuple[int, int],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        """Raycast forward along direction until encountering a barrier or boundary.

        Returns:
            (final_pos, path_traversed)
        """
        H, W = grid_shape
        dr, dc = direction
        if dr == 0 and dc == 0:
            return start_pos, [start_pos]

        sgn_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
        sgn_c = 1 if dc > 0 else (-1 if dc < 0 else 0)

        curr_r, curr_c = start_pos
        path: list[tuple[int, int]] = [start_pos]

        while True:
            nr = curr_r + sgn_r
            nc = curr_c + sgn_c
            # Boundary check
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                break
            # Barrier check
            if (nr, nc) in barriers:
                break
            curr_r, curr_c = nr, nc
            path.append((curr_r, curr_c))

        return (curr_r, curr_c), path

    def plan_sliding_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        max_slides: int = 30,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        """Plan a sequence of sliding impulses to reach goal (or pass through goal).

        Returns:
            List of (direction_vector, end_position) pairs.
        """
        if start == goal:
            return []

        # BFS over resting positions
        # Queue item: (current_pos, list of (direction, destination))
        queue = collections.deque([(start, [])])
        visited: set[tuple[int, int]] = {start}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr_pos, path = queue.popleft()
            if len(path) >= max_slides:
                continue

            for d in directions:
                dest, traj = self.simulate_slide(curr_pos, d, barriers, grid_shape)
                if dest == curr_pos:
                    continue  # Blocked in this direction

                # Check if goal was hit along the trajectory
                if goal in traj:
                    return path + [(d, dest)]

                if dest not in visited:
                    visited.add(dest)
                    queue.append((dest, path + [(d, dest)]))

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for kinematic sliding environment recognition."""
        if self.model.is_sliding_environment:
            return True
        if metadata and metadata.get("is_sliding"):
            return True
        return False

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        """Standardized interface plan generation for sliding momentum routes."""
        if not metadata or "start" not in metadata or "goal" not in metadata:
            return []
        start = metadata["start"]
        goal = metadata["goal"]
        barriers = metadata.get("barriers", set())
        grid_shape = (int(grid.shape[-2]), int(grid.shape[-1]))
        path = self.plan_sliding_path(start, goal, barriers, grid_shape)
        if not path:
            return []
        # Convert delta directions to action IDs (1: UP, 2: DOWN, 3: LEFT, 4: RIGHT)
        delta_to_action = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
        return [delta_to_action[d] for d, _ in path if d in delta_to_action]
