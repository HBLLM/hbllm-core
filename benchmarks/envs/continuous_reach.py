"""Closed-Loop Continuous 2D Reach Environment.

Continuous 2D kinematic point-robot navigation with obstacle avoidance:
- Continuous pose in bounded workspace [0, 10] x [0, 10].
- Formal spatial ReferenceFrame attached to observations.
- Circular obstacle avoidance.
- Vector steering oracle baseline.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from benchmarks.envs.protocol import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.grounding.contracts.observation import CoordinateType, ReferenceFrame
from hbllm.hcir.world.percept import EmbodiedObservation


class ContinuousReachEnvironment(BenchmarkEnvironment):
    """Continuous 2D point navigation task with obstacle avoidance."""

    def __init__(
        self,
        bounds: tuple[float, float] = (10.0, 10.0),
        step_size: float = 0.5,
        goal_radius: float = 0.6,
        max_steps: int = 60,
    ) -> None:
        self.bounds = bounds
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps

        self.agent_pos: np.ndarray = np.array([1.0, 1.0], dtype=float)
        self.goal_pos: np.ndarray = np.array([9.0, 9.0], dtype=float)
        self.obstacle_center: np.ndarray = np.array([5.0, 5.0], dtype=float)
        self.obstacle_radius: float = 1.8

        self.reference_frame = ReferenceFrame.generic_nd(
            dimensions=2,
            coordinate_type=CoordinateType.CONTINUOUS,
            frame_id="continuous_cartesian_2d",
            axes=("x", "y"),
            bounds=((0.0, bounds[0]), (0.0, bounds[1])),
            unit="meter",
            origin_description="bottom_left_metric",
        )

        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset environment to initial continuous pose generated from seed."""
        self.current_seed = seed if seed is not None else random.randint(0, 1000000)
        rng = random.Random(self.current_seed)

        self.agent_pos = np.array([rng.uniform(1.0, 2.5), rng.uniform(1.0, 2.5)], dtype=float)
        self.goal_pos = np.array([rng.uniform(7.5, 9.0), rng.uniform(7.5, 9.0)], dtype=float)
        self.obstacle_center = np.array([rng.uniform(4.5, 5.5), rng.uniform(4.5, 5.5)], dtype=float)
        self.obstacle_radius = 1.6

        self.steps = 0
        self.done = False
        return self._make_observation()

    def _render_lattice(self) -> np.ndarray:
        """Render discrete 10x10 lattice representation of the continuous workspace."""
        grid = np.zeros((10, 10), dtype=int)
        # Perimeter walls
        grid[0, :] = 5
        grid[-1, :] = 5
        grid[:, 0] = 5
        grid[:, -1] = 5

        # Render obstacle circle
        for r in range(1, 9):
            for c in range(1, 9):
                pos = np.array([c + 0.5, 9.5 - r], dtype=float)
                if np.linalg.norm(pos - self.obstacle_center) <= self.obstacle_radius:
                    grid[r, c] = 5

        # Render goal
        gr = int(np.clip(round(9.5 - self.goal_pos[1]), 1, 8))
        gc = int(np.clip(round(self.goal_pos[0] - 0.5), 1, 8))
        grid[gr, gc] = 8

        # Render agent
        ar = int(np.clip(round(9.5 - self.agent_pos[1]), 1, 8))
        ac = int(np.clip(round(self.agent_pos[0] - 0.5), 1, 8))
        grid[ar, ac] = 1

        return grid

    def _make_observation(self) -> EmbodiedObservation:
        lattice = self._render_lattice()
        return EmbodiedObservation(
            lattice=lattice,
            available_actions=[
                ACTION_UP,
                ACTION_DOWN,
                ACTION_LEFT,
                ACTION_RIGHT,
            ],
            context_id=f"continuous_reach_{self.current_seed}",
            episode_id=f"ep_{self.steps}",
            entities=(),
            metadata={
                "reference_frame": self.reference_frame,
                "agent_pos": self.agent_pos.copy(),
                "goal_pos": self.goal_pos.copy(),
                "obstacle_center": self.obstacle_center.copy(),
                "obstacle_radius": self.obstacle_radius,
            },
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return self._make_observation(), 0.0, True, {"success": False, "steps": self.steps}

        self.steps += 1
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)

        vel = np.array([0.0, 0.0], dtype=float)
        if act_id == ACTION_UP:
            vel[1] = self.step_size
        elif act_id == ACTION_DOWN:
            vel[1] = -self.step_size
        elif act_id == ACTION_LEFT:
            vel[0] = -self.step_size
        elif act_id == ACTION_RIGHT:
            vel[0] = self.step_size

        new_pos = self.agent_pos + vel
        new_pos[0] = np.clip(new_pos[0], 0.5, self.bounds[0] - 0.5)
        new_pos[1] = np.clip(new_pos[1], 0.5, self.bounds[1] - 0.5)

        # Check collision with obstacle
        dist_to_obs = np.linalg.norm(new_pos - self.obstacle_center)
        if dist_to_obs > self.obstacle_radius:
            self.agent_pos = new_pos

        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.02
        success = False

        if dist_to_goal <= self.goal_radius:
            self.done = True
            success = True
            reward = 1.0

        if self.steps >= self.max_steps:
            self.done = True

        info = {
            "success": success,
            "dist_to_goal": float(dist_to_goal),
            "steps": self.steps,
        }
        return self._make_observation(), reward, self.done, info

    def get_oracle_action(self) -> int | None:
        """Exact continuous A* search over kinematic state (x, y) with obstacle avoidance."""
        import heapq

        start = (round(float(self.agent_pos[0]), 3), round(float(self.agent_pos[1]), 3))
        queue: list[tuple[float, float, tuple[float, float], list[int]]] = []
        h0 = float(np.linalg.norm(np.array(start) - self.goal_pos))
        heapq.heappush(queue, (h0, 0.0, start, []))
        visited: set[tuple[float, float]] = {start}

        actions = [
            (ACTION_UP, np.array([0.0, self.step_size])),
            (ACTION_DOWN, np.array([0.0, -self.step_size])),
            (ACTION_LEFT, np.array([-self.step_size, 0.0])),
            (ACTION_RIGHT, np.array([self.step_size, 0.0])),
        ]

        while queue:
            _, cost, (cx, cy), path = heapq.heappop(queue)
            cpos = np.array([cx, cy])
            if np.linalg.norm(cpos - self.goal_pos) <= self.goal_radius:
                return path[0] if path else ACTION_UP

            for act, vel in actions:
                npos = cpos + vel
                if (
                    0.5 <= npos[0] <= self.bounds[0] - 0.5
                    and 0.5 <= npos[1] <= self.bounds[1] - 0.5
                ):
                    if np.linalg.norm(npos - self.obstacle_center) > self.obstacle_radius:
                        key = (round(float(npos[0]), 3), round(float(npos[1]), 3))
                        if key not in visited:
                            visited.add(key)
                            h = float(np.linalg.norm(npos - self.goal_pos))
                            heapq.heappush(
                                queue,
                                (
                                    cost + self.step_size + h,
                                    cost + self.step_size,
                                    key,
                                    path + [act],
                                ),
                            )

        return ACTION_UP
