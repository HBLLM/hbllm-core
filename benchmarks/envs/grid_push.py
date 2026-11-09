"""Closed-Loop Procedural Grid Push (Sokoban-style) Environment.

Generates seedable block-pushing tasks with raw sensory lattices (no oracle entity roles):
- Avatar (value 1)
- Movable Box (value 4)
- Target Switch / Goal (value 8)
- Walls (value 5)
- Empty / Walkable (value 0)
"""

from __future__ import annotations

import heapq
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
from hbllm.hcir.world.percept import EmbodiedObservation


class GridPushEnvironment(BenchmarkEnvironment):
    """Seedable procedural grid push task where agent must maneuver and push a block to a goal."""

    def __init__(self, size: int = 8, max_steps: int = 100) -> None:
        self.size = size
        self.max_steps = max_steps
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.avatar_pos: tuple[int, int] = (1, 1)
        self.box_pos: tuple[int, int] = (2, 2)
        self.goal_pos: tuple[int, int] = (size - 2, size - 2)
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset environment to initial state deterministically from seed."""
        self.current_seed = seed if seed is not None else random.randint(0, 1000000)
        rng = random.Random(self.current_seed)

        self.size = 8
        self.grid = np.zeros((self.size, self.size), dtype=int)
        # Perimeter walls
        self.grid[0, :] = 5
        self.grid[-1, :] = 5
        self.grid[:, 0] = 5
        self.grid[:, -1] = 5

        # Place target goal away from corners so pushing is always solvable
        self.goal_pos = (rng.randint(2, self.size - 3), rng.randint(2, self.size - 3))

        # Place box in interior, distinct from goal
        self.box_pos = (rng.randint(2, self.size - 3), rng.randint(2, self.size - 3))
        while self.box_pos == self.goal_pos:
            self.box_pos = (rng.randint(2, self.size - 3), rng.randint(2, self.size - 3))

        # Place avatar in interior, distinct from box and goal
        self.avatar_pos = (rng.randint(1, self.size - 2), rng.randint(1, self.size - 2))
        while self.avatar_pos in (self.box_pos, self.goal_pos):
            self.avatar_pos = (rng.randint(1, self.size - 2), rng.randint(1, self.size - 2))

        # Place 1-2 interior obstacle walls that don't trap the box
        for _ in range(2):
            wr, wc = rng.randint(1, self.size - 2), rng.randint(1, self.size - 2)
            if (wr, wc) not in (self.avatar_pos, self.box_pos, self.goal_pos):
                self.grid[wr, wc] = 5

        # Render entities
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 8
        self.grid[self.box_pos[0], self.box_pos[1]] = 4
        self.grid[self.avatar_pos[0], self.avatar_pos[1]] = 1

        self.steps = 0
        self.done = False
        return self._make_observation()

    def _make_observation(self) -> EmbodiedObservation:
        return EmbodiedObservation(
            lattice=self.grid.copy(),
            available_actions=[
                ACTION_UP,
                ACTION_DOWN,
                ACTION_LEFT,
                ACTION_RIGHT,
            ],
            context_id=f"grid_push_{self.current_seed}",
            episode_id=f"ep_{self.steps}",
            entities=(),  # Zero oracle role labels
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return self._make_observation(), 0.0, True, {"success": False, "steps": self.steps}

        self.steps += 1
        r, c = self.avatar_pos
        dr, dc = 0, 0

        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)

        if act_id == ACTION_UP:
            dr, dc = -1, 0
        elif act_id == ACTION_DOWN:
            dr, dc = 1, 0
        elif act_id == ACTION_LEFT:
            dr, dc = 0, -1
        elif act_id == ACTION_RIGHT:
            dr, dc = 0, 1

        next_avatar = (r + dr, c + dc)
        reward = -0.01
        success = False

        if 0 <= next_avatar[0] < self.size and 0 <= next_avatar[1] < self.size:
            target_cell = self.grid[next_avatar[0], next_avatar[1]]

            if next_avatar == self.box_pos:
                # Attempt to push box
                next_box = (self.box_pos[0] + dr, self.box_pos[1] + dc)
                if (
                    0 <= next_box[0] < self.size
                    and 0 <= next_box[1] < self.size
                    and self.grid[next_box[0], next_box[1]] not in (5, 4)
                ):
                    # Box can move
                    self.grid[self.box_pos[0], self.box_pos[1]] = 0
                    self.box_pos = next_box
                    self.grid[self.box_pos[0], self.box_pos[1]] = 4

                    # Avatar moves into box's previous spot
                    self.grid[self.avatar_pos[0], self.avatar_pos[1]] = (
                        8 if self.avatar_pos == self.goal_pos else 0
                    )
                    self.avatar_pos = next_avatar
                    self.grid[self.avatar_pos[0], self.avatar_pos[1]] = 1

                    # Check if box reached goal
                    if self.box_pos == self.goal_pos:
                        self.done = True
                        success = True
                        reward = 1.0
                        return (
                            self._make_observation(),
                            reward,
                            True,
                            {
                                "success": True,
                                "reason": "goal_reached",
                                "steps": self.steps,
                            },
                        )
            elif target_cell != 5:
                # Avatar moves into empty cell or goal
                self.grid[self.avatar_pos[0], self.avatar_pos[1]] = (
                    8 if self.avatar_pos == self.goal_pos else 0
                )
                self.avatar_pos = next_avatar
                self.grid[self.avatar_pos[0], self.avatar_pos[1]] = 1

        if self.steps >= self.max_steps:
            self.done = True

        info = {
            "success": success,
            "steps": self.steps,
            "box_at_goal": self.box_pos == self.goal_pos,
        }
        return self._make_observation(), reward, self.done, info

    def get_oracle_action(self) -> int | None:
        """Compute optimal action using A* over state (avatar_pos, box_pos)."""
        start_state = (self.avatar_pos, self.box_pos)
        target_box = self.goal_pos

        # Priority queue: (f_score, g_score, state, path)
        queue: list[tuple[int, int, tuple[tuple[int, int], tuple[int, int]], list[int]]] = []
        h_start = abs(self.box_pos[0] - target_box[0]) + abs(self.box_pos[1] - target_box[1])
        heapq.heappush(queue, (h_start, 0, start_state, []))
        visited: set[tuple[tuple[int, int], tuple[int, int]]] = {start_state}

        while queue:
            _, g, (av, box), path = heapq.heappop(queue)
            if box == target_box:
                return path[0] if path else ACTION_UP

            r, c = av
            for action_dir, dr, dc in (
                (ACTION_UP, -1, 0),
                (ACTION_DOWN, 1, 0),
                (ACTION_LEFT, 0, -1),
                (ACTION_RIGHT, 0, 1),
            ):
                nav = (r + dr, c + dc)
                if not (0 <= nav[0] < self.size and 0 <= nav[1] < self.size):
                    continue
                if self.grid[nav[0], nav[1]] == 5 and nav != box:
                    continue

                if nav == box:
                    # Pushing box
                    nbox = (box[0] + dr, box[1] + dc)
                    if (
                        0 <= nbox[0] < self.size
                        and 0 <= nbox[1] < self.size
                        and self.grid[nbox[0], nbox[1]] != 5
                    ):
                        next_state = (nav, nbox)
                        if next_state not in visited:
                            visited.add(next_state)
                            h = abs(nbox[0] - target_box[0]) + abs(nbox[1] - target_box[1])
                            heapq.heappush(
                                queue,
                                (g + 1 + h, g + 1, next_state, path + [action_dir]),
                            )
                else:
                    # Avatar moving freely
                    next_state = (nav, box)
                    if next_state not in visited:
                        visited.add(next_state)
                        h = abs(box[0] - target_box[0]) + abs(box[1] - target_box[1])
                        heapq.heappush(
                            queue,
                            (g + 1 + h, g + 1, next_state, path + [action_dir]),
                        )

        return ACTION_UP
