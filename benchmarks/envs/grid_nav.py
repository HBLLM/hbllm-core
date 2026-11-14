"""Closed-Loop Procedural Grid Navigation Environment.

Generates seedable maze navigation tasks with raw sensory lattices (no oracle entity roles):
- Avatar (value 1)
- Locked Door (value 2)
- Key (value 3)
- Walls (value 5)
- Goal (value 8)
- Hazard (value 9)
- Empty / Walkable (value 0)
"""

from __future__ import annotations

import heapq
import random
from typing import Any

import numpy as np

from benchmarks.envs.protocol import (
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.world.percept import EmbodiedObservation, TerminalSignal


class GridNavEnvironment(BenchmarkEnvironment):
    """Seedable procedural grid navigation with key-door puzzle and hazard traps."""

    def __init__(self, size: int = 10, max_steps: int = 100) -> None:
        self.size = size
        self.max_steps = max_steps
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.avatar_pos: tuple[int, int] = (1, 1)
        self.key_pos: tuple[int, int] = (1, 1)
        self.door_pos: tuple[int, int] = (1, 5)
        self.goal_pos: tuple[int, int] = (size - 2, size - 2)
        self.hazards: set[tuple[int, int]] = set()
        self.has_key: bool = False
        self.door_unlocked: bool = False
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset environment to initial state generated deterministically from seed."""
        self.current_seed = seed if seed is not None else random.randint(0, 1000000)
        rng = random.Random(self.current_seed)

        self.size = 10
        self.grid = np.zeros((self.size, self.size), dtype=int)
        # Perimeter boundary walls
        self.grid[0, :] = 5
        self.grid[-1, :] = 5
        self.grid[:, 0] = 5
        self.grid[:, -1] = 5

        # Dividing wall at col 5 with one doorway
        door_r = rng.randint(2, self.size - 3)
        self.door_pos = (door_r, 5)
        for r in range(1, self.size - 1):
            if r == door_r:
                self.grid[r, 5] = 2  # Locked Door
            else:
                self.grid[r, 5] = 5  # Wall

        # Avatar in Room 1 (cols 1..4)
        self.avatar_pos = (rng.randint(1, self.size - 2), rng.randint(1, 3))
        while self.grid[self.avatar_pos[0], self.avatar_pos[1]] != 0:
            self.avatar_pos = (rng.randint(1, self.size - 2), rng.randint(1, 3))

        # Key in Room 1 (cols 1..4), distinct from avatar
        self.key_pos = (rng.randint(1, self.size - 2), rng.randint(1, 4))
        while self.grid[self.key_pos[0], self.key_pos[1]] != 0 or self.key_pos == self.avatar_pos:
            self.key_pos = (rng.randint(1, self.size - 2), rng.randint(1, 4))
        self.grid[self.key_pos[0], self.key_pos[1]] = 3  # Key

        # Goal in Room 2 (cols 6..8)
        self.goal_pos = (rng.randint(1, self.size - 2), rng.randint(6, self.size - 2))
        while self.grid[self.goal_pos[0], self.goal_pos[1]] != 0:
            self.goal_pos = (rng.randint(1, self.size - 2), rng.randint(6, self.size - 2))
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 8  # Goal

        # Place 2-3 isolated hazard tiles (value 9) in empty spots
        self.hazards.clear()
        for _ in range(2):
            hr, hc = rng.randint(1, self.size - 2), rng.randint(1, self.size - 2)
            if (
                self.grid[hr, hc] == 0
                and (hr, hc) != self.avatar_pos
                and (hr, hc) != self.key_pos
                and (hr, hc) != self.goal_pos
                and (hr, hc) != (door_r, 4)
                and (hr, hc) != (door_r, 6)
            ):
                self.grid[hr, hc] = 9
                self.hazards.add((hr, hc))

        # Render avatar on grid
        self.grid[self.avatar_pos[0], self.avatar_pos[1]] = 1

        self.has_key = False
        self.door_unlocked = False
        self.steps = 0
        self.done = False

        return self._make_observation()

    def _make_observation(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        """Construct raw observation lattice without oracle role labels."""
        return EmbodiedObservation(
            lattice=self.grid.copy(),
            available_actions=[
                ACTION_UP,
                ACTION_DOWN,
                ACTION_LEFT,
                ACTION_RIGHT,
                ACTION_INTERACT,
            ],
            terminal=terminal,
            context_id=f"grid_nav_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),  # Zero oracle entity roles
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        """Advance simulation by executing proposed motor action."""
        if self.done:
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "steps": self.steps},
            )

        self.steps += 1
        r, c = self.avatar_pos
        dr, dc = 0, 0

        # Coerce action to integer if wrapped
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
        elif act_id == ACTION_INTERACT:
            # Check adjacent for key pick-up
            for ar, ac in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                adj_pos = (r + ar, c + ac)
                if adj_pos == self.key_pos and not self.has_key:
                    self._collect_key()

        target_r, target_c = r + dr, c + dc
        target_cell = self.grid[target_r, target_c]

        # Check collision with walls or locked doors
        is_blocked = target_cell == 5 or (target_cell == 2 and not self.door_unlocked)

        reward = -0.01  # Small step penalty
        success = False

        if not is_blocked:
            # Clear old avatar position
            if self.avatar_pos == self.key_pos and not self.has_key:
                self.grid[r, c] = 3
            else:
                self.grid[r, c] = 0

            # Step onto new position
            self.avatar_pos = (target_r, target_c)

            # Auto-collect key on stepping
            if self.avatar_pos == self.key_pos and not self.has_key:
                self._collect_key()

            # Stepped into hazard trap
            if self.avatar_pos in self.hazards:
                self.done = True
                reward = -1.0
                self.grid[target_r, target_c] = 9
                return (
                    self._make_observation(terminal=TerminalSignal.FAILURE),
                    reward,
                    True,
                    {
                        "success": False,
                        "reason": "hazard_collision",
                        "steps": self.steps,
                    },
                )

            # Stepped onto goal
            if self.avatar_pos == self.goal_pos:
                self.done = True
                success = True
                reward = 1.0
                self.grid[target_r, target_c] = 1
                return (
                    self._make_observation(terminal=TerminalSignal.SUCCESS),
                    reward,
                    True,
                    {
                        "success": True,
                        "reason": "goal_reached",
                        "steps": self.steps,
                    },
                )

            # Update avatar on grid
            self.grid[target_r, target_c] = 1

        terminal = None
        if self.steps >= self.max_steps:
            self.done = True
            terminal = TerminalSignal.TIMEOUT

        info = {
            "success": success,
            "has_key": self.has_key,
            "door_unlocked": self.door_unlocked,
            "steps": self.steps,
        }
        return self._make_observation(terminal=terminal), reward, self.done, info

    def _collect_key(self) -> None:
        """Collect key and unlock barrier door."""
        self.has_key = True
        self.door_unlocked = True
        dr, dc = self.door_pos
        self.grid[dr, dc] = 0  # Door opens

    def get_oracle_action(self) -> int | None:
        """Compute optimal ground truth action using A* pathfinding."""
        cur = self.avatar_pos
        target = self.key_pos if not self.has_key else self.goal_pos

        # A* search
        queue: list[tuple[int, int, tuple[int, int], list[int]]] = []
        heapq.heappush(queue, (0, 0, cur, []))
        visited: set[tuple[int, int]] = {cur}

        while queue:
            _, cost, pos, path = heapq.heappop(queue)
            if pos == target:
                return path[0] if path else ACTION_INTERACT

            r, c = pos
            for action_dir, dr, dc in (
                (ACTION_UP, -1, 0),
                (ACTION_DOWN, 1, 0),
                (ACTION_LEFT, 0, -1),
                (ACTION_RIGHT, 0, 1),
            ):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    cell = self.grid[nr, nc]
                    can_walk = cell != 5 and cell != 9 and (cell != 2 or self.door_unlocked)
                    if can_walk and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        h = abs(nr - target[0]) + abs(nc - target[1])
                        heapq.heappush(
                            queue, (cost + 1 + h, cost + 1, (nr, nc), path + [action_dir])
                        )

        return ACTION_UP
