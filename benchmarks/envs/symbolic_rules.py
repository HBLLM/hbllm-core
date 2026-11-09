"""Closed-Loop Procedural Symbolic Rule Induction Environment.

Generates seedable rule-induction tasks with raw sensory lattices (no oracle entity roles):
- Switches of different colors/values (e.g., values 2, 3, 4)
- Cue / Indicator lamp (value 7) indicating required activation target
- Exit Door / Goal (value 8) which opens upon correct switch activation
- Avatar (value 1)
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
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_TOGGLE,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.world.percept import EmbodiedObservation


class SymbolicRulesEnvironment(BenchmarkEnvironment):
    """Seedable symbolic rule-induction environment where agent must induce the switch rule."""

    def __init__(self, size: int = 9, max_steps: int = 80) -> None:
        self.size = size
        self.max_steps = max_steps
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.avatar_pos: tuple[int, int] = (1, 1)
        self.switches: dict[int, tuple[int, int]] = {}  # color_id -> (r, c)
        self.target_color: int = 2
        self.switch_states: dict[int, bool] = {}  # color_id -> toggled
        self.goal_pos: tuple[int, int] = (size - 2, size - 2)
        self.door_pos: tuple[int, int] = (size - 3, size - 2)
        self.door_open: bool = False
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        self.current_seed = seed if seed is not None else random.randint(0, 1000000)
        rng = random.Random(self.current_seed)

        self.size = 9
        self.grid = np.zeros((self.size, self.size), dtype=int)
        # Perimeter walls
        self.grid[0, :] = 5
        self.grid[-1, :] = 5
        self.grid[:, 0] = 5
        self.grid[:, -1] = 5

        # Door blocking the goal room
        self.door_pos = (self.size - 3, self.size - 2)
        self.grid[self.door_pos[0], self.door_pos[1]] = 5  # Barrier wall until rule satisfied
        self.goal_pos = (self.size - 2, self.size - 2)
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 8

        # Cue indicator showing the target color (at top wall row 1, col 4)
        colors = [2, 3, 4]
        self.target_color = rng.choice(colors)
        self.grid[1, 4] = self.target_color  # Visual cue in world

        # Place 3 color switches in the room
        self.switches = {}
        self.switch_states = {c: False for c in colors}
        switch_locs = [(2, 2), (4, 2), (6, 2)]
        rng.shuffle(switch_locs)
        for color, loc in zip(colors, switch_locs):
            self.switches[color] = loc
            self.grid[loc[0], loc[1]] = color

        # Avatar initial position
        self.avatar_pos = (rng.randint(2, 6), rng.randint(4, 6))
        while self.grid[self.avatar_pos[0], self.avatar_pos[1]] != 0:
            self.avatar_pos = (rng.randint(2, 6), rng.randint(4, 6))
        self.grid[self.avatar_pos[0], self.avatar_pos[1]] = 1

        self.door_open = False
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
                ACTION_TOGGLE,
                ACTION_INTERACT,
            ],
            context_id=f"symbolic_rules_{self.current_seed}",
            episode_id=f"ep_{self.steps}",
            entities=(),
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

        reward = -0.01
        success = False

        if act_id in (ACTION_TOGGLE, ACTION_INTERACT):
            # Check if adjacent to any switch
            for color, (sr, sc) in self.switches.items():
                if abs(r - sr) + abs(c - sc) <= 1:
                    self.switch_states[color] = not self.switch_states[color]
                    # If target color switch is active, open door
                    if self.switch_states[self.target_color]:
                        self.door_open = True
                        self.grid[self.door_pos[0], self.door_pos[1]] = 0
                    else:
                        self.door_open = False
                        self.grid[self.door_pos[0], self.door_pos[1]] = 5
        else:
            if act_id == ACTION_UP:
                dr, dc = -1, 0
            elif act_id == ACTION_DOWN:
                dr, dc = 1, 0
            elif act_id == ACTION_LEFT:
                dr, dc = 0, -1
            elif act_id == ACTION_RIGHT:
                dr, dc = 0, 1

            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                cell = self.grid[nr, nc]
                # Can step onto empty space or goal (if open)
                if (nr, nc) == self.goal_pos and self.door_open:
                    self.done = True
                    success = True
                    reward = 1.0
                    self.grid[r, c] = 0
                    self.avatar_pos = (nr, nc)
                    self.grid[nr, nc] = 1
                    return (
                        self._make_observation(),
                        reward,
                        True,
                        {
                            "success": True,
                            "steps": self.steps,
                        },
                    )
                elif cell == 0:
                    self.grid[r, c] = 0
                    self.avatar_pos = (nr, nc)
                    self.grid[nr, nc] = 1

        if self.steps >= self.max_steps:
            self.done = True

        info = {
            "success": success,
            "door_open": self.door_open,
            "steps": self.steps,
        }
        return self._make_observation(), reward, self.done, info

    def get_oracle_action(self) -> int | None:
        """Compute optimal oracle action: navigate to target switch, toggle it, then navigate to goal."""
        target_switch_pos = self.switches[self.target_color]

        if not self.door_open:
            # Need to get adjacent to target switch
            r, c = self.avatar_pos
            sr, sc = target_switch_pos
            if abs(r - sr) + abs(c - sc) <= 1:
                return ACTION_TOGGLE

            # Find path to cell adjacent to target_switch_pos
            target_adj = (sr, sc + 1) if self.grid[sr, sc + 1] == 0 else (sr + 1, sc)
            return self._astar_first_step(self.avatar_pos, target_adj)
        else:
            # Navigate to goal
            return self._astar_first_step(self.avatar_pos, self.goal_pos)

    def _astar_first_step(self, start: tuple[int, int], target: tuple[int, int]) -> int:
        queue: list[tuple[int, int, tuple[int, int], list[int]]] = []
        heapq.heappush(queue, (0, 0, start, []))
        visited: set[tuple[int, int]] = {start}

        while queue:
            _, cost, pos, path = heapq.heappop(queue)
            if pos == target:
                return path[0] if path else ACTION_UP

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
                    can_walk = cell == 0 or (nr, nc) == target
                    if can_walk and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        h = abs(nr - target[0]) + abs(nc - target[1])
                        heapq.heappush(
                            queue,
                            (cost + 1 + h, cost + 1, (nr, nc), path + [action_dir]),
                        )

        return ACTION_UP
