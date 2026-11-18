"""Sealed Held-Out v2 Environments — Novel Capability Families.

Introduces two task families absent from dev suites:
1. PartiallyObservableGridEnv (family: partial_observability)
   - Cells outside local Chebyshev visibility radius R=2 are masked as fog (value 7).
   - Requires spatial working memory and systematic exploration across rooms.
2. StochasticPatrolEnv (family: stochastic_dynamic)
   - Moving hazard entity patrolling a central corridor on a deterministic cycle,
     combined with seeded transition slip noise (p=0.15 perpendicular drift).
   - Requires closed-loop recovery and dynamic hazard avoidance.
"""

from __future__ import annotations

import random
from collections import deque
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

_DIRS: dict[int, tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class PartiallyObservableGridEnv(BenchmarkEnvironment):
    """Multi-room key-door navigation under local fog-of-war (Chebyshev radius 2)."""

    FOG_VAL = 7

    def __init__(self, size: int = 11, view_radius: int = 2, max_steps: int = 140) -> None:
        self.size = size
        self.view_radius = view_radius
        self.max_steps = max_steps
        self.true_grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.avatar_pos: tuple[int, int] = (1, 1)
        self.key_pos: tuple[int, int] = (1, 2)
        self.door_pos: tuple[int, int] = (5, 5)
        self.goal_pos: tuple[int, int] = (9, 9)
        self.has_key: bool = False
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        rng = random.Random(self.current_seed)
        n = self.size
        self.true_grid = np.zeros((n, n), dtype=int)
        self.true_grid[0, :] = 5
        self.true_grid[-1, :] = 5
        self.true_grid[:, 0] = 5
        self.true_grid[:, -1] = 5

        mid = n // 2
        door_r = rng.randint(2, n - 3)
        self.door_pos = (door_r, mid)
        self.true_grid[1:-1, mid] = 5
        self.true_grid[door_r, mid] = 2

        self.avatar_pos = (rng.randint(1, n - 2), rng.randint(1, mid - 1))
        self.key_pos = self.avatar_pos
        while self.key_pos == self.avatar_pos:
            self.key_pos = (rng.randint(1, n - 2), rng.randint(1, mid - 1))
        self.true_grid[self.key_pos] = 3

        self.goal_pos = (rng.randint(1, n - 2), rng.randint(mid + 1, n - 2))
        self.true_grid[self.goal_pos] = 8

        self.has_key = False
        self.steps = 0
        self.done = False
        return self._make_obs()

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        n = self.size
        lat = np.full((n, n), self.FOG_VAL, dtype=int)
        ar, ac = self.avatar_pos
        r0 = max(0, ar - self.view_radius)
        r1 = min(n, ar + self.view_radius + 1)
        c0 = max(0, ac - self.view_radius)
        c1 = min(n, ac + self.view_radius + 1)
        lat[r0:r1, c0:c1] = self.true_grid[r0:r1, c0:c1]
        lat[ar, ac] = 1
        return EmbodiedObservation(
            lattice=lat,
            available_actions=[
                ACTION_UP,
                ACTION_DOWN,
                ACTION_LEFT,
                ACTION_RIGHT,
                ACTION_INTERACT,
            ],
            terminal=terminal,
            context_id=f"partial_obs_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return (
                self._make_obs(TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)
        self.steps += 1
        if act_id in _DIRS:
            dr, dc = _DIRS[act_id]
            nr, nc = self.avatar_pos[0] + dr, self.avatar_pos[1] + dc
            cell = int(self.true_grid[nr, nc])
            blocked = cell == 5 or (cell == 2 and not self.has_key)
            if not blocked:
                self.avatar_pos = (nr, nc)
                if self.avatar_pos == self.key_pos and not self.has_key:
                    self.has_key = True
                    self.true_grid[self.key_pos] = 0
                    self.true_grid[self.door_pos] = 0
                if self.avatar_pos == self.goal_pos:
                    self.done = True
                    return (
                        self._make_obs(TerminalSignal.SUCCESS),
                        1.0,
                        True,
                        {"success": True, "reason": "goal_reached", "steps": self.steps},
                    )
        if self.steps >= self.max_steps:
            self.done = True
            return (
                self._make_obs(TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )
        return self._make_obs(), -0.01, False, {"success": False, "steps": self.steps}

    def get_oracle_action(self) -> int | None:
        target = self.key_pos if not self.has_key else self.goal_pos
        q: deque[tuple[tuple[int, int], list[int]]] = deque([(self.avatar_pos, [])])
        seen = {self.avatar_pos}
        while q:
            cur, path = q.popleft()
            if cur == target:
                return path[0] if path else ACTION_INTERACT
            for act, (dr, dc) in _DIRS.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in seen:
                    cell = int(self.true_grid[nr, nc])
                    if cell != 5 and not (cell == 2 and not self.has_key):
                        seen.add((nr, nc))
                        q.append(((nr, nc), [*path, act]))
        return ACTION_UP


class StochasticPatrolEnv(BenchmarkEnvironment):
    """Navigation with seeded 15% perpendicular slip noise and a bouncing patrol hazard."""

    def __init__(self, size: int = 10, slip_prob: float = 0.15, max_steps: int = 120) -> None:
        self.size = size
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.base_grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.avatar_pos: tuple[int, int] = (1, 1)
        self.goal_pos: tuple[int, int] = (8, 8)
        self.patrol_r: int = 2
        self.patrol_c: int = 5
        self.patrol_dr: int = 1
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        rng = random.Random(self.current_seed)
        n = self.size
        self.base_grid = np.zeros((n, n), dtype=int)
        self.base_grid[0, :] = 5
        self.base_grid[-1, :] = 5
        self.base_grid[:, 0] = 5
        self.base_grid[:, -1] = 5

        # Partial internal pillar obstacles leaving wide paths around patrol column
        for r in (3, 6):
            self.base_grid[r, 3] = 5
            self.base_grid[r, 7] = 5

        self.avatar_pos = (rng.randint(1, n - 2), rng.randint(1, 2))
        self.goal_pos = (rng.randint(1, n - 2), rng.randint(n - 3, n - 2))
        self.base_grid[self.goal_pos] = 8

        self.patrol_c = n // 2
        self.patrol_r = rng.randint(2, n - 3)
        self.patrol_dr = rng.choice([-1, 1])
        self.steps = 0
        self.done = False
        return self._make_obs()

    def _next_patrol(self, pr: int, pdr: int) -> tuple[int, int]:
        nr = pr + pdr
        if nr <= 1 or nr >= self.size - 2:
            pdr = -pdr
            nr = pr + pdr
        return nr, pdr

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        lat = self.base_grid.copy()
        lat[self.patrol_r, self.patrol_c] = 9
        lat[self.avatar_pos] = 1
        return EmbodiedObservation(
            lattice=lat,
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT],
            terminal=terminal,
            context_id=f"stoch_patrol_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return (
                self._make_obs(TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)
        self.steps += 1

        # Seeded transition slip (reproducible per seed & step)
        step_rng = random.Random(self.current_seed * 10007 + self.steps)
        if act_id in _DIRS and step_rng.random() < self.slip_prob:
            if act_id in (ACTION_UP, ACTION_DOWN):
                act_id = step_rng.choice([ACTION_LEFT, ACTION_RIGHT])
            else:
                act_id = step_rng.choice([ACTION_UP, ACTION_DOWN])

        prev_avatar = self.avatar_pos
        if act_id in _DIRS:
            dr, dc = _DIRS[act_id]
            nr, nc = self.avatar_pos[0] + dr, self.avatar_pos[1] + dc
            if int(self.base_grid[nr, nc]) != 5:
                self.avatar_pos = (nr, nc)

        prev_patrol = (self.patrol_r, self.patrol_c)
        self.patrol_r, self.patrol_dr = self._next_patrol(self.patrol_r, self.patrol_dr)
        curr_patrol = (self.patrol_r, self.patrol_c)

        # Check direct or swap collision with patroller
        hit_patrol = self.avatar_pos == curr_patrol or (
            self.avatar_pos == prev_patrol and prev_avatar == curr_patrol
        )
        if hit_patrol:
            self.done = True
            return (
                self._make_obs(TerminalSignal.FAILURE),
                -1.0,
                True,
                {"success": False, "reason": "hazard_collision", "steps": self.steps},
            )

        if self.avatar_pos == self.goal_pos:
            self.done = True
            return (
                self._make_obs(TerminalSignal.SUCCESS),
                1.0,
                True,
                {"success": True, "reason": "goal_reached", "steps": self.steps},
            )

        if self.steps >= self.max_steps:
            self.done = True
            return (
                self._make_obs(TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )
        return self._make_obs(), -0.01, False, {"success": False, "steps": self.steps}

    def get_oracle_action(self) -> int | None:
        # Space-time BFS with 1-cell safety margin around the patroller
        q: deque[tuple[tuple[int, int], int, int, list[int]]] = deque(
            [(self.avatar_pos, self.patrol_r, self.patrol_dr, [])]
        )
        seen = {(self.avatar_pos, self.patrol_r, self.patrol_dr)}
        while q:
            cur, pr, pdr, path = q.popleft()
            if cur == self.goal_pos:
                return path[0] if path else ACTION_RIGHT
            if len(path) >= 40:
                continue
            npr, npdr = self._next_patrol(pr, pdr)
            for act, (dr, dc) in _DIRS.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                nxt = (nr, nc) if int(self.base_grid[nr, nc]) != 5 else cur
                if abs(nxt[0] - npr) + abs(nxt[1] - self.patrol_c) <= 1 and nxt != self.goal_pos:
                    continue
                state = (nxt, npr, npdr)
                if state not in seen:
                    seen.add(state)
                    q.append((nxt, npr, npdr, [*path, act]))
        return ACTION_RIGHT
