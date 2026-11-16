"""Dev-side Novelty and Invariance Environments.

Promotes the 2026-11-15 generalization probes into reproducible BenchmarkEnvironment
classes (dev split only) with ground-truth oracle solvers and zero entity-role leakage:
- PaletteSwapGridNavEnv: role-to-value bijective permutation
- NonzeroBgGridNavEnv: non-zero floor value (floor=6, walls=0)
- TransposedGridNavEnv: horizontal dividing wall via lattice transposition
- ActionRelabelGridNavEnv: permuted directional action IDs
- TwoKeysGridEnv: 3 rooms, 2 distinct locked doors and keys (depth-2 prerequisite chain)
- SwitchGateGridEnv: remote floor switch opens an ordinary-looking wall cell
- DecoyHazardGridEnv: two unique tiles (one fatal, one goal) randomized per seed
- PerfectMazeGridEnv: 15x15 DFS perfect maze
"""

from __future__ import annotations

import heapq
import random
from collections import deque
from dataclasses import replace
from typing import Any

import numpy as np

from benchmarks.envs.grid_nav import GridNavEnvironment
from benchmarks.envs.protocol import (
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.world.percept import EmbodiedObservation, TerminalSignal

_MOVES: dict[int, tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class _GridNavTransformBase(BenchmarkEnvironment):
    """Base wrapper applying a sensory/motor bijection over GridNavEnvironment."""

    palette: dict[int, int] = {}
    transpose: bool = False
    act_to_inner: dict[int, int] = {}

    def __init__(self, size: int = 10, max_steps: int = 100) -> None:
        self._inner = GridNavEnvironment(size=size, max_steps=max_steps)
        self._inner_to_act = {v: k for k, v in self.act_to_inner.items()}

    def _transform_obs(self, obs: EmbodiedObservation) -> EmbodiedObservation:
        lat = np.asarray(obs.lattice)
        if self.palette:
            out = lat.copy()
            for src_val, dst_val in self.palette.items():
                out[lat == src_val] = dst_val
            lat = out
        if self.transpose:
            lat = lat.T.copy()
        return replace(obs, lattice=lat, entities=())

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        return self._transform_obs(self._inner.reset(seed=seed))

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)
        mapped = self.act_to_inner.get(act_id, act_id)
        if self.transpose and isinstance(mapped, int):
            mapped = {
                ACTION_UP: ACTION_LEFT,
                ACTION_LEFT: ACTION_UP,
                ACTION_DOWN: ACTION_RIGHT,
                ACTION_RIGHT: ACTION_DOWN,
            }.get(mapped, mapped)
        obs, reward, done, info = self._inner.step(mapped)
        return self._transform_obs(obs), reward, done, info

    def get_oracle_action(self) -> int | None:
        inner_act = self._inner.get_oracle_action()
        if inner_act is None:
            return None
        if self.transpose:
            inner_act = {
                ACTION_UP: ACTION_LEFT,
                ACTION_LEFT: ACTION_UP,
                ACTION_DOWN: ACTION_RIGHT,
                ACTION_RIGHT: ACTION_DOWN,
            }.get(inner_act, inner_act)
        return self._inner_to_act.get(inner_act, inner_act)


class PaletteSwapGridNavEnv(_GridNavTransformBase):
    palette = {5: 7, 3: 6, 2: 4, 8: 3, 9: 2, 1: 9}


class NonzeroBgGridNavEnv(_GridNavTransformBase):
    palette = {0: 6, 5: 0, 3: 7, 2: 4, 8: 9, 9: 3, 1: 2}


class TransposedGridNavEnv(_GridNavTransformBase):
    transpose = True


class ActionRelabelGridNavEnv(_GridNavTransformBase):
    act_to_inner = {
        ACTION_UP: ACTION_RIGHT,
        ACTION_RIGHT: ACTION_DOWN,
        ACTION_DOWN: ACTION_LEFT,
        ACTION_LEFT: ACTION_UP,
    }


class _CustomTileWorld(BenchmarkEnvironment):
    """Minimal deterministic grid world base with zero oracle entity leakage."""

    world_name: str = "tile_world"
    floor_val: int = 0
    wall_val: int = 4
    avatar_val: int = 6
    goal_val: int = 7

    def __init__(self, max_steps: int = 150) -> None:
        self.max_steps = max_steps
        self.grid: np.ndarray = np.zeros((10, 10), dtype=int)
        self.pos: tuple[int, int] = (1, 1)
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def _build(self, rng: random.Random) -> None:
        raise NotImplementedError

    def _is_blocked(self, cell_val: int) -> bool:
        return cell_val == self.wall_val

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == self.goal_val:
            return "win"
        return None

    def _box(self, h: int, w: int) -> None:
        self.grid = np.full((h, w), self.floor_val, dtype=int)
        self.grid[0, :] = self.wall_val
        self.grid[-1, :] = self.wall_val
        self.grid[:, 0] = self.wall_val
        self.grid[:, -1] = self.wall_val

    def _free_cell(
        self, rng: random.Random, rows: tuple[int, int], cols: tuple[int, int]
    ) -> tuple[int, int]:
        while True:
            cand = (rng.randint(rows[0], rows[1]), rng.randint(cols[0], cols[1]))
            if self.grid[cand] == self.floor_val and cand != self.pos:
                return cand

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        lat = self.grid.copy()
        lat[self.pos] = self.avatar_val
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
            context_id=f"{self.world_name}_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.steps = 0
        self.done = False
        self.pos = (-1, -1)
        self._build(random.Random(self.current_seed))
        return self._make_obs()

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
        if act_id in _MOVES:
            dr, dc = _MOVES[act_id]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            cell_val = int(self.grid[nr, nc])
            if not self._is_blocked(cell_val):
                self.pos = (nr, nc)
                outcome = self._on_enter(cell_val, (nr, nc))
                if outcome == "win":
                    self.done = True
                    return (
                        self._make_obs(TerminalSignal.SUCCESS),
                        1.0,
                        True,
                        {"success": True, "reason": "goal_reached", "steps": self.steps},
                    )
                if outcome == "die":
                    self.done = True
                    return (
                        self._make_obs(TerminalSignal.FAILURE),
                        -1.0,
                        True,
                        {"success": False, "reason": "hazard_collision", "steps": self.steps},
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

    def _bfs_next_action(
        self, target: tuple[int, int], passable_extra: set[int] | None = None
    ) -> int:
        extra = passable_extra or set()
        q: deque[tuple[tuple[int, int], list[int]]] = deque([(self.pos, [])])
        seen = {self.pos}
        h, w = self.grid.shape
        while q:
            cur, path = q.popleft()
            if cur == target:
                return path[0] if path else ACTION_INTERACT
            for act, (dr, dc) in _MOVES.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                    v = int(self.grid[nr, nc])
                    if not self._is_blocked(v) or v in extra or (nr, nc) == target:
                        seen.add((nr, nc))
                        q.append(((nr, nc), [*path, act]))
        return ACTION_UP


class TwoKeysGridEnv(_CustomTileWorld):
    world_name = "two_keys"

    def __init__(self, max_steps: int = 150) -> None:
        super().__init__(max_steps=max_steps)
        self.key1_pos = (1, 1)
        self.key2_pos = (1, 6)
        self.goal_pos = (1, 12)
        self.has_k1 = False
        self.has_k2 = False

    def _build(self, rng: random.Random) -> None:
        self._box(8, 15)
        self.has_k1 = False
        self.has_k2 = False
        for col, door_val in ((5, 2), (10, 5)):
            self.grid[1:-1, col] = self.wall_val
            self.grid[rng.randint(1, 6), col] = door_val
        self.pos = self._free_cell(rng, (1, 6), (1, 4))
        self.key1_pos = self._free_cell(rng, (1, 6), (1, 4))
        self.grid[self.key1_pos] = 3
        self.key2_pos = self._free_cell(rng, (1, 6), (6, 9))
        self.grid[self.key2_pos] = 9
        self.goal_pos = self._free_cell(rng, (1, 6), (11, 13))
        self.grid[self.goal_pos] = self.goal_val

    def _is_blocked(self, cell_val: int) -> bool:
        return cell_val in (self.wall_val, 2, 5)

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == 3:
            self.has_k1 = True
            self.grid[rc] = self.floor_val
            self.grid[self.grid == 2] = self.floor_val
        elif cell_val == 9:
            self.has_k2 = True
            self.grid[rc] = self.floor_val
            self.grid[self.grid == 5] = self.floor_val
        elif cell_val == self.goal_val:
            return "win"
        return None

    def get_oracle_action(self) -> int | None:
        target = (
            self.key1_pos
            if not self.has_k1
            else (self.key2_pos if not self.has_k2 else self.goal_pos)
        )
        return self._bfs_next_action(target)


class SwitchGateGridEnv(_CustomTileWorld):
    world_name = "switch_gate"

    def __init__(self, max_steps: int = 150) -> None:
        super().__init__(max_steps=max_steps)
        self.switch_pos = (1, 1)
        self.gate_pos = (4, 5)
        self.goal_pos = (1, 7)
        self.gate_open = False

    def _build(self, rng: random.Random) -> None:
        self._box(10, 10)
        self.gate_open = False
        self.grid[1:-1, 5] = self.wall_val
        self.gate_pos = (rng.randint(2, 7), 5)
        self.pos = self._free_cell(rng, (1, 8), (1, 4))
        self.switch_pos = self._free_cell(rng, (1, 8), (1, 4))
        self.grid[self.switch_pos] = 3
        self.goal_pos = self._free_cell(rng, (1, 8), (6, 8))
        self.grid[self.goal_pos] = self.goal_val

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == 3:
            self.gate_open = True
            self.grid[self.gate_pos] = self.floor_val
        elif cell_val == self.goal_val:
            return "win"
        return None

    def get_oracle_action(self) -> int | None:
        target = self.switch_pos if not self.gate_open else self.goal_pos
        return self._bfs_next_action(target)


class DecoyHazardGridEnv(_CustomTileWorld):
    world_name = "decoy_hazard"

    def __init__(self, max_steps: int = 80) -> None:
        super().__init__(max_steps=max_steps)
        self.win_val = 8
        self.die_val = 9
        self.win_pos = (1, 1)
        self.die_pos = (2, 2)

    def _build(self, rng: random.Random) -> None:
        self._box(10, 10)
        self.win_val, self.die_val = rng.sample([2, 3, 5, 8, 9], 2)
        self.pos = self._free_cell(rng, (1, 8), (1, 8))
        self.win_pos = self._free_cell(rng, (1, 8), (1, 8))
        self.grid[self.win_pos] = self.win_val
        self.die_pos = self._free_cell(rng, (1, 8), (1, 8))
        self.grid[self.die_pos] = self.die_val

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == self.win_val:
            return "win"
        if cell_val == self.die_val:
            return "die"
        return None

    def get_oracle_action(self) -> int | None:
        q: list[tuple[int, tuple[int, int], list[int]]] = [(0, self.pos, [])]
        seen = {self.pos}
        h, w = self.grid.shape
        while q:
            cost, cur, path = heapq.heappop(q)
            if cur == self.win_pos:
                return path[0] if path else ACTION_INTERACT
            for act, (dr, dc) in _MOVES.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                    v = int(self.grid[nr, nc])
                    if v != self.wall_val and v != self.die_val:
                        seen.add((nr, nc))
                        heapq.heappush(q, (cost + 1, (nr, nc), [*path, act]))
        return ACTION_UP


class PerfectMazeGridEnv(_CustomTileWorld):
    world_name = "perfect_maze"

    def __init__(self, size: int = 15, max_steps: int = 250) -> None:
        super().__init__(max_steps=max_steps)
        self.maze_size = size if size % 2 == 1 else size + 1
        self.goal_pos = (self.maze_size - 2, self.maze_size - 2)

    def _build(self, rng: random.Random) -> None:
        n = self.maze_size
        self.grid = np.full((n, n), self.wall_val, dtype=int)
        stack = [(1, 1)]
        self.grid[1, 1] = self.floor_val
        while stack:
            r, c = stack[-1]
            neighbors = [
                (r + dr, c + dc, r + dr // 2, c + dc // 2)
                for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 0 < r + dr < n - 1
                and 0 < c + dc < n - 1
                and self.grid[r + dr, c + dc] == self.wall_val
            ]
            if not neighbors:
                stack.pop()
                continue
            nr, nc, mr, mc = rng.choice(neighbors)
            self.grid[mr, mc] = self.floor_val
            self.grid[nr, nc] = self.floor_val
            stack.append((nr, nc))
        self.pos = (1, 1)
        self.goal_pos = (n - 2, n - 2)
        self.grid[self.goal_pos] = self.goal_val

    def get_oracle_action(self) -> int | None:
        return self._bfs_next_action(self.goal_pos)
