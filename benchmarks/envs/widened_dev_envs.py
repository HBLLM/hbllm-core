"""Widened Dev Benchmark Environments for R9 Regimes.

Covers the 6 generalization regimes in Review §5 (R9):
1. PartialObsMazeEnv (partial_observability): 7x7 local FOV shroud over 11x11 maze
2. StochasticWindNavEnv (stochastic_dynamics): 25% orthogonal transition slip + hazard pit
3. MultiAgentCorridorEnv (multi_agent): autonomous mobile blocker + patroller agents
4. LongHorizonQuestEnv (long_horizon): 4-chamber 3-prerequisite hierarchical quest (13x13)
5. ContinuousMultiWaypointEnv (continuous_control): 2-obstacle chicane + sequential 2-leg control
6. LanguageGoalSelectEnv (language_goals): 3 distractor-free targets disambiguated only by mission text
"""

from __future__ import annotations

import heapq
import random
from collections import deque
from typing import Any

import numpy as np

from benchmarks.envs.novelty_envs import _CustomTileWorld
from benchmarks.envs.protocol import (
    ACTION_DOWN,
    ACTION_INTERACT,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.grounding.contracts.observation import CoordinateType, ReferenceFrame
from hbllm.hcir.world.percept import EmbodiedObservation, TerminalSignal

_MOVES: dict[int, tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class PartialObsMazeEnv(_CustomTileWorld):
    """11x11 two-room key-door maze with 7x7 egocentric visibility shroud (token 9)."""

    world_name = "partial_obs_maze"
    shroud_val: int = 9
    key_val: int = 3
    door_val: int = 5

    def __init__(self, max_steps: int = 140, vision_radius: int = 3) -> None:
        super().__init__(max_steps=max_steps)
        self.vision_radius = vision_radius
        self.has_key = False
        self.door_open = False
        self.key_pos = (1, 1)
        self.door_pos = (5, 5)
        self.goal_pos = (9, 9)

    def _build(self, rng: random.Random) -> None:
        self._box(11, 11)
        for r in range(1, 10):
            self.grid[r, 5] = self.wall_val
        door_r = rng.randint(2, 8)
        self.door_pos = (door_r, 5)
        self.grid[self.door_pos] = self.door_val
        self.pos = self._free_cell(rng, (1, 9), (1, 4))
        self.key_pos = self._free_cell(rng, (1, 9), (1, 4))
        self.grid[self.key_pos] = self.key_val
        self.goal_pos = self._free_cell(rng, (1, 9), (6, 9))
        self.grid[self.goal_pos] = self.goal_val
        self.has_key = False
        self.door_open = False

    def _is_blocked(self, cell_val: int) -> bool:
        return cell_val == self.wall_val or (cell_val == self.door_val and not self.has_key)

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == self.key_val:
            self.has_key = True
            self.grid[rc] = self.floor_val
        elif cell_val == self.door_val and self.has_key:
            self.door_open = True
            self.grid[rc] = self.floor_val
        elif cell_val == self.goal_val:
            return "win"
        return None

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        lat = np.full_like(self.grid, self.shroud_val)
        h, w = self.grid.shape
        pr, pc = self.pos
        r0 = max(0, pr - self.vision_radius)
        r1 = min(h, pr + self.vision_radius + 1)
        c0 = max(0, pc - self.vision_radius)
        c1 = min(w, pc + self.vision_radius + 1)
        lat[r0:r1, c0:c1] = self.grid[r0:r1, c0:c1]
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

    def get_oracle_action(self) -> int | None:
        target = (
            self.key_pos
            if not self.has_key
            else (self.door_pos if not self.door_open else self.goal_pos)
        )
        extra = {self.door_val} if self.has_key else set()
        return self._bfs_next_action(target, passable_extra=extra)


class StochasticWindNavEnv(_CustomTileWorld):
    """10x10 grid with hazards where movement has 25% orthogonal transition slip."""

    world_name = "stochastic_wind"
    wall_val: int = 5
    avatar_val: int = 1
    goal_val: int = 8
    hazard_val: int = 9

    def __init__(self, max_steps: int = 120, slip_prob: float = 0.25) -> None:
        super().__init__(max_steps=max_steps)
        self.slip_prob = slip_prob
        self.goal_pos = (8, 8)
        self.hazards: set[tuple[int, int]] = set()
        self._rng = random.Random(0)

    def _build(self, rng: random.Random) -> None:
        self._rng = random.Random(self.current_seed * 7919 + 13)
        self._box(10, 10)
        wall_col = 5
        gap_r = rng.randint(2, 7)
        for r in range(1, 9):
            if r != gap_r and r != gap_r + 1:
                self.grid[r, wall_col] = self.wall_val
        self.pos = (rng.randint(1, 3), rng.randint(1, 3))
        self.goal_pos = (rng.randint(6, 8), rng.randint(7, 8))
        self.grid[self.goal_pos] = self.goal_val
        self.hazards = set()
        forbidden = {
            self.pos,
            self.goal_pos,
            (gap_r, wall_col),
            (gap_r + 1, wall_col),
        }
        while len(self.hazards) < 3:
            hz = self._free_cell(rng, (2, 7), (2, 8))
            if hz[1] in (4, 5, 6):
                continue
            if any(abs(hz[0] - fr) + abs(hz[1] - fc) <= 1 for fr, fc in forbidden):
                continue
            self.hazards.add(hz)
            self.grid[hz] = self.hazard_val

    def _apply_slip(self, act_id: int) -> int:
        if act_id not in _MOVES or self._rng.random() >= self.slip_prob:
            return act_id
        if act_id in (ACTION_UP, ACTION_DOWN):
            return self._rng.choice([ACTION_LEFT, ACTION_RIGHT])
        return self._rng.choice([ACTION_UP, ACTION_DOWN])

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == self.hazard_val or rc in self.hazards:
            return "die"
        if cell_val == self.goal_val:
            return "win"
        return None

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)
        return super().step(self._apply_slip(act_id))

    def get_oracle_action(self) -> int | None:
        # Risk-aware Dijkstra avoiding hazard-adjacent cells unless unavoidable
        q: list[tuple[float, tuple[int, int], list[int]]] = [(0.0, self.pos, [])]
        best: dict[tuple[int, int], float] = {self.pos: 0.0}
        h, w = self.grid.shape
        while q:
            cost, cur, path = heapq.heappop(q)
            if cost > best.get(cur, float("inf")):
                continue
            if cur == self.goal_pos:
                return path[0] if path else ACTION_RIGHT
            for act, (dr, dc) in _MOVES.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(self.grid[nr, nc])
                    if v == self.wall_val or v == self.hazard_val:
                        continue
                    hz_adj = any(abs(nr - hr) + abs(nc - hc) <= 1 for hr, hc in self.hazards)
                    ncost = cost + (500.0 if hz_adj else 1.0)
                    if ncost < best.get((nr, nc), float("inf")):
                        best[(nr, nc)] = ncost
                        heapq.heappush(q, (ncost, (nr, nc), [*path, act]))
        return ACTION_RIGHT


class MultiAgentCorridorEnv(BenchmarkEnvironment):
    """11x11 environment with autonomous neutral blocker (6) and mobile patroller hazard (7)."""

    wall_val = 5
    floor_val = 0
    avatar_val = 1
    blocker_val = 6
    patroller_val = 7
    goal_val = 8

    def __init__(self, size: int = 11, max_steps: int = 120) -> None:
        self.size = size
        self.max_steps = max_steps
        self.base_grid = np.zeros((size, size), dtype=int)
        self.pos = (1, 1)
        self.goal_pos = (9, 9)
        self.blocker_pos = (5, 5)
        self.blocker_dir = (1, 0)
        self.patroller_pos = (7, 7)
        self.patroller_dir = (0, 1)
        self.steps = 0
        self.done = False
        self.current_seed = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        rng = random.Random(self.current_seed)
        n = self.size
        self.base_grid = np.zeros((n, n), dtype=int)
        self.base_grid[0, :] = self.wall_val
        self.base_grid[-1, :] = self.wall_val
        self.base_grid[:, 0] = self.wall_val
        self.base_grid[:, -1] = self.wall_val
        for r in range(1, n - 1):
            if r not in (3, 7):
                self.base_grid[r, 5] = self.wall_val
        self.pos = (rng.randint(1, 2), rng.randint(1, 3))
        self.goal_pos = (rng.randint(7, 9), rng.randint(7, 9))
        self.base_grid[self.goal_pos] = self.goal_val
        self.blocker_pos = (3, rng.randint(3, 6))
        self.blocker_dir = (0, 1)
        self.patroller_pos = (rng.randint(4, 6), rng.randint(6, 9))
        while self.base_grid[self.patroller_pos] != 0 or self.patroller_pos == self.goal_pos:
            self.patroller_pos = (rng.randint(4, 6), rng.randint(6, 9))
        self.patroller_dir = (1, 0)
        self.steps = 0
        self.done = False
        return self._make_obs()

    def _advance_agent(
        self, pos: tuple[int, int], direction: tuple[int, int], forbidden: set[tuple[int, int]]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        n = self.size
        if (
            1 <= nr < n - 1
            and 1 <= nc < n - 1
            and self.base_grid[nr, nc] == self.floor_val
            and (nr, nc) not in forbidden
        ):
            return (nr, nc), direction
        rev = (-direction[0], -direction[1])
        rr, rc = pos[0] + rev[0], pos[1] + rev[1]
        if (
            1 <= rr < n - 1
            and 1 <= rc < n - 1
            and self.base_grid[rr, rc] == self.floor_val
            and (rr, rc) not in forbidden
        ):
            return (rr, rc), rev
        return pos, rev

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        lat = self.base_grid.copy()
        lat[self.blocker_pos] = self.blocker_val
        lat[self.patroller_pos] = self.patroller_val
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
            context_id=f"multi_agent_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return self._make_obs(TerminalSignal.TIMEOUT), 0.0, True, {"success": False}
        act_id = getattr(action, "value", action)
        if isinstance(act_id, str) and act_id.isdigit():
            act_id = int(act_id)
        self.steps += 1
        if act_id in _MOVES:
            dr, dc = _MOVES[act_id]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if (
                0 <= nr < self.size
                and 0 <= nc < self.size
                and self.base_grid[nr, nc] != self.wall_val
                and (nr, nc) != self.blocker_pos
            ):
                self.pos = (nr, nc)
        if self.pos == self.patroller_pos:
            self.done = True
            return (
                self._make_obs(TerminalSignal.FAILURE),
                -1.0,
                True,
                {"success": False, "reason": "hazard_collision", "steps": self.steps},
            )
        if self.pos == self.goal_pos:
            self.done = True
            return (
                self._make_obs(TerminalSignal.SUCCESS),
                1.0,
                True,
                {"success": True, "reason": "goal_reached", "steps": self.steps},
            )
        # Other autonomous agents step independently
        self.blocker_pos, self.blocker_dir = self._advance_agent(
            self.blocker_pos, self.blocker_dir, {self.pos, self.goal_pos, self.patroller_pos}
        )
        self.patroller_pos, self.patroller_dir = self._advance_agent(
            self.patroller_pos, self.patroller_dir, {self.goal_pos, self.blocker_pos}
        )
        if self.pos == self.patroller_pos:
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

    def get_oracle_action(self) -> int | None:
        # Static-safe BFS avoiding patroller neighborhood and current blocker
        q: deque[tuple[tuple[int, int], list[int]]] = deque([(self.pos, [])])
        seen = {self.pos}
        while q:
            cur, path = q.popleft()
            if cur == self.goal_pos:
                return path[0] if path else ACTION_INTERACT
            for act, (dr, dc) in _MOVES.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                nxt = (nr, nc)
                if 0 <= nr < self.size and 0 <= nc < self.size and nxt not in seen:
                    if self.base_grid[nr, nc] == self.wall_val or nxt == self.blocker_pos:
                        continue
                    if (
                        not path
                        and abs(nr - self.patroller_pos[0]) + abs(nc - self.patroller_pos[1]) <= 1
                    ):
                        continue
                    seen.add(nxt)
                    q.append((nxt, [*path, act]))
        return ACTION_INTERACT


class LongHorizonQuestEnv(_CustomTileWorld):
    """13x13 four-chamber long-horizon task: key1 -> door1 -> key2 -> door2 -> switch -> gate -> exit."""

    world_name = "long_horizon_quest"
    key1_val = 2
    door1_val = 3
    key2_val = 5
    door2_val = 9
    switch_val = 10
    gate_val = 11
    hazard_val = 12

    def __init__(self, max_steps: int = 220) -> None:
        super().__init__(max_steps=max_steps)
        self.has_k1 = False
        self.door1_open = False
        self.has_k2 = False
        self.door2_open = False
        self.gate_open = False
        self.k1_pos = (2, 2)
        self.d1_pos = (3, 6)
        self.k2_pos = (2, 9)
        self.d2_pos = (6, 9)
        self.sw_pos = (9, 9)
        self.gate_pos = (9, 6)
        self.goal_pos = (9, 3)

    def _build(self, rng: random.Random) -> None:
        self._box(13, 13)
        for c in range(1, 12):
            self.grid[6, c] = self.wall_val
        for r in range(1, 12):
            self.grid[r, 6] = self.wall_val
        self.has_k1 = False
        self.door1_open = False
        self.has_k2 = False
        self.door2_open = False
        self.gate_open = False

        self.d1_pos = (rng.randint(2, 4), 6)
        self.d2_pos = (6, rng.randint(8, 10))
        self.gate_pos = (rng.randint(8, 10), 6)
        self.grid[self.d1_pos] = self.door1_val
        self.grid[self.d2_pos] = self.door2_val
        self.grid[self.gate_pos] = self.gate_val

        self.pos = (1, 1)
        self.k1_pos = self._free_cell(rng, (2, 5), (2, 5))
        self.grid[self.k1_pos] = self.key1_val
        self.k2_pos = self._free_cell(rng, (1, 5), (7, 11))
        self.grid[self.k2_pos] = self.key2_val
        self.sw_pos = self._free_cell(rng, (7, 11), (7, 11))
        self.grid[self.sw_pos] = self.switch_val
        self.goal_pos = self._free_cell(rng, (7, 11), (1, 4))
        self.grid[self.goal_pos] = self.goal_val
        while True:
            hz = self._free_cell(rng, (1, 5), (7, 11))
            if abs(hz[0] - self.d1_pos[0]) + abs(hz[1] - self.d1_pos[1]) >= 2 and (
                abs(hz[0] - self.d2_pos[0]) + abs(hz[1] - self.d2_pos[1]) >= 2
            ):
                self.grid[hz] = self.hazard_val
                break

    def _is_blocked(self, cell_val: int) -> bool:
        if cell_val == self.wall_val:
            return True
        if cell_val == self.door1_val and not self.has_k1:
            return True
        if cell_val == self.door2_val and not self.has_k2:
            return True
        if cell_val == self.gate_val and not self.gate_open:
            return True
        return False

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == self.hazard_val:
            return "die"
        if cell_val == self.key1_val:
            self.has_k1 = True
            self.grid[rc] = self.floor_val
        elif cell_val == self.door1_val and self.has_k1:
            self.door1_open = True
            self.grid[rc] = self.floor_val
        elif cell_val == self.key2_val:
            self.has_k2 = True
            self.grid[rc] = self.floor_val
        elif cell_val == self.door2_val and self.has_k2:
            self.door2_open = True
            self.grid[rc] = self.floor_val
        elif cell_val == self.switch_val:
            self.gate_open = True
            self.grid[rc] = self.floor_val
            self.grid[self.gate_pos] = self.floor_val
        elif cell_val == self.goal_val:
            return "win"
        return None

    def get_oracle_action(self) -> int | None:
        extra: set[int] = set()
        if self.has_k1:
            extra.add(self.door1_val)
        if self.has_k2:
            extra.add(self.door2_val)
        if self.gate_open:
            extra.add(self.gate_val)
        if not self.has_k1:
            target = self.k1_pos
        elif not self.door1_open:
            target = self.d1_pos
        elif not self.has_k2:
            target = self.k2_pos
        elif not self.door2_open:
            target = self.d2_pos
        elif not self.gate_open:
            target = self.sw_pos
        else:
            target = self.goal_pos
        # Exclude hazard tiles during BFS
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
                    if v == self.hazard_val or (
                        v in {self.wall_val, self.door1_val, self.door2_val, self.gate_val}
                        and v not in extra
                    ):
                        continue
                    seen.add((nr, nc))
                    q.append(((nr, nc), [*path, act]))
        return ACTION_RIGHT


class ContinuousMultiWaypointEnv(BenchmarkEnvironment):
    """Continuous 2D [0,12]x[0,12] chicane with 2 obstacles and sequential waypoint -> exit control."""

    def __init__(
        self,
        bounds: tuple[float, float] = (12.0, 12.0),
        step_size: float = 0.5,
        goal_radius: float = 0.65,
        max_steps: int = 120,
    ) -> None:
        self.bounds = bounds
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.reference_frame = ReferenceFrame.generic_nd(
            dimensions=2,
            coordinate_type=CoordinateType.CONTINUOUS,
            frame_id="continuous_multi_waypoint_2d",
            axes=("x", "y"),
            bounds=((0.0, bounds[0]), (0.0, bounds[1])),
            unit="meter",
            origin_description="bottom_left_metric",
        )
        self.agent_pos = np.array([1.2, 1.2], dtype=float)
        self.waypoint_pos = np.array([6.0, 9.5], dtype=float)
        self.goal_pos = np.array([10.5, 2.0], dtype=float)
        self.obstacles: list[tuple[np.ndarray, float]] = []
        self.waypoint_reached = False
        self.steps = 0
        self.done = False
        self.current_seed = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        rng = random.Random(self.current_seed)
        self.agent_pos = np.array([rng.uniform(1.0, 2.2), rng.uniform(1.0, 2.2)], dtype=float)
        self.waypoint_pos = np.array([rng.uniform(5.0, 7.0), rng.uniform(9.2, 10.6)], dtype=float)
        self.goal_pos = np.array([rng.uniform(9.5, 10.8), rng.uniform(1.4, 3.0)], dtype=float)
        self.obstacles = [
            (np.array([rng.uniform(3.6, 4.4), rng.uniform(5.2, 6.4)], dtype=float), 1.45),
            (np.array([rng.uniform(7.6, 8.4), rng.uniform(5.2, 6.4)], dtype=float), 1.45),
        ]
        self.waypoint_reached = False
        self.steps = 0
        self.done = False
        return self._make_obs()

    def _in_obstacle(self, pos: np.ndarray) -> bool:
        return any(float(np.linalg.norm(pos - c)) <= r for c, r in self.obstacles)

    def _render_lattice(self) -> np.ndarray:
        grid = np.zeros((12, 12), dtype=int)
        grid[0, :] = 5
        grid[-1, :] = 5
        grid[:, 0] = 5
        grid[:, -1] = 5
        for r in range(1, 11):
            for c in range(1, 11):
                pos = np.array([c + 0.5, 11.5 - r], dtype=float)
                if self._in_obstacle(pos):
                    grid[r, c] = 5
        if not self.waypoint_reached:
            wr = int(np.clip(round(11.5 - self.waypoint_pos[1]), 1, 10))
            wc = int(np.clip(round(self.waypoint_pos[0] - 0.5), 1, 10))
            grid[wr, wc] = 3
        else:
            gr = int(np.clip(round(11.5 - self.goal_pos[1]), 1, 10))
            gc = int(np.clip(round(self.goal_pos[0] - 0.5), 1, 10))
            grid[gr, gc] = 8
        ar = int(np.clip(round(11.5 - self.agent_pos[1]), 1, 10))
        ac = int(np.clip(round(self.agent_pos[0] - 0.5), 1, 10))
        grid[ar, ac] = 1
        return grid

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        return EmbodiedObservation(
            lattice=self._render_lattice(),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT],
            terminal=terminal,
            context_id=f"cont_multi_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
            metadata={
                "reference_frame": self.reference_frame,
                "agent_pos": self.agent_pos.copy(),
            },
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        if self.done:
            return self._make_obs(TerminalSignal.TIMEOUT), 0.0, True, {"success": False}
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
        nxt = self.agent_pos + vel
        nxt[0] = float(np.clip(nxt[0], 0.5, self.bounds[0] - 0.5))
        nxt[1] = float(np.clip(nxt[1], 0.5, self.bounds[1] - 0.5))
        if not self._in_obstacle(nxt):
            self.agent_pos = nxt
        if not self.waypoint_reached:
            if float(np.linalg.norm(self.agent_pos - self.waypoint_pos)) <= self.goal_radius:
                self.waypoint_reached = True
        elif float(np.linalg.norm(self.agent_pos - self.goal_pos)) <= self.goal_radius:
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
        target = self.waypoint_pos if not self.waypoint_reached else self.goal_pos
        start = (round(float(self.agent_pos[0]), 3), round(float(self.agent_pos[1]), 3))
        q: list[tuple[float, float, tuple[float, float], list[int]]] = [
            (float(np.linalg.norm(np.array(start) - target)), 0.0, start, [])
        ]
        seen = {start}
        actions = [
            (ACTION_UP, np.array([0.0, self.step_size])),
            (ACTION_DOWN, np.array([0.0, -self.step_size])),
            (ACTION_LEFT, np.array([-self.step_size, 0.0])),
            (ACTION_RIGHT, np.array([self.step_size, 0.0])),
        ]
        while q:
            _, cost, (cx, cy), path = heapq.heappop(q)
            cpos = np.array([cx, cy])
            if float(np.linalg.norm(cpos - target)) <= self.goal_radius:
                return path[0] if path else ACTION_UP
            for act, vel in actions:
                npos = cpos + vel
                if (
                    0.5 <= npos[0] <= self.bounds[0] - 0.5
                    and 0.5 <= npos[1] <= self.bounds[1] - 0.5
                ):
                    if not self._in_obstacle(npos):
                        k = (round(float(npos[0]), 3), round(float(npos[1]), 3))
                        if k not in seen:
                            seen.add(k)
                            h = float(np.linalg.norm(npos - target))
                            heapq.heappush(
                                q,
                                (cost + self.step_size + h, cost + self.step_size, k, [*path, act]),
                            )
        return ACTION_UP


class LanguageGoalSelectEnv(_CustomTileWorld):
    """10x10 room with 3 visually distinct goals; mission text selects the unique valid target."""

    world_name = "language_goal_select"
    wall_val: int = 5
    avatar_val: int = 1

    def __init__(self, max_steps: int = 80) -> None:
        super().__init__(max_steps=max_steps)
        self.candidates = [2, 3, 8]
        self.target_feature = 3
        self.target_pos = (2, 2)
        self.distractor_positions: set[tuple[int, int]] = set()
        self.mission = "reach feature 3"

    def _build(self, rng: random.Random) -> None:
        self._box(10, 10)
        self.pos = self._free_cell(rng, (4, 5), (4, 5))
        chosen = rng.choice(self.candidates)
        self.target_feature = chosen
        templates = (
            f"go to feature {chosen}",
            f"reach target {chosen}",
            f"navigate to symbol {chosen}",
        )
        self.mission = rng.choice(templates)
        quads = [((1, 3), (1, 3)), ((1, 3), (6, 8)), ((6, 8), (1, 8))]
        rng.shuffle(quads)
        self.distractor_positions = set()
        for feat, (rr, cc) in zip(self.candidates, quads):
            loc = self._free_cell(rng, rr, cc)
            self.grid[loc] = feat
            if feat == chosen:
                self.target_pos = loc
            else:
                self.distractor_positions.add(loc)

    def _on_enter(self, cell_val: int, rc: tuple[int, int]) -> str | None:
        if cell_val == self.target_feature:
            return "win"
        if cell_val in self.candidates:
            return "die"
        return None

    def _make_obs(self, terminal: TerminalSignal | None = None) -> EmbodiedObservation:
        lat = self.grid.copy()
        lat[self.pos] = self.avatar_val
        return EmbodiedObservation(
            lattice=lat,
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT],
            terminal=terminal,
            context_id=f"{self.world_name}_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
            metadata={"mission": self.mission, "instruction": self.mission},
        )

    def get_oracle_action(self) -> int | None:
        q: deque[tuple[tuple[int, int], list[int]]] = deque([(self.pos, [])])
        seen = {self.pos}
        h, w = self.grid.shape
        while q:
            cur, path = q.popleft()
            if cur == self.target_pos:
                return path[0] if path else ACTION_UP
            for act, (dr, dc) in _MOVES.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                    v = int(self.grid[nr, nc])
                    if v == self.wall_val or (v in self.candidates and v != self.target_feature):
                        continue
                    seen.add((nr, nc))
                    q.append(((nr, nc), [*path, act]))
        return ACTION_UP
