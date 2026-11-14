"""Independent Third-Party Closed-Loop Benchmark Environments.

Adapts external, unmodified Gymnasium and MiniGrid environments behind HBLLM's
``BenchmarkEnvironment`` protocol so generality can be evaluated on tasks not
authored alongside the cognitive engine. Feature IDs are scrambled deterministically
per seed and observations expose zero oracle entity roles.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments with gymnasium
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from minigrid.core.constants import OBJECT_TO_IDX

from benchmarks.envs.protocol import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BenchmarkEnvironment,
)
from hbllm.hcir.world.percept import EmbodiedObservation, TerminalSignal

_CARDINAL_DELTAS: tuple[tuple[int, int, int], ...] = (
    (ACTION_UP, -1, 0),
    (ACTION_DOWN, 1, 0),
    (ACTION_LEFT, 0, -1),
    (ACTION_RIGHT, 0, 1),
)


def _coerce_action_id(action: Any) -> int | None:
    """Extract an integer action identifier from an action or enum wrapper."""
    raw = getattr(action, "value", action)
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


class GymFrozenLakeBenchmarkEnv(BenchmarkEnvironment):
    """Benchmark adapter for Gymnasium's unmodified ``FrozenLake-v1`` environment."""

    _ACTION_TO_FL: dict[int, int] = {
        ACTION_LEFT: 0,
        ACTION_DOWN: 1,
        ACTION_RIGHT: 2,
        ACTION_UP: 3,
    }

    def __init__(
        self,
        size: int = 5,
        p_frozen: float = 0.82,
        max_steps: int = 60,
        is_slippery: bool = False,
    ) -> None:
        self.size = size
        self.p_frozen = p_frozen
        self.max_steps = max_steps
        self.is_slippery = is_slippery
        self._env: gym.Env[Any, Any] | None = None
        self._desc: np.ndarray = np.empty((size, size), dtype="S1")
        self._state: int = 0
        self._nrow: int = size
        self._ncol: int = size
        self._feature_map: dict[str, int] = {}
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset to a deterministic procedural FrozenLake board derived from ``seed``."""
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        if self._env is not None:
            self._env.close()

        board = generate_random_map(size=self.size, p=self.p_frozen, seed=self.current_seed)
        self._env = gym.make("FrozenLake-v1", desc=board, is_slippery=self.is_slippery)
        obs_state, _ = self._env.reset(seed=self.current_seed)

        unwrapped: Any = self._env.unwrapped
        self._desc = np.asarray(unwrapped.desc)
        self._nrow = int(unwrapped.nrow)
        self._ncol = int(unwrapped.ncol)
        self._state = int(obs_state)
        self.steps = 0
        self.done = False

        rng = random.Random(self.current_seed)
        agent_id, goal_id, hole_id, wall_id = rng.sample(range(10, 95), 4)
        self._feature_map = {
            "empty": 0,
            "agent": agent_id,
            "goal": goal_id,
            "hole": hole_id,
            "wall": wall_id,
        }
        return self._make_observation(terminal=None)

    def _render_lattice(self) -> np.ndarray:
        """Render current FrozenLake state into a 2D integer lattice with zero entity tags."""
        lattice = np.full((self._nrow, self._ncol), self._feature_map["empty"], dtype=int)
        for r in range(self._nrow):
            for c in range(self._ncol):
                tile = bytes(self._desc[r, c])
                if tile == b"H":
                    lattice[r, c] = self._feature_map["hole"]
                elif tile == b"G":
                    lattice[r, c] = self._feature_map["goal"]
        ar, ac = divmod(self._state, self._ncol)
        lattice[ar, ac] = self._feature_map["agent"]
        return lattice

    def _make_observation(self, terminal: TerminalSignal | None) -> EmbodiedObservation:
        return EmbodiedObservation(
            lattice=self._render_lattice(),
            available_actions=(ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT),
            terminal=terminal,
            context_id=f"gym_frozenlake_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        """Execute one cardinal action against the underlying ``FrozenLake-v1`` environment."""
        if self._env is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if self.done:
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "steps": self.steps},
            )

        self.steps += 1
        act_id = _coerce_action_id(action)
        gym_act = self._ACTION_TO_FL.get(act_id) if act_id is not None else None

        if gym_act is None:
            if self.steps >= self.max_steps:
                self.done = True
                return (
                    self._make_observation(terminal=TerminalSignal.TIMEOUT),
                    0.0,
                    True,
                    {"success": False, "reason": "timeout", "steps": self.steps},
                )
            return (
                self._make_observation(terminal=None),
                0.0,
                False,
                {"success": False, "steps": self.steps},
            )

        next_state, reward, terminated, truncated, _ = self._env.step(gym_act)
        self._state = int(next_state)
        reward_f = float(reward)

        if terminated and reward_f > 0.0:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.SUCCESS),
                reward_f,
                True,
                {"success": True, "reason": "goal_reached", "steps": self.steps},
            )

        if terminated:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.FAILURE),
                reward_f,
                True,
                {"success": False, "reason": "hole_hazard", "steps": self.steps},
            )

        if self.steps >= self.max_steps or truncated:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )

        return (
            self._make_observation(terminal=None),
            reward_f,
            False,
            {"success": False, "steps": self.steps},
        )

    def get_oracle_action(self) -> int:
        """Compute optimal cardinal step to `'G'` avoiding `'H'` holes via BFS."""
        start = divmod(self._state, self._ncol)
        queue: deque[tuple[tuple[int, int], int | None]] = deque([(start, None)])
        visited: set[tuple[int, int]] = {start}

        while queue:
            (r, c), first_act = queue.popleft()
            if bytes(self._desc[r, c]) == b"G":
                return first_act if first_act is not None else ACTION_RIGHT

            for act_id, dr, dc in _CARDINAL_DELTAS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._nrow and 0 <= nc < self._ncol and (nr, nc) not in visited:
                    if bytes(self._desc[nr, nc]) != b"H":
                        visited.add((nr, nc))
                        queue.append(((nr, nc), first_act if first_act is not None else act_id))

        return ACTION_RIGHT


class MiniGridEmptyBenchmarkEnv(BenchmarkEnvironment):
    """Benchmark adapter for unmodified MiniGrid environments (default ``MiniGrid-Empty-6x6-v0``)."""

    _CARDINAL_TO_MINIGRID_DIR: dict[int, int] = {
        ACTION_RIGHT: 0,
        ACTION_DOWN: 1,
        ACTION_LEFT: 2,
        ACTION_UP: 3,
    }
    _MG_LEFT = 0
    _MG_RIGHT = 1
    _MG_FORWARD = 2

    def __init__(
        self,
        env_id: str = "MiniGrid-Empty-6x6-v0",
        max_steps: int = 60,
        cardinal_mode: bool = True,
    ) -> None:
        self.env_id = env_id
        self.max_steps = max_steps
        self.cardinal_mode = cardinal_mode
        self._env: gym.Env[Any, Any] | None = None
        self._feature_map: dict[int, int] = {}
        self._agent_id: int = 10
        self.steps: int = 0
        self.done: bool = False
        self.current_seed: int = 0

    def reset(self, seed: int | None = None) -> EmbodiedObservation:
        """Reset underlying MiniGrid environment with seed-scrambled visual feature IDs."""
        self.current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        if self._env is not None:
            self._env.close()

        self._env = gym.make(self.env_id)
        self._env.reset(seed=self.current_seed)
        self.steps = 0
        self.done = False

        rng = random.Random(self.current_seed)
        non_empty_names = ("wall", "door", "key", "ball", "box", "goal", "lava", "floor")
        sampled = rng.sample(range(10, 95), len(non_empty_names) + 1)
        self._agent_id = sampled[0]
        self._feature_map = {
            OBJECT_TO_IDX["unseen"]: 0,
            OBJECT_TO_IDX["empty"]: 0,
        }
        for name, feat_id in zip(non_empty_names, sampled[1:]):
            self._feature_map[OBJECT_TO_IDX[name]] = feat_id

        return self._make_observation(terminal=None)

    def _raw_object_lattice(self) -> np.ndarray:
        """Return MiniGrid object IDs oriented as ``(row=y, col=x)``."""
        if self._env is None:
            raise RuntimeError("Environment must be reset() before reading lattice.")
        unwrapped: Any = self._env.unwrapped
        encoded = np.asarray(unwrapped.grid.encode()[:, :, 0], dtype=int)
        return encoded.T

    def _render_lattice(self) -> np.ndarray:
        """Render MiniGrid world with seed-scrambled object features and agent overlay."""
        if self._env is None:
            raise RuntimeError("Environment must be reset() before rendering.")
        raw = self._raw_object_lattice()
        lattice = np.zeros_like(raw, dtype=int)
        for obj_idx, feat_id in self._feature_map.items():
            if feat_id != 0:
                lattice[raw == obj_idx] = feat_id

        unwrapped: Any = self._env.unwrapped
        ax, ay = int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])
        lattice[ay, ax] = self._agent_id
        return lattice

    def _make_observation(self, terminal: TerminalSignal | None) -> EmbodiedObservation:
        avail: tuple[int, ...] = (
            (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
            if self.cardinal_mode
            else (self._MG_LEFT, self._MG_RIGHT, self._MG_FORWARD)
        )
        return EmbodiedObservation(
            lattice=self._render_lattice(),
            available_actions=avail,
            terminal=terminal,
            context_id=f"minigrid_{self.env_id}_{self.current_seed}",
            episode_id=str(self.current_seed),
            entities=(),
        )

    def _step_raw_minigrid(self, mg_action: int) -> tuple[float, bool, bool]:
        assert self._env is not None
        _, reward, terminated, truncated, _ = self._env.step(mg_action)
        return float(reward), bool(terminated), bool(truncated)

    def step(self, action: Any) -> tuple[EmbodiedObservation, float, bool, dict[str, Any]]:
        """Execute one step against the unmodified MiniGrid environment."""
        if self._env is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if self.done:
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "steps": self.steps},
            )

        self.steps += 1
        act_id = _coerce_action_id(action)
        reward_f = 0.0
        terminated = False
        truncated = False

        if self.cardinal_mode:
            target_dir = self._CARDINAL_TO_MINIGRID_DIR.get(act_id) if act_id is not None else None
            if target_dir is not None:
                unwrapped: Any = self._env.unwrapped
                cur_dir = int(unwrapped.agent_dir)
                diff = (target_dir - cur_dir) % 4
                turn_seq: tuple[int, ...] = (
                    (self._MG_RIGHT,)
                    if diff == 1
                    else (self._MG_RIGHT, self._MG_RIGHT)
                    if diff == 2
                    else (self._MG_LEFT,)
                    if diff == 3
                    else ()
                )
                for turn_act in turn_seq:
                    reward_f, terminated, truncated = self._step_raw_minigrid(turn_act)
                    if terminated or truncated:
                        break
                if not terminated and not truncated:
                    reward_f, terminated, truncated = self._step_raw_minigrid(self._MG_FORWARD)
        elif act_id in (self._MG_LEFT, self._MG_RIGHT, self._MG_FORWARD):
            reward_f, terminated, truncated = self._step_raw_minigrid(int(act_id))

        if terminated and reward_f > 0.0:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.SUCCESS),
                reward_f,
                True,
                {"success": True, "reason": "goal_reached", "steps": self.steps},
            )

        if terminated:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.FAILURE),
                reward_f,
                True,
                {"success": False, "reason": "hazard_collision", "steps": self.steps},
            )

        if self.steps >= self.max_steps or truncated:
            self.done = True
            return (
                self._make_observation(terminal=TerminalSignal.TIMEOUT),
                0.0,
                True,
                {"success": False, "reason": "timeout", "steps": self.steps},
            )

        return (
            self._make_observation(terminal=None),
            reward_f,
            False,
            {"success": False, "steps": self.steps},
        )

    def get_oracle_action(self) -> int:
        """Compute optimal step toward MiniGrid goal cell via BFS."""
        if self._env is None:
            raise RuntimeError("Environment must be reset() before querying oracle.")

        raw = self._raw_object_lattice()
        height, width = raw.shape
        unwrapped: Any = self._env.unwrapped
        start = (int(unwrapped.agent_pos[1]), int(unwrapped.agent_pos[0]))
        cur_dir = int(unwrapped.agent_dir)

        blocked = {OBJECT_TO_IDX["wall"], OBJECT_TO_IDX["lava"]}
        goal_idx = OBJECT_TO_IDX["goal"]

        queue: deque[tuple[tuple[int, int], int | None]] = deque([(start, None)])
        visited: set[tuple[int, int]] = {start}
        best_cardinal: int | None = None

        while queue:
            (r, c), first_act = queue.popleft()
            if int(raw[r, c]) == goal_idx:
                best_cardinal = first_act
                break

            for act_id, dr, dc in _CARDINAL_DELTAS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                    if int(raw[nr, nc]) not in blocked:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), first_act if first_act is not None else act_id))

        target_cardinal = best_cardinal if best_cardinal is not None else ACTION_RIGHT
        if self.cardinal_mode:
            return target_cardinal

        target_dir = self._CARDINAL_TO_MINIGRID_DIR[target_cardinal]
        if cur_dir == target_dir:
            return self._MG_FORWARD
        diff = (target_dir - cur_dir) % 4
        return self._MG_LEFT if diff == 3 else self._MG_RIGHT
