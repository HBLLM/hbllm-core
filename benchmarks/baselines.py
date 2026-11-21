"""Non-LLM Classical Baselines for HBLLM Closed-Loop Benchmarks.

Provides strong, standard reference baselines:
1. FatalMemoryBFSAgent: Shortest-path BFS with persistent fatal-cell memory.
2. TabularDynaQAgent: Classical model-based reinforcement learning with Q-planning.
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
    ACTION_TOGGLE,
    ACTION_UP,
)
from hbllm.hcir.world.percept import EmbodiedObservation


class FatalMemoryBFSAgent:
    """Goal-directed BFS pathfinder that remembers fatal cells and impassable barriers across steps."""

    def __init__(self, retain_fatal_memory: bool = True) -> None:
        self.retain_fatal_memory = retain_fatal_memory
        self.fatal_cells: set[tuple[int, int]] = set()
        self.fatal_features: set[int] = set()
        self.known_barriers: set[tuple[int, int]] = set()
        self.last_pos: tuple[int, int] | None = None
        self.last_action: int | None = None
        self.rng = random.Random(42)

    def reset_episode(self, reset_memory: bool = False) -> None:
        self.last_pos = None
        self.last_action = None
        if reset_memory or not self.retain_fatal_memory:
            self.fatal_cells.clear()
            self.fatal_features.clear()
            self.known_barriers.clear()

    def on_boundary(self, boundary: Any = None) -> None:
        """Handle episodic boundary signal across levels or episodes."""
        self.reset_episode(reset_memory=False)

    def is_action_calibrated(self, action: int) -> bool:
        """Whether selected action is calibrated/confident."""
        return True

    def record_transition(
        self,
        prev_obs: EmbodiedObservation | None,
        action: int,
        next_obs: EmbodiedObservation,
        died: bool,
        info: dict[str, Any],
    ) -> None:
        if died:
            if self.last_pos is not None:
                r, c = self.last_pos
                dr, dc = self._action_delta(action)
                target = (r + dr, c + dc)
                self.fatal_cells.add(target)
                if prev_obs is not None and prev_obs.lattice is not None:
                    grid = prev_obs.lattice
                    if 0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]:
                        feat = int(grid[target[0], target[1]])
                        if feat != 0 and feat != 1:
                            self.fatal_features.add(feat)

    def _action_delta(self, action: int) -> tuple[int, int]:
        if action == ACTION_UP:
            return (-1, 0)
        if action == ACTION_DOWN:
            return (1, 0)
        if action == ACTION_LEFT:
            return (0, -1)
        if action == ACTION_RIGHT:
            return (0, 1)
        return (0, 0)

    def step(self, obs: EmbodiedObservation) -> int:
        available = list(getattr(obs, "available_actions", [1, 2, 3, 4]) or [1, 2, 3, 4])
        grid = obs.lattice
        if grid is None or grid.ndim != 2:
            # Non-grid continuous or vector domain: fallback to sensor-based greedy steering
            return self._step_continuous(obs, available)

        H, W = grid.shape
        bg = self._estimate_background(grid)
        avatar_pos = self._find_avatar(grid, bg)
        if avatar_pos is None:
            avatar_pos = self.last_pos or (1, 1)
        self.last_pos = avatar_pos

        # Identify candidate targets (e.g. goals, keys, non-background non-wall tiles)
        targets = self._find_candidate_targets(grid, bg, avatar_pos)
        if not targets:
            return self.rng.choice(available)

        # Plan shortest path via BFS to nearest valid target avoiding fatal cells & known walls
        path = self._bfs_path(grid, avatar_pos, targets, bg)
        if path:
            act = path[0]
            self.last_action = act
            return act

        # If blocked, try interact / toggle or an unvisited safe neighbor
        for act in (ACTION_INTERACT, ACTION_TOGGLE):
            if act in available:
                self.last_action = act
                return act

        # Explore safe adjacent cell
        r, c = avatar_pos
        for act in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            if act not in available:
                continue
            dr, dc = self._action_delta(act)
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if (nr, nc) not in self.fatal_cells and grid[nr, nc] != 5:
                    self.last_action = act
                    return act

        act = self.rng.choice(available)
        self.last_action = act
        return act

    def _step_continuous(self, obs: EmbodiedObservation, available: list[int]) -> int:
        # For continuous reach point-robot navigation
        meta = obs.metadata or {}
        agent_pos = meta.get("agent_pos")
        goal_pos = meta.get("goal_pos", (9.0, 9.0))
        if agent_pos is not None:
            dx = goal_pos[0] - agent_pos[0]
            dy = goal_pos[1] - agent_pos[1]
            if abs(dx) > abs(dy):
                return ACTION_RIGHT if dx > 0 else ACTION_LEFT
            else:
                return ACTION_DOWN if dy > 0 else ACTION_UP
        return self.rng.choice(available)

    def _estimate_background(self, grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _find_avatar(self, grid: np.ndarray, bg: int) -> tuple[int, int] | None:
        # Check standard avatar value 1
        pos = np.argwhere(grid == 1)
        if len(pos) == 1:
            return (int(pos[0][0]), int(pos[0][1]))
        if self.last_pos is not None:
            r, c = self.last_pos
            if self.last_action is not None:
                dr, dc = self._action_delta(self.last_action)
                exp = (r + dr, c + dc)
                if 0 <= exp[0] < grid.shape[0] and 0 <= exp[1] < grid.shape[1]:
                    if grid[exp[0], exp[1]] != bg:
                        return exp
            return self.last_pos
        # Fallback to single non-background interior cell
        for r in range(1, grid.shape[0] - 1):
            for c in range(1, grid.shape[1] - 1):
                if grid[r, c] == 1:
                    return (r, c)
        return None

    def _find_candidate_targets(
        self, grid: np.ndarray, bg: int, avatar_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        H, W = grid.shape
        targets = []
        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if val != bg and val != 1 and val != 5 and val not in self.fatal_features:
                    if (r, c) not in self.fatal_cells and (r, c) != avatar_pos:
                        targets.append((r, c))
        return targets

    def _bfs_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        bg: int,
    ) -> list[int] | None:
        H, W = grid.shape
        target_set = set(targets)
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
        visited: set[tuple[int, int]] = {start}

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) in target_set:
                return path

            for act, dr, dc in (
                (ACTION_UP, -1, 0),
                (ACTION_DOWN, 1, 0),
                (ACTION_LEFT, 0, -1),
                (ACTION_RIGHT, 0, 1),
            ):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) in visited:
                        continue
                    if (nr, nc) in self.fatal_cells:
                        continue
                    val = int(grid[nr, nc])
                    if val in self.fatal_features:
                        continue
                    # Wall (value 5) or perimeter
                    if val == 5:
                        continue
                    # If target is non-walkable barrier/key, can reach it as endpoint
                    if (nr, nc) in target_set:
                        return path + [act]

                    # Empty / walkable check
                    if val == bg or val == 0 or val == 1:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))

        return None


class TabularDynaQAgent:
    """Tabular Dyna-Q model-based reinforcement learning baseline."""

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        planning_steps: int = 10,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.q_table: dict[tuple[Any, ...], dict[int, float]] = {}
        self.model: dict[tuple[tuple[Any, ...], int], tuple[float, tuple[Any, ...], bool]] = {}
        self.visited_states: set[tuple[Any, ...]] = set()
        self.last_state: tuple[Any, ...] | None = None
        self.last_action: int | None = None
        self.rng = random.Random(42)

    def reset_episode(self, reset_memory: bool = False) -> None:
        self.last_state = None
        self.last_action = None
        if reset_memory:
            self.q_table.clear()
            self.model.clear()
            self.visited_states.clear()

    def on_boundary(self, boundary: Any = None) -> None:
        """Handle episodic boundary signal across levels or episodes."""
        self.reset_episode(reset_memory=False)

    def is_action_calibrated(self, action: int) -> bool:
        """Whether selected action is calibrated/confident."""
        return True

    def _encode_state(self, obs: EmbodiedObservation) -> tuple[Any, ...]:
        grid = obs.lattice
        if grid is not None and grid.ndim == 2:
            # Avatar location + hash of interactive components
            pos = np.argwhere(grid == 1)
            av = (int(pos[0][0]), int(pos[0][1])) if len(pos) == 1 else (0, 0)
            non_bg = tuple(sorted(set(grid[grid > 1].tolist())))
            return (av, non_bg[:4])
        meta = obs.metadata or {}
        pos_cont = meta.get("agent_pos")
        if pos_cont is not None:
            return (round(pos_cont[0], 1), round(pos_cont[1], 1))
        return (0, 0)

    def step(self, obs: EmbodiedObservation) -> int:
        state = self._encode_state(obs)
        available = list(getattr(obs, "available_actions", [1, 2, 3, 4]) or [1, 2, 3, 4])
        self.visited_states.add(state)
        q_s = self.q_table.setdefault(state, {a: 0.0 for a in available})

        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(available)
        else:
            best_val = max(q_s.get(a, 0.0) for a in available)
            candidates = [a for a in available if q_s.get(a, 0.0) == best_val]
            action = self.rng.choice(candidates)

        self.last_state = state
        self.last_action = action
        return action

    def record_transition(
        self,
        prev_obs: EmbodiedObservation | None,
        action: int,
        next_obs: EmbodiedObservation,
        reward: float,
        done: bool,
    ) -> None:
        if self.last_state is None:
            return
        s = self.last_state
        s_next = self._encode_state(next_obs)
        avail_next = list(getattr(next_obs, "available_actions", [1, 2, 3, 4]) or [1, 2, 3, 4])
        q_next = self.q_table.setdefault(s_next, {a: 0.0 for a in avail_next})
        max_q_next = max(q_next.values()) if q_next else 0.0

        # Q-learning real step update
        q_s = self.q_table.setdefault(s, {action: 0.0})
        old_val = q_s.get(action, 0.0)
        target = reward + (0.0 if done else self.gamma * max_q_next)
        q_s[action] = old_val + self.alpha * (target - old_val)

        # Model update
        self.model[(s, action)] = (reward, s_next, done)

        # Dyna-Q simulated planning updates
        if self.planning_steps > 0 and self.model:
            model_keys = list(self.model.keys())
            n_samples = min(self.planning_steps, len(model_keys))
            for s_sim, a_sim in self.rng.choices(model_keys, k=n_samples):
                r_sim, s_prime, d_sim = self.model[(s_sim, a_sim)]
                q_prime = self.q_table.setdefault(s_prime, {})
                max_q_prime = max(q_prime.values()) if q_prime else 0.0
                q_sim = self.q_table.setdefault(s_sim, {})
                old_sim = q_sim.get(a_sim, 0.0)
                sim_target = r_sim + (0.0 if d_sim else self.gamma * max_q_prime)
                q_sim[a_sim] = old_sim + self.alpha * (sim_target - old_sim)
