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


class RulesKnownPlannerAgent:
    """Domain-rules-aware classical planner baseline.

    Knows the rules and mechanics of each benchmark domain (geometry, pushing, switches),
    but has zero prior knowledge of hidden seed layouts or fatal hazards.
    """

    def __init__(self, retain_fatal_memory: bool = True) -> None:
        self.retain_fatal_memory = retain_fatal_memory
        self.fatal_cells: set[tuple[int, int]] = set()
        self.fatal_features: set[int] = set()
        self.cached_plan: list[int] = []
        self.last_pos: tuple[int, int] | None = None
        self.last_action: int | None = None
        self.rng = random.Random(42)

    def reset_episode(self, reset_memory: bool = False) -> None:
        self.cached_plan.clear()
        self.last_pos = None
        self.last_action = None
        if reset_memory or not self.retain_fatal_memory:
            self.fatal_cells.clear()
            self.fatal_features.clear()

    def on_boundary(self, boundary: Any = None) -> None:
        self.reset_episode(reset_memory=False)

    def is_action_calibrated(self, action: int) -> bool:
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
            self.cached_plan.clear()
            if self.last_pos is not None:
                r, c = self.last_pos
                dr, dc = self._action_delta(action)
                target = (r + dr, c + dc)
                self.fatal_cells.add(target)
                if prev_obs is not None and prev_obs.lattice is not None:
                    grid = prev_obs.lattice
                    if 0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]:
                        feat = int(grid[target[0], target[1]])
                        if feat not in (0, 1):
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
        if self.cached_plan:
            act = self.cached_plan.pop(0)
            if act in available:
                self.last_action = act
                return act
            self.cached_plan.clear()

        grid = obs.lattice
        if grid is None or grid.ndim != 2:
            return self._step_continuous(obs, available)
        meta = obs.metadata or {}
        if "reference_frame" in meta or bool(obs.context_id and "continuous" in obs.context_id):
            return self._step_continuous(obs, available)

        H, W = grid.shape
        pos = np.argwhere(grid == 1)
        avatar: tuple[int, int] = (
            (int(pos[0][0]), int(pos[0][1])) if len(pos) == 1 else self.last_pos or (1, 1)
        )
        self.last_pos = avatar

        # Case 1: Symbolic Rules (has cue at (1, 4) in (2, 3, 4) and ACTION_TOGGLE available)
        if H >= 7 and W >= 7 and int(grid[1, 4]) in (2, 3, 4) and ACTION_TOGGLE in available:
            act = self._plan_symbolic_rules(grid, avatar, available)
            if act is not None:
                self.last_action = act
                return act

        # Case 2: Grid Push (has movable box 4 and goal 8)
        if 4 in grid and 8 in grid and ACTION_TOGGLE not in available:
            plan = self._plan_push(grid, avatar)
            if plan:
                self.cached_plan = plan[1:]
                self.last_action = plan[0]
                return plan[0]

        # Case 3: Grid Nav (Key-Door-Goal puzzle)
        # If key 3 is present on the grid, navigate to key 3 first!
        if 3 in grid:
            key_positions: list[tuple[int, int]] = [
                (int(p[0]), int(p[1])) for p in np.argwhere(grid == 3)
            ]
            path = self._bfs_path(grid, avatar, key_positions, passable={0, 1})
            if path:
                self.cached_plan = path[1:]
                self.last_action = path[0]
                return path[0]

        # Once key is collected (or not present), door 2 is passable, navigate to goal 8
        goal_positions: list[tuple[int, int]] = [
            (int(p[0]), int(p[1])) for p in np.argwhere(grid == 8)
        ]
        if goal_positions:
            path = self._bfs_path(grid, avatar, goal_positions, passable={0, 1, 2})
            if path:
                self.cached_plan = path[1:]
                self.last_action = path[0]
                return path[0]

        # Fallback: step safe neighbor
        r, c = avatar
        for act in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            if act in available:
                dr, dc = self._action_delta(act)
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in self.fatal_cells and grid[nr, nc] not in (5, 4):
                        self.last_action = act
                        return act

        act = self.rng.choice(available)
        self.last_action = act
        return act

    def _step_continuous(self, obs: EmbodiedObservation, available: list[int]) -> int:
        grid = obs.lattice
        if grid is not None and grid.ndim == 2:
            pos = np.argwhere(grid == 1)
            avatar_c: tuple[int, int] = (
                (int(pos[0][0]), int(pos[0][1])) if len(pos) == 1 else self.last_pos or (8, 1)
            )
            self.last_pos = avatar_c
            g_pos = np.argwhere(grid == 8)
            if len(g_pos) >= 1:
                target_c: tuple[int, int] = (int(g_pos[0][0]), int(g_pos[0][1]))
                path = self._bfs_path(grid, avatar_c, [target_c], passable={0, 1})
                if path:
                    return path[0]

        meta = obs.metadata or {}
        agent_pos = meta.get("agent_pos")
        if agent_pos is None:
            return self.rng.choice(available)
        import heapq

        cpos = np.array(agent_pos, dtype=float)
        grid = obs.lattice
        if grid is not None and 8 in grid:
            g_pos = np.argwhere(grid == 8)
            gr, gc = g_pos[0]
            goal_pos = np.array([gc + 0.5, 9.5 - gr], dtype=float)
        else:
            goal_pos = np.array([8.5, 8.5], dtype=float)

        obs_center = np.array([5.0, 5.0], dtype=float)
        obs_radius = 1.7
        start = (round(float(cpos[0]), 2), round(float(cpos[1]), 2))
        actions = [
            (ACTION_UP, np.array([0.0, 0.5])),
            (ACTION_DOWN, np.array([0.0, -0.5])),
            (ACTION_LEFT, np.array([-0.5, 0.0])),
            (ACTION_RIGHT, np.array([0.5, 0.0])),
        ]
        q: list[tuple[float, float, tuple[float, float], list[int]]] = [
            (float(np.linalg.norm(cpos - goal_pos)), 0.0, start, [])
        ]
        v: set[tuple[float, float]] = {start}
        while q:
            _, cost, (cx, cy), path = heapq.heappop(q)
            curr = np.array([cx, cy])
            if np.linalg.norm(curr - goal_pos) <= 0.8:
                return path[0] if path else ACTION_UP
            if len(path) >= 28:
                continue
            for act, vel in actions:
                np_pos = curr + vel
                if 0.5 <= np_pos[0] <= 9.5 and 0.5 <= np_pos[1] <= 9.5:
                    if np.linalg.norm(np_pos - obs_center) > obs_radius:
                        key = (round(float(np_pos[0]), 2), round(float(np_pos[1]), 2))
                        if key not in v:
                            v.add(key)
                            h = float(np.linalg.norm(np_pos - goal_pos))
                            heapq.heappush(q, (cost + 0.5 + h, cost + 0.5, key, path + [act]))
        return ACTION_UP

    def _plan_push(self, grid: np.ndarray, avatar: tuple[int, int]) -> list[int] | None:
        H, W = grid.shape
        b_pos = np.argwhere(grid == 4)
        g_pos = np.argwhere(grid == 8)
        if len(b_pos) != 1 or len(g_pos) != 1:
            return None
        box = (int(b_pos[0][0]), int(b_pos[0][1]))
        goal = (int(g_pos[0][0]), int(g_pos[0][1]))

        start_state = (box[0], box[1], avatar[0], avatar[1])
        queue: deque[tuple[tuple[int, int, int, int], list[int]]] = deque([(start_state, [])])
        visited: set[tuple[int, int, int, int]] = {start_state}
        walls = {(r, c) for r in range(H) for c in range(W) if grid[r, c] == 5}

        while queue:
            (br, bc, ar, ac), path = queue.popleft()
            if (br, bc) == goal:
                return path

            for act, dr, dc in (
                (ACTION_UP, -1, 0),
                (ACTION_DOWN, 1, 0),
                (ACTION_LEFT, 0, -1),
                (ACTION_RIGHT, 0, 1),
            ):
                nar, nac = ar + dr, ac + dc
                if not (0 <= nar < H and 0 <= nac < W) or (nar, nac) in walls:
                    continue
                if (nar, nac) in self.fatal_cells:
                    continue

                if (nar, nac) == (br, bc):
                    nbr, nbc = br + dr, bc + dc
                    if not (0 <= nbr < H and 0 <= nbc < W) or (nbr, nbc) in walls:
                        continue
                    if (nbr, nbc) in self.fatal_cells:
                        continue
                    next_state = (nbr, nbc, nar, nac)
                else:
                    next_state = (br, bc, nar, nac)

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [act]))
        return None

    def _plan_symbolic_rules(
        self, grid: np.ndarray, avatar: tuple[int, int], available: list[int]
    ) -> int | None:
        H, W = grid.shape
        cue = int(grid[1, 4])
        g_pos = np.argwhere(grid == 8)
        if len(g_pos) != 1:
            return None
        goal = (int(g_pos[0][0]), int(g_pos[0][1]))
        door_pos = (H - 3, W - 2)
        door_open = grid[door_pos[0], door_pos[1]] == 0

        # Check if door is open; only then navigate directly to goal
        if door_open:
            direct = self._bfs_path(grid, avatar, [goal])
            if direct:
                self.cached_plan = direct[1:]
                return direct[0]

        # Locate target switch matching cue
        switches: dict[int, tuple[int, int]] = {}
        for r in range(H):
            for c in range(W):
                if (r, c) != (1, 4) and grid[r, c] in (2, 3, 4):
                    switches[int(grid[r, c])] = (r, c)

        if cue not in switches:
            return None
        sw_pos = switches[cue]

        # If adjacent to switch, toggle it
        if abs(avatar[0] - sw_pos[0]) + abs(avatar[1] - sw_pos[1]) <= 1:
            return ACTION_TOGGLE

        # Navigate adjacent to target switch
        adj_cells = [
            (sw_pos[0] + dr, sw_pos[1] + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= sw_pos[0] + dr < H
            and 0 <= sw_pos[1] + dc < W
            and grid[sw_pos[0] + dr, sw_pos[1] + dc] in (0, 1)
            and (sw_pos[0] + dr, sw_pos[1] + dc) not in self.fatal_cells
        ]
        if adj_cells:
            path = self._bfs_path(grid, avatar, adj_cells)
            if path:
                self.cached_plan = path[1:]
                return path[0]
        return None

    def _bfs_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        passable: set[int] | None = None,
    ) -> list[int] | None:
        if passable is None:
            passable = {0, 1}
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
                    if (nr, nc) in visited or (nr, nc) in self.fatal_cells:
                        continue
                    val = int(grid[nr, nc])
                    if val in self.fatal_features or val == 9:
                        continue
                    if (nr, nc) in target_set:
                        return path + [act]
                    if val in passable:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))
        return None


class ObjectCentricHypothesisAgent:
    """Object-centric learning agent with hypothesis testing over entity affordances.

    Maintains empirical hypotheses over feature affordances:
    - movable features (discovered by stepping into them)
    - switch features (discovered by toggling near them)
    - goal features (discovered by reward / terminal success)
    - fatal features / cells (discovered by death)
    """

    def __init__(self) -> None:
        self.movable_features: set[int] = set()
        self.switch_features: set[int] = set()
        self.goal_features: set[int] = {8}
        self.fatal_features: set[int] = set()
        self.fatal_cells: set[tuple[int, int]] = set()
        self.tested_toggles: set[tuple[int, int]] = set()
        self.blocked_targets: set[tuple[int, int]] = set()
        self.last_pos: tuple[int, int] | None = None
        self.last_action: int | None = None
        self.cached_plan: list[int] = []
        self.rng = random.Random(42)

    def reset_episode(self, reset_memory: bool = False) -> None:
        self.cached_plan.clear()
        self.last_pos = None
        self.last_action = None
        self.blocked_targets.clear()
        if reset_memory:
            self.movable_features.clear()
            self.switch_features.clear()
            self.goal_features = {8}
            self.fatal_features.clear()
            self.fatal_cells.clear()
            self.tested_toggles.clear()

    def on_boundary(self, boundary: Any = None) -> None:
        self.reset_episode(reset_memory=False)

    def is_action_calibrated(self, action: int) -> bool:
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
            self.cached_plan.clear()
            if self.last_pos is not None:
                r, c = self.last_pos
                dr, dc = self._action_delta(action)
                target = (r + dr, c + dc)
                self.fatal_cells.add(target)
                if prev_obs is not None and prev_obs.lattice is not None:
                    grid = prev_obs.lattice
                    if 0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]:
                        feat = int(grid[target[0], target[1]])
                        if feat not in (0, 1):
                            self.fatal_features.add(feat)

        if info.get("success") or info.get("reward", 0.0) > 0.5:
            if prev_obs is not None and prev_obs.lattice is not None and self.last_pos is not None:
                dr, dc = self._action_delta(action)
                tg = (self.last_pos[0] + dr, self.last_pos[1] + dc)
                if (
                    0 <= tg[0] < prev_obs.lattice.shape[0]
                    and 0 <= tg[1] < prev_obs.lattice.shape[1]
                ):
                    self.goal_features.add(int(prev_obs.lattice[tg[0], tg[1]]))

        # Discover movable objects
        if (
            prev_obs is not None
            and prev_obs.lattice is not None
            and next_obs.lattice is not None
            and self.last_pos is not None
        ):
            dr, dc = self._action_delta(action)
            target = (self.last_pos[0] + dr, self.last_pos[1] + dc)
            p_grid = prev_obs.lattice
            n_grid = next_obs.lattice
            if 0 <= target[0] < p_grid.shape[0] and 0 <= target[1] < p_grid.shape[1]:
                target_val = int(p_grid[target[0], target[1]])
                if target_val not in (0, 1, 5):
                    next_box = (target[0] + dr, target[1] + dc)
                    if (
                        0 <= next_box[0] < n_grid.shape[0]
                        and 0 <= next_box[1] < n_grid.shape[1]
                        and int(n_grid[next_box[0], next_box[1]]) == target_val
                    ):
                        self.movable_features.add(target_val)

        # Discover switches and track blocked affordance feedback
        if prev_obs is not None and next_obs is not None:
            if prev_obs.lattice is not None and next_obs.lattice is not None:
                p_grid = prev_obs.lattice.copy()
                p_grid[p_grid == 1] = 0
                n_grid = next_obs.lattice.copy()
                n_grid[n_grid == 1] = 0
                # If non-avatar world state changed (e.g. door unblocked / state changed), clear blocked goals
                if not np.array_equal(p_grid, n_grid):
                    self.blocked_targets.clear()

                # If movement action failed to move avatar, mark attempted cell as blocked
                if self.last_pos is not None and action in (
                    ACTION_UP,
                    ACTION_DOWN,
                    ACTION_LEFT,
                    ACTION_RIGHT,
                ):
                    dr, dc = self._action_delta(action)
                    attempted = (self.last_pos[0] + dr, self.last_pos[1] + dc)
                    p_pos = np.argwhere(prev_obs.lattice == 1)
                    n_pos = np.argwhere(next_obs.lattice == 1)
                    if len(p_pos) == 1 and len(n_pos) == 1 and tuple(p_pos[0]) == tuple(n_pos[0]):
                        self.blocked_targets.add(attempted)

        if (
            action in (ACTION_TOGGLE, ACTION_INTERACT)
            and prev_obs is not None
            and prev_obs.lattice is not None
            and next_obs.lattice is not None
        ):
            diff = np.where(prev_obs.lattice != next_obs.lattice)
            if len(diff[0]) > 0 and self.last_pos is not None:
                r, c = self.last_pos
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < prev_obs.lattice.shape[0] and 0 <= nc < prev_obs.lattice.shape[1]:
                        sw_val = int(prev_obs.lattice[nr, nc])
                        if sw_val in (2, 3, 4):
                            self.switch_features.add(sw_val)

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
        if self.cached_plan:
            act = self.cached_plan.pop(0)
            if act in available:
                self.last_action = act
                return act
            self.cached_plan.clear()

        grid = obs.lattice
        if grid is None or grid.ndim != 2:
            return self._step_continuous(obs, available)
        meta = obs.metadata or {}
        if "reference_frame" in meta or bool(obs.context_id and "continuous" in obs.context_id):
            return self._step_continuous(obs, available)

        H, W = grid.shape
        pos = np.argwhere(grid == 1)
        avatar: tuple[int, int] = (
            (int(pos[0][0]), int(pos[0][1])) if len(pos) == 1 else self.last_pos or (1, 1)
        )
        self.last_pos = avatar

        # Check for confirmed/candidate goals
        goal_cells: list[tuple[int, int]] = [
            (int(r), int(c))
            for r, c in np.argwhere(np.isin(grid, list(self.goal_features)))
            if (int(r), int(c)) not in self.blocked_targets
        ]

        # 1. If movable objects are known and goal exists without toggle actions, push box first
        movable_cells: list[tuple[int, int]] = [
            (int(r), int(c))
            for r, c in np.argwhere(np.isin(grid, list(self.movable_features | {4})))
        ]
        if movable_cells and goal_cells and ACTION_TOGGLE not in available:
            box = movable_cells[0]
            goal = goal_cells[0]
            plan = self._plan_push(grid, avatar, box, goal)
            if plan:
                self.cached_plan = plan[1:]
                self.last_action = plan[0]
                return plan[0]

        # 2. Direct path to goal if reachable
        if goal_cells:
            path = self._bfs_path(grid, avatar, goal_cells, passable={0, 1})
            if path:
                self.cached_plan = path[1:]
                self.last_action = path[0]
                return path[0]

        # 3. If switches are present and ACTION_TOGGLE available, test / toggle switch
        switches: list[tuple[int, int]] = [
            (int(r), int(c))
            for r, c in np.argwhere(np.isin(grid, [2, 3, 4]))
            if (int(r), int(c)) != (1, 4)
        ]
        if switches and ACTION_TOGGLE in available:
            cue = int(grid[1, 4]) if H >= 7 and W >= 7 and int(grid[1, 4]) in (2, 3, 4) else None
            # If adjacent to switch, toggle it
            for sw in switches:
                if abs(avatar[0] - sw[0]) + abs(avatar[1] - sw[1]) <= 1:
                    if cue is None or grid[sw[0], sw[1]] == cue:
                        if sw not in self.tested_toggles:
                            self.tested_toggles.add(sw)
                            self.last_action = ACTION_TOGGLE
                            return ACTION_TOGGLE
            # Else navigate adjacent to target switch
            target_sws = []
            if cue is not None:
                target_sws = [
                    sw
                    for sw in switches
                    if grid[sw[0], sw[1]] == cue and sw not in self.tested_toggles
                ]
            if not target_sws:
                untested = [sw for sw in switches if sw not in self.tested_toggles]
                target_sws = untested or switches
            adj_sw = [
                (sw[0] + dr, sw[1] + dc)
                for sw in target_sws
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= sw[0] + dr < H
                and 0 <= sw[1] + dc < W
                and grid[sw[0] + dr, sw[1] + dc] in (0, 1)
                and (sw[0] + dr, sw[1] + dc) not in self.fatal_cells
            ]
            if adj_sw:
                path = self._bfs_path(grid, avatar, adj_sw, passable={0, 1})
                if path:
                    self.cached_plan = path[1:]
                    self.last_action = path[0]
                    return path[0]

        # 4. Explore candidate objects (test movable hypothesis)
        unknown_objs: list[tuple[int, int]] = [
            (int(r), int(c))
            for r, c in np.argwhere(~np.isin(grid, [0, 1, 5]))
            if (int(r), int(c)) not in self.fatal_cells
        ]
        if unknown_objs:
            path = self._bfs_path(grid, avatar, unknown_objs, passable={0, 1})
            if path:
                self.cached_plan = path[1:]
                self.last_action = path[0]
                return path[0]

        # Fallback safe neighbor
        r, c = avatar
        for act in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            if act in available:
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
        grid = obs.lattice
        if grid is not None and grid.ndim == 2:
            pos = np.argwhere(grid == 1)
            avatar_c: tuple[int, int] = (
                (int(pos[0][0]), int(pos[0][1])) if len(pos) == 1 else self.last_pos or (8, 1)
            )
            self.last_pos = avatar_c
            g_pos = np.argwhere(grid == 8)
            if len(g_pos) >= 1:
                target_c: tuple[int, int] = (int(g_pos[0][0]), int(g_pos[0][1]))
                path = self._bfs_path(grid, avatar_c, [target_c], passable={0, 1})
                if path:
                    return path[0]

        meta = obs.metadata or {}
        agent_pos = meta.get("agent_pos")
        if agent_pos is None:
            return self.rng.choice(available)
        import heapq

        cpos = np.array(agent_pos, dtype=float)
        grid = obs.lattice
        if grid is not None and 8 in grid:
            g_pos = np.argwhere(grid == 8)
            gr, gc = g_pos[0]
            goal_pos = np.array([gc + 0.5, 9.5 - gr], dtype=float)
        else:
            goal_pos = np.array([8.5, 8.5], dtype=float)

        obs_center = np.array([5.0, 5.0], dtype=float)
        obs_radius = 1.7
        start = (round(float(cpos[0]), 2), round(float(cpos[1]), 2))
        actions = [
            (ACTION_UP, np.array([0.0, 0.5])),
            (ACTION_DOWN, np.array([0.0, -0.5])),
            (ACTION_LEFT, np.array([-0.5, 0.0])),
            (ACTION_RIGHT, np.array([0.5, 0.0])),
        ]
        q: list[tuple[float, float, tuple[float, float], list[int]]] = [
            (float(np.linalg.norm(cpos - goal_pos)), 0.0, start, [])
        ]
        v: set[tuple[float, float]] = {start}
        while q:
            _, cost, (cx, cy), path = heapq.heappop(q)
            curr = np.array([cx, cy])
            if np.linalg.norm(curr - goal_pos) <= 0.8:
                return path[0] if path else ACTION_UP
            if len(path) >= 28:
                continue
            for act, vel in actions:
                np_pos = curr + vel
                if 0.5 <= np_pos[0] <= 9.5 and 0.5 <= np_pos[1] <= 9.5:
                    if np.linalg.norm(np_pos - obs_center) > obs_radius:
                        key = (round(float(np_pos[0]), 2), round(float(np_pos[1]), 2))
                        if key not in v:
                            v.add(key)
                            h = float(np.linalg.norm(np_pos - goal_pos))
                            heapq.heappush(q, (cost + 0.5 + h, cost + 0.5, key, path + [act]))
        return ACTION_UP

    def _plan_push(
        self,
        grid: np.ndarray,
        avatar: tuple[int, int],
        box: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[int] | None:
        H, W = grid.shape
        start_state = (box[0], box[1], avatar[0], avatar[1])
        queue: deque[tuple[tuple[int, int, int, int], list[int]]] = deque([(start_state, [])])
        visited: set[tuple[int, int, int, int]] = {start_state}
        walls = {(r, c) for r in range(H) for c in range(W) if grid[r, c] == 5}

        while queue:
            (br, bc, ar, ac), path = queue.popleft()
            if (br, bc) == goal:
                return path

            for act, dr, dc in (
                (ACTION_UP, -1, 0),
                (ACTION_DOWN, 1, 0),
                (ACTION_LEFT, 0, -1),
                (ACTION_RIGHT, 0, 1),
            ):
                nar, nac = ar + dr, ac + dc
                if not (0 <= nar < H and 0 <= nac < W) or (nar, nac) in walls:
                    continue
                if (nar, nac) in self.fatal_cells:
                    continue

                if (nar, nac) == (br, bc):
                    nbr, nbc = br + dr, bc + dc
                    if not (0 <= nbr < H and 0 <= nbc < W) or (nbr, nbc) in walls:
                        continue
                    if (nbr, nbc) in self.fatal_cells:
                        continue
                    next_state = (nbr, nbc, nar, nac)
                else:
                    next_state = (br, bc, nar, nac)

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [act]))
        return None

    def _bfs_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        passable: set[int] | None = None,
    ) -> list[int] | None:
        if passable is None:
            passable = {0, 1}
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
                    if (nr, nc) in visited or (nr, nc) in self.fatal_cells:
                        continue
                    val = int(grid[nr, nc])
                    if val in self.fatal_features:
                        continue
                    if (nr, nc) in target_set:
                        return path + [act]
                    if val in passable:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))
        return None
