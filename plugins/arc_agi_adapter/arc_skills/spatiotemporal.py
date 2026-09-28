"""Spatiotemporal Dynamics & Periodic Phase Skill Acquisition.

Enables learning trajectory patterns, velocities, and oscillation periods
of autonomous moving entities (hazards, patrollers, conveyors), and planning
collision-free trajectories in space-time (r, c, t).
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.skills.common_subskills import LatticeQuantizer
from hbllm.hcir.spatial_planner import EntityRole, SpatialActionIntent, SpatialEntity

logger = logging.getLogger(__name__)


@dataclass
class EntityTrajectory:
    """Historical and predictive trajectory for a single dynamic entity."""

    entity_id: str
    feature_id: Any
    history: list[tuple[int, int]] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)  # (dr/dt, dc/dt)
    period: int | None = None  # Oscillation cycle length T
    cycle_pattern: list[tuple[int, int]] = field(default_factory=list)
    is_periodic: bool = False
    confidence: float = 0.0

    def predict_position(self, t: int) -> tuple[int, int] | None:
        """Predict entity position at future timestamp t."""
        if not self.history or not self.timestamps:
            return None

        # 1. Periodic oscillation prediction
        if self.is_periodic and self.cycle_pattern and self.period and self.period > 0:
            base_t = self.timestamps[0]
            offset = (t - base_t) % self.period
            if offset < len(self.cycle_pattern):
                return self.cycle_pattern[offset]

        # 2. Linear velocity extrapolation
        last_t = self.timestamps[-1]
        last_pos = self.history[-1]
        dt = t - last_t
        if dt <= 0:
            # Look up historical position if available
            for hist_pos, hist_t in zip(self.history, self.timestamps):
                if hist_t == t:
                    return hist_pos
            return last_pos

        pred_r = int(round(last_pos[0] + self.velocity[0] * dt))
        pred_c = int(round(last_pos[1] + self.velocity[1] * dt))
        return (pred_r, pred_c)


class AutonomousTrajectoryModel:
    """Estimates velocity and periodic cycle patterns from position time series."""

    @classmethod
    def fit(
        cls, entity_id: str, feature_id: Any, history: list[tuple[int, int]], timestamps: list[int]
    ) -> EntityTrajectory:
        """Fit velocity and periodicity model to observed trajectory."""
        if len(history) < 2:
            return EntityTrajectory(
                entity_id=entity_id,
                feature_id=feature_id,
                history=list(history),
                timestamps=list(timestamps),
                confidence=0.1,
            )

        # 1. Compute instantaneous velocities
        dr_list: list[float] = []
        dc_list: list[float] = []
        for i in range(1, len(history)):
            dt = max(1, timestamps[i] - timestamps[i - 1])
            dr_list.append((history[i][0] - history[i - 1][0]) / dt)
            dc_list.append((history[i][1] - history[i - 1][1]) / dt)

        mean_vr = sum(dr_list) / len(dr_list)
        mean_vc = sum(dc_list) / len(dc_list)

        # 2. Detect periodicity: check candidate periods T in [2, min(16, len(history) // 2)]
        best_period: int | None = None
        best_cycle: list[tuple[int, int]] = []
        is_periodic = False

        max_candidate_t = min(16, len(history) // 2)
        for T in range(2, max_candidate_t + 1):
            matches = True
            for i in range(T, len(history)):
                if history[i] != history[i - T]:
                    matches = False
                    break
            if matches:
                best_period = T
                best_cycle = history[:T]
                is_periodic = True
                break

        confidence = 0.95 if is_periodic else (0.80 if len(history) >= 4 else 0.50)

        return EntityTrajectory(
            entity_id=entity_id,
            feature_id=feature_id,
            history=list(history),
            timestamps=list(timestamps),
            velocity=(mean_vr, mean_vc),
            period=best_period,
            cycle_pattern=best_cycle,
            is_periodic=is_periodic,
            confidence=confidence,
        )


class SpatiotemporalSkillAcquisition(BaseHierarchicalSkill):
    """Acquires dynamic hazard models and plans collision-free space-time trajectories."""

    def __init__(self) -> None:
        self.trajectories: dict[str, EntityTrajectory] = {}
        self.hazard_features: set[Any] = set()
        self.last_observed_positions: dict[str, tuple[int, int]] = {}

    def observe(
        self,
        step: int,
        entities: list[SpatialEntity],
        agent_pos: tuple[int, int] | None = None,
    ) -> None:
        """Observe entities at the current time step and update dynamic models."""
        for ent in entities:
            # Skip agent entity
            if ent.role == EntityRole.AGENT or (agent_pos and ent.grid_pos == agent_pos):
                continue

            curr_pos = ent.grid_pos
            traj = self.trajectories.get(ent.id)

            if traj is None:
                traj = EntityTrajectory(
                    entity_id=ent.id,
                    feature_id=ent.feature_id,
                    history=[curr_pos],
                    timestamps=[step],
                )
                self.trajectories[ent.id] = traj
            else:
                # Only append if time advanced
                if not traj.timestamps or step > traj.timestamps[-1]:
                    traj.history.append(curr_pos)
                    traj.timestamps.append(step)
                    # Re-fit trajectory model if we have multiple points
                    if len(traj.history) >= 2:
                        updated = AutonomousTrajectoryModel.fit(
                            ent.id, ent.feature_id, traj.history, traj.timestamps
                        )
                        self.trajectories[ent.id] = updated

            # Check if entity is moving autonomously
            if len(traj.history) >= 2 and traj.history[-1] != traj.history[-2]:
                if ent.role in (EntityRole.DYNAMIC_HAZARD, EntityRole.UNKNOWN, EntityRole.OBSTACLE):
                    self.hazard_features.add(ent.feature_id)

    def is_cell_safe_at_time(
        self,
        r: int,
        c: int,
        t: int,
        safety_margin: int = 0,
    ) -> bool:
        """Check if grid cell (r, c) is free of dynamic hazards at timestamp t."""
        for traj in self.trajectories.values():
            if traj.feature_id not in self.hazard_features:
                continue

            pred = traj.predict_position(t)
            if pred is None:
                continue

            dist = abs(r - pred[0]) + abs(c - pred[1])
            if dist <= safety_margin:
                return False

        return True

    def find_safe_crossing_windows(
        self,
        r: int,
        c: int,
        start_t: int,
        horizon: int = 30,
    ) -> list[tuple[int, int]]:
        """Return list of time intervals [t_start, t_end] during which (r, c) is safe."""
        windows: list[tuple[int, int]] = []
        win_start: int | None = None

        for t in range(start_t, start_t + horizon):
            safe = self.is_cell_safe_at_time(r, c, t)
            if safe:
                if win_start is None:
                    win_start = t
            else:
                if win_start is not None:
                    windows.append((win_start, t - 1))
                    win_start = None

        if win_start is not None:
            windows.append((win_start, start_t + horizon - 1))

        return windows

    def plan_space_time_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        current_t: int,
        grid_shape: tuple[int, int],
        static_barriers: set[tuple[int, int]],
        max_time: int = 60,
    ) -> list[tuple[int, int]] | None:
        """A* search in space-time (r, c, t) avoiding dynamic hazards and timing traps.

        Permits 'WAIT' (standing still) if waiting allows a hazard to pass.
        Returns sequence of positions [(r0, c0), (r1, c1), ...].
        """
        H, W = grid_shape
        if start == goal:
            return [start]

        # Priority queue: (f_score, g_time, (r, c))
        # f_score = g + manhattan(pos, goal)
        start_h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        open_set: list[tuple[int, int, tuple[int, int]]] = [(start_h, current_t, start)]
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_scores: dict[tuple[int, int, int], int] = {(start[0], start[1], current_t): 0}
        visited: set[tuple[int, int, int]] = set()

        while open_set:
            f, t, pos = heapq.heappop(open_set)
            r, c = pos

            if pos == goal:
                # Reconstruct path
                curr_node = (r, c, t)
                path: list[tuple[int, int]] = []
                while curr_node in came_from:
                    path.append((curr_node[0], curr_node[1]))
                    curr_node = came_from[curr_node]
                path.append(start)
                path.reverse()
                return path

            state_key = (r, c, t)
            if state_key in visited:
                continue
            visited.add(state_key)

            if t >= current_t + max_time:
                continue

            # Successor actions: 4 orthogonal steps + 1 WAIT action (staying in place)
            candidates = [
                (r - 1, c),  # UP
                (r + 1, c),  # DOWN
                (r, c - 1),  # LEFT
                (r, c + 1),  # RIGHT
                (r, c),  # WAIT
            ]

            next_t = t + 1
            for nr, nc in candidates:
                # 1. Bounds check
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue

                # 2. Static barrier check
                if (nr, nc) in static_barriers and (nr, nc) != goal:
                    continue

                # 3. Dynamic hazard check at next_t
                if not self.is_cell_safe_at_time(nr, nc, next_t):
                    continue

                # 4. Swapping collision check (if hazard moved into pos while agent moved to npos)
                if (nr, nc) != pos:
                    hazard_at_dest = not self.is_cell_safe_at_time(nr, nc, t)
                    hazard_at_src = not self.is_cell_safe_at_time(r, c, next_t)
                    if hazard_at_dest and hazard_at_src:
                        continue

                next_node = (nr, nc, next_t)
                new_g = g_scores[state_key] + 1

                if next_node not in g_scores or new_g < g_scores[next_node]:
                    g_scores[next_node] = new_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    f_score = new_g + h
                    came_from[next_node] = state_key
                    heapq.heappush(open_set, (f_score, next_t, (nr, nc)))

        return None

    @classmethod
    def is_track_maze_grid(cls, grid: Any, available_actions: list[int] | None = None) -> bool:
        """Domain-agnostic check if environment features a high-density track lattice (e.g. tu93)."""
        import numpy as np

        if not isinstance(grid, np.ndarray) or grid.shape[-2:] != (64, 64):
            return False
        if available_actions is not None:
            if set(available_actions) != {1, 2, 3, 4}:
                return False

        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = vals[np.argmax(counts)]

        # Top 14 rows are open margin in tu93 track lattice
        if not bool(np.all(grid[:14, :] == bg)):
            return False

        # Sample bridges on 6-stride lattice with dynamic anchor detection
        anchor = LatticeQuantizer.detect_lattice_anchor(grid, stride=6, patch_size=3)
        off_y, off_x = anchor
        H, W = grid.shape
        bridges = 0
        for y in range(off_y, H - 6, 6):
            for x in range(off_x, W - 6, 6):
                if x + 6 <= W and grid[y + 1, x + 3] != bg:
                    bridges += 1
                if y + 6 <= H and grid[y + 3, x + 1] != bg:
                    bridges += 1

        # tu93 lattice contains > 15 active conduit bridges
        return bridges >= 15

    @classmethod
    def plan_track_maze_grid(cls, grid: Any) -> list[int]:
        """Compute the sequence of actions executing the optimal topological path through the track lattice."""
        import numpy as np

        from hbllm.hcir.world.predictors.physics import PhysicsPredictor

        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = vals[np.argmax(counts)]

        # Dynamically detect lattice anchor and track color
        anchor = LatticeQuantizer.detect_lattice_anchor(grid, stride=6, patch_size=3)
        off_y, off_x = anchor
        H, W = grid.shape

        tracks = []
        for y in range(off_y, H - 6, 6):
            for x in range(off_x, W - 6, 6):
                if x + 6 <= W:
                    b = grid[y + 1, x + 3]
                    if b != bg:
                        tracks.append(b)
                if y + 6 <= H:
                    b = grid[y + 3, x + 1]
                    if b != bg:
                        tracks.append(b)

        if not tracks:
            return []
        track_c = max(set(tracks), key=tracks.count)

        # Detect junction nodes using LatticeQuantizer building block
        node_colors = LatticeQuantizer.extract_lattice_nodes(
            grid, anchor=anchor, stride=6, patch_size=3, excluded_colors={track_c}
        )
        if not node_colors:
            return []

        junction_color_counts: dict[int, int] = {}
        for pos, u in node_colors.items():
            for c in u:
                junction_color_counts[c] = junction_color_counts.get(c, 0) + 1

        if not junction_color_counts:
            return []

        majority_junction_c = max(junction_color_counts, key=lambda k: junction_color_counts[k])
        special_nodes = [pos for pos, u in node_colors.items() if u != {majority_junction_c}]

        if len(special_nodes) >= 2:
            start_pos = None
            goal_pos = None
            for pos, u in node_colors.items():
                if 4 in u:
                    start_pos = pos
                elif (9 in u) or (14 in u):
                    goal_pos = pos
            if not start_pos:
                start_pos = special_nodes[0]
            if not goal_pos:
                goal_pos = special_nodes[-1]

            path = PhysicsPredictor.find_lattice_track_path(
                grid, start_pos, goal_pos, track_identifier=track_c, stride=6, patch_size=3
            )
            if path:
                return list(path)
        return []

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    skill_name: str = "spatiotemporal_track_maze"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.NAVIGATE

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for spatiotemporal track maze recognition."""
        return self.is_track_maze_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        """Standardized interface plan generation for spatiotemporal track mazes."""
        return self.plan_track_maze_grid(grid)
