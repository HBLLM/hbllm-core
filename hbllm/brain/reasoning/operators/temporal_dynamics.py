"""Spatio-Temporal Hazard & Velocity Dynamics Model for HCIR.

Tracks dynamic obstacles and autonomous hazards across time ticks:
1. Fits linear velocities v = (dx/dt, dy/dt).
2. Detects periodic cyclical patterns (e.g. moving lasers, patrol sentries)
   with period T and discrete cycle trajectory phases.
3. Forecasts future hazard occupancy H(t + dt) to enable collision-free path planning
   and safe wait-state synchronization.

Independence Level: L1 (100% deterministic, 0 LLM tokens).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EntityTrajectory:
    """Historical spatio-temporal trajectory points for an entity."""

    entity_id: str
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, t)
    velocity: tuple[float, float] = (0.0, 0.0)
    is_periodic: bool = False
    detected_period: float = 0.0
    cycle_positions: list[tuple[float, float]] = field(default_factory=list)

    def add_point(self, x: float, y: float, t: float | None = None) -> None:
        timestamp = time.time() if t is None else t
        self.history.append((float(x), float(y), float(timestamp)))
        if len(self.history) > 50:
            self.history.pop(0)
        self._update_kinematics()

    def _update_kinematics(self) -> None:
        if len(self.history) < 2:
            return

        # 1. Instantaneous / moving average velocity
        p_prev = self.history[-2]
        p_curr = self.history[-1]
        dt = p_curr[2] - p_prev[2]
        if dt > 1e-5:
            dx = (p_curr[0] - p_prev[0]) / dt
            dy = (p_curr[1] - p_prev[1]) / dt
            # Exponential smoothing
            self.velocity = (0.6 * dx + 0.4 * self.velocity[0], 0.6 * dy + 0.4 * self.velocity[1])
        else:
            # Step-based displacement
            self.velocity = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])

        # 2. Periodic cycle detection (e.g. oscillating patrol paths or cycling lasers)
        if len(self.history) >= 6:
            positions = [(p[0], p[1]) for p in self.history]
            current = positions[-1]

            # Search for recurrence of the current position earlier in history
            revisit_indices = [
                i
                for i, pos in enumerate(positions[:-1])
                if math.isclose(pos[0], current[0], abs_tol=0.5)
                and math.isclose(pos[1], current[1], abs_tol=0.5)
            ]

            if revisit_indices:
                period_steps = len(positions) - 1 - revisit_indices[-1]
                if period_steps >= 2:
                    cycle = positions[revisit_indices[-1] : -1]
                    if len(cycle) >= 2:
                        self.is_periodic = True
                        self.detected_period = float(period_steps)
                        self.cycle_positions = cycle


class TemporalVelocityModel:
    """Predictive trajectory and dynamic collision forecasting engine."""

    def __init__(self) -> None:
        self.trajectories: dict[str, EntityTrajectory] = {}

    def record_observation(
        self,
        entity_id: str,
        position: tuple[float, float],
        timestamp: float | None = None,
    ) -> EntityTrajectory:
        """Record entity observation and update its trajectory model."""
        if entity_id not in self.trajectories:
            self.trajectories[entity_id] = EntityTrajectory(entity_id=entity_id)
        traj = self.trajectories[entity_id]
        traj.add_point(position[0], position[1], timestamp)
        return traj

    def predict_position(self, entity_id: str, future_dt: float) -> tuple[float, float]:
        """Predict position of entity at future offset future_dt."""
        traj = self.trajectories.get(entity_id)
        if traj is None or not traj.history:
            return (0.0, 0.0)

        curr_x, curr_y, _ = traj.history[-1]

        # If periodic with a known cycle, project along discrete cycle points
        if traj.is_periodic and traj.cycle_positions and traj.detected_period > 0:
            step_offset = int(round(future_dt)) % len(traj.cycle_positions)
            return traj.cycle_positions[step_offset]

        # Otherwise use linear velocity projection
        vx, vy = traj.velocity
        return (curr_x + vx * future_dt, curr_y + vy * future_dt)

    def is_collision_imminent(
        self,
        agent_pos: tuple[float, float],
        future_dt: float = 1.0,
        safe_margin: float = 1.5,
    ) -> list[str]:
        """Return IDs of entities predicted to collide with agent within future_dt."""
        colliding_entities: list[str] = []
        for eid, _ in self.trajectories.items():
            pred_x, pred_y = self.predict_position(eid, future_dt)
            dist = math.hypot(pred_x - agent_pos[0], pred_y - agent_pos[1])
            if dist < safe_margin:
                colliding_entities.append(eid)
        return colliding_entities

    def find_safe_wait_time(
        self,
        target_pos: tuple[float, float],
        entity_id: str,
        max_lookahead: int = 10,
        safe_margin: float = 1.5,
    ) -> int | None:
        """Find the earliest future step offset where target_pos is safe from entity."""
        for step in range(max_lookahead + 1):
            pred_x, pred_y = self.predict_position(entity_id, float(step))
            dist = math.hypot(pred_x - target_pos[0], pred_y - target_pos[1])
            if dist >= safe_margin:
                return step
        return None
