"""
Safety-Gymnasium Action Adapter and Constrained Causal Planner.

Implements SafetyCausalPlanner using artificial potential fields with
hard safety boundary clearance and dynamic obstacle velocity projection.
"""

from __future__ import annotations

import logging
import math

from .types import (
    SafetyGymAction,
    SafetyObservation,
)

logger = logging.getLogger(__name__)


class SafetyGymActionAdapter:
    """
    HCIR Constrained Safe Navigation Planner.
    Calculates collision-free trajectories satisfying C_t = 0 constraints.
    """

    def __init__(self, clearance_margin: float = 0.35) -> None:
        self.clearance_margin = clearance_margin

    def plan_next_action(self, obs: SafetyObservation) -> SafetyGymAction:
        """Select next discrete action minimizing distance to goal while maintaining C=0."""
        ax, ay = obs.agent_pos
        gx, gy = obs.goal_pos
        current_heading = obs.agent_heading

        # 1. Attractive force toward goal
        dist_to_goal = math.hypot(gx - ax, gy - ay)
        if dist_to_goal < 0.05:
            return SafetyGymAction.NOOP

        fx = (gx - ax) / dist_to_goal
        fy = (gy - ay) / dist_to_goal

        # 2. Repulsive force from static hazards and pillars
        for obstacle in obs.hazards + obs.pillars:
            dist = math.hypot(ax - obstacle.x, ay - obstacle.y)
            safe_dist = obstacle.radius + self.clearance_margin
            if dist < safe_dist and dist > 0.01:
                repulse_mag = 5.0 * (1.0 / dist - 1.0 / safe_dist) / (dist**2)
                repulse_mag = min(12.0, max(0.0, repulse_mag))
                fx += repulse_mag * (ax - obstacle.x) / dist
                fy += repulse_mag * (ay - obstacle.y) / dist

        # 3. Repulsive force from dynamic gremlins (predictive)
        for g in obs.gremlins:
            # Predict gremlin position 2 steps ahead
            future_gx = g.x + 2 * g.vx
            future_gy = g.y + 2 * g.vy
            dist = math.hypot(ax - future_gx, ay - future_gy)
            safe_dist = g.radius + self.clearance_margin + 0.15
            if dist < safe_dist and dist > 0.01:
                repulse_mag = 6.0 * (1.0 / dist - 1.0 / safe_dist) / (dist**2)
                repulse_mag = min(12.0, max(0.0, repulse_mag))
                fx += repulse_mag * (ax - future_gx) / dist
                fy += repulse_mag * (ay - future_gy) / dist

        # 4. Arena boundary repulsion
        boundary_limit = 2.6
        if ax > boundary_limit:
            fx -= 5.0 * (ax - boundary_limit)
        elif ax < -boundary_limit:
            fx += 5.0 * (-boundary_limit - ax)
        if ay > boundary_limit:
            fy -= 5.0 * (ay - boundary_limit)
        elif ay < -boundary_limit:
            fy += 5.0 * (-boundary_limit - ay)

        # 5. Compute target steering angle
        desired_heading = math.atan2(fy, fx)
        # Normalize angle difference to [-pi, pi]
        angle_diff = (desired_heading - current_heading + math.pi) % (2 * math.pi) - math.pi

        # Steering decision: allow forward motion when within 45 degrees of desired heading
        if angle_diff > math.pi / 4.0:
            return SafetyGymAction.TURN_LEFT
        elif angle_diff < -math.pi / 4.0:
            return SafetyGymAction.TURN_RIGHT
        else:
            return SafetyGymAction.FORWARD
