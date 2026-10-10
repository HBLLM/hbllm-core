"""Anterior Cingulate Cortex (ACC / BA24 / BA32) Conflict & Frustration Monitor.

Models the mammalian anterior cingulate cortex:
1. Conflict & Prediction-Error Monitoring: Tracks repeated failure / lethal death events
   in localized coordinate clusters.
2. Frustration Accumulation: When consecutive deaths cluster within spatial radius <= 3,
   frustration F in [0.0, 1.0] scales non-linearly.
3. Executive Policy Modulation:
   - Conflict-induced Heuristic Gating: Suppresses greedy straight-line distance heuristics
     when F >= 0.5 to prevent repetitive suicide loops into lethal chokepoints.
   - Topological Detour Bias: Directs search toward orthogonal perimeter corridors and
     unvisited topological homologies.
   - Habenular IOR Gain Modulation: Synergistically amplifies negative inhibition priors
     around persistent failure zones.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FailureCluster:
    """Cluster of recurring lethal / failure events."""

    center_r: float
    center_c: float
    count: int = 1
    recent_steps: list[int] = field(default_factory=list)
    lethal_actions: list[int] = field(default_factory=list)


class AnteriorCingulateConflictMonitor:
    """Anterior Cingulate Cortex (ACC) conflict, error, and frustration monitor."""

    def __init__(
        self,
        cluster_radius: float = 3.5,
        frustration_decay: float = 0.90,
        frustration_threshold: float = 0.50,
    ) -> None:
        self.cluster_radius = cluster_radius
        self.frustration_decay = frustration_decay
        self.frustration_threshold = frustration_threshold

        self.clusters: list[FailureCluster] = []
        self.frustration: float = 0.0
        self.active_conflict: bool = False
        self.failure_history: list[tuple[float, float, int]] = []  # (r, c, step)
        self.detour_target: tuple[int, int] | None = None

    def reset_episode(self, retain_long_term: bool = True) -> None:
        """Reset transient conflict state, optionally preserving long-term failure clusters."""
        self.active_conflict = False
        self.detour_target = None
        if not retain_long_term:
            self.clusters.clear()
            self.frustration = 0.0
            self.failure_history.clear()
        else:
            # Mild decay between trials
            self.frustration *= self.frustration_decay
            if self.frustration < 0.1:
                self.frustration = 0.0

    def register_death_event(
        self,
        actor_pos: tuple[float, float],
        step: int,
        last_action: int | None = None,
    ) -> float:
        """Record a lethal / reset event and compute updated frustration level."""
        ar, ac = actor_pos
        self.failure_history.append((ar, ac, step))

        matched = False
        for cl in self.clusters:
            dist = math.hypot(cl.center_r - ar, cl.center_c - ac)
            if dist <= self.cluster_radius:
                cl.count += 1
                # Update moving center
                cl.center_r = (cl.center_r * (cl.count - 1) + ar) / cl.count
                cl.center_c = (cl.center_c * (cl.count - 1) + ac) / cl.count
                cl.recent_steps.append(step)
                if last_action is not None:
                    cl.lethal_actions.append(last_action)
                matched = True
                break

        if not matched:
            self.clusters.append(
                FailureCluster(
                    center_r=ar,
                    center_c=ac,
                    count=1,
                    recent_steps=[step],
                    lethal_actions=[last_action] if last_action is not None else [],
                )
            )

        # Compute frustration based on maximum cluster recurrence
        max_recurrence = max((cl.count for cl in self.clusters), default=0)
        if max_recurrence <= 1:
            self.frustration = 0.0
        else:
            # Sigmoidal growth: count 2 -> ~0.50, count 3 -> ~0.76, count 4+ -> ~0.90+
            self.frustration = 1.0 - math.exp(-0.7 * (max_recurrence - 1))

        self.active_conflict = self.frustration >= self.frustration_threshold
        if self.active_conflict:
            logger.warning(
                "ACC Conflict Monitor: High frustration detected (F=%.2f, max_cluster_deaths=%d). "
                "Triggering topological detour and suppressing greedy heuristics.",
                self.frustration,
                max_recurrence,
            )
        return self.frustration

    def compute_heuristic_weight(self, default_weight: float = 1.0) -> float:
        """Compute modulated A* goal heuristic weight.

        When frustration is high, suppresses greedy distance heuristics so the agent
        is not inexorably pulled into the same lethal trap.
        """
        if not self.active_conflict:
            return default_weight
        # Attenuate heuristic: F=0.5 -> 0.5x, F=0.8 -> 0.2x
        suppression = max(0.1, 1.0 - self.frustration)
        return default_weight * suppression

    def get_chokepoint_penalty(
        self,
        candidate_pos: tuple[int, int],
        base_penalty: float = 25.0,
    ) -> float:
        """Compute extra routing penalty for stepping near recurring death clusters."""
        if not self.clusters:
            return 0.0

        cr, cc = candidate_pos
        total_penalty = 0.0
        for cl in self.clusters:
            if cl.count < 2:
                continue
            dist = math.hypot(cl.center_r - cr, cl.center_c - cc)
            if dist <= self.cluster_radius:
                # Severity scales with cluster recurrence count and proximity
                proximity_gain = max(0.0, 1.0 - (dist / self.cluster_radius))
                total_penalty += (
                    base_penalty * (cl.count - 1) * proximity_gain * (1.0 + self.frustration)
                )

        return total_penalty

    def synthesize_perimeter_detour(
        self,
        current_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        grid_shape: tuple[int, int],
        traversable: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Find an orthogonal perimeter waypoint that routes away from the active death cluster."""
        if not self.active_conflict or not self.clusters:
            return None

        # Find most lethal cluster
        worst_cluster = max(self.clusters, key=lambda c: c.count)
        if worst_cluster.count < 2:
            return None

        H, W = grid_shape
        cr, cc = current_pos
        gr, gc = goal_pos
        lr, lc = worst_cluster.center_r, worst_cluster.center_c

        # Vector from lethal center to current position (repulsive direction)
        dr = cr - lr
        dc = cc - lc
        norm = math.hypot(dr, dc)
        if norm < 1e-4:
            dr, dc = 1.0, 0.0
            norm = 1.0

        # Evaluate candidate detour waypoints in traversable cells that maximize distance
        # from lethal center while maintaining progress
        best_candidate: tuple[int, int] | None = None
        best_score = -float("inf")

        for r, c in traversable:
            dist_to_lethal = math.hypot(r - lr, c - lc)
            if dist_to_lethal <= self.cluster_radius:
                continue  # Skip inside lethal cone

            dist_to_goal = math.hypot(r - gr, c - gc)
            dist_to_current = math.hypot(r - cr, c - cc)

            # Detour score: prefer far from lethal cluster and accessible from current pos
            score = (dist_to_lethal * 2.0) - (dist_to_current * 0.5) - (dist_to_goal * 0.2)
            if score > best_score:
                best_score = score
                best_candidate = (r, c)

        self.detour_target = best_candidate
        return best_candidate
