"""
Stochastic Action Adapter.

Implements closed-loop, surprise-resilient pathfinding with automatic re-planning
when sensory or actuator discrepancies occur.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from .perception import StochasticPerceptionAdapter
from .types import StochasticAction, StochasticObservation

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = [
    (StochasticAction.UP, -1, 0),
    (StochasticAction.DOWN, 1, 0),
    (StochasticAction.LEFT, 0, -1),
    (StochasticAction.RIGHT, 0, 1),
]


class StochasticActionAdapter:
    """Surprise-reactive closed-loop navigation planner."""

    def __init__(self) -> None:
        self.planned_actions: list[StochasticAction] = []

    def reset(self) -> None:
        """Clear action buffer."""
        self.planned_actions.clear()

    def select_action(
        self,
        obs: StochasticObservation,
        perception: StochasticPerceptionAdapter,
        perception_data: dict[str, Any],
    ) -> StochasticAction:
        """Select action, triggering instant re-plan if surprise detected or path invalidated."""
        surprise = perception_data.get("surprise_detected", False)
        target_pos = perception_data.get("target_pos")

        # Re-plan if surprise detected (actuator slip or sudden drift) or plan empty
        if surprise or not self.planned_actions:
            self._plan_path(obs.player_pos, target_pos, perception_data.get("obstacles", set()))

        action = self.planned_actions.pop(0) if self.planned_actions else StochasticAction.NOOP

        # Compute expected transition and register with perception
        pr, pc = obs.player_pos
        dr, dc = (0, 0)
        for act, r_delta, c_delta in DIRECTION_DELTAS:
            if act == action:
                dr, dc = r_delta, c_delta
                break

        expected_pos = (pr + dr, pc + dc)
        perception.register_expected_transition(expected_pos)

        return action

    def _plan_path(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int] | None,
        obstacles: set[tuple[int, int]],
    ) -> None:
        """Find shortest BFS path from current pos to believed target pos."""
        self.planned_actions.clear()
        if target_pos is None or start_pos == target_pos:
            return

        queue: deque[tuple[tuple[int, int], list[StochasticAction]]] = deque([(start_pos, [])])
        visited = {start_pos}

        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == target_pos:
                self.planned_actions = path
                return

            cr, cc = curr_pos
            for act, dr, dc in DIRECTION_DELTAS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    if (nr, nc) not in obstacles and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))
