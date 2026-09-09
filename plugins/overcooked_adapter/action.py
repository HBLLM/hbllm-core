"""
Overcooked Action Adapter.

Implements causal recipe scheduling (Onion -> Pot -> Soup -> Plate -> Serve)
and multi-agent collision-avoiding pathfinding in kitchen topologies.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from .types import (
    CulinaryItem,
    KitchenTile,
    OvercookedAction,
    OvercookedObservation,
)

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = [
    (OvercookedAction.UP, -1, 0),
    (OvercookedAction.DOWN, 1, 0),
    (OvercookedAction.LEFT, 0, -1),
    (OvercookedAction.RIGHT, 0, 1),
]


class OvercookedActionAdapter:
    """Causal recipe executor and pathfinding adapter for kitchen coordination."""

    def __init__(self) -> None:
        self.planned_actions: list[OvercookedAction] = []
        self.expected_pos: tuple[int, int] | None = None

    def reset(self) -> None:
        """Reset action plan queue and position expectations."""
        self.planned_actions.clear()
        self.expected_pos = None

    def select_action(
        self,
        obs: OvercookedObservation,
        perception_data: dict[str, Any],
    ) -> OvercookedAction:
        """Select next kitchen action, detecting unexpected bumps and dynamic partner collisions."""
        # If agent did not reach expected pos (bumped into partner), invalidate plan
        if self.expected_pos is not None and obs.agent.pos != self.expected_pos:
            self.planned_actions.clear()

        # Check if next planned action steps into current partner position
        if self.planned_actions and obs.partner is not None:
            first_act = self.planned_actions[0]
            for act, dr, dc in DIRECTION_DELTAS:
                if act == first_act:
                    dest = (obs.agent.pos[0] + dr, obs.agent.pos[1] + dc)
                    if dest == obs.partner.pos:
                        self.planned_actions.clear()
                        self.expected_pos = obs.agent.pos
                        return OvercookedAction.STAY

        if not self.planned_actions:
            self._plan_next_step(obs, perception_data)

        if self.planned_actions:
            action = self.planned_actions.pop(0)
            # Update expected pos for next turn
            pr, pc = obs.agent.pos
            dr, dc = (0, 0)
            for act, r_delta, c_delta in DIRECTION_DELTAS:
                if act == action:
                    dr, dc = r_delta, c_delta
                    break
            nr, nc = pr + dr, pc + dc
            if (
                0 <= nr < len(obs.grid)
                and 0 <= nc < len(obs.grid[0])
                and obs.grid[nr][nc] == int(KitchenTile.FLOOR)
            ):
                self.expected_pos = (nr, nc)
            else:
                self.expected_pos = (pr, pc)
            return action

        self.expected_pos = obs.agent.pos
        return OvercookedAction.STAY

    def _plan_next_step(
        self,
        obs: OvercookedObservation,
        perception_data: dict[str, Any],
    ) -> None:
        """Determine next recipe objective and navigate to appliance."""
        held = obs.agent.held_item

        ready_pots = perception_data.get("ready_pots", [])
        filling_pots = perception_data.get("filling_pots", [])
        cooking_pots = perception_data.get("cooking_pots", [])
        onion_dispensers = perception_data.get("onion_dispensers", [])
        dish_dispensers = perception_data.get("dish_dispensers", [])
        serving_stations = perception_data.get("serving_stations", [])

        # Priority 1: Deliver ready soup
        if held == CulinaryItem.SOUP and serving_stations:
            self._plan_interact_with(obs, serving_stations[0])
            return

        # Priority 2: Scoop ready soup if holding dish
        if held == CulinaryItem.DISH and ready_pots:
            self._plan_interact_with(obs, ready_pots[0].pos)
            return

        # Priority 3: Ignite full pot (3 onions) to start cooking
        ready_to_cook = perception_data.get("ready_to_cook_pots", [])
        if ready_to_cook:
            target_pot = ready_to_cook[0]
            if held == CulinaryItem.NONE:
                self._plan_interact_with(obs, target_pot.pos)
                return
            elif held == CulinaryItem.ONION:
                empty_counter = self._find_empty_counter(obs)
                if empty_counter is not None:
                    self._plan_interact_with(obs, empty_counter)
                    return

        # Priority 4: Fetch dish if pot is cooking or ready
        if held == CulinaryItem.NONE and (ready_pots or cooking_pots) and dish_dispensers:
            if self._is_reachable(obs, dish_dispensers[0]):
                self._plan_interact_with(obs, dish_dispensers[0])
                return

        # Priority 5: If holding dish and pot is cooking, wait adjacent to pot
        if held == CulinaryItem.DISH and cooking_pots:
            pot_pos = cooking_pots[0].pos
            self._plan_interact_with(obs, pot_pos)
            if self.planned_actions and self.planned_actions[-1] == OvercookedAction.INTERACT:
                self.planned_actions.pop()
            return

        # Priority 6: Fill pot with onions
        if filling_pots:
            target_pot = filling_pots[0]
            if held == CulinaryItem.ONION:
                if self._is_reachable(obs, target_pot.pos):
                    self._plan_interact_with(obs, target_pot.pos)
                else:
                    # Asymmetric room: place onion on shared counter (2, 3)
                    self._plan_interact_with(obs, (2, 3))
                return
            elif held == CulinaryItem.NONE and onion_dispensers:
                self._plan_interact_with(obs, onion_dispensers[0])
                return

        # Default wait
        self.planned_actions = [OvercookedAction.STAY]

    def _find_empty_counter(self, obs: OvercookedObservation) -> tuple[int, int] | None:
        """Find nearest accessible counter tile not currently holding an item."""
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        best_counter = None
        best_dist = float("inf")
        ar, ac = obs.agent.pos
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(KitchenTile.COUNTER) and (r, c) not in obs.counter_items:
                    if self._is_reachable(obs, (r, c)):
                        dist = abs(ar - r) + abs(ac - c)
                        if dist < best_dist:
                            best_dist = dist
                            best_counter = (r, c)
        return best_counter

    def _is_reachable(self, obs: OvercookedObservation, appliance_pos: tuple[int, int]) -> bool:
        """Check if any adjacent interaction cell is reachable by agent."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (appliance_pos[0] + dr, appliance_pos[1] + dc)
            if self._bfs_path(obs, obs.agent.pos, adj) is not None:
                return True
        return False

    def _plan_interact_with(
        self, obs: OvercookedObservation, appliance_pos: tuple[int, int]
    ) -> None:
        """Find path to adjacent interaction cell and face appliance to interact."""
        best_path: list[OvercookedAction] | None = None
        best_interact_turn: OvercookedAction | None = None

        for act, dr, dc in DIRECTION_DELTAS:
            adj = (appliance_pos[0] - dr, appliance_pos[1] - dc)
            path = self._bfs_path(obs, obs.agent.pos, adj)
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_interact_turn = act

        if best_path is not None and best_interact_turn is not None:
            # Path to adjacent tile + orient towards appliance + INTERACT
            self.planned_actions = list(best_path) + [best_interact_turn, OvercookedAction.INTERACT]
        else:
            self.planned_actions = [OvercookedAction.STAY]

    def _bfs_path(
        self,
        obs: OvercookedObservation,
        start_pos: tuple[int, int],
        goal_pos: tuple[int, int],
    ) -> list[OvercookedAction] | None:
        """BFS navigation avoiding partner collision and counter obstacles."""
        if start_pos == goal_pos:
            return []

        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0

        # Goal pos must be valid floor
        gr, gc = goal_pos
        if not (0 <= gr < height and 0 <= gc < width) or obs.grid[gr][gc] != int(KitchenTile.FLOOR):
            return None

        partner_pos = obs.partner.pos if obs.partner else None

        queue: deque[tuple[tuple[int, int], list[OvercookedAction]]] = deque([(start_pos, [])])
        visited = {start_pos}

        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == goal_pos:
                return path

            cr, cc = curr_pos
            for act, dr, dc in DIRECTION_DELTAS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if obs.grid[nr][nc] == int(KitchenTile.FLOOR) and (nr, nc) != partner_pos:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [act]))

        return None
