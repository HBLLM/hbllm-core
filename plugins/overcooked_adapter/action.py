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
        # If agent bumped into partner or destination was blocked, invalidate plan
        if self.expected_pos is not None and obs.agent.pos != self.expected_pos:
            self.planned_actions.clear()
            if obs.agent.agent_id == 1:
                # Agent 1 yields for 1 tick to let Agent 0 clear the bottleneck
                self.expected_pos = obs.agent.pos
                return OvercookedAction.STAY

        # Check if next planned action steps into current partner position or mutual collision
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
        """Determine next recipe objective and navigate to appliance with multi-agent coordination."""
        held = obs.agent.held_item

        ready_pots = perception_data.get("ready_pots", [])
        filling_pots = perception_data.get("filling_pots", [])
        cooking_pots = perception_data.get("cooking_pots", [])
        ready_to_cook = perception_data.get("ready_to_cook_pots", [])
        onion_dispensers = perception_data.get("onion_dispensers", [])
        dish_dispensers = perception_data.get("dish_dispensers", [])
        serving_stations = perception_data.get("serving_stations", [])

        can_reach_pots = any(self._is_reachable(obs, p.pos, ignore_partner=True) for p in obs.pots)
        can_reach_serving = any(
            self._is_reachable(obs, s, ignore_partner=True) for s in serving_stations
        )
        can_reach_onions = any(
            self._is_reachable(obs, o, ignore_partner=True) for o in onion_dispensers
        )
        can_reach_dishes = any(
            self._is_reachable(obs, d, ignore_partner=True) for d in dish_dispensers
        )

        # -------------------------------------------------------------
        # ROLE 1: SUPPLIER / PREP AGENT (Partitioned from pots)
        # -------------------------------------------------------------
        if not can_reach_pots and obs.partner is not None:
            # If holding item, place on shared counter
            if held in (CulinaryItem.ONION, CulinaryItem.DISH):
                shared_counter = self._find_shared_counter(obs, empty_only=True)
                if shared_counter is not None:
                    self._plan_interact_with(obs, shared_counter)
                    return
                # If no empty shared counter, wait
                self.planned_actions = [OvercookedAction.STAY]
                return

            # Holding NONE: decide whether to pass onions or dish
            if held == CulinaryItem.NONE:
                # Count onions on shared counter
                shared_counters = self._get_shared_counters(obs)
                onions_on_counter = sum(
                    1 for c in shared_counters if obs.counter_items.get(c) == CulinaryItem.ONION
                )
                dishes_on_counter = sum(
                    1 for c in shared_counters if obs.counter_items.get(c) == CulinaryItem.DISH
                )
                total_pot_onions = sum(p.onions_in_pot for p in obs.pots)

                if (total_pot_onions + onions_on_counter) < 3 and can_reach_onions:
                    self._plan_interact_with(obs, onion_dispensers[0])
                    return
                elif dishes_on_counter == 0 and can_reach_dishes:
                    self._plan_interact_with(obs, dish_dispensers[0])
                    return
                else:
                    self.planned_actions = [OvercookedAction.STAY]
                    return

        # -------------------------------------------------------------
        # ROLE 2: CHEF / SERVER (Can reach pots)
        # -------------------------------------------------------------
        # Priority 1: Deliver ready soup
        if held == CulinaryItem.SOUP:
            if can_reach_serving and serving_stations:
                self._plan_interact_with(obs, serving_stations[0])
                return
            else:
                shared_counter = self._find_shared_counter(obs, empty_only=True)
                if shared_counter is not None:
                    self._plan_interact_with(obs, shared_counter)
                    return

        # Priority 2: Scoop ready soup if holding dish
        if held == CulinaryItem.DISH and ready_pots:
            self._plan_interact_with(obs, ready_pots[0].pos)
            return

        # Priority 3: Ignite full pot (3 onions) to start cooking
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
        if held == CulinaryItem.NONE and (ready_pots or cooking_pots):
            if can_reach_dishes and dish_dispensers:
                self._plan_interact_with(obs, dish_dispensers[0])
                return
            # Pick up dish from shared counter
            shared_dishes = [
                pos
                for pos, item in obs.counter_items.items()
                if item == CulinaryItem.DISH and self._is_reachable(obs, pos)
            ]
            if shared_dishes:
                self._plan_interact_with(obs, shared_dishes[0])
                return

        # Priority 5: If holding dish and pot is cooking, wait near pot
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
                self._plan_interact_with(obs, target_pot.pos)
                return
            elif held == CulinaryItem.NONE:
                if can_reach_onions and onion_dispensers:
                    self._plan_interact_with(obs, onion_dispensers[0])
                    return
                # Pick up onion from shared counter
                shared_onions = [
                    pos
                    for pos, item in obs.counter_items.items()
                    if item == CulinaryItem.ONION and self._is_reachable(obs, pos)
                ]
                if shared_onions:
                    self._plan_interact_with(obs, shared_onions[0])
                    return

        # Courtesy Yield: If empty-handed and standing adjacent to pot/station that partner needs
        if (
            held == CulinaryItem.NONE
            and obs.partner is not None
            and obs.partner.held_item != CulinaryItem.NONE
        ):
            blocking_pot = any(
                abs(obs.agent.pos[0] - p.pos[0]) + abs(obs.agent.pos[1] - p.pos[1]) == 1
                for p in obs.pots
            )
            if blocking_pot and obs.partner.held_item in (
                CulinaryItem.ONION,
                CulinaryItem.DISH,
            ):
                for act, dr, dc in DIRECTION_DELTAS:
                    nr, nc = obs.agent.pos[0] + dr, obs.agent.pos[1] + dc
                    if (
                        0 <= nr < len(obs.grid)
                        and 0 <= nc < len(obs.grid[0])
                        and obs.grid[nr][nc] == int(KitchenTile.FLOOR)
                    ):
                        if (nr, nc) != obs.partner.pos and not any(
                            abs(nr - p.pos[0]) + abs(nc - p.pos[1]) == 1 for p in obs.pots
                        ):
                            self.planned_actions = [act]
                            return

        # Default wait
        self.planned_actions = [OvercookedAction.STAY]

    def _get_shared_counters(self, obs: OvercookedObservation) -> list[tuple[int, int]]:
        """Get all counter locations reachable by both agents."""
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        shared = []
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(KitchenTile.COUNTER):
                    if self._is_reachable(
                        obs, (r, c), ignore_partner=True, from_partner=False
                    ) and (
                        obs.partner is None
                        or self._is_reachable(obs, (r, c), ignore_partner=True, from_partner=True)
                    ):
                        shared.append((r, c))
        return shared

    def _find_shared_counter(
        self, obs: OvercookedObservation, empty_only: bool = True
    ) -> tuple[int, int] | None:
        """Find nearest shared counter accessible by both agent and partner."""
        shared = self._get_shared_counters(obs)
        ar, ac = obs.agent.pos
        best_counter = None
        best_dist = float("inf")

        for pos in shared:
            if empty_only and pos in obs.counter_items:
                continue
            if self._is_reachable(obs, pos):
                dist = abs(ar - pos[0]) + abs(ac - pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_counter = pos
        return best_counter

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

    def _is_reachable(
        self,
        obs: OvercookedObservation,
        appliance_pos: tuple[int, int],
        ignore_partner: bool = False,
        from_partner: bool = False,
    ) -> bool:
        """Check if any adjacent interaction cell is reachable by agent."""
        start_pos = obs.partner.pos if from_partner and obs.partner is not None else obs.agent.pos
        other_pos = (
            None
            if ignore_partner
            else (obs.agent.pos if from_partner else (obs.partner.pos if obs.partner else None))
        )
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (appliance_pos[0] + dr, appliance_pos[1] + dc)
            if (
                self._bfs_path(
                    obs,
                    start_pos,
                    adj,
                    ignore_partner=ignore_partner,
                    other_pos=other_pos,
                )
                is not None
            ):
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
            self.planned_actions = list(best_path) + [
                best_interact_turn,
                OvercookedAction.INTERACT,
            ]
            return

        # If primary interaction cell is occupied by partner, queue adjacent to that cell
        if obs.partner is not None:
            for act, dr, dc in DIRECTION_DELTAS:
                adj = (appliance_pos[0] - dr, appliance_pos[1] - dc)
                if obs.partner.pos == adj:
                    for _, ndr, ndc in DIRECTION_DELTAS:
                        nadj = (adj[0] + ndr, adj[1] + ndc)
                        path = self._bfs_path(obs, obs.agent.pos, nadj)
                        if path is not None and len(path) > 0:
                            self.planned_actions = list(path)
                            return

        self.planned_actions = [OvercookedAction.STAY]

    def _bfs_path(
        self,
        obs: OvercookedObservation,
        start_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        ignore_partner: bool = False,
        other_pos: tuple[int, int] | None = None,
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

        blocked_pos = (
            None
            if ignore_partner
            else (
                other_pos if other_pos is not None else (obs.partner.pos if obs.partner else None)
            )
        )

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
                    if obs.grid[nr][nc] == int(KitchenTile.FLOOR) and (nr, nc) != blocked_pos:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [act]))

        return None
