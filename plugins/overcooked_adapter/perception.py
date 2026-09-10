"""
Overcooked Perception Adapter.

Parses kitchen grid, culinary objects, pot statuses, and partner poses to construct
epistemic state estimates and partner intent hypotheses.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)

from .types import CulinaryItem, KitchenTile, OvercookedObservation, PotStatus

logger = logging.getLogger(__name__)


class OvercookedPerceptionAdapter:
    """Extracts topological layout, culinary stages, and partner intentions."""

    def __init__(self) -> None:
        self.onion_dispensers: list[tuple[int, int]] = []
        self.dish_dispensers: list[tuple[int, int]] = []
        self.serving_stations: list[tuple[int, int]] = []
        self._analyzed_layout = False

    def reset(self) -> None:
        """Reset internal layout cache."""
        self.onion_dispensers.clear()
        self.dish_dispensers.clear()
        self.serving_stations.clear()
        self._analyzed_layout = False

    def process_observation(self, obs: OvercookedObservation) -> dict[str, Any]:
        """Convert raw kitchen observation into typed perception graph."""
        if not self._analyzed_layout:
            self._scan_static_layout(obs.grid)
            self._analyzed_layout = True

        partner_intent = "none"
        if obs.partner is not None:
            if obs.partner.held_item == CulinaryItem.ONION:
                partner_intent = "filling_pot"
            elif obs.partner.held_item == CulinaryItem.DISH:
                partner_intent = "scooping_soup"
            elif obs.partner.held_item == CulinaryItem.SOUP:
                partner_intent = "serving_soup"
            else:
                partner_intent = "idle_or_navigating"

        ready_pots = [p for p in obs.pots if p.status == PotStatus.READY]
        ready_to_cook_pots = [
            p
            for p in obs.pots
            if p.onions_in_pot >= p.required_onions
            and p.status not in (PotStatus.COOKING, PotStatus.READY)
        ]
        filling_pots = [
            p
            for p in obs.pots
            if p.onions_in_pot < p.required_onions
            and p.status in (PotStatus.EMPTY, PotStatus.FILLING)
        ]
        cooking_pots = [p for p in obs.pots if p.status == PotStatus.COOKING]

        return {
            "agent_pos": obs.agent.pos,
            "agent_item": obs.agent.held_item,
            "partner_pos": obs.partner.pos if obs.partner else None,
            "partner_item": obs.partner.held_item if obs.partner else None,
            "partner_intent": partner_intent,
            "onion_dispensers": list(self.onion_dispensers),
            "dish_dispensers": list(self.dish_dispensers),
            "serving_stations": list(self.serving_stations),
            "ready_pots": ready_pots,
            "ready_to_cook_pots": ready_to_cook_pots,
            "filling_pots": filling_pots,
            "cooking_pots": cooking_pots,
            "counter_items": dict(obs.counter_items),
            "soups_delivered": obs.soups_delivered,
            "step_count": obs.step_count,
        }

    def _scan_static_layout(self, grid: list[list[int]]) -> None:
        """Locate stationary kitchen appliances."""
        for r, row in enumerate(grid):
            for c, tile in enumerate(row):
                if tile == int(KitchenTile.ONION_DISPENSER):
                    self.onion_dispensers.append((r, c))
                elif tile == int(KitchenTile.DISH_DISPENSER):
                    self.dish_dispensers.append((r, c))
                elif tile == int(KitchenTile.SERVING_STATION):
                    self.serving_stations.append((r, c))

    def _get_reachable_floors(
        self, grid: list[list[int]], start: tuple[int, int]
    ) -> set[tuple[int, int]]:
        """Compute all floor tiles reachable from start position."""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        if not (0 <= start[0] < height and 0 <= start[1] < width):
            return set()
        visited = {start}
        queue = [start]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                    if grid[nr][nc] == int(KitchenTile.FLOOR):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return visited

    def _is_appliance_reachable(
        self, reachable_floors: set[tuple[int, int]], pos: tuple[int, int]
    ) -> bool:
        """Check if any adjacent cell of an appliance is in reachable floors."""
        return any(
            (pos[0] + dr, pos[1] + dc) in reachable_floors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )

    def to_cognitive_graph(
        self, obs: OvercookedObservation, perception_data: dict[str, Any]
    ) -> CognitiveGraph:
        """Construct a CognitiveGraph with appliances, pots, agent state, and active recipe goal."""
        if not self._analyzed_layout:
            self._scan_static_layout(obs.grid)
            self._analyzed_layout = True

        graph = CognitiveGraph()
        pr, pc = obs.agent.pos
        held_name = obs.agent.held_item.name.lower() if obs.agent.held_item else "none"
        reachable_floors = self._get_reachable_floors(obs.grid, (pr, pc))

        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="cook_player",
            entity_type="agent",
            properties={
                "x": pr,
                "y": pc,
                "coords": (pr, pc),
                "held_object_id": held_name,
                "carrying": {"type": held_name},
                "reach_distance": 1.5,
                "soups_delivered": obs.soups_delivered,
                "won": obs.won,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        graph.add_node(agent_node)

        # Ingest reachable appliances
        for r, c in self.onion_dispensers:
            if self._is_appliance_reachable(reachable_floors, (r, c)):
                node = PhysicalEntityNode(
                    id=f"dispenser_onion_{r}_{c}",
                    entity_name="onion_dispenser",
                    entity_type="appliance",
                    properties={
                        "coords": (r, c),
                        "x": r,
                        "y": c,
                        "distance": abs(pr - r) + abs(pc - c),
                    },
                )
                graph.add_node(node)

        for r, c in self.dish_dispensers:
            if self._is_appliance_reachable(reachable_floors, (r, c)):
                node = PhysicalEntityNode(
                    id=f"dispenser_dish_{r}_{c}",
                    entity_name="dish_dispenser",
                    entity_type="appliance",
                    properties={
                        "coords": (r, c),
                        "x": r,
                        "y": c,
                        "distance": abs(pr - r) + abs(pc - c),
                    },
                )
                graph.add_node(node)

        for r, c in self.serving_stations:
            if self._is_appliance_reachable(reachable_floors, (r, c)):
                node = PhysicalEntityNode(
                    id=f"serving_station_{r}_{c}",
                    entity_name="serving_station",
                    entity_type="appliance",
                    properties={
                        "coords": (r, c),
                        "x": r,
                        "y": c,
                        "distance": abs(pr - r) + abs(pc - c),
                    },
                )
                graph.add_node(node)

        can_reach_pots = False
        for p in obs.pots:
            r, c = p.pos
            if self._is_appliance_reachable(reachable_floors, (r, c)):
                can_reach_pots = True
                node = PhysicalEntityNode(
                    id=f"pot_{r}_{c}",
                    entity_name="pot",
                    entity_type="pot",
                    properties={
                        "coords": (r, c),
                        "x": r,
                        "y": c,
                        "distance": abs(pr - r) + abs(pc - c),
                        "num_onions": p.onions_in_pot,
                        "is_cooked": p.status == PotStatus.READY,
                        "status": p.status.name.lower(),
                    },
                )
                graph.add_node(node)

        # Ingest counter items
        for (cr, cc), item in obs.counter_items.items():
            item_name = item.name.lower()
            node = PhysicalEntityNode(
                id=f"counter_item_{cr}_{cc}",
                entity_name=f"{item_name}_on_counter",
                entity_type="counter_item",
                properties={
                    "coords": (cr, cc),
                    "x": cr,
                    "y": cc,
                    "distance": abs(pr - cr) + abs(pc - cc),
                    "item": item_name,
                },
            )
            graph.add_node(node)

        # Ingest reachable shared counters
        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(KitchenTile.COUNTER) and self._is_appliance_reachable(
                    reachable_floors, (r, c)
                ):
                    node = PhysicalEntityNode(
                        id=f"counter_{r}_{c}",
                        entity_name="shared_counter",
                        entity_type="counter",
                        properties={
                            "coords": (r, c),
                            "x": r,
                            "y": c,
                            "distance": abs(pr - r) + abs(pc - c),
                            "has_item": (r, c) in obs.counter_items,
                        },
                    )
                    graph.add_node(node)

        # Active GoalNode based on cognitive role and kitchen state
        can_reach_serving = any(
            self._is_appliance_reachable(reachable_floors, s) for s in self.serving_stations
        )
        can_reach_dish = any(
            self._is_appliance_reachable(reachable_floors, d) for d in self.dish_dispensers
        )
        pot_ready = any(p.status == PotStatus.READY for p in obs.pots)
        pot_cooking = any(p.status == PotStatus.COOKING for p in obs.pots)
        # Check if a dish is already on a shared counter reachable by this agent
        dish_on_shared_counter = any(
            item == CulinaryItem.DISH
            for pos, item in obs.counter_items.items()
            if self._is_appliance_reachable(reachable_floors, pos)
        )

        if not can_reach_pots and not can_reach_serving:
            # Supplier side: can reach dispensers (onion/dish) but not pots or serving.
            if pot_ready or pot_cooking:
                # Pot is cooking/ready — switch from supplying onions to supplying dish.
                if can_reach_dish and not dish_on_shared_counter:
                    target_conditions = ["dish_on_counter"]
                else:
                    # Dish already on counter or no dish dispenser — idle.
                    target_conditions = ["onion_on_counter"]  # will resolve to stay/wait
            else:
                # Pot still needs onions.
                target_conditions = ["onion_on_counter"]
        elif not can_reach_serving and obs.partner is not None:
            # Chef partitioned from serving station: plate soup and place on shared counter.
            target_conditions = ["soup_on_counter"]
        else:
            # Chef / Server who can reach serving station.
            target_conditions = ["soup_served"]

        goal_node = GoalNode(
            id="goal_active",
            properties={"target_conditions": target_conditions},
        )
        graph.add_node(goal_node)

        return graph
