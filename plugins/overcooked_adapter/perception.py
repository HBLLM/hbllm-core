"""
Overcooked Perception Adapter.

Parses kitchen grid, culinary objects, pot statuses, and partner poses to construct
epistemic state estimates and partner intent hypotheses.
"""

from __future__ import annotations

import logging
from typing import Any

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
