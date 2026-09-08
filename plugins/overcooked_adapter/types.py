"""
Overcooked-AI Adapter Types.

Defines strongly-typed representations of kitchen layout tiles, culinary items,
cooking pot states, partner intent hypotheses, and cooperative benchmark tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class OvercookedAction(IntEnum):
    """6 discrete primitive actions in Overcooked."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    STAY = 5


class KitchenTile(IntEnum):
    """Grid tile types in kitchen."""

    FLOOR = 0
    COUNTER = 1
    ONION_DISPENSER = 2
    DISH_DISPENSER = 3
    POT = 4
    SERVING_STATION = 5


class CulinaryItem(StrEnum):
    """Culinary objects held by agents or resting on counters."""

    NONE = "none"
    ONION = "onion"
    DISH = "dish"
    SOUP = "soup"


class PotStatus(StrEnum):
    """Cooking states of a soup pot."""

    EMPTY = "empty"
    FILLING = "filling"  # 1-2 onions placed
    COOKING = "cooking"  # 3 onions placed, cooking timer ticking
    READY = "ready"  # Soup finished, waiting for dish scoop


class OvercookedTier(StrEnum):
    """Cooperative benchmark evaluation tiers."""

    TIER_1_CRAMPED_ROOM_SOLO = "tier_1_cramped_room_solo"
    TIER_2_ASYMMETRIC_COORDINATION = "tier_2_asymmetric_coordination"
    TIER_3_CORRIDOR_CONTENTION = "tier_3_corridor_contention"
    TIER_4_DYNAMIC_PARTNER_ADAPTATION = "tier_4_dynamic_partner_adaptation"
    TIER_5_MULTI_ORDER_SURGE = "tier_5_multi_order_surge"


@dataclass
class PotState:
    """State of a cooking pot."""

    pos: tuple[int, int]
    onions_in_pot: int = 0
    required_onions: int = 3
    cooking_timer: int = 0
    cooking_duration: int = 4
    status: PotStatus = PotStatus.EMPTY


@dataclass
class AgentState:
    """Agent pose and inventory in kitchen."""

    agent_id: int
    pos: tuple[int, int]
    orientation: tuple[int, int] = (0, 1)  # Facing direction
    held_item: CulinaryItem = CulinaryItem.NONE


@dataclass
class OvercookedObservation:
    """Complete observation of cooperative kitchen state."""

    grid: list[list[int]]
    agent: AgentState
    partner: AgentState | None
    pots: list[PotState]
    counter_items: dict[tuple[int, int], CulinaryItem]
    soups_delivered: int
    step_count: int
    max_steps: int
    done: bool = False
    won: bool = False
    info: dict[str, Any] = field(default_factory=dict)
