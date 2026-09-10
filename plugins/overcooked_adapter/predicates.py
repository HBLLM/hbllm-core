"""
Domain-specific predicates for Overcooked environments.

Plugs into EmbodiedCausalOperator via the open Predicate Extension Registry,
keeping HCIR core cleanly decoupled from kitchen/culinary mechanics.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.reasoning.operators.base import FrozenGraphView
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator
from hbllm.hcir.graph import HCIRNodeType

logger = logging.getLogger(__name__)


def eval_pot_has_items(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    if len(args) < 3:
        return False
    pot_id, _, count_str = args[0], args[1], args[2]
    target_count = int(count_str)
    pot = view.get_node(pot_id)
    if pot and hasattr(pot, "properties"):
        cur = pot.properties.get("num_onions", pot.properties.get("items_count", 0))
        return cur >= target_count
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_name == pot_id or n.entity_type == pot_id or pot_id in n.id:
            cur = n.properties.get("num_onions", n.properties.get("items_count", 0))
            if cur >= target_count:
                return True
    return False


def eval_is_cooked(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    if not args:
        return False
    pot_id = args[0]
    pot = view.get_node(pot_id)
    if pot and hasattr(pot, "properties"):
        return (
            bool(pot.properties.get("is_cooked", False)) or pot.properties.get("status") == "ready"
        )
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_name == pot_id or n.entity_type == pot_id or pot_id in n.id:
            if bool(n.properties.get("is_cooked", False)) or n.properties.get("status") == "ready":
                return True
    return False


def eval_is_cooking(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    if not args:
        return False
    pot_id = args[0]
    pot = view.get_node(pot_id)
    if pot and hasattr(pot, "properties"):
        return pot.properties.get("status") == "cooking"
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_name == pot_id or n.entity_type == pot_id or pot_id in n.id:
            if n.properties.get("status") == "cooking":
                return True
    return False


def eval_counter_has(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    item_name = args[0] if args else "onion"
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_type == "counter_item" or "counter" in n.entity_name:
            if n.properties.get("item") == item_name or item_name in n.entity_name:
                return True
    return False


def eval_onion_on_counter(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_type == "counter_item" or "counter" in n.entity_name:
            if n.properties.get("item") == "onion" or "onion" in n.entity_name:
                return True
    return False


def eval_soup_on_counter(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_type == "counter_item" or "counter" in n.entity_name:
            if n.properties.get("item") == "soup" or "soup" in n.entity_name:
                return True
    return False


def eval_dish_on_counter(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    for n in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if n.entity_type == "counter_item" or "counter" in n.entity_name:
            if n.properties.get("item") == "dish" or "dish" in n.entity_name:
                return True
    return False


def eval_soup_served(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    req_soups = int(args[0]) if args else 1
    cur_soups = int(agent_props.get("soups_delivered", 0))
    return cur_soups >= req_soups or bool(agent_props.get("won", False))


OVERCOOKED_PREDICATES = {
    "pot_has_items": eval_pot_has_items,
    "is_cooked": eval_is_cooked,
    "is_cooking": eval_is_cooking,
    "counter_has": eval_counter_has,
    "has_counter_item": eval_counter_has,
    "onion_on_counter": eval_onion_on_counter,
    "soup_on_counter": eval_soup_on_counter,
    "dish_on_counter": eval_dish_on_counter,
    "soup_served": eval_soup_served,
}


def register_overcooked_predicates() -> None:
    """Register all Overcooked-specific causal predicates with EmbodiedCausalOperator."""
    for name, handler in OVERCOOKED_PREDICATES.items():
        EmbodiedCausalOperator.register_predicate(name, handler)
    logger.debug(
        "Registered %d Overcooked predicates with EmbodiedCausalOperator",
        len(OVERCOOKED_PREDICATES),
    )
