"""
Domain-specific predicates for NetHack environments.

Plugs into EmbodiedCausalOperator via the open Predicate Extension Registry,
keeping HCIR core cleanly decoupled from roguelike dungeon mechanics.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.reasoning.operators.base import FrozenGraphView
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator
from hbllm.hcir.graph import HCIRNodeType

logger = logging.getLogger(__name__)


def eval_vitals_safe(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    vital_name = args[0] if args else "health"
    min_val = float(args[1]) if len(args) > 1 else 4.0
    cur_val = float(agent_props.get(vital_name, 10.0))
    return cur_val >= min_val


def eval_safe_from_monster(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    agent_pos = agent_props.get("coords") or (
        agent_props.get("x", 0),
        agent_props.get("y", 0),
    )
    for node in view.iter_nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY):
        if getattr(node, "entity_type", "") in ("monster", "entity", "mob"):
            props = getattr(node, "properties", {})
            if props.get("obj_type") in (15, 16):  # Zombie, Skeleton
                m_pos = props.get("coords") or (props.get("x", 0), props.get("y", 0))
                dist = abs(agent_pos[0] - m_pos[0]) + abs(agent_pos[1] - m_pos[1])
                if dist <= 1:
                    return False
            elif "monster" in node.entity_name.lower():
                dist = float(props.get("distance", float("inf")))
                if dist <= 1.5:
                    return False
    return True


NETHACK_PREDICATES = {
    "vitals_safe": eval_vitals_safe,
    "safe_from_monster": eval_safe_from_monster,
    "no_mobs_adjacent": eval_safe_from_monster,
}


def register_nethack_predicates() -> None:
    """Register all NetHack-specific causal predicates with EmbodiedCausalOperator."""
    for name, handler in NETHACK_PREDICATES.items():
        EmbodiedCausalOperator.register_predicate(name, handler)
    logger.debug(
        "Registered %d NetHack predicates with EmbodiedCausalOperator", len(NETHACK_PREDICATES)
    )
