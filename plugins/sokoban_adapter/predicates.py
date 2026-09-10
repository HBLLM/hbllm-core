"""
Domain-specific predicates for Sokoban environments.

Plugs into EmbodiedCausalOperator via the open Predicate Extension Registry,
keeping HCIR core cleanly decoupled from puzzle-specific tile mechanics.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.reasoning.operators.base import FrozenGraphView
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator

logger = logging.getLogger(__name__)


def eval_box_on_target(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    if len(args) < 2:
        return False
    box_id, target_id = args[0], args[1]
    box = view.get_node(box_id)
    tgt = view.get_node(target_id)
    if not box or not tgt or not hasattr(box, "properties") or not hasattr(tgt, "properties"):
        return False
    b_pos = box.properties.get("coords") or (box.properties.get("x", 0), box.properties.get("y", 0))
    t_pos = tgt.properties.get("coords") or (tgt.properties.get("x", 0), tgt.properties.get("y", 0))
    return b_pos == t_pos


SOKOBAN_PREDICATES = {
    "box_on_target": eval_box_on_target,
}


def register_sokoban_predicates() -> None:
    """Register Sokoban-specific causal predicates with EmbodiedCausalOperator."""
    for name, handler in SOKOBAN_PREDICATES.items():
        EmbodiedCausalOperator.register_predicate(name, handler)
    logger.debug(
        "Registered %d Sokoban predicates with EmbodiedCausalOperator", len(SOKOBAN_PREDICATES)
    )
