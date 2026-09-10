"""
Domain-specific causal predicates for Stochastic and Epistemic Environments.

Plugs into EmbodiedCausalOperator via the open Predicate Extension Registry,
keeping HCIR core cleanly decoupled from grid navigation and belief uncertainty mechanics.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.reasoning.operators.base import FrozenGraphView
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator

logger = logging.getLogger(__name__)


def eval_surprise_detected(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    """Predicate: surprise_detected()."""
    agent = view.get_node("agent")
    if not agent or not hasattr(agent, "properties"):
        return False
    return bool(agent.properties.get("surprise_detected", False))


def eval_target_occluded(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    """Predicate: target_occluded()."""
    target = view.get_node("target_goal")
    if not target or not hasattr(target, "properties"):
        return False
    return bool(target.properties.get("occluded", False))


def eval_at_target(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    """Predicate: at_target(agent_id, target_id)."""
    agent_id = args[0] if len(args) > 0 else "agent"
    target_id = args[1] if len(args) > 1 else "target_goal"
    agent = view.get_node(agent_id)
    target = view.get_node(target_id)
    if (
        not agent
        or not target
        or not hasattr(agent, "properties")
        or not hasattr(target, "properties")
    ):
        return False
    a_pos = agent.properties.get("pos")
    t_pos = target.properties.get("pos")
    return a_pos is not None and t_pos is not None and a_pos == t_pos


STOCHASTIC_PREDICATES = {
    "surprise_detected": eval_surprise_detected,
    "target_occluded": eval_target_occluded,
    "at_target": eval_at_target,
}


def register_stochastic_predicates() -> None:
    """Register stochastic/epistemic predicates with EmbodiedCausalOperator."""
    for name, handler in STOCHASTIC_PREDICATES.items():
        EmbodiedCausalOperator.register_predicate(name, handler)
    logger.debug(
        "Registered %d stochastic predicates with EmbodiedCausalOperator",
        len(STOCHASTIC_PREDICATES),
    )
