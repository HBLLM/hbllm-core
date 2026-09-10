"""
Domain-specific causal predicates for Semantic Ambiguity Environments.

Plugs into EmbodiedCausalOperator via the open Predicate Extension Registry,
keeping HCIR core cleanly decoupled from natural language grounding and ambiguity mechanics.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.reasoning.operators.base import FrozenGraphView
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator

logger = logging.getLogger(__name__)


def eval_is_canonical(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    """Predicate: is_canonical()."""
    agent = view.get_node("agent")
    if not agent or not hasattr(agent, "properties"):
        return False
    return bool(agent.properties.get("is_canonical", False))


def eval_ambiguity_detected(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    """Predicate: ambiguity_detected()."""
    agent = view.get_node("agent")
    if not agent or not hasattr(agent, "properties"):
        return False
    return bool(agent.properties.get("ambiguity_detected", False))


def eval_subgoal_completed(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    """Predicate: subgoal_completed(subgoal_name)."""
    if not args:
        return False
    subgoal = args[0]
    agent = view.get_node("agent")
    if not agent or not hasattr(agent, "properties"):
        return False
    completed = agent.properties.get("completed_subgoals", [])
    return subgoal in completed


SEMANTIC_PREDICATES = {
    "is_canonical": eval_is_canonical,
    "ambiguity_detected": eval_ambiguity_detected,
    "subgoal_completed": eval_subgoal_completed,
}


def register_semantic_predicates() -> None:
    """Register semantic domain predicates with EmbodiedCausalOperator."""
    for name, handler in SEMANTIC_PREDICATES.items():
        EmbodiedCausalOperator.register_predicate(name, handler)
    logger.debug(
        "Registered %d semantic predicates with EmbodiedCausalOperator", len(SEMANTIC_PREDICATES)
    )
