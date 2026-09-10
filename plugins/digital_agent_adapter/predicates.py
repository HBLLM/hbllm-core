"""
Domain-specific causal predicates for Digital Agent environments.

Plugs into EmbodiedCausalOperator via the open Predicate Extension Registry,
keeping HCIR core cleanly decoupled from operating system and DOM mechanics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.brain.reasoning.operators.base import FrozenGraphView
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator

logger = logging.getLogger(__name__)

DESTRUCTIVE_PATTERNS = [
    r"rm\s+-rf\s+/",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
    r"curl.*\|\s*(?:bash|sh)",
    r"chmod\s+-R\s+777\s+/",
    r">\s*/dev/sd[a-z]",
]


def eval_file_exists(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    """Predicate: file_exists(filename)."""
    if not args:
        return False
    path = args[0]
    node_id = f"file_{path}"
    return view.has_node(node_id) or view.has_node(path)


def eval_has_error(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    """Predicate: has_error()."""
    agent = view.get_node("agent")
    if not agent or not hasattr(agent, "properties"):
        return False
    return bool(agent.properties.get("has_error", False))


def eval_assertion_failed(
    args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]
) -> bool:
    """Predicate: assertion_failed()."""
    agent = view.get_node("agent")
    if not agent or not hasattr(agent, "properties"):
        return False
    return bool(agent.properties.get("is_assertion_error", False))


def eval_command_safe(args: list[str], view: FrozenGraphView, agent_props: dict[str, Any]) -> bool:
    """Predicate: command_safe(cmd)."""
    if not args:
        return True
    cmd = args[0]
    for pattern in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, cmd):
            return False
    return True


DIGITAL_PREDICATES = {
    "file_exists": eval_file_exists,
    "has_error": eval_has_error,
    "assertion_failed": eval_assertion_failed,
    "command_safe": eval_command_safe,
}


def register_digital_predicates() -> None:
    """Register digital domain predicates with EmbodiedCausalOperator."""
    for name, handler in DIGITAL_PREDICATES.items():
        EmbodiedCausalOperator.register_predicate(name, handler)
    logger.debug(
        "Registered %d digital predicates with EmbodiedCausalOperator", len(DIGITAL_PREDICATES)
    )
