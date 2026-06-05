from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hbllm.network.messages import Message, MessageType

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus

logger = logging.getLogger(__name__)


async def get_dynamic_system_prompt(bus: MessageBus, tenant_id: str, source_node_id: str) -> str:
    """
    Dynamically retrieve the system prompt, goals, constraints, and active capabilities
    of the system for the given tenant over the message bus.
    """
    persona_name = "Sentra"
    system_prompt_base = "You are Sentra, an advanced cognitive AI assistant powered by the HBLLM modular architecture."
    goals = []
    constraints = []

    # 1. Query identity profile from IdentityNode
    try:
        id_msg = Message(
            type=MessageType.QUERY,
            source_node_id=source_node_id,
            tenant_id=tenant_id,
            topic="identity.query",
            payload={},
        )
        id_resp = await bus.request("identity.query", id_msg, timeout=2.0)
        if id_resp.payload.get("found"):
            profile = id_resp.payload.get("profile")
            if profile:
                persona_name = profile.get("persona_name", persona_name)
                system_prompt_base = (
                    profile.get("system_prompt")
                    or f"You are {persona_name}, an advanced cognitive AI assistant powered by the HBLLM modular architecture."
                )
                goals = profile.get("goals") or []
                constraints = profile.get("constraints") or []
    except Exception as e:
        logger.warning("Failed to retrieve identity profile: %s. Using default identity.", e)

    # 2. Query active registry capabilities
    has_browser = False
    has_execution = False
    has_logic = False
    has_memory = False

    try:
        discover_msg = Message(
            type=MessageType.QUERY,
            source_node_id=source_node_id,
            tenant_id=tenant_id,
            topic="registry.discover",
            payload={},
        )
        reg_resp = await bus.request("registry.discover", discover_msg, timeout=2.0)
        nodes = reg_resp.payload.get("nodes", [])
        for n in nodes:
            node_id = n.get("node_id", "")
            caps = n.get("capabilities", [])
            if "web_search" in caps or "browser" in node_id:
                has_browser = True
            if "exec" in node_id or "execution" in node_id or "shell_executor" in node_id:
                has_execution = True
            if "theorem_proving" in caps or "logic" in node_id:
                has_logic = True
            if "memory" in caps or "memory" in node_id:
                has_memory = True
    except Exception as e:
        logger.warning("Failed to discover active nodes from registry: %s", e)

    capabilities_parts = []
    if has_browser:
        capabilities_parts.append(
            "a BrowserNode (which allows you to browse the web and search for real-time information)"
        )
    if has_execution:
        capabilities_parts.append("an ExecutionNode (for running Python code in a secure sandbox)")
    if has_logic:
        capabilities_parts.append("a LogicNode (powered by Z3 for symbolic reasoning)")
    if has_memory:
        capabilities_parts.append("a persistent memory node")

    system_prompt = system_prompt_base
    if capabilities_parts:
        if len(capabilities_parts) == 1:
            caps_str = capabilities_parts[0]
        elif len(capabilities_parts) == 2:
            caps_str = " and ".join(capabilities_parts)
        else:
            caps_str = ", ".join(capabilities_parts[:-1]) + ", and " + capabilities_parts[-1]
        system_prompt += (
            f" You have access to various cognitive and tool modules, including {caps_str}."
        )

    # Append goals
    if goals:
        system_prompt += "\n\nGoals:\n- " + "\n- ".join(goals)

    # Append constraints
    if constraints:
        system_prompt += "\n\nConstraints:\n- " + "\n- ".join(constraints)

    system_prompt += " Be helpful, precise, and accurate."
    return system_prompt
