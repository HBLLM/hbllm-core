"""HCIR-backed chat prompt and memory context helpers for serving & domain modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import HCIRNodeType
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuration parameters for the serving HTTP/WebSocket layer."""

    data_dir: str = "data"
    api_timeout: float = 30.0
    stream_timeout: float = 60.0
    system_prompt: str = ""
    inject_memory: bool = True
    inject_identity: bool = True
    inject_curiosity: bool = True
    inject_perception: bool = True
    inject_fuzzy_logic: bool = True
    inject_symbolic_logic: bool = True


@dataclass
class ChatContext:
    """Bounded context recalled from tenant HCIR tiers and session buffers."""

    recent_turns: str = ""
    recalled_context: str = ""
    summary_context: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


async def get_dynamic_system_prompt(
    bus: Any | None,
    tenant_id: str,
    source_node_id: str,
) -> str:
    """Build dynamic system prompt from tenant HCIR TieredWorkspace + optional bus registry."""
    persona_name = "Sentra"
    system_prompt_base = (
        "You are Sentra, an advanced cognitive AI assistant powered by the HBLLM HCIR Cognitive OS."
    )
    goals: list[str] = []

    try:
        from hbllm.serving.cognitive_registry import get_cognitive_registry

        registry = get_cognitive_registry()
        sys_obj = registry.get_or_create(tenant_id or "default")
        for ws in (sys_obj.tiered_workspace.persistent, sys_obj.tiered_workspace.brain):
            for node in ws.graph.nodes_by_type(HCIRNodeType.GOAL):
                desc = getattr(node, "description", "") or getattr(node, "goal_name", "")
                if desc:
                    goals.append(str(desc))
            for node in ws.graph.nodes_by_type(HCIRNodeType.VALUE):
                label = getattr(node, "value_name", "") or getattr(node, "description", "")
                if label and label not in goals:
                    goals.append(f"Value: {label}")
    except Exception as exc:
        logger.debug("HCIR dynamic prompt inspection skipped: %s", exc)

    caps_lines: list[str] = []
    if bus is not None and hasattr(bus, "request"):
        try:
            discover_msg = Message(
                type=MessageType.QUERY,
                source_node_id=source_node_id,
                tenant_id=tenant_id,
                topic="registry.discover",
                payload={},
            )
            reg_resp = await bus.request("registry.discover", discover_msg, timeout=1.0)
            nodes = reg_resp.payload.get("nodes", [])
            if nodes:
                caps_lines.append(
                    "Active service nodes: " + ", ".join(str(n.get("node_id", "")) for n in nodes)
                )
        except Exception as exc:
            logger.debug("Optional bus discovery skipped: %s", exc)

    parts = [system_prompt_base]
    if persona_name:
        parts.append(f"Persona: {persona_name}")
    if goals:
        parts.append("Active Goals/Values: " + "; ".join(goals[:5]))
    parts.extend(caps_lines)
    return "\n".join(parts)


async def get_chat_memories(
    bus: Any | None,
    tenant_id: str,
    session_id: str,
    user_text: str,
    history: list[dict[str, Any]] | None = None,
) -> ChatContext:
    """Recall relevant session turns and HCIR knowledge context for chat inference."""
    ctx = ChatContext()
    try:
        from hbllm.serving.cognitive_registry import get_cognitive_registry

        registry = get_cognitive_registry()
        turns = registry.get_session_turns(tenant_id or "default", session_id, limit=8)
        if not turns and history:
            turns = list(history)[-8:]
        if turns:
            ctx.recent_turns = "\n".join(
                f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns
            )

        sys_obj = registry.get_or_create(tenant_id or "default")
        query_words = {w.lower() for w in (user_text or "").split() if len(w) > 2}
        recalled_items: list[str] = []
        for ws in (sys_obj.tiered_workspace.brain, sys_obj.tiered_workspace.persistent):
            for node in ws.graph.all_nodes():
                if len(recalled_items) >= 8:
                    break
                text_val = (
                    getattr(node, "description", "")
                    or getattr(node, "condition", "")
                    or getattr(node, "name", "")
                    or getattr(node, "content", "")
                )
                if not text_val:
                    continue
                if not query_words or any(w in str(text_val).lower() for w in query_words):
                    recalled_items.append(str(text_val))
        if recalled_items:
            ctx.recalled_context = "\n".join(recalled_items)
    except Exception as exc:
        logger.debug("HCIR memory recall failed (non-fatal): %s", exc)

    if not ctx.recalled_context and bus is not None and hasattr(bus, "request") and user_text:
        try:
            search_msg = Message(
                type=MessageType.QUERY,
                source_node_id="api_server",
                tenant_id=tenant_id,
                session_id=session_id,
                topic="memory.search",
                payload={"query": user_text, "tenant_id": tenant_id, "top_k": 5},
            )
            resp = await bus.request("memory.search", search_msg, timeout=1.5)
            results = resp.payload.get("results", [])
            snippets = [str(r.get("content", "")) for r in results if r.get("content")]
            if snippets:
                ctx.recalled_context = "\n".join(snippets[:5])
        except Exception:
            pass

    return ctx
