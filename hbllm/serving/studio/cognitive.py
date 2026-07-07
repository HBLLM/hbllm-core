"""
Studio Cognitive Endpoints — Notifications, Habits, Digest, Threads.

Exposes the cognitive-layer nodes for observability and manual
interaction from the Studio dashboard.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from hbllm.serving.studio.helpers import (
    get_node,
    get_tenant_id,
    require_bus,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Notification Gateway ─────────────────────────────────────────────────────


@router.get("/studio/notifications")
async def studio_notifications_status(request: Request):
    """Notification gateway status, channel health, and delivery stats."""

    node = get_node("NotificationGateway")
    if not node:
        return {
            "status": "not_loaded",
            "channels": [],
            "total_sent": 0,
            "total_failed": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    channels = []
    channel_registry = getattr(node, "_channels", {})
    for name, ch in channel_registry.items():
        channels.append(
            {
                "name": name,
                "enabled": getattr(ch, "enabled", True),
                "type": type(ch).__name__,
            }
        )

    return {
        "status": "active",
        "channels": channels,
        **stats,
    }


@router.post("/studio/notifications/test")
async def studio_notifications_test(request: Request):
    """Send a test notification through the gateway.

    Body::

        {
            "channel": "push",
            "title": "Test",
            "body": "Hello from Studio"
        }
    """
    bus = require_bus()
    body = await request.json()
    tenant_id = get_tenant_id(request)

    from hbllm.network.messages import Message, MessageType

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="system.notification",
        payload={
            "channel": body.get("channel", "push"),
            "title": body.get("title", "Studio Test"),
            "body": body.get("body", "Test notification from Studio dashboard"),
            "priority": body.get("priority", "normal"),
        },
    )
    await bus.publish("system.notification", msg)

    return {"status": "sent", "channel": body.get("channel", "push")}


# ── Habit Tracker ────────────────────────────────────────────────────────────


@router.get("/studio/habits")
async def studio_habits_status(request: Request):
    """Habit tracker status and active habits."""

    node = get_node("HabitTracker")
    if not node:
        return {
            "status": "not_loaded",
            "habits": [],
            "total_tracked": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    habits = []
    habit_registry = getattr(node, "_habits", {})
    for tenant_habits in habit_registry.values():
        if isinstance(tenant_habits, dict):
            for habit_id, habit in tenant_habits.items():
                habits.append(
                    {
                        "id": habit_id,
                        "name": getattr(habit, "name", habit_id),
                        "frequency": getattr(habit, "frequency", "unknown"),
                        "streak": getattr(habit, "streak", 0),
                        "last_triggered": getattr(habit, "last_triggered", 0),
                    }
                )

    return {
        "status": "active",
        "habits": habits,
        **stats,
    }


@router.post("/studio/habits")
async def studio_habits_create(request: Request):
    """Create or update a habit via the bus.

    Body::

        {
            "name": "Morning standup",
            "cron": "0 9 * * 1-5",
            "action": "remind"
        }
    """
    bus = require_bus()
    body = await request.json()
    tenant_id = get_tenant_id(request)

    from hbllm.network.messages import Message, MessageType

    msg = Message(
        type=MessageType.COMMAND,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="habits.create",
        payload={
            "name": body.get("name", "Unnamed habit"),
            "cron": body.get("cron"),
            "frequency": body.get("frequency", "daily"),
            "action": body.get("action", "remind"),
            "metadata": body.get("metadata", {}),
        },
    )
    await bus.publish("habits.create", msg)

    return {"status": "created", "name": body.get("name")}


# ── Activity Digest ──────────────────────────────────────────────────────────


@router.get("/studio/digest")
async def studio_digest_status(request: Request):
    """Latest activity digest summary."""

    node = get_node("ActivityDigest")
    if not node:
        return {
            "status": "not_loaded",
            "latest": None,
            "total_generated": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    latest = getattr(node, "_latest_digest", None)

    return {
        "status": "active",
        "latest": latest.to_dict() if latest and hasattr(latest, "to_dict") else latest,
        **stats,
    }


@router.post("/studio/digest/generate")
async def studio_digest_generate(request: Request):
    """Force-generate an activity digest."""
    bus = require_bus()
    tenant_id = get_tenant_id(request)

    from hbllm.network.messages import Message, MessageType

    msg = Message(
        type=MessageType.COMMAND,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="digest.generate",
        payload={"force": True},
    )
    await bus.publish("digest.generate", msg)

    return {"status": "generating"}


# ── Conversation Threads ─────────────────────────────────────────────────────


@router.get("/studio/threads")
async def studio_threads_status(request: Request):
    """Active conversation threads per tenant."""

    node = get_node("ConversationThread")
    if not node:
        return {
            "status": "not_loaded",
            "threads": [],
            "total_messages": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    threads = []
    thread_registry = getattr(node, "_threads", {})
    for tid, thread in thread_registry.items():
        messages = getattr(thread, "messages", [])
        threads.append(
            {
                "thread_id": tid,
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "last_activity": getattr(thread, "last_activity", 0),
            }
        )

    return {
        "status": "active",
        "threads": threads,
        **stats,
    }


# ── Cognitive State V3 ────────────────────────────────────────────────────────


@router.get("/studio/cognitive/state")
async def get_cognitive_state(request: Request):
    """Retrieve the current CognitiveState from the orchestrators."""
    node = get_node("CognitiveExecutiveController")
    if node and hasattr(node, "active_states"):
        return {
            "status": "active",
            "active_states": {
                cid: state.to_dict() if hasattr(state, "to_dict") else str(state)
                for cid, state in node.active_states.items()
            },
        }

    cortex = get_node("ExecutiveCortex")
    if cortex and hasattr(cortex, "snapshot"):
        return {
            "status": "active",
            "cortex_snapshot": cortex.snapshot(),
        }

    return {
        "status": "not_loaded",
        "active_states": {},
        "cortex_snapshot": None,
    }


# ── Network Topology (SNN + LoRA + LLM) ──────────────────────────────────────


@router.get("/studio/topology")
async def get_studio_topology(request: Request):
    """Unified network topology graph (SNN + LoRA + LLM Router) for visualization."""
    from hbllm.network.metrics import MetricsCollector
    from hbllm.serving.state import _state
    from hbllm.serving.studio.helpers import get_node_map

    collector = MetricsCollector.get_instance()

    # 1. Gather SNN Data
    categories = ["physics", "math", "coding", "finance", "personal", "general"]
    brain = _state.get("brain")
    memory_node = None
    if brain:
        node_map = get_node_map()
        memory_node = node_map.get("MemoryNode")

    all_cats = list(categories)
    if memory_node and hasattr(memory_node, "primer"):
        for cat in memory_node.primer.categories.keys():
            if cat not in all_cats:
                all_cats.append(cat)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # SNN Group Node
    nodes.append(
        {
            "id": "snn_layer",
            "label": "SNN Priming Layer",
            "type": "layer",
            "group": "snn",
            "details": {"description": "Spiking Neural Network category-priming layer"},
        }
    )

    # Add SNN category nodes
    for cat in all_cats:
        neuron_id = f"priming_{cat}"
        pot = collector._mem_gauges.get(f"snn_potential:{neuron_id}", 0.0)
        threshold = 1.0
        label = cat
        if cat.startswith("cluster_") and memory_node and hasattr(memory_node, "semantic_db"):
            try:
                cluster_id = int(cat.split("_")[1])
                label = memory_node.semantic_db.cluster_manager.get_cluster_label(
                    cluster_id, memory_node.semantic_db.documents
                )
            except (ValueError, IndexError):
                pass
        if memory_node and hasattr(memory_node, "primer"):
            acc = memory_node.primer.categories.get(cat)
            if acc:
                pot = acc.get_potential()
                threshold = acc.neuron.config.threshold

        nodes.append(
            {
                "id": f"snn_{cat}",
                "label": f"SNN: {label}",
                "type": "neuron",
                "group": "snn",
                "details": {
                    "potential": round(pot, 3),
                    "threshold": threshold,
                    "is_active": pot >= threshold,
                },
            }
        )
        # Link category neurons to the main layer
        edges.append(
            {
                "source": "snn_layer",
                "target": f"snn_{cat}",
                "relation": "contains",
            }
        )

    # Attention Fatigue
    attn_pot = collector._mem_gauges.get("snn_potential:human_attention_fatigue", 0.0)
    nodes.append(
        {
            "id": "snn_attention_fatigue",
            "label": "Attention Fatigue",
            "type": "neuron",
            "group": "snn",
            "details": {
                "potential": round(attn_pot, 3),
                "threshold": 0.8,
                "is_refractory": attn_pot >= 0.8,
            },
        }
    )
    edges.append(
        {
            "source": "snn_layer",
            "target": "snn_attention_fatigue",
            "relation": "contains",
        }
    )

    # 2. Gather LoRA Data
    import glob
    import os

    data_dir = os.environ.get("HBLLM_DATA_DIR", "data")
    lora_dir = os.path.join(data_dir, "lora")

    nodes.append(
        {
            "id": "lora_layer",
            "label": "LoRA Adapter Registry",
            "type": "layer",
            "group": "lora",
            "details": {"description": "Domain-specific LoRA adapters"},
        }
    )

    active_adapters = []
    if os.path.exists(lora_dir):
        for pt_file in glob.glob(os.path.join(lora_dir, "**/*.pt"), recursive=True):
            name = os.path.basename(pt_file)
            size_mb = os.path.getsize(pt_file) / (1024 * 1024)
            if not name.endswith(".pending.pt"):
                active_adapters.append((name, pt_file, size_mb))

    for name, path, size in active_adapters:
        adapter_id = f"lora_{name.replace('.pt', '')}"
        nodes.append(
            {
                "id": adapter_id,
                "label": f"LoRA: {name}",
                "type": "adapter",
                "group": "lora",
                "details": {
                    "size_mb": round(size, 2),
                    "path": path,
                },
            }
        )
        edges.append(
            {
                "source": "lora_layer",
                "target": adapter_id,
                "relation": "contains",
            }
        )

        # Dynamically link primed SNN categories to matching LoRA adapters
        for cat in all_cats:
            if cat in name.lower():
                edges.append(
                    {
                        "source": f"snn_{cat}",
                        "target": adapter_id,
                        "relation": "primes",
                        "weight": 1.0,
                    }
                )

    # 3. Gather Dual LLM Router Data
    nodes.append(
        {
            "id": "llm_router",
            "label": "Dual LLM Router",
            "type": "router",
            "group": "llm",
            "details": {"description": "Decides whether to route to local or external LLM"},
        }
    )

    router_node = get_node("DualLLMRouter")
    router_stats = {}
    local_model = "local"
    external_model = "external"
    complexity_threshold = 0.4

    if router_node:
        if hasattr(router_node, "stats") and router_node.stats:
            router_stats = router_node.stats.to_dict()
        if hasattr(router_node, "_local_name") and router_node._local_name:
            local_model = getattr(router_node._local_name, "name", "local")
        if hasattr(router_node, "_external_name") and router_node._external_name:
            external_model = getattr(router_node._external_name, "name", "external")
        if hasattr(router_node, "complexity_threshold"):
            complexity_threshold = router_node.complexity_threshold

    # Local LLM Node
    nodes.append(
        {
            "id": "llm_local",
            "label": f"Local LLM ({local_model})",
            "type": "llm",
            "group": "llm",
            "details": {
                "model_name": local_model,
                "tier": "local",
            },
        }
    )
    edges.append(
        {
            "source": "llm_router",
            "target": "llm_local",
            "relation": "routes_to",
        }
    )

    # External LLM Node
    nodes.append(
        {
            "id": "llm_external",
            "label": f"External LLM ({external_model})",
            "type": "llm",
            "group": "llm",
            "details": {
                "model_name": external_model,
                "tier": "external",
            },
        }
    )
    edges.append(
        {
            "source": "llm_router",
            "target": "llm_external",
            "relation": "routes_to",
        }
    )

    # Link all active LoRA adapters to Local LLM (where they are applied)
    for name, _, _ in active_adapters:
        adapter_id = f"lora_{name.replace('.pt', '')}"
        edges.append(
            {
                "source": adapter_id,
                "target": "llm_local",
                "relation": "extends",
            }
        )

    # Link SNN Attention fatigue to router
    edges.append(
        {
            "source": "snn_attention_fatigue",
            "target": "llm_router",
            "relation": "biases",
        }
    )

    # Update LLM router node details
    for node in nodes:
        if node["id"] == "llm_router":
            node["details"].update(
                {
                    "stats": router_stats,
                    "complexity_threshold": complexity_threshold,
                    "local_model": local_model,
                    "external_model": external_model,
                }
            )

    return {
        "status": "success",
        "nodes": nodes,
        "edges": edges,
    }
