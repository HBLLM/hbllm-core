import glob
import json
import logging
import os
import pathlib
import shutil
import sqlite3
import time
import urllib.request
import zipfile
from io import BytesIO
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hbllm.network.messages import Message, MessageType
from hbllm.serving.state import _get_node_map, _state

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Studio Compatibility Endpoints ───────────────────────────────────────────


@router.get("/api/emotion/state")
async def get_emotion_state(agent_name: str = "assistant", tenant_id: str = "default"):
    brain = _state.get("brain")
    if not brain:
        return {"valence": 0.0, "arousal": 0.0, "emotion_label": "neutral", "status": "not_loaded"}
    node_map = _get_node_map(brain)
    emotion_node = node_map.get("EmotionNode")
    if emotion_node:
        valence = getattr(emotion_node, "current_valence", 0.0)
        arousal = getattr(emotion_node, "current_arousal", 0.0)
        return {
            "valence": valence,
            "arousal": arousal,
            "emotion_label": "neutral" if valence == 0.0 else ("happy" if valence > 0.0 else "sad"),
            "status": "active",
        }
    return {"valence": 0.0, "arousal": 0.0, "emotion_label": "neutral", "status": "not_loaded"}


@router.get("/api/swarm/status")
async def get_swarm_status():
    brain = _state.get("brain")
    if not brain:
        return {"agents": [], "active_delegations": [], "status": "standby"}
    node_map = _get_node_map(brain)
    agents = []
    for name, node in sorted(node_map.items()):
        if hasattr(node, "_running") or hasattr(node, "get_info"):
            agents.append(
                {
                    "name": name.replace("Node", "").replace("Manager", " Mgr"),
                    "tenant_id": getattr(node, "tenant_id", "default"),
                    "status": "healthy" if getattr(node, "_running", True) else "unhealthy",
                }
            )
    cn = node_map.get("CollectiveNode")
    active_dels = (
        list(cn.active_delegations.values()) if cn and hasattr(cn, "active_delegations") else []
    )

    return {
        "agents": agents,
        "active_delegations": active_dels,
        "status": "active" if len(agents) > 1 else "standby",
    }


@router.get("/api/temporal/timeline")
async def get_temporal_timeline():
    db_path = os.path.join(os.environ.get("HBLLM_DATA_DIR", "data"), "scheduler.db")
    if not os.path.exists(db_path):
        return {"timeline": [], "count": 0}
    try:
        now = time.time()
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT task_id, trigger_time, payload, route_topic, status FROM scheduled_tasks ORDER BY trigger_time ASC LIMIT 50"
            )
            rows = []
            for r in cursor.fetchall():
                payload_str = r["payload"]
                task_prompt = r["route_topic"]
                try:
                    p = json.loads(payload_str)
                    if isinstance(p, dict):
                        task_prompt = p.get("prompt") or p.get("text") or r["route_topic"]
                except Exception:
                    pass

                rows.append(
                    {
                        "id": r["task_id"],
                        "execute_at": r["trigger_time"],
                        "task_prompt": task_prompt,
                        "status": r["status"],
                        "is_overdue": r["trigger_time"] < now and r["status"] == "pending",
                        "seconds_until": max(0.0, r["trigger_time"] - now),
                    }
                )
            return {"timeline": rows, "count": len(rows)}
    except Exception as e:
        logger.error("Failed to query scheduler.db: %s", e)
        return {"timeline": [], "count": 0}


@router.get("/api/synapsis/config")
async def get_synapsis_config():
    return {
        "synapsis": {
            "enabled": True,
            "network_name": "hbllm-default",
            "node_role": "hub",
            "bus_backend": _state.get("bus_type", "inprocess"),
            "redis_url": os.getenv("HBLLM_REDIS_URL", "redis://localhost:6379"),
            "hub_url": "",
            "device_tier": "server",
            "heartbeat_interval": 10.0,
            "node_timeout": 30.0,
        }
    }


@router.put("/api/synapsis/config")
async def update_synapsis_config(request: Request):
    body = await request.json()
    return {"status": "success", "synapsis": body}


@router.get("/api/synapsis/status")
async def get_synapsis_status():
    gateway = _state.get("synapse_gateway")
    connected = len(gateway.active_connections) if gateway else 0
    return {
        "enabled": True,
        "role": "hub",
        "bus_backend": _state.get("bus_type", "inprocess"),
        "uplink_active": False,
        "connected_edges": connected,
    }


@router.post("/api/synapsis/test")
async def test_synapsis_connection():
    return {"status": "success", "message": "Synapsis Gateway is active and healthy."}


@router.post("/api/synapsis/connect")
async def connect_synapsis():
    return {"status": "success", "message": "Connected to Synapsis network."}


@router.post("/api/synapsis/disconnect")
async def disconnect_synapsis():
    return {"status": "success", "message": "Disconnected from Synapsis network."}


@router.get("/api/persona/profile")
async def get_persona_profile(request: Request):
    tenant_id = getattr(request.state, "tenant_id", "default")
    db_path = os.path.join(os.environ.get("HBLLM_DATA_DIR", "data"), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT traits_json FROM identities WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
            if row:
                traits = json.loads(row["traits_json"])
                return {
                    "verbosity": traits.get("verbosity", "balanced"),
                    "tone": traits.get("tone", "neutral"),
                    "emoji_preference": traits.get("emoji_preference", "minimal"),
                    "interaction_count": traits.get("interaction_count", 5),
                    "topics_of_interest": traits.get(
                        "topics_of_interest", ["AI", "cognitive architecture"]
                    ),
                }
    except Exception:
        pass
    return {
        "verbosity": "balanced",
        "tone": "neutral",
        "emoji_preference": "minimal",
        "interaction_count": 5,
        "topics_of_interest": ["AI", "cognitive architecture"],
    }


@router.put("/api/persona/override")
async def override_persona(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "default")
    db_path = os.path.join(os.environ.get("HBLLM_DATA_DIR", "data"), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT traits_json FROM identities WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
            traits = {}
            if row and row["traits_json"]:
                traits = json.loads(row["traits_json"])
            traits.update(body)
            conn.execute(
                "INSERT INTO identities (tenant_id, traits_json, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now')) ON CONFLICT(tenant_id) DO UPDATE SET traits_json = excluded.traits_json, updated_at = excluded.updated_at",
                (tenant_id, json.dumps(traits)),
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to override persona: %s", e)
    return {"status": "ok"}


@router.post("/api/persona/reset")
async def reset_persona(request: Request):
    tenant_id = getattr(request.state, "tenant_id", "default")
    db_path = os.path.join(os.environ.get("HBLLM_DATA_DIR", "data"), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE identities SET traits_json = '{}' WHERE tenant_id = ?", (tenant_id,)
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to reset persona: %s", e)
    return {"status": "ok"}


@router.delete("/api/persona/override")
async def clear_persona_overrides(request: Request):
    tenant_id = getattr(request.state, "tenant_id", "default")
    db_path = os.path.join(os.environ.get("HBLLM_DATA_DIR", "data"), "identity.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE identities SET traits_json = '{}' WHERE tenant_id = ?", (tenant_id,)
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to clear persona overrides: %s", e)
    return {"status": "ok"}


@router.get("/api/memory/stats")
async def get_memory_stats(request: Request):
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="memory.stats",
        payload={"tenant_id": tenant_id},
    )
    reply = await brain.bus.request("memory.stats", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload


@router.post("/api/memory/browse")
async def browse_memories(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="memory.browse",
        payload={
            "offset": body.get("offset", 0),
            "limit": body.get("limit", 20),
            "session_id": body.get("session_id"),
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("memory.browse", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload


@router.post("/api/memory/search")
async def search_memories(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="memory.search",
        payload={
            "query_text": body.get("query", ""),
            "top_k": body.get("top_k", 5),
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("memory.search", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload


@router.post("/api/memory/forget")
async def forget_memories(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.COMMAND,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="memory.forget",
        payload={
            "query": body.get("query"),
            "session_id": body.get("session_id"),
            "before": body.get("before"),
            "after": body.get("after"),
            "entry_ids": body.get("entry_ids", []),
            "forget_semantic": body.get("forget_semantic", True),
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("memory.forget", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload


@router.get("/api/memory/export")
async def export_memories(request: Request):
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="memory.browse",
        payload={
            "offset": 0,
            "limit": 1000,
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("memory.browse", msg, timeout=15.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload


@router.get("/api/knowledge-graph/entities")
async def get_knowledge_graph_entities(request: Request, limit: int = 100):
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="knowledge.query",
        payload={
            "action": "all_entities",
            "limit": limit,
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("knowledge.query", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload.get("entities", [])


@router.get("/api/knowledge-graph/stats")
async def get_knowledge_graph_stats(request: Request):
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="knowledge.query",
        payload={
            "action": "stats",
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("knowledge.query", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
    return reply.payload


@router.post("/api/knowledge-graph/neighbors")
async def get_knowledge_graph_neighbors(request: Request):
    body = await request.json()
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    if not brain or not brain.bus:
        raise HTTPException(status_code=503, detail="Brain or Bus not initialized")

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        topic="knowledge.query",
        payload={
            "action": "neighbors",
            "entity": body.get("entity", ""),
            "direction": body.get("direction", "both"),
            "tenant_id": tenant_id,
        },
    )
    reply = await brain.bus.request("knowledge.query", msg, timeout=10.0)
    if reply.type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))

    mapped_neighbors = []
    for n in reply.payload.get("neighbors", []):
        mapped_neighbors.append(
            {
                "label": n.get("entity"),
                "relation_type": n.get("relation"),
                "weight": n.get("weight"),
                "direction": n.get("direction"),
            }
        )
    return {"neighbors": mapped_neighbors}


@router.get("/studio/stats")
async def studio_stats() -> Any:
    """Aggregated cognitive subsystem stats for HBLLM Studio dashboard."""
    brain = _state.get("brain")
    nodes = getattr(brain, "nodes", [])
    result: dict[str, Any] = {
        "mode": _state.get("mode", "unknown"),
        "node_count": len(nodes),
    }

    node_map = _get_node_map(brain)

    # ── Node health ──
    node_health = []
    for name, node in sorted(node_map.items()):
        if not hasattr(node, "get_info"):
            continue
        info = node.get_info()
        node_health.append(
            {
                "id": info.node_id,
                "name": name.replace("Node", "").replace("Manager", " Mgr"),
                "status": "healthy" if getattr(node, "_running", True) else "unhealthy",
                "type": info.node_type.value
                if hasattr(info.node_type, "value")
                else str(info.node_type),
            }
        )
    result["nodes"] = node_health

    # ── Cognitive metrics ──
    from hbllm.brain.cognitive_metrics import CognitiveMetrics

    cm = node_map.get("CognitiveMetrics")
    if cm and isinstance(cm, CognitiveMetrics):
        result["metrics"] = cm.get_dashboard_metrics()

    # ── Self model ──
    from hbllm.brain.self_model import SelfModel

    sm = node_map.get("SelfModel")
    if sm and isinstance(sm, SelfModel):
        result["self_model"] = sm.get_metrics()

    # ── Skill registry ──
    from hbllm.brain.skill_registry import SkillRegistry

    sr = node_map.get("SkillRegistry")
    if sr and isinstance(sr, SkillRegistry):
        result["skills"] = sr.stats()

    # ── Goals ──
    from hbllm.brain.goal_manager import GoalManager

    gm = node_map.get("GoalManager")
    if gm and isinstance(gm, GoalManager):
        result["goals"] = gm.stats()

    # ── Evaluation ──
    from hbllm.brain.evaluation_node import EvaluationNode

    ev = node_map.get("EvaluationNode")
    if ev and isinstance(ev, EvaluationNode):
        result["evaluation"] = ev.stats()

    # ── Attention ──
    from hbllm.brain.attention_manager import AttentionManager

    am = node_map.get("AttentionManager")
    if am and isinstance(am, AttentionManager):
        result["attention"] = am.stats()

    # ── Load manager ──
    from hbllm.brain.load_manager import LoadManager

    lm = node_map.get("LoadManager")
    if lm and isinstance(lm, LoadManager):
        result["load_manager"] = lm.stats()

    # ── Collective ──
    from hbllm.brain.collective_node import CollectiveNode

    cn = node_map.get("CollectiveNode")
    if cn and isinstance(cn, CollectiveNode):
        collective_stats = cn.stats
        result["collective"] = {
            "instance_id": cn.instance_id,
            "stats": dict(collective_stats),
            "peers": [
                {
                    "instance_id": p.instance_id,
                    "domains": p.domains,
                    "load": p.load,
                    "performance": p.performance,
                }
                for p in cn.peer_profiles.values()
            ],
            "recent_activity": cn.recent_activity if hasattr(cn, "recent_activity") else [],
        }

    # ── Reflection ──
    from hbllm.brain.reflection_node import ReflectionNode

    rn = node_map.get("ReflectionNode")
    if rn and isinstance(rn, ReflectionNode):
        result["reflection"] = rn.stats()

    # ── Skill compiler ──
    from hbllm.brain.skill_compiler_node import SkillCompilerNode

    sc = node_map.get("SkillCompilerNode")
    if sc and isinstance(sc, SkillCompilerNode):
        result["skill_compiler"] = sc.stats()

    # ── Bus metrics ──
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if bus and hasattr(bus, "metrics"):
        result["bus_metrics"] = bus.metrics.snapshot()

    return result


@router.get("/studio/memory")
async def studio_memory() -> Any:
    """Memory subsystem stats for Studio — episodic, semantic, procedural, value."""
    brain = _state.get("brain")
    node_map = _get_node_map(brain)

    result: dict[str, Any] = {}

    from hbllm.memory.memory_node import MemoryNode

    mem = node_map.get("MemoryNode")
    if mem and isinstance(mem, MemoryNode):
        # Episodic
        try:
            ep_stats = mem.db.stats() if hasattr(mem.db, "stats") else {}
            result["episodic"] = (
                ep_stats
                if ep_stats
                else {
                    "db_path": str(mem.db.db_path),
                    "status": "active",
                }
            )
        except Exception:
            result["episodic"] = {"status": "active"}

        # Semantic
        try:
            sem = mem.semantic_db
            result["semantic"] = {
                "total_entries": len(sem._entries) if hasattr(sem, "_entries") else 0,
                "priority_entries": sum(
                    1
                    for e in (sem._entries if hasattr(sem, "_entries") else [])
                    if getattr(e, "is_priority", False)
                ),
                "status": "active",
            }
        except Exception:
            result["semantic"] = {"status": "active", "total_entries": 0}

        # Procedural
        try:
            proc = mem.procedural_db
            result["procedural"] = {
                "db_path": str(proc.db_path) if hasattr(proc, "db_path") else "N/A",
                "status": "active",
            }
        except Exception:
            result["procedural"] = {"status": "active"}

        # Value
        try:
            val = mem.value_db
            result["value"] = {
                "db_path": str(val.db_path) if hasattr(val, "db_path") else "N/A",
                "status": "active",
            }
        except Exception:
            result["value"] = {"status": "active"}

        # Knowledge Graph summary
        try:
            kg = mem.knowledge_graph
            result["knowledge_graph"] = {
                "entity_count": kg.entity_count,
                "relation_count": kg.relation_count,
            }
        except Exception:
            result["knowledge_graph"] = {"entity_count": 0, "relation_count": 0}

    return result


@router.get("/studio/knowledge")
async def studio_knowledge() -> Any:
    """Knowledge Graph contents for Studio — entities, relations, subgraphs."""
    brain = _state.get("brain")
    node_map = _get_node_map(brain)

    result: dict[str, Any] = {"entities": [], "relations": [], "stats": {}}

    from hbllm.memory.memory_node import MemoryNode

    mem = node_map.get("MemoryNode")
    if mem and isinstance(mem, MemoryNode):
        kg = mem.knowledge_graph
        result["stats"] = {
            "entity_count": kg.entity_count,
            "relation_count": kg.relation_count,
        }
        # Entities
        if hasattr(kg, "_entities"):
            result["entities"] = [
                {
                    "id": e.id,
                    "label": e.label,
                    "type": e.entity_type,
                }
                for e in list(kg._entities.values())[:100]
            ]
        # Relations
        if hasattr(kg, "_relations"):
            result["relations"] = [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type,
                    "weight": r.weight,
                }
                for r in list(kg._relations)[:200]
            ]

    return result


@router.get("/studio/lora")
async def studio_lora() -> Any:
    """LoRA adapter status for Studio — pending, active, rejected."""
    data_dir = os.environ.get("HBLLM_DATA_DIR", "data")
    lora_dir = os.path.join(data_dir, "lora")

    result: dict[str, Any] = {
        "lora_dir": lora_dir,
        "active_adapters": [],
        "pending_adapters": [],
        "rejected_count": 0,
        "self_improve_status": "idle",
    }

    # Scan for LoRA files
    if os.path.exists(lora_dir):
        for pt_file in glob.glob(os.path.join(lora_dir, "**/*.pt"), recursive=True):
            name = os.path.basename(pt_file)
            size_mb = os.path.getsize(pt_file) / (1024 * 1024)
            mtime = os.path.getmtime(pt_file)
            entry = {
                "name": name,
                "path": pt_file,
                "size_mb": round(size_mb, 2),
                "modified": mtime,
            }
            if name.endswith(".pending.pt"):
                result["pending_adapters"].append(entry)
            else:
                result["active_adapters"].append(entry)

    # Check self-improve worker status
    brain = _state.get("brain")
    node_map = _get_node_map(brain)
    sleep_node = node_map.get("SleepCycleNode") or node_map.get("SleepNode")
    if sleep_node:
        result["self_improve_status"] = (
            "active" if getattr(sleep_node, "_running", False) else "idle"
        )
        if hasattr(sleep_node, "_dpo_cycles"):
            result["dpo_cycles"] = sleep_node._dpo_cycles
        if hasattr(sleep_node, "_consolidation_cycles"):
            result["consolidation_cycles"] = sleep_node._consolidation_cycles

    return result


# ─── Plugin Management Endpoints ───────────────────────────────────────────


@router.get("/api/plugins")
async def list_plugins():
    pm = _state.get("plugin_manager")
    if not pm:
        return {"plugins": []}

    # Call discover() to pick up any newly installed/deleted folders
    pm.discover()

    raw_plugins = pm.list_plugins()
    mapped = []
    for p in raw_plugins:
        mapped.append(
            {
                "name": p["name"],
                "enabled": p["loaded"],
                "loaded": p["loaded"],
                "description": p["description"] or "No description available",
                "version": p["version"] or "0.1.0",
                "path": p["path"],
                "error": p["error"],
            }
        )
    return {"plugins": mapped}


@router.post("/api/plugins/{plugin_name}/toggle")
async def toggle_plugin(plugin_name: str):
    pm = _state.get("plugin_manager")
    if not pm:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")

    enabled = await pm.toggle_plugin(plugin_name)
    return {"plugin": plugin_name, "enabled": enabled}


class PluginMarketplace:
    def __init__(self, plugins_dir: pathlib.Path):
        self.plugins_dir = plugins_dir
        self.registry_path = (
            pathlib.Path(__file__).resolve().parent.parent.parent.parent
            / "sentra-plugins"
            / "registry.json"
        )

    async def list_available(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            logger.warning("Marketplace registry file not found: %s", self.registry_path)
            return []
        try:
            with open(self.registry_path) as f:
                data = json.load(f)
            plugins = data.get("plugins", [])
            for p in plugins:
                p["installed"] = (self.plugins_dir / p["name"]).exists()
            return plugins
        except Exception as e:
            logger.error("Failed to read registry: %s", e)
            return []

    async def install(self, plugin_name: str) -> dict[str, Any]:
        plugins = await self.list_available()
        plugin_info = next((p for p in plugins if p["name"] == plugin_name), None)
        if not plugin_info:
            return {"status": "error", "error": f"Plugin '{plugin_name}' not found"}

        download_url = plugin_info.get("download_url")

        try:
            local_source = self.registry_path.parent / "plugins" / plugin_name
            target_dir = self.plugins_dir / plugin_name
            if target_dir.exists():
                shutil.rmtree(target_dir)

            if local_source.exists() and local_source.is_dir():
                shutil.copytree(local_source, target_dir)
                logger.info("Installed plugin %s locally from sentra-plugins", plugin_name)
            elif download_url:
                req = urllib.request.Request(
                    download_url, headers={"User-Agent": "HBLLM-Marketplace/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()

                with zipfile.ZipFile(BytesIO(data)) as zf:
                    zf.extractall(target_dir)
                logger.info("Installed plugin %s from URL", plugin_name)
            else:
                return {"status": "error", "error": "No download source available"}

            return {
                "status": "installed",
                "name": plugin_name,
                "version": plugin_info.get("version"),
            }
        except Exception as e:
            logger.error("Install failed for %s: %s", plugin_name, e)
            return {"status": "error", "error": str(e)}

    def uninstall(self, plugin_name: str) -> dict[str, Any]:
        target_dir = self.plugins_dir / plugin_name
        if not target_dir.exists():
            return {"status": "error", "error": "Not installed"}
        try:
            shutil.rmtree(target_dir)
            return {"status": "uninstalled", "name": plugin_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}


@router.get("/api/plugins/marketplace")
async def marketplace_list():
    pm = _state.get("plugin_manager")
    if not pm or not pm._plugin_dirs:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")
    plugins_dir = pathlib.Path(pm._plugin_dirs[0])
    mp = PluginMarketplace(plugins_dir)
    available = await mp.list_available()
    return {"plugins": available}


@router.post("/api/plugins/install")
async def marketplace_install(request: Request):
    pm = _state.get("plugin_manager")
    if not pm or not pm._plugin_dirs:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")
    body = await request.json()
    plugin_name = body.get("name")
    if not plugin_name:
        return {"status": "error", "error": "Missing 'name'"}

    plugins_dir = pathlib.Path(pm._plugin_dirs[0])
    mp = PluginMarketplace(plugins_dir)
    result = await mp.install(plugin_name)

    if result.get("status") == "installed":
        brain = _state.get("brain")
        if brain and hasattr(brain, "plugin_manager") and brain.plugin_manager:
            try:
                await brain.plugin_manager.load_bundle(plugins_dir / plugin_name)
            except Exception as e:
                logger.error("Failed to hot-load installed plugin %s: %s", plugin_name, e)

    pm.discover()
    return result


@router.delete("/api/plugins/uninstall")
async def marketplace_uninstall(request: Request):
    pm = _state.get("plugin_manager")
    if not pm or not pm._plugin_dirs:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")
    body = await request.json()
    plugin_name = body.get("name")
    if not plugin_name:
        return {"status": "error", "error": "Missing 'name'"}

    info = pm._plugins.get(plugin_name)
    if info and info.loaded:
        await pm.toggle_plugin(plugin_name)
    else:
        brain = _state.get("brain")
        if brain and hasattr(brain, "plugin_manager") and brain.plugin_manager:
            if plugin_name in brain.plugin_manager.bundles:
                try:
                    await brain.plugin_manager.unload_bundle(plugin_name)
                except Exception as e:
                    logger.error("Failed to unload bundle %s during uninstall: %s", plugin_name, e)

    plugins_dir = pathlib.Path(pm._plugin_dirs[0])
    mp = PluginMarketplace(plugins_dir)
    result = mp.uninstall(plugin_name)

    pm.discover()
    return result
