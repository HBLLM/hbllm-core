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

    # Try brain node map first
    node_map = _get_node_map(brain)
    emotion_node = node_map.get("EmotionNode")

    # Fall back to plugin manager's loaded nodes
    if not emotion_node:
        pm = _state.get("plugin_manager")
        if pm:
            for node in getattr(pm, "_loaded_nodes", []):
                if hasattr(node, "current_valence") and hasattr(node, "current_arousal"):
                    emotion_node = node
                    break

    if emotion_node:
        valence = getattr(emotion_node, "current_valence", 0.0)
        arousal = getattr(emotion_node, "current_arousal", 0.0)
        label = "neutral"
        if valence > 0.3:
            label = "happy"
        elif valence > 0.0:
            label = "content"
        elif valence < -0.3:
            label = "sad"
        elif valence < 0.0:
            label = "uneasy"
        return {
            "valence": valence,
            "arousal": arousal,
            "emotion_label": label,
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


@router.get("/api/snn/status")
async def get_snn_status():
    from hbllm.network.metrics import MetricsCollector

    collector = MetricsCollector.get_instance()

    # Extract priming category potentials
    categories = ["physics", "math", "coding", "finance", "personal", "general"]
    priming_potentials = {}

    # Try to extract exact real-time potentials from MemoryNode primer if available
    brain = _state.get("brain")
    memory_node = None
    if brain:
        node_map = _get_node_map(brain)
        memory_node = node_map.get("MemoryNode")

    for cat in categories:
        neuron_id = f"priming_{cat}"
        # Fall back to metrics collector value
        pot = collector._mem_gauges.get(f"snn_potential:{neuron_id}", 0.0)
        threshold = 1.0

        # If memory node is active, get precise current values
        if memory_node and hasattr(memory_node, "primer"):
            acc = memory_node.primer.categories.get(cat)
            if acc:
                pot = acc.get_potential()
                threshold = acc.neuron.config.threshold

        priming_potentials[cat] = {
            "potential": pot,
            "threshold": threshold,
            "history": collector.get_snn_history(neuron_id),
        }

    # Extract attention fatigue potential
    attn_pot = collector._mem_gauges.get("snn_potential:human_attention_fatigue", 0.0)
    attn_threshold = 0.8
    refractory_time = 0.0

    attention_fatigue = {
        "potential": attn_pot,
        "threshold": attn_threshold,
        "refractory_time_remaining": refractory_time,
        "history": collector.get_snn_history("human_attention_fatigue"),
    }

    # Extract any reflex rules from metrics collector
    reflexes = {}
    for key in list(collector._mem_gauges.keys()):
        if key.startswith("snn_potential:reflex_"):
            neuron_id = key.replace("snn_potential:", "")
            reflexes[neuron_id] = {
                "potential": collector._mem_gauges[key],
                "threshold": 1.0,
                "history": collector.get_snn_history(neuron_id),
            }

    return {
        "status": "success",
        "priming_categories": priming_potentials,
        "attention_fatigue": attention_fatigue,
        "reflex_rules": reflexes,
    }


@router.post("/api/snn/stimulate")
async def stimulate_snn_neuron(request: Request):
    body = await request.json()
    category = body.get("category")
    charge = float(body.get("charge", 0.5))

    if not category:
        raise HTTPException(status_code=400, detail="Category is required")

    brain = _state.get("brain")
    if brain:
        node_map = _get_node_map(brain)
        memory_node = node_map.get("MemoryNode")
        if memory_node and hasattr(memory_node, "primer"):
            memory_node.primer.stimulate_category(category, charge)
            # Force update metrics collector immediately
            pot = memory_node.primer.categories[category].get_potential()
            from hbllm.network.metrics import MetricsCollector

            MetricsCollector.get_instance().record_snn_potential(f"priming_{category}", pot)
            return {
                "status": "success",
                "message": f"Stimulated category {category} with {charge} charge.",
            }

    # Fallback if MemoryNode is not loaded
    from hbllm.network.metrics import MetricsCollector

    collector = MetricsCollector.get_instance()
    neuron_id = f"priming_{category}"
    cur_pot = collector._mem_gauges.get(f"snn_potential:{neuron_id}", 0.0)
    new_pot = min(1.0, cur_pot + charge)
    collector.record_snn_potential(neuron_id, new_pot)
    return {
        "status": "success",
        "message": f"Stimulated fallback metrics for category {category}.",
    }


@router.post("/api/snn/replay")
async def replay_cognitive_search(request: Request):
    body = await request.json()
    query = body.get("query")
    priming_state = body.get("priming_state", {})
    tenant_id = getattr(request.state, "tenant_id", "default")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    brain = _state.get("brain")
    memory_node = None
    if brain:
        node_map = _get_node_map(brain)
        memory_node = node_map.get("MemoryNode")

    if not memory_node:
        # Fallback Mock Comparison if brain is not running or MemoryNode is not loaded
        mock_unprimed = [
            {
                "id": "doc1",
                "content": "Introduction to string theory and quantum loop gravity.",
                "metadata": {"domain": "physics"},
                "score": 0.75,
                "score_breakdown": {
                    "similarity": 0.75,
                    "usefulness_boost": 0.0,
                    "reward_boost": 0.0,
                    "priming_boost": 0.0,
                },
            },
            {
                "id": "doc2",
                "content": "Writing clean asynchronous Python code and modules.",
                "metadata": {"domain": "coding"},
                "score": 0.62,
                "score_breakdown": {
                    "similarity": 0.62,
                    "usefulness_boost": 0.0,
                    "reward_boost": 0.0,
                    "priming_boost": 0.0,
                },
            },
        ]

        # Stimulate coding in primed results
        coding_prime = priming_state.get("coding", 0.0)
        coding_boost = 0.15 * coding_prime
        mock_primed = [
            {
                "id": "doc2",
                "content": "Writing clean asynchronous Python code and modules.",
                "metadata": {"domain": "coding"},
                "score": 0.62 + coding_boost,
                "score_breakdown": {
                    "similarity": 0.62,
                    "usefulness_boost": 0.0,
                    "reward_boost": 0.0,
                    "priming_boost": coding_boost,
                },
            },
            {
                "id": "doc1",
                "content": "Introduction to string theory and quantum loop gravity.",
                "metadata": {"domain": "physics"},
                "score": 0.75,
                "score_breakdown": {
                    "similarity": 0.75,
                    "usefulness_boost": 0.0,
                    "reward_boost": 0.0,
                    "priming_boost": 0.0,
                },
            },
        ]
        if mock_primed[0]["score"] < mock_primed[1]["score"]:
            mock_primed = [mock_primed[1], mock_primed[0]]

        from hbllm.memory.semantic import SemanticMemory

        sm = SemanticMemory()
        differentials = sm.get_ranking_differential(mock_primed)
        return {
            "status": "success",
            "unprimed": mock_unprimed,
            "primed": mock_primed,
            "differentials": differentials,
        }

    sem_db = memory_node.semantic_db

    # 1. Unprimed Search (baseline)
    unprimed_env = sem_db.search(
        query=query, top_k=5, tenant_id=tenant_id, priming_boosts=None, explain=True
    )

    # 2. Primed Search (replayed state)
    primed_env = sem_db.search(
        query=query, top_k=5, tenant_id=tenant_id, priming_boosts=priming_state, explain=True
    )

    unprimed_results = unprimed_env["results"] if isinstance(unprimed_env, dict) else unprimed_env
    primed_results = primed_env["results"] if isinstance(primed_env, dict) else primed_env

    # Compute ranking differentials
    differentials = sem_db.get_ranking_differential(primed_results)

    return {
        "status": "success",
        "unprimed": unprimed_results,
        "primed": primed_results,
        "differentials": differentials,
    }


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
        status = "healthy"
        if hasattr(node, "health_check"):
            try:
                h_report = await node.health_check()
                status = (
                    h_report.status.value
                    if hasattr(h_report.status, "value")
                    else str(h_report.status)
                )
            except Exception as e:
                logger.error("Failed to run health check for node %s: %s", name, e)
                status = "unhealthy"
        else:
            status = "healthy" if getattr(node, "_running", True) else "unhealthy"

        node_health.append(
            {
                "id": info.node_id,
                "name": name.replace("Node", "").replace("Manager", " Mgr"),
                "status": status,
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

    # ── Learning (LearnerNode) ──
    from hbllm.brain.learner_node import LearnerNode

    ln = node_map.get("LearnerNode")
    if ln and isinstance(ln, LearnerNode):
        learning_stats = ln.micro_learning_stats()
        # Add DPO queue depth from disk
        dpo_queue_depth = 0
        try:
            dpo_path = pathlib.Path(ln.queue_path)
            if dpo_path.exists():
                with dpo_path.open() as f:
                    dpo_queue_depth = len(json.load(f))
        except Exception:
            pass
        learning_stats["dpo_queue_depth"] = dpo_queue_depth
        result["learning"] = learning_stats

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
                "total_entries": len(sem.documents) if hasattr(sem, "documents") else 0,
                "priority_entries": sum(
                    1
                    for e in (sem.documents.values() if hasattr(sem, "documents") else [])
                    if e.get("metadata", {}).get("is_priority", False)
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
                    "category": e.attributes.get("category", "other"),
                    "name": e.attributes.get("name", e.label),
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
                for r in list(kg._relations.values())[:200]
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


# ─── Learning Pipeline Endpoints ──────────────────────────────────────────


@router.get("/studio/learning")
async def studio_learning() -> Any:
    """Detailed self-learning pipeline status for real-time observability.

    Combines LearnerNode stats, EvaluationNode aggregate, ReflectionNode
    insights, and DPO queue preview into a single diagnostic view.
    """
    brain = _state.get("brain")
    node_map = _get_node_map(brain)
    result: dict[str, Any] = {"status": "active"}

    # ── LearnerNode stats ──
    from hbllm.brain.learner_node import LearnerNode

    ln = node_map.get("LearnerNode")
    if ln and isinstance(ln, LearnerNode):
        result["learner"] = ln.micro_learning_stats()
        # DPO queue on disk
        dpo_queue_depth = 0
        dpo_preview: list[str] = []
        try:
            dpo_path = pathlib.Path(ln.queue_path)
            if dpo_path.exists():
                with dpo_path.open() as f:
                    queue = json.load(f)
                    dpo_queue_depth = len(queue)
                    # Preview: first 5 prompt prefixes
                    for entry in queue[:5]:
                        if isinstance(entry, (list, tuple)) and len(entry) > 0:
                            dpo_preview.append(str(entry[0])[:80])
        except Exception:
            pass
        result["learner"]["dpo_queue_depth"] = dpo_queue_depth
        result["learner"]["dpo_queue_preview"] = dpo_preview
        # Micro-learn queue preview
        micro_queue = ln.get_micro_learn_queue()
        result["learner"]["micro_queue_preview"] = [
            {"query": item.get("query", "")[:80], "score": item.get("score", 0.0)}
            for item in micro_queue[:5]
        ]
    else:
        result["learner"] = {"status": "not_found"}

    # ── EvaluationNode aggregate ──
    from hbllm.brain.evaluation_node import EvaluationNode

    ev = node_map.get("EvaluationNode")
    if ev and isinstance(ev, EvaluationNode):
        result["evaluation"] = ev.stats()
    else:
        result["evaluation"] = {"status": "not_found"}

    # ── ReflectionNode insights ──
    from hbllm.brain.reflection_node import ReflectionNode

    rn = node_map.get("ReflectionNode")
    if rn and isinstance(rn, ReflectionNode):
        result["reflection"] = rn.stats()
    else:
        result["reflection"] = {"status": "not_found"}

    # ── Synaptic Plasticity Weights ──
    synaptic_weights = {}
    memory_node = None
    if brain:
        memory_node = node_map.get("MemoryNode")
    if memory_node and hasattr(memory_node, "semantic_db"):
        synaptic_weights = memory_node.semantic_db.synaptic_weights
    else:
        categories = ["physics", "math", "coding", "finance", "personal", "general"]
        for cat in categories:
            synaptic_weights[cat] = {other: 1.0 if cat == other else 0.0 for other in categories}

    if "learner" not in result or not isinstance(result["learner"], dict):
        result["learner"] = {}
    result["learner"]["synaptic_weights"] = synaptic_weights

    return result


@router.post("/studio/learning/trigger")
async def studio_learning_trigger(request: Request) -> Any:
    """Inject a synthetic evaluation event to test the learning pipeline.

    This publishes a system.evaluation event on the bus, which the LearnerNode
    listens to. Use low scores (<0.3) to trigger micro-learn queueing, then
    follow up with a high score (>0.85) on the same query to trigger actual
    micro-learning correction.

    Body:
        {
            "query": "What is the capital of France?",
            "response": "I'm not sure, maybe London?",
            "score": 0.15
        }
    """
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    body = await request.json()
    query = body.get("query", "")
    response = body.get("response", "")
    score = float(body.get("score", 0.5))

    if not query or not response:
        raise HTTPException(status_code=400, detail="query and response are required")

    import uuid

    eval_msg = Message(
        type=MessageType.EVENT,
        source_node_id="api_server",
        topic="system.evaluation",
        payload={
            "correlation_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "task_success": score,
            "plan_validity": score,
            "tool_accuracy": 0.8,
            "memory_usage": 0.5,
            "confidence_error": max(0.0, 1.0 - score),
            "overall_score": score,
            "query": query,
            "response": response,
            "flags": ["synthetic_trigger"],
            "dimensions": {
                "task_success": score,
                "plan_validity": score,
            },
        },
    )
    await bus.publish("system.evaluation", eval_msg)

    # Determine what should happen based on score
    from hbllm.brain.learner_node import LearnerNode

    node_map = _get_node_map(brain)
    ln = node_map.get("LearnerNode")
    threshold_info = {}
    if ln and isinstance(ln, LearnerNode):
        threshold_info = {
            "micro_learn_threshold": ln.micro_learn_threshold,
            "distillation_threshold": ln.distillation_threshold,
        }

    expected_action = "no_action"
    if score < (threshold_info.get("micro_learn_threshold", 0.3)):
        expected_action = "queued_for_micro_learning"
    elif score > (threshold_info.get("distillation_threshold", 0.85)):
        expected_action = "banked_for_distillation"

    return {
        "status": "published",
        "score": score,
        "expected_action": expected_action,
        "thresholds": threshold_info,
        "tip": "To trigger micro-learning: send a low-score event, then a high-score event with the same query.",
    }


@router.post("/studio/learning/reset_weights")
async def studio_learning_reset_weights() -> Any:
    """Reset Hebbian synaptic weight matrix to defaults."""
    brain = _state.get("brain")
    memory_node = None
    if brain:
        node_map = _get_node_map(brain)
        memory_node = node_map.get("MemoryNode")

    categories = ["physics", "math", "coding", "finance", "personal", "general"]
    if memory_node and hasattr(memory_node, "semantic_db"):
        db = memory_node.semantic_db
        with db._lock:
            db.synaptic_weights = {}
            for cat in categories:
                db.synaptic_weights[cat] = {
                    other: 1.0 if cat == other else 0.0 for other in categories
                }
            db._retrieval_priming_history = {}
            db._priming_history_keys = []

            try:
                db.save_to_disk(memory_node._persistence_dir / "semantic")
            except Exception as e:
                logger.error("Failed to save reset synaptic weights: %s", e)

    return {"status": "success", "message": "Synaptic connection weights reset to default."}


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
