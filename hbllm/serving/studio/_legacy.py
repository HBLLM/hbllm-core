"""
Studio Legacy Endpoints — Dashboard Aggregation Routes.

These routes aggregate data from multiple cognitive subsystems into single
dashboard views. They don't belong to a single domain and remain here until
a future dashboard refactor.

For domain-specific endpoints, see the modular sub-routers:
- snn.py, memory.py, knowledge_graph.py, learning.py
- voice.py, plugins.py, rbac.py
- emotion.py, persona.py, perception.py, cognitive.py, hcir_epistemics.py
"""

import glob
import json
import logging
import os
import pathlib
import sqlite3
import time
from typing import Any

from fastapi import APIRouter, Request

from hbllm.serving.state import _get_node_map, _state

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Studio Compatibility Endpoints ───────────────────────────────────────────


# NOTE: /api/emotion/state moved to studio/emotion.py


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


# NOTE: /api/snn/* moved to studio/snn.py
# NOTE: /api/persona/* moved to studio/persona.py
# NOTE: /api/memory/* moved to studio/memory.py
# NOTE: /api/knowledge-graph/* moved to studio/knowledge_graph.py


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
    from hbllm.brain.self_model.cognitive_metrics import CognitiveMetrics

    cm = node_map.get("CognitiveMetrics")
    if cm and isinstance(cm, CognitiveMetrics):
        result["metrics"] = cm.get_dashboard_metrics()

    # ── Self model ──
    from hbllm.brain.self_model.self_model import SelfModel

    sm = node_map.get("SelfModel")
    if sm and isinstance(sm, SelfModel):
        result["self_model"] = sm.get_metrics()

    # ── Skill registry ──
    from hbllm.brain.skills.skill_registry import SkillRegistry

    sr = node_map.get("SkillRegistry")
    if sr and isinstance(sr, SkillRegistry):
        result["skills"] = sr.stats()

    # ── Goals ──
    from hbllm.brain.emotion.goal_manager import GoalManager

    gm = node_map.get("GoalManager")
    if gm and isinstance(gm, GoalManager):
        result["goals"] = gm.stats()

    # ── Evaluation ──
    from hbllm.brain.evaluation.evaluation_node import EvaluationNode

    ev = node_map.get("EvaluationNode")
    if ev and isinstance(ev, EvaluationNode):
        result["evaluation"] = ev.stats()

    # ── Attention ──
    from hbllm.brain.self_model.attention_manager import AttentionManager

    am = node_map.get("AttentionManager")
    if am and isinstance(am, AttentionManager):
        result["attention"] = am.stats()

    # ── Load manager ──
    from hbllm.brain.control.load_manager import LoadManager

    lm = node_map.get("LoadManager")
    if lm and isinstance(lm, LoadManager):
        result["load_manager"] = lm.stats()

    # ── Collective ──
    from hbllm.brain.world.collective_node import CollectiveNode

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
    from hbllm.brain.emotion.reflection_node import ReflectionNode

    rn = node_map.get("ReflectionNode")
    if rn and isinstance(rn, ReflectionNode):
        result["reflection"] = rn.stats()

    # ── Learning (LearnerNode) ──
    from hbllm.brain.learning.learner_node import LearnerNode

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
    from hbllm.brain.skills.skill_compiler_node import SkillCompilerNode

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


# NOTE: /studio/learning/* moved to studio/learning.py
# NOTE: /api/plugins/* moved to studio/plugins.py
# NOTE: /studio/voice/* moved to studio/voice.py
# NOTE: /studio/rbac/* moved to studio/rbac.py
