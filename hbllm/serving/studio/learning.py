"""
Studio Learning Pipeline Endpoints.

Exposes the self-learning pipeline: DPO queue status, micro-learning
trigger/preview, synaptic weight reset, evaluation and reflection stats.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hbllm.network.messages import Message, MessageType
from hbllm.serving.studio.helpers import get_brain, get_node_map, require_bus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/studio/learning")
async def studio_learning() -> Any:
    """Detailed self-learning pipeline status for real-time observability.

    Combines LearnerNode stats, EvaluationNode aggregate, ReflectionNode
    insights, and DPO queue preview into a single diagnostic view.
    """
    brain = get_brain()
    node_map = get_node_map()
    result: dict[str, Any] = {"status": "active"}

    # ── LearnerNode stats ──
    from hbllm.brain.learning.learner_node import LearnerNode

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
    from hbllm.brain.evaluation.evaluation_node import EvaluationNode

    ev = node_map.get("EvaluationNode")
    if ev and isinstance(ev, EvaluationNode):
        result["evaluation"] = ev.stats()
    else:
        result["evaluation"] = {"status": "not_found"}

    # ── ReflectionNode insights ──
    from hbllm.brain.emotion.reflection_node import ReflectionNode

    rn = node_map.get("ReflectionNode")
    if rn and isinstance(rn, ReflectionNode):
        result["reflection"] = rn.stats()
    else:
        result["reflection"] = {"status": "not_found"}

    # ── Synaptic Plasticity Weights ──
    synaptic_weights = {}
    cluster_stats = {}
    cluster_labels = {}
    memory_node = None
    if brain:
        memory_node = node_map.get("MemoryNode")
    if memory_node and hasattr(memory_node, "semantic_db"):
        synaptic_weights = memory_node.semantic_db.synaptic_weights
        cluster_stats = memory_node.semantic_db.cluster_manager.cluster_stats
        # Generate friendly labels for all categories/clusters
        for cat in list(synaptic_weights.keys()):
            if cat.startswith("cluster_"):
                try:
                    cluster_id = int(cat.split("_")[1])
                    cluster_labels[cat] = memory_node.semantic_db.cluster_manager.get_cluster_label(
                        cluster_id, memory_node.semantic_db.documents
                    )
                except (ValueError, IndexError):
                    cluster_labels[cat] = cat
            else:
                cluster_labels[cat] = cat
    else:
        categories = ["physics", "math", "coding", "finance", "personal", "general"]
        for cat in categories:
            synaptic_weights[cat] = {other: 1.0 if cat == other else 0.0 for other in categories}
            cluster_labels[cat] = cat

    if "learner" not in result or not isinstance(result["learner"], dict):
        result["learner"] = {}
    result["learner"]["synaptic_weights"] = synaptic_weights
    result["learner"]["cluster_stats"] = cluster_stats
    result["learner"]["cluster_labels"] = cluster_labels

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
    bus = require_bus()

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
    from hbllm.brain.learning.learner_node import LearnerNode

    node_map = get_node_map()
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
    brain = get_brain()
    memory_node = None
    if brain:
        node_map = get_node_map()
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
