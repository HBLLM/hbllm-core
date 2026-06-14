"""
Memory, Knowledge & Sync Routes.

Endpoints:
  GET  /v1/memory/{session_id}       — Retrieve conversation history
  POST /v1/sync/episodic             — Sync episodic memories from edge
  POST /v1/sync/semantic             — Sync semantic knowledge from edge
  POST /v1/feedback                  — Submit RLHF feedback
  GET  /v1/knowledge/{entity}        — Knowledge graph neighbors
  GET  /v1/knowledge/path            — Shortest path between entities
  GET  /v1/knowledge/subgraph/{entity} — Subgraph extraction
  GET  /v1/knowledge/stats           — Knowledge graph statistics
  GET  /v1/rules                     — List extracted rules
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from hbllm.network.messages import Message, MessageType
from hbllm.serving.state import _get_node_map, _state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["memory"])


# ─── Schemas ─────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """User feedback on a response (thumbs up/down for RLHF loop)."""

    tenant_id: str = Field(..., description="Tenant who sent the feedback")
    message_id: str = Field(..., description="Correlation ID of the response being rated")
    rating: int = Field(..., ge=-1, le=1, description="-1 (bad), 0 (neutral), 1 (good)")
    prompt: str | None = Field(default=None, description="Original prompt text")
    response: str | None = Field(default=None, description="Response text that was rated")
    comment: str | None = Field(default=None, description="Optional user comment")


class SyncEpisodicRequest(BaseModel):
    """Batch of episodic memories to sync upstream."""

    memories: list[dict[str, Any]] = Field(
        ..., description="List of episodic memory dictionaries to append"
    )


class SyncSemanticRequest(BaseModel):
    """Batch of semantic knowledge items to sync upstream."""

    knowledge_items: list[dict[str, Any]] = Field(
        ..., description="List of knowledge items to append"
    )


# ─── Dedup Cache ─────────────────────────────────────────────────────────────

_sync_dedup_cache: set[str] = set()


# ─── Helper ──────────────────────────────────────────────────────────────────


async def _bus_request(
    bus: Any, topic: str, payload: dict, response_topic: str, timeout: float = 5.0
) -> dict:
    """Send a bus query and wait for response on a specific topic."""
    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe(response_topic, handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        topic=topic,
        payload=payload,
        correlation_id=correlation_id,
    )
    await bus.publish(topic, query)

    try:
        result = await asyncio.wait_for(response_future, timeout=timeout)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {}


def _require_bus() -> Any:
    """Get the bus or raise 503."""
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")
    return bus


# ─── Memory Endpoints ────────────────────────────────────────────────────────


@router.get("/memory/{session_id}")
async def get_memory(request: Request, session_id: str, limit: int = 20) -> Any:
    """Retrieve recent conversation history for a tenant's session."""
    tenant_id = getattr(request.state, "tenant_id", "default")
    bus = _require_bus()

    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def memory_handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("memory.retrieve_recent.response", memory_handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        session_id=session_id,
        topic="memory.retrieve_recent",
        payload={"session_id": session_id, "tenant_id": tenant_id, "limit": limit},
        correlation_id=correlation_id,
    )
    await bus.publish("memory.retrieve_recent", query)

    try:
        result = await asyncio.wait_for(response_future, timeout=5.0)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {"session_id": session_id, "turns": []}


@router.post("/sync/episodic")
async def sync_episodic(api_req: Request, request: SyncEpisodicRequest) -> Any:
    """Sync a batch of episodic memories from an edge device (append strategy)."""
    tenant_id = getattr(api_req.state, "tenant_id", "default")
    user_id = getattr(api_req.state, "user_id", "default")
    device_id = getattr(api_req.state, "device_id", "default")
    bus = _require_bus()

    synced_count = 0
    for mem in request.memories:
        payload = dict(mem)

        content_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if content_hash in _sync_dedup_cache:
            continue

        _sync_dedup_cache.add(content_hash)
        if len(_sync_dedup_cache) > 10000:
            _sync_dedup_cache.pop()

        payload["tenant_id"] = tenant_id
        payload["user_id"] = user_id
        payload["device_id"] = device_id

        msg = Message(
            type=MessageType.EVENT,
            source_node_id=f"edge_sync_{device_id}",
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            session_id=mem.get("session_id", "sync_session"),
            topic="memory.store",
            payload=payload,
        )
        await bus.publish("memory.store", msg)
        synced_count += 1

    return {
        "status": "success",
        "synced": synced_count,
        "skipped": len(request.memories) - synced_count,
    }


@router.post("/sync/semantic")
async def sync_semantic(api_req: Request, request: SyncSemanticRequest) -> Any:
    """Sync a batch of semantic knowledge items from an edge device (append strategy)."""
    tenant_id = getattr(api_req.state, "tenant_id", "default")
    user_id = getattr(api_req.state, "user_id", "default")
    device_id = getattr(api_req.state, "device_id", "default")
    bus = _require_bus()

    synced_count = 0
    for item in request.knowledge_items:
        payload = dict(item)

        content_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if content_hash in _sync_dedup_cache:
            continue

        _sync_dedup_cache.add(content_hash)
        if len(_sync_dedup_cache) > 10000:
            _sync_dedup_cache.pop()

        payload["tenant_id"] = tenant_id
        payload["user_id"] = user_id
        payload["device_id"] = device_id

        msg = Message(
            type=MessageType.EVENT,
            source_node_id=f"edge_sync_{device_id}",
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            session_id="sync_session",
            topic="knowledge.store",
            payload=payload,
        )
        await bus.publish("knowledge.store", msg)
        synced_count += 1

    return {
        "status": "success",
        "synced": synced_count,
        "skipped": len(request.knowledge_items) - synced_count,
    }


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> Any:
    """Submit user feedback on a response for RLHF / DPO continuous learning."""
    bus = _require_bus()

    feedback_msg = Message(
        type=MessageType.FEEDBACK,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        topic="system.feedback",
        payload={
            "message_id": request.message_id,
            "rating": request.rating,
            "prompt": request.prompt,
            "response": request.response,
            "comment": request.comment,
        },
    )
    await bus.publish("system.feedback", feedback_msg)

    return {
        "status": "accepted",
        "message_id": request.message_id,
        "rating": request.rating,
    }


# ─── Knowledge Graph Endpoints ───────────────────────────────────────────────


@router.get("/knowledge/{entity}")
async def knowledge_neighbors(
    entity: str, direction: str = "both", relation_type: str | None = None
) -> Any:
    """Query KnowledgeGraph neighbors for an entity."""
    bus = _require_bus()
    result = await _bus_request(
        bus,
        "knowledge.query",
        {
            "action": "neighbors",
            "entity": entity,
            "direction": direction,
            "relation_type": relation_type,
        },
        "knowledge.response",
    )
    return result or {"neighbors": [], "entity": entity}


@router.get("/knowledge/path")
async def knowledge_path(from_entity: str, to_entity: str, max_depth: int = 5) -> Any:
    """Find shortest path between two entities in the KnowledgeGraph."""
    bus = _require_bus()
    result = await _bus_request(
        bus,
        "knowledge.query",
        {"action": "path", "from": from_entity, "to": to_entity, "max_depth": max_depth},
        "knowledge.response",
    )
    return result or {"path": None, "from": from_entity, "to": to_entity}


@router.get("/knowledge/subgraph/{entity}")
async def knowledge_subgraph(entity: str, depth: int = 2) -> Any:
    """Extract a subgraph around an entity."""
    bus = _require_bus()
    result = await _bus_request(
        bus,
        "knowledge.query",
        {"action": "subgraph", "entity": entity, "depth": depth},
        "knowledge.response",
    )
    return result or {"subgraph": {"entities": [], "relations": []}, "entity": entity}


@router.get("/knowledge/stats")
async def knowledge_stats() -> Any:
    """Get KnowledgeGraph statistics."""
    bus = _require_bus()
    result = await _bus_request(
        bus,
        "knowledge.query",
        {"action": "stats"},
        "knowledge.response",
    )
    return result or {"entity_count": 0, "relation_count": 0}


# ─── Rules ───────────────────────────────────────────────────────────────────


@router.get("/rules")
async def list_rules() -> Any:
    """List extracted if→then rules from the RuleExtractorNode."""
    brain = _state.get("brain")
    node_map = _get_node_map(brain)
    for node in node_map.values():
        if hasattr(node, "rules"):
            return {
                "rules": [r.to_dict() for r in node.rules],
                "total": len(node.rules),
            }
    return {"rules": [], "total": 0}
