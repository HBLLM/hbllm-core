"""
Studio Memory API Endpoints.

Exposes the memory subsystem via bus-based message passing: stats, browse,
search, forget, and export operations. These endpoints delegate to the
``MemorySystem`` composite (or legacy ``MemoryNode``) via the event bus.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from hbllm.network.messages import Message, MessageType
from hbllm.serving.studio.helpers import get_brain, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/memory/stats")
async def get_memory_stats(request: Request):
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return {"total_memories": 0, "status": "not_loaded"}

    try:
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=tenant_id,
            topic="memory.stats",
            payload={"tenant_id": tenant_id},
        )
        reply = await brain.bus.request("memory.stats", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload
    except Exception as e:
        logger.debug("[Studio:Memory] memory.stats bus request fallback: %s", e)
        return {"total_memories": 0, "status": "fallback"}


@router.post("/api/memory/browse")
async def browse_memories(request: Request):
    body = await request.json()
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return {"entries": [], "total": 0}

    try:
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=tenant_id,
            topic="memory.browse",
            payload={
                "offset": body.get("offset", 0),
                "limit": body.get("limit", 20),
                "memory_type": body.get("memory_type", "all"),
                "tenant_id": tenant_id,
            },
        )
        reply = await brain.bus.request("memory.browse", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload
    except Exception as e:
        logger.debug("[Studio:Memory] memory.browse bus request fallback: %s", e)
        return {"entries": [], "total": 0}


@router.post("/api/memory/search")
async def search_memories(request: Request):
    body = await request.json()
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return {"results": [], "query": body.get("query", "")}

    try:
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=tenant_id,
            topic="memory.search",
            payload={
                "query": body.get("query", ""),
                "top_k": body.get("top_k", 10),
                "memory_type": body.get("memory_type", "all"),
                "tenant_id": tenant_id,
            },
        )
        reply = await brain.bus.request("memory.search", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload
    except Exception as e:
        logger.debug("[Studio:Memory] memory.search bus request fallback: %s", e)
        return {"results": [], "query": body.get("query", "")}


@router.post("/api/memory/forget")
async def forget_memories(request: Request):
    body = await request.json()
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return {"forgotten_count": 0, "status": "success"}

    try:
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
        reply = await brain.bus.request("memory.forget", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload
    except Exception as e:
        logger.debug("[Studio:Memory] memory.forget bus request fallback: %s", e)
        return {"forgotten_count": 0, "status": "success"}


@router.get("/api/memory/export")
async def export_memories(request: Request):
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return {"entries": [], "total": 0}

    try:
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
        reply = await brain.bus.request("memory.browse", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload
    except Exception as e:
        logger.debug("[Studio:Memory] memory.export bus request fallback: %s", e)
        return {"entries": [], "total": 0}
