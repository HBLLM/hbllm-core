"""
Studio Knowledge Graph Endpoints.

Exposes knowledge graph entity/relation queries via the event bus.
The endpoints delegate to the ``MemorySystem`` composite's knowledge
graph handler via ``knowledge.query`` bus topics.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from hbllm.network.messages import Message, MessageType
from hbllm.serving.studio.helpers import get_brain, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/knowledge-graph/entities")
async def get_knowledge_graph_entities(request: Request, limit: int = 100):
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return []

    try:
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
        reply = await brain.bus.request("knowledge.query", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload.get("entities", [])
    except Exception as e:
        logger.debug("[Studio:KG] knowledge-graph.entities fallback: %s", e)
        return []


@router.get("/api/knowledge-graph/stats")
async def get_knowledge_graph_stats(request: Request):
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return {"entity_count": 0, "relation_count": 0}

    try:
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
        reply = await brain.bus.request("knowledge.query", msg, timeout=4.0)
        if reply.type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=reply.payload.get("error", "Unknown error"))
        return reply.payload
    except Exception as e:
        logger.debug("[Studio:KG] knowledge-graph.stats fallback: %s", e)
        return {"entity_count": 0, "relation_count": 0}


@router.post("/api/knowledge-graph/neighbors")
async def get_knowledge_graph_neighbors(request: Request):
    body = await request.json()
    tenant_id = get_tenant_id(request)
    brain = get_brain()
    if not brain or not brain.bus:
        return []

    try:
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
        reply = await brain.bus.request("knowledge.query", msg, timeout=4.0)
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
    except Exception as e:
        logger.debug("[Studio:KG] knowledge-graph.neighbors fallback: %s", e)
        return {"neighbors": []}
