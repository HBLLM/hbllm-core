"""Recall Handler for MemoryNode.

Handles memory retrieval, semantic RAG search, skill lookup, preference/reward querying,
and priming integrations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hbllm.memory.interface import MemoryType, SearchResult
from hbllm.network.messages import MemoryRetrievePayload, MemorySearchPayload, Message
from hbllm.security.tenant_guard import require_tenant

logger = logging.getLogger(__name__)


class RecallHandler:
    """Handles memory retrieval, query, and search operations across different scopes."""

    def __init__(self, node: Any) -> None:
        self.node = node

    async def retrieve(self, memory_type: MemoryType, query: Any, **kwargs: Any) -> list[Any]:
        """Core retrieval implementation for UnifiedMemoryInterface."""
        if memory_type == MemoryType.EPISODIC:
            return await self.node.db.retrieve_recent(
                session_id=kwargs.get("session_id", ""),
                limit=kwargs.get("limit", 10),
                tenant_id=kwargs.get("tenant_id", "default"),
            )
        return []

    async def search(
        self, query: str, memory_types: list[MemoryType] | None = None, **kwargs: Any
    ) -> list[SearchResult]:
        """Core search implementation for UnifiedMemoryInterface."""
        types_to_search = memory_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC]
        results = []

        tenant_id = kwargs.get("tenant_id", "default")
        limit = kwargs.get("limit", 5)

        if MemoryType.SEMANTIC in types_to_search:
            sem_res = await asyncio.to_thread(
                self.node.semantic_db.search,
                query,
                top_k=limit,
                tenant_id=tenant_id,
                primer=self.node.primer,
            )
            for r in sem_res:
                if not isinstance(r, dict):
                    continue
                results.append(
                    SearchResult(
                        memory_type=MemoryType.SEMANTIC,
                        id=r.get("id", ""),
                        content=r.get("content"),
                        score=r.get("score", 1.0),
                        metadata=r,
                    )
                )

        # Episodic search (mocked as retrieve for unified demo)
        if MemoryType.EPISODIC in types_to_search:
            ep_res = await self.node.db.retrieve_recent(
                session_id=kwargs.get("session_id", ""), limit=limit, tenant_id=tenant_id
            )
            for i, r in enumerate(ep_res):
                results.append(
                    SearchResult(
                        memory_type=MemoryType.EPISODIC,
                        id=str(i),
                        content=r.get("content"),
                        score=1.0,
                        metadata=r,
                    )
                )

        return sorted(results, key=lambda x: x.score, reverse=True)[:limit]

    @require_tenant
    async def handle_retrieve(self, message: Message) -> Message | None:
        """Handles `memory.retrieve_recent` topics."""
        # CapBAC: Check permissions (retrieve defaults to episodic if not specified)
        if self.node.registry:
            scope = message.payload.get("scope", "episodic")
            if not await self.node.registry.has_permission(message.source_node_id, scope):
                return message.create_error(f"Access denied to scope '{scope}'", code="FORBIDDEN")

        try:
            payload = MemoryRetrievePayload(**message.payload)
            session_id = payload.session_id
            limit = payload.limit

            turns = await self.node.db.retrieve_recent(
                session_id,
                limit=limit,
                tenant_id=payload.tenant_id or (message.tenant_id or "default"),
            )

            return message.create_response(
                {
                    "session_id": session_id,
                    "turns": turns,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory retrieval failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_search(self, message: Message) -> Message | None:
        """Handles `memory.search` topics for long-term semantic RAG."""
        try:
            payload = MemorySearchPayload(**message.payload)
            query = payload.query_text
            limit = payload.top_k

            if not query:
                return message.create_error("Missing 'query_text'")

            # Fetch current SNN working memory priming boosts
            priming_boosts = self.node.primer.get_boosts()

            results = await asyncio.to_thread(
                self.node.semantic_db.search,
                query,
                limit,
                priming_boosts=priming_boosts,
                tenant_id=message.tenant_id,
                user_id=message.user_id,
                device_id=message.device_id,
                primer=self.node.primer,
            )

            return message.create_response(
                {
                    "results": results,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Semantic search failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_skill_find(self, message: Message) -> Message | None:
        """Handles `memory.skill.find` topics."""
        try:
            payload = message.payload
            query = payload.get("query", "")
            top_k = int(payload.get("top_k", 3))
            tenant_id = message.tenant_id or "default"

            if not query:
                return message.create_error("Missing 'query'")

            skills = await self.node.procedural_db.find_skill(tenant_id, query, top_k)
            return message.create_response({"skills": skills})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Skill find failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_reward_query(self, message: Message) -> Message | None:
        """Handles `memory.reward.query` topics."""
        try:
            payload = message.payload
            topic = payload.get("topic")
            top_k = int(payload.get("top_k", 5))
            tenant_id = message.tenant_id or "default"

            if topic:
                preferences = await self.node.value_db.get_preference(
                    tenant_id, topic, message.user_id, message.device_id
                )
                return message.create_response({"topic": topic, "preferences": preferences})
            else:
                top_prefs = await self.node.value_db.get_top_preferences(
                    tenant_id, top_k, message.user_id, message.device_id
                )
                return message.create_response({"top_preferences": top_prefs})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Reward query failed: %s", e)
            return message.create_error(str(e))

    async def _on_router_query(self, message: Message) -> None:
        """Stimulate the primer using text/content of the user's query."""
        text = message.payload.get("text", "")
        if text:
            await asyncio.to_thread(self.node.primer.stimulate_by_text, text)

    async def _on_workspace_thought(self, message: Message) -> None:
        """Stimulate the primer based on reasoning thoughts."""
        content = message.payload.get("content", "")
        if content:
            await asyncio.to_thread(self.node.primer.stimulate_by_text, content)

        domain = message.payload.get("domain")
        if domain:
            await asyncio.to_thread(self.node.primer.stimulate_category, domain, 0.5)
