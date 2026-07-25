"""Storage Handler for MemoryNode.

Handles writing memory (EPISODIC, SEMANTIC, PROCEDURAL, VALUE), sensitive memory guarding,
PII redaction, browsing episodic history, selective amnesia (forgetting), skill storing,
and reward signal recording.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from hbllm.memory.interface import MemoryType
from hbllm.network.messages import MemoryStorePayload, Message
from hbllm.security.tenant_guard import require_tenant

logger = logging.getLogger(__name__)


class StorageHandler:
    """Handles memory store, browse, forget, skill store, and reward record logic."""

    def __init__(self, node: Any) -> None:
        self.node = node

    async def store(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        """Core storage implementation for UnifiedMemoryInterface."""
        # PII Protection: redact sensitive data before persistence
        if isinstance(data, str) and self.node._pii_redactor:
            try:
                data, _pii_matches = self.node._pii_redactor.redact(
                    data, tenant_id=kwargs.get("tenant_id", "default")
                )
            except Exception as e:
                logger.debug("PII redaction error (proceeding): %s", e)

        if memory_type == MemoryType.EPISODIC:
            return str(
                await self.node.db.store_turn(
                    session_id=kwargs.get("session_id", ""),
                    role=kwargs.get("role", "user"),
                    content=data,
                    tenant_id=kwargs.get("tenant_id", "default"),
                )
            )
        elif memory_type == MemoryType.SEMANTIC:
            await asyncio.to_thread(
                self.node.semantic_db.store,
                data,
                kwargs.get("metadata", {}),
                is_priority=kwargs.get("is_priority", False),
                tenant_id=kwargs.get("tenant_id", "default"),
            )
            return "stored"
        elif memory_type == MemoryType.PROCEDURAL:
            await self.node.procedural_db.store_skill(
                tenant_id=kwargs.get("tenant_id", "default"),
                skill_name=kwargs.get("name", ""),
                trigger_pattern=data,
                steps=kwargs.get("steps") or kwargs.get("code") or [],
            )
            return "stored"
        elif memory_type == MemoryType.VALUE:
            await self.node.value_db.record_reward(
                tenant_id=kwargs.get("tenant_id", "default"),
                topic=kwargs.get("topic", "general"),
                action=kwargs.get("action_name", ""),
                reward=data,
                user_id=kwargs.get("session_id", ""),
            )
            return "stored"
        elif memory_type == MemoryType.KNOWLEDGE_GRAPH:
            return "stored"
        return ""

    @require_tenant
    async def handle_store(self, message: Message) -> Message | None:
        """Handles `memory.store` topics."""
        # CapBAC: Check permissions
        if self.node.registry:
            payload_data = message.payload
            scope = payload_data.get("scope", "episodic")
            if not await self.node.registry.has_permission(message.source_node_id, scope):
                logger.warning(
                    "[MemoryNode] Permission denied: Node '%s' cannot store in scope '%s'",
                    message.source_node_id,
                    scope,
                )
                return message.create_error(f"Access denied to scope '{scope}'", code="FORBIDDEN")

        try:
            payload = MemoryStorePayload(**message.payload)
            session_id = payload.session_id
            role = payload.role
            content = payload.content

            turn_id = await self.node.db.store_turn(
                session_id=session_id,
                role=role,
                content=content,
                domain=payload.domain,
                metadata=payload.metadata,
                tenant_id=payload.tenant_id or (message.tenant_id or "default"),
                user_id=payload.user_id or message.user_id,
                device_id=payload.device_id or message.device_id,
                scope=payload.scope,
                vector_clock=message.vector_clock,
                authority_score=message.payload.get("authority_score", 50),
                parent_memory_id=payload.parent_memory_id,
            )

            # Sensitive Memory Guard: Skip semantic indexing for sensitive/working data
            if payload.scope not in ["sensitive", "working"]:
                # Offload semantic storage to a background thread with error handling
                task = asyncio.create_task(
                    asyncio.to_thread(
                        self.node.semantic_db.store,
                        content,
                        {
                            "session_id": session_id,
                            "role": role,
                            "tenant_id": payload.tenant_id or message.tenant_id,
                            "user_id": payload.user_id or message.user_id,
                            "device_id": payload.device_id or message.device_id,
                            "scope": payload.scope,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            **(payload.metadata or {}),
                        },
                        is_priority=False,
                        tenant_id=payload.tenant_id or message.tenant_id,
                        user_id=payload.user_id or message.user_id,
                        device_id=payload.device_id or message.device_id,
                        vector_clock=message.vector_clock,
                        authority_score=message.payload.get("authority_score", 50),
                    )
                )
                task.add_done_callback(self.node._handle_background_task_result)
                # Track for graceful shutdown
                self.node._pending_tasks.add(task)
                task.add_done_callback(self.node._pending_tasks.discard)
            else:
                logger.debug(
                    "[MemoryNode] Skipping semantic storage for %s scoped entry", payload.scope
                )

            # Fire and forget mostly, but we reply with success
            return message.create_response({"status": "stored", "turn_id": turn_id})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory store failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_browse(self, message: Message) -> Message | None:
        """Handles `memory.browse` — paginated episodic memory retrieval."""
        try:
            payload = message.payload
            offset = int(payload.get("offset", 0))
            limit = min(int(payload.get("limit", 20)), 100)
            session_id = payload.get("session_id")
            tenant_id = payload.get("tenant_id") or message.tenant_id or "default"

            async with self.node.db.pool.acquire() as conn:
                conn.row_factory = sqlite3.Row

                if session_id:
                    async with conn.execute(
                        "SELECT * FROM turns WHERE tenant_id = ? AND session_id = ? "
                        "ORDER BY timestamp_iso DESC LIMIT ? OFFSET ?",
                        (tenant_id, session_id, limit, offset),
                    ) as cursor:
                        rows = await cursor.fetchall()
                    async with conn.execute(
                        "SELECT COUNT(*) as cnt FROM turns WHERE tenant_id = ? AND session_id = ?",
                        (tenant_id, session_id),
                    ) as cursor:
                        total_row = await cursor.fetchone()
                else:
                    async with conn.execute(
                        "SELECT * FROM turns WHERE tenant_id = ? "
                        "ORDER BY timestamp_iso DESC LIMIT ? OFFSET ?",
                        (tenant_id, limit, offset),
                    ) as cursor:
                        rows = await cursor.fetchall()
                    async with conn.execute(
                        "SELECT COUNT(*) as cnt FROM turns WHERE tenant_id = ?",
                        (tenant_id,),
                    ) as cursor:
                        total_row = await cursor.fetchone()

                conn.row_factory = None

            entries = []
            for row in rows:
                entries.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "role": row["role"],
                        "content": row["content"],
                        "domain": row["domain"],
                        "timestamp": row["timestamp_iso"],
                        "metadata": _json.loads(row["metadata"]) if row["metadata"] else {},
                    }
                )

            total = total_row["cnt"] if total_row else 0
            return message.create_response(
                {
                    "entries": entries,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory browse failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_forget(self, message: Message) -> Message | None:
        """Handles `memory.forget` — selective amnesia."""
        try:
            payload = message.payload
            tenant_id = payload.get("tenant_id") or message.tenant_id or "default"
            query = payload.get("query")
            session_id = payload.get("session_id")
            before = payload.get("before")
            after = payload.get("after")
            entry_ids = payload.get("entry_ids", [])
            forget_semantic = payload.get("forget_semantic", True)

            deleted_episodic = 0
            deleted_semantic = 0

            # Delete specific entries by ID
            if entry_ids:
                async with self.node.db.pool.acquire() as conn:
                    placeholders = ",".join("?" * len(entry_ids))
                    cursor = await conn.execute(
                        f"DELETE FROM turns WHERE tenant_id = ? AND id IN ({placeholders})",
                        [tenant_id] + list(entry_ids),
                    )
                    deleted_episodic += cursor.rowcount
                    await conn.commit()

            # Delete by session
            if session_id:
                deleted_episodic += await self.node.db.clear_session(session_id, tenant_id)

            # Delete by content query
            if query:
                async with self.node.db.pool.acquire() as conn:
                    # First find matching IDs for reporting
                    conn.row_factory = sqlite3.Row
                    async with conn.execute(
                        "SELECT id, content FROM turns WHERE tenant_id = ? AND content LIKE ?",
                        (tenant_id, f"%{query}%"),
                    ) as cursor:
                        matches = await cursor.fetchall()
                    conn.row_factory = None

                    if matches:
                        match_ids = [m["id"] for m in matches]
                        placeholders = ",".join("?" * len(match_ids))
                        cursor = await conn.execute(
                            f"DELETE FROM turns WHERE id IN ({placeholders})",
                            match_ids,
                        )
                        deleted_episodic += cursor.rowcount
                        await conn.commit()

                # Also delete from semantic memory
                if forget_semantic:
                    results = await asyncio.to_thread(
                        self.node.semantic_db.search, query, top_k=50, tenant_id=tenant_id
                    )
                    for r in results:
                        doc_id = r.get("id")
                        if doc_id and self.node.semantic_db.delete(doc_id):
                            deleted_semantic += 1

            # Delete by time range
            if before or after:
                async with self.node.db.pool.acquire() as conn:
                    conditions = ["tenant_id = ?"]
                    params: list[Any] = [tenant_id]
                    if before:
                        conditions.append("timestamp_iso < ?")
                        params.append(before)
                    if after:
                        conditions.append("timestamp_iso > ?")
                        params.append(after)

                    where = " AND ".join(conditions)
                    cursor = await conn.execute(f"DELETE FROM turns WHERE {where}", params)
                    deleted_episodic += cursor.rowcount
                    await conn.commit()

            logger.info(
                "[MemoryNode] Forget completed: %d episodic, %d semantic entries removed",
                deleted_episodic,
                deleted_semantic,
            )
            # Reset the synaptic working memory primer state
            self.node.primer.reset()
            return message.create_response(
                {
                    "status": "forgotten",
                    "deleted_episodic": deleted_episodic,
                    "deleted_semantic": deleted_semantic,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory forget failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_skill_store(self, message: Message) -> Message | None:
        """Handles `memory.skill.store` topics."""
        try:
            payload = message.payload
            skill_name = payload.get("skill_name")
            trigger_pattern = payload.get("trigger_pattern", "")
            steps = payload.get("steps", [])
            tenant_id = message.tenant_id or "default"

            if not skill_name or not steps:
                return message.create_error("Missing 'skill_name' or 'steps'")

            skill_id = await self.node.procedural_db.store_skill(
                tenant_id=tenant_id,
                skill_name=skill_name,
                trigger_pattern=trigger_pattern,
                steps=steps,
                source_node=message.source_node_id,
            )
            return message.create_response({"status": "stored", "skill_id": skill_id})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Skill store failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_reward_record(self, message: Message) -> Message | None:
        """Handles `memory.reward.record` topics."""
        try:
            payload = message.payload
            topic = payload.get("topic", "general")
            action = payload.get("action", "")
            reward = float(payload.get("reward", 0.0))
            tenant_id = message.tenant_id or "default"

            if not action:
                return message.create_error("Missing 'action'")

            reward_id = await self.node.value_db.record_reward(
                tenant_id=tenant_id,
                topic=topic,
                action=action,
                reward=reward,
                context=payload.get("context"),
                user_id=message.user_id,
                device_id=message.device_id,
            )
            return message.create_response({"status": "recorded", "reward_id": reward_id})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Reward record failed: %s", e)
            return message.create_error(str(e))
