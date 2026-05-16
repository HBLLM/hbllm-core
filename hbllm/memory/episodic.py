"""
Episodic Memory storage using SQLite.

This module persists conversation turns across sessions, allowing the
MemoryNode to recall context even if the system restarts or the prompt
is routed to a different DomainModule.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hbllm.memory.pool import DatabasePool

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Lightweight SQLite storage for conversation history.
    """

    def __init__(self, db_path: str | Path = "working_memory.db"):
        self.db_path = Path(db_path)
        self.pool = DatabasePool(str(self.db_path))

    async def init_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                user_id TEXT,
                device_id TEXT,
                scope TEXT NOT NULL DEFAULT 'episodic',
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                domain TEXT,
                timestamp_iso TEXT NOT NULL,
                metadata TEXT,
                vector_clock TEXT,
                authority_score INTEGER DEFAULT 50,
                parent_memory_id TEXT
            )
        """)
            # Index for fast retrieval of latest turns per tenant+session
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tenant_session_time
                ON turns(tenant_id, session_id, timestamp_iso DESC)
            """)
            try:
                await conn.execute("SELECT parent_memory_id FROM turns LIMIT 1")
            except Exception:
                await conn.execute("ALTER TABLE turns ADD COLUMN parent_memory_id TEXT")
            await conn.commit()
        logger.debug("Initialized EpisodicMemory at %s", self.db_path)

    async def close(self) -> None:
        """Close the persistent database connection."""
        await self.pool.close_all()

    async def store_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
        user_id: str | None = None,
        device_id: str | None = None,
        scope: str = "episodic",
        vector_clock: dict[str, int] | None = None,
        authority_score: int = 50,
        parent_memory_id: str | None = None,
    ) -> str:
        """
        Store a single conversation turn.

        Args:
            session_id: Identifier for the conversation thread.
            role: "user" | "assistant" | "system".
            content: The text of the turn.
            domain: Which domain module generated this (if assistant).
            metadata: Additional JSON serializable data.
            tenant_id: Tenant identifier for multi-tenant isolation.

        Returns:
            The generated ID of the turn.
        """
        turn_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata) if metadata else "{}"

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO turns (id, tenant_id, user_id, device_id, scope, session_id, role, content, domain, timestamp_iso, metadata, vector_clock, authority_score, parent_memory_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    turn_id,
                    tenant_id,
                    user_id,
                    device_id,
                    scope,
                    session_id,
                    role,
                    content,
                    domain,
                    now_iso,
                    meta_str,
                    json.dumps(vector_clock) if vector_clock else None,
                    authority_score,
                    parent_memory_id,
                ),
            )
            await conn.commit()

        return turn_id

    async def retrieve_recent(
        self, session_id: str, limit: int = 10, tenant_id: str = "default"
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most recent turns for a given tenant's session.

        Returns a list of dicts ordered chronologically (oldest first).
        """
        async with self.pool.acquire() as conn:
            conn.row_factory = sqlite3.Row
            async with conn.execute(
                """
                SELECT * FROM turns
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY timestamp_iso DESC
                LIMIT ?
            """,
                (tenant_id, session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        results = []
        for row in reversed(list(rows)):
            results.append(
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "domain": row["domain"],
                    "timestamp": row["timestamp_iso"],
                    "metadata": json.loads(row["metadata"]),
                    "vector_clock": json.loads(row["vector_clock"])
                    if row["vector_clock"]
                    else None,
                    "authority_score": row["authority_score"],
                    "parent_memory_id": row["parent_memory_id"],
                }
            )

        return results

    async def clear_session(self, session_id: str, tenant_id: str = "default") -> int:
        """Delete all turns for a tenant's session. Returns deleted count."""
        async with self.pool.acquire() as conn:
            cursor = await conn.execute(
                "DELETE FROM turns WHERE tenant_id = ? AND session_id = ?", (tenant_id, session_id)
            )
            deleted = cursor.rowcount
            await conn.commit()
        return deleted

    async def search_by_content(
        self,
        query: str,
        tenant_id: str = "default",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search across all sessions for turns containing the query string.

        Args:
            query: Substring to search for (case-insensitive).
            tenant_id: Tenant scope.
            limit: Max results.

        Returns:
            List of matching turn dicts ordered by recency.
        """
        async with self.pool.acquire() as conn:
            async with conn.execute(
                """SELECT * FROM turns
                   WHERE tenant_id = ? AND content LIKE ?
                   ORDER BY timestamp_iso DESC
                   LIMIT ?""",
                (tenant_id, f"%{query}%", limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "domain": row["domain"],
                "timestamp": row["timestamp_iso"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def retrieve_by_domain(
        self,
        domain: str,
        tenant_id: str = "default",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve turns tagged with a specific domain across all sessions.

        Useful for cross-session context — e.g. recalling all "coding"
        conversations regardless of session.
        """
        async with self.pool.acquire() as conn:
            async with conn.execute(
                """SELECT * FROM turns
                   WHERE tenant_id = ? AND domain = ?
                   ORDER BY timestamp_iso DESC
                   LIMIT ?""",
                (tenant_id, domain, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "domain": row["domain"],
                "timestamp": row["timestamp_iso"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    async def cleanup_old_turns(self, days: int = 90, tenant_id: str | None = None) -> int:
        """
        Delete turns older than `days` to prevent unbounded growth.

        Args:
            days: Age threshold in days.
            tenant_id: If set, only clean this tenant. Otherwise all tenants.

        Returns:
            Number of deleted turns.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        async with self.pool.acquire() as conn:
            if tenant_id:
                cursor = await conn.execute(
                    "DELETE FROM turns WHERE tenant_id = ? AND timestamp_iso < ?",
                    (tenant_id, cutoff),
                )
            else:
                cursor = await conn.execute(
                    "DELETE FROM turns WHERE timestamp_iso < ?",
                    (cutoff,),
                )
            deleted = cursor.rowcount
            await conn.commit()

        logger.info("Cleaned up %d turns older than %d days", deleted, days)
        return deleted

    async def get_session_count(self, tenant_id: str = "default") -> int:
        """Count distinct sessions for a tenant."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM turns WHERE tenant_id = ?",
                (tenant_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_turn_count(self, tenant_id: str = "default") -> int:
        """Count total turns for a tenant."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM turns WHERE tenant_id = ?",
                (tenant_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0
