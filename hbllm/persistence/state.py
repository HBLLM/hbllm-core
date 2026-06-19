"""
Brain State — SQLite-backed state storage for HBLLM brain sessions.

Supports:
- Key-value store for configuration snapshots
- Conversation history
- Memory checkpoints
- Tool execution logs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Any

from hbllm.persistence.db_pool import DBPool
from hbllm.security.tenant_guard import require_tenant

logger = logging.getLogger(__name__)


class BrainState:
    """
    Persistent state manager for HBLLM brain sessions.

    Usage:
        state = BrainState(path="./data/brain_state.db")
        state.save("config", {"name": "researcher"})
        config = state.load("config")

        # Conversation history
        state.append_message("user", "What is AI?")
        state.append_message("assistant", "AI is...")
        history = state.get_messages(limit=10)

        # Checkpointing
        state.checkpoint({"step": 42, "memory_count": 100})
        latest = state.latest_checkpoint()
    """

    def __init__(self, path: str = "./brain_state.db"):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # Enable 5-second busy timeout to wait for transient SQLite locks
        self._conn = sqlite3.connect(self.path, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        # Enable WAL (Write-Ahead Logging) and safe synchronous modes
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                parent_id TEXT REFERENCES tenants(id),
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kv_store (
                tenant_id TEXT DEFAULT '',
                user_id TEXT DEFAULT '',
                device_id TEXT DEFAULT '',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (tenant_id, user_id, device_id, key)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT DEFAULT '',
                user_id TEXT DEFAULT '',
                device_id TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT DEFAULT '',
                user_id TEXT DEFAULT '',
                device_id TEXT DEFAULT '',
                data TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT DEFAULT '',
                user_id TEXT DEFAULT '',
                device_id TEXT DEFAULT '',
                tool_name TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                duration_ms REAL DEFAULT 0,
                created_at REAL NOT NULL
            );

            -- Performance indexes for tenant-scoped queries
            CREATE INDEX IF NOT EXISTS idx_messages_tenant_time
                ON messages(tenant_id, user_id, device_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_tenant
                ON checkpoints(tenant_id, user_id, device_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_tool_logs_tenant_name
                ON tool_logs(tenant_id, user_id, device_id, tool_name);
        """)
        self._conn.commit()

    # ── Key-Value Store ───────────────────────────────────────────────────

    @require_tenant
    def save(
        self,
        key: str,
        value: Any,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> None:
        """Save a value to the key-value store."""
        self._conn.execute(
            "INSERT OR REPLACE INTO kv_store (tenant_id, user_id, device_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (tenant_id, user_id, device_id, key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    @require_tenant
    def load(
        self,
        key: str,
        default: Any = None,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> Any:
        """Load a value from the key-value store."""
        row = self._conn.execute(
            "SELECT value FROM kv_store WHERE tenant_id = ? AND user_id = ? AND device_id = ? AND key = ?",
            (tenant_id, user_id, device_id, key),
        ).fetchone()
        if row:
            return json.loads(row[0])
        return default

    @require_tenant
    def delete(
        self, key: str, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> None:
        """Delete a key from the store."""
        self._conn.execute(
            "DELETE FROM kv_store WHERE tenant_id = ? AND user_id = ? AND device_id = ? AND key = ?",
            (tenant_id, user_id, device_id, key),
        )
        self._conn.commit()

    # ── Conversation History ──────────────────────────────────────────────

    @require_tenant
    def append_message(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> int:
        """Append a message to conversation history."""
        cursor = self._conn.execute(
            "INSERT INTO messages (tenant_id, user_id, device_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tenant_id, user_id, device_id, role, content, json.dumps(metadata or {}), time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    @require_tenant
    def get_messages(
        self,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> list[dict]:
        """Get recent messages from conversation history."""
        rows = self._conn.execute(
            "SELECT id, role, content, metadata, created_at FROM messages "
            "WHERE tenant_id = ? AND user_id = ? AND device_id = ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (tenant_id, user_id, device_id, limit, offset),
        ).fetchall()
        return [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]),
                "created_at": row[4],
            }
            for row in reversed(rows)  # Return in chronological order
        ]

    @require_tenant
    def clear_messages(
        self, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> None:
        """Clear all conversation history."""
        self._conn.execute(
            "DELETE FROM messages WHERE tenant_id = ? AND user_id = ? AND device_id = ?",
            (tenant_id, user_id, device_id),
        )
        self._conn.commit()

    # ── Checkpoints ───────────────────────────────────────────────────────

    @require_tenant
    def checkpoint(
        self, data: dict, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> int:
        """Save a checkpoint."""
        cursor = self._conn.execute(
            "INSERT INTO checkpoints (tenant_id, user_id, device_id, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (tenant_id, user_id, device_id, json.dumps(data), time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    @require_tenant
    def latest_checkpoint(
        self, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> dict | None:
        """Get the most recent checkpoint."""
        row = self._conn.execute(
            "SELECT data, created_at FROM checkpoints WHERE tenant_id = ? AND user_id = ? AND device_id = ? ORDER BY id DESC LIMIT 1",
            (tenant_id, user_id, device_id),
        ).fetchone()
        if row:
            return {"data": json.loads(row[0]), "created_at": row[1]}
        return None

    @require_tenant
    def list_checkpoints(
        self, limit: int = 10, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> list[dict]:
        """List recent checkpoints."""
        rows = self._conn.execute(
            "SELECT id, data, created_at FROM checkpoints WHERE tenant_id = ? AND user_id = ? AND device_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, user_id, device_id, limit),
        ).fetchall()
        return [{"id": row[0], "data": json.loads(row[1]), "created_at": row[2]} for row in rows]

    # ── Tool Logs ─────────────────────────────────────────────────────────

    @require_tenant
    def log_tool_call(
        self,
        tool_name: str,
        input_data: str,
        output: str,
        duration_ms: float = 0,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> None:
        """Log a tool invocation."""
        self._conn.execute(
            "INSERT INTO tool_logs (tenant_id, user_id, device_id, tool_name, input, output, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tenant_id,
                user_id,
                device_id,
                tool_name,
                input_data,
                output[:5000]
                if len(output) <= 5000
                else output[:4950] + f"\n... [truncated from {len(output)} chars]",
                duration_ms,
                time.time(),
            ),
        )
        self._conn.commit()

    @require_tenant
    def get_tool_logs(
        self,
        tool_name: str | None = None,
        limit: int = 20,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> list[dict]:
        """Get tool execution logs."""
        if tool_name:
            rows = self._conn.execute(
                "SELECT tool_name, input, output, duration_ms, created_at FROM tool_logs "
                "WHERE tenant_id = ? AND user_id = ? AND device_id = ? AND tool_name = ? ORDER BY id DESC LIMIT ?",
                (tenant_id, user_id, device_id, tool_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT tool_name, input, output, duration_ms, created_at FROM tool_logs "
                "WHERE tenant_id = ? AND user_id = ? AND device_id = ? ORDER BY id DESC LIMIT ?",
                (tenant_id, user_id, device_id, limit),
            ).fetchall()
        return [
            {
                "tool": row[0],
                "input": row[1],
                "output": row[2],
                "duration_ms": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @property
    def stats(self) -> dict:
        """Get state storage statistics."""
        kv_count = self._conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
        msg_count = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        cp_count = self._conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        tool_count = self._conn.execute("SELECT COUNT(*) FROM tool_logs").fetchone()[0]
        return {
            "kv_entries": kv_count,
            "messages": msg_count,
            "checkpoints": cp_count,
            "tool_logs": tool_count,
        }


class AsyncBrainState:
    """
    Async Postgres-backed persistent state manager for HBLLM brain sessions.
    Falls back to synchronous sqlite-backed BrainState if Postgres is unavailable.
    """

    def __init__(self, path: str = "./brain_state.db"):
        self.path = path
        self._fallback = BrainState(path)
        self._pg_tables_created = False

    async def _ensure_pg_tables(self) -> Any | None:
        """Lazily create the PostgreSQL tables on first write."""
        pool = await DBPool.get_pool()
        if pool is None:
            return None

        if not self._pg_tables_created:
            async with pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS tenants (
                        id TEXT PRIMARY KEY,
                        parent_id TEXT REFERENCES tenants(id),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS kv_store (
                        tenant_id TEXT DEFAULT '',
                        user_id TEXT DEFAULT '',
                        device_id TEXT DEFAULT '',
                        key TEXT NOT NULL,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (tenant_id, user_id, device_id, key)
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        tenant_id TEXT DEFAULT '',
                        user_id TEXT DEFAULT '',
                        device_id TEXT DEFAULT '',
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id SERIAL PRIMARY KEY,
                        tenant_id TEXT DEFAULT '',
                        user_id TEXT DEFAULT '',
                        device_id TEXT DEFAULT '',
                        data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS tool_logs (
                        id SERIAL PRIMARY KEY,
                        tenant_id TEXT DEFAULT '',
                        user_id TEXT DEFAULT '',
                        device_id TEXT DEFAULT '',
                        tool_name TEXT NOT NULL,
                        input TEXT NOT NULL,
                        output TEXT NOT NULL,
                        duration_ms REAL DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Performance indexes for tenant-scoped queries
                    CREATE INDEX IF NOT EXISTS idx_pg_messages_tenant_time
                        ON messages(tenant_id, user_id, device_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_pg_checkpoints_tenant
                        ON checkpoints(tenant_id, user_id, device_id, id DESC);
                    CREATE INDEX IF NOT EXISTS idx_pg_tool_logs_tenant_name
                        ON tool_logs(tenant_id, user_id, device_id, tool_name);
                """)
            self._pg_tables_created = True
        return pool

    # ── Key-Value Store ───────────────────────────────────────────────────

    @require_tenant
    async def save(
        self,
        key: str,
        value: Any,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO kv_store (tenant_id, user_id, device_id, key, value) VALUES ($1, $2, $3, $4, $5) "
                    "ON CONFLICT (tenant_id, user_id, device_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
                    tenant_id,
                    user_id,
                    device_id,
                    key,
                    value,
                )
        else:
            await asyncio.to_thread(self._fallback.save, key, value, tenant_id, user_id, device_id)

    @require_tenant
    async def load(
        self,
        key: str,
        default: Any = None,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> Any:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT value FROM kv_store WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 AND key = $4",
                    tenant_id,
                    user_id,
                    device_id,
                    key,
                )
                if row:
                    return row["value"]
                return default
        return await asyncio.to_thread(
            self._fallback.load, key, default, tenant_id, user_id, device_id
        )

    @require_tenant
    async def delete(
        self, key: str, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM kv_store WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 AND key = $4",
                    tenant_id,
                    user_id,
                    device_id,
                    key,
                )
        else:
            await asyncio.to_thread(self._fallback.delete, key, tenant_id, user_id, device_id)

    # ── Conversation History ──────────────────────────────────────────────

    @require_tenant
    async def append_message(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> int:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row_id = await conn.fetchval(
                    "INSERT INTO messages (tenant_id, user_id, device_id, role, content, metadata) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                    tenant_id,
                    user_id,
                    device_id,
                    role,
                    content,
                    metadata or {},
                )
                return int(row_id) if row_id else 0
        return await asyncio.to_thread(
            self._fallback.append_message, role, content, metadata, tenant_id, user_id, device_id
        )

    @require_tenant
    async def get_messages(
        self,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> list[dict]:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, role, content, metadata, extract(epoch from created_at) as created_at "
                    "FROM messages WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 "
                    "ORDER BY id DESC LIMIT $4 OFFSET $5",
                    tenant_id,
                    user_id,
                    device_id,
                    limit,
                    offset,
                )
                return [
                    {
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"],
                    }
                    for row in reversed(rows)
                ]
        return await asyncio.to_thread(
            self._fallback.get_messages, limit, offset, tenant_id, user_id, device_id
        )

    @require_tenant
    async def clear_messages(
        self, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM messages WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3",
                    tenant_id,
                    user_id,
                    device_id,
                )
        else:
            await asyncio.to_thread(self._fallback.clear_messages, tenant_id, user_id, device_id)

    # ── Checkpoints ───────────────────────────────────────────────────────

    @require_tenant
    async def checkpoint(
        self, data: dict, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> int:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row_id = await conn.fetchval(
                    "INSERT INTO checkpoints (tenant_id, user_id, device_id, data) VALUES ($1, $2, $3, $4) RETURNING id",
                    tenant_id,
                    user_id,
                    device_id,
                    data,
                )
                return int(row_id) if row_id else 0
        return await asyncio.to_thread(
            self._fallback.checkpoint, data, tenant_id, user_id, device_id
        )

    @require_tenant
    async def latest_checkpoint(
        self, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> dict | None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data, extract(epoch from created_at) as created_at "
                    "FROM checkpoints WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 ORDER BY id DESC LIMIT 1",
                    tenant_id,
                    user_id,
                    device_id,
                )
                if row:
                    return {"data": row["data"], "created_at": row["created_at"]}
                return None
        return await asyncio.to_thread(
            self._fallback.latest_checkpoint, tenant_id, user_id, device_id
        )

    @require_tenant
    async def list_checkpoints(
        self, limit: int = 10, tenant_id: str = "default", user_id: str = "", device_id: str = ""
    ) -> list[dict]:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, data, extract(epoch from created_at) as created_at "
                    "FROM checkpoints WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 ORDER BY id DESC LIMIT $4",
                    tenant_id,
                    user_id,
                    device_id,
                    limit,
                )
                return [
                    {"id": row["id"], "data": row["data"], "created_at": row["created_at"]}
                    for row in rows
                ]
        return await asyncio.to_thread(
            self._fallback.list_checkpoints, limit, tenant_id, user_id, device_id
        )

    # ── Tool Logs ─────────────────────────────────────────────────────────

    @require_tenant
    async def log_tool_call(
        self,
        tool_name: str,
        input_data: str,
        output: str,
        duration_ms: float = 0,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO tool_logs (tenant_id, user_id, device_id, tool_name, input, output, duration_ms) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    tenant_id,
                    user_id,
                    device_id,
                    tool_name,
                    input_data,
                    output[:5000],
                    duration_ms,
                )
        else:
            await asyncio.to_thread(
                self._fallback.log_tool_call,
                tool_name,
                input_data,
                output,
                duration_ms,
                tenant_id,
                user_id,
                device_id,
            )

    @require_tenant
    async def get_tool_logs(
        self,
        tool_name: str | None = None,
        limit: int = 20,
        tenant_id: str = "default",
        user_id: str = "",
        device_id: str = "",
    ) -> list[dict]:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                if tool_name:
                    rows = await conn.fetch(
                        "SELECT tool_name, input, output, duration_ms, extract(epoch from created_at) as created_at "
                        "FROM tool_logs WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 AND tool_name = $4 ORDER BY id DESC LIMIT $5",
                        tenant_id,
                        user_id,
                        device_id,
                        tool_name,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT tool_name, input, output, duration_ms, extract(epoch from created_at) as created_at "
                        "FROM tool_logs WHERE tenant_id = $1 AND user_id = $2 AND device_id = $3 ORDER BY id DESC LIMIT $4",
                        tenant_id,
                        user_id,
                        device_id,
                        limit,
                    )
                return [
                    {
                        "tool": row["tool_name"],
                        "input": row["input"],
                        "output": row["output"],
                        "duration_ms": row["duration_ms"],
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]
        return await asyncio.to_thread(
            self._fallback.get_tool_logs, tool_name, limit, tenant_id, user_id, device_id
        )

    async def close(self) -> None:
        # DBPool handles global connection closure, but we close the fallback
        await asyncio.to_thread(self._fallback.close)

    async def get_stats(self) -> dict:
        """Return storage statistics."""
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                kv_count = await conn.fetchval("SELECT COUNT(*) FROM kv_store")
                msg_count = await conn.fetchval("SELECT COUNT(*) FROM messages")
                cp_count = await conn.fetchval("SELECT COUNT(*) FROM checkpoints")
                tool_count = await conn.fetchval("SELECT COUNT(*) FROM tool_logs")
                return {
                    "kv_entries": kv_count,
                    "messages": msg_count,
                    "checkpoints": cp_count,
                    "tool_logs": tool_count,
                }
        return await asyncio.to_thread(lambda: self._fallback.stats)
