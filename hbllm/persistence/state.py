"""
Brain State — SQLite-backed state storage for HBLLM brain sessions.

Supports:
- Key-value store for configuration snapshots
- Conversation history
- Memory checkpoints
- Tool execution logs
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any

from hbllm.persistence.db_pool import DBPool

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
        self._conn = sqlite3.connect(self.path)
        self._init_tables()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                duration_ms REAL DEFAULT 0,
                created_at REAL NOT NULL
            );
        """)
        self._conn.commit()

    # ── Key-Value Store ───────────────────────────────────────────────────

    def save(self, key: str, value: Any) -> None:
        """Save a value to the key-value store."""
        self._conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    def load(self, key: str, default: Any = None) -> Any:
        """Load a value from the key-value store."""
        row = self._conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        return default

    def delete(self, key: str) -> None:
        """Delete a key from the store."""
        self._conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        self._conn.commit()

    # ── Conversation History ──────────────────────────────────────────────

    def append_message(self, role: str, content: str, metadata: dict | None = None) -> int:
        """Append a message to conversation history."""
        cursor = self._conn.execute(
            "INSERT INTO messages (role, content, metadata, created_at) VALUES (?, ?, ?, ?)",
            (role, content, json.dumps(metadata or {}), time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_messages(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get recent messages from conversation history."""
        rows = self._conn.execute(
            "SELECT id, role, content, metadata, created_at FROM messages "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
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

    def clear_messages(self) -> None:
        """Clear all conversation history."""
        self._conn.execute("DELETE FROM messages")
        self._conn.commit()

    # ── Checkpoints ───────────────────────────────────────────────────────

    def checkpoint(self, data: dict) -> int:
        """Save a checkpoint."""
        cursor = self._conn.execute(
            "INSERT INTO checkpoints (data, created_at) VALUES (?, ?)",
            (json.dumps(data), time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def latest_checkpoint(self) -> dict | None:
        """Get the most recent checkpoint."""
        row = self._conn.execute(
            "SELECT data, created_at FROM checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return {"data": json.loads(row[0]), "created_at": row[1]}
        return None

    def list_checkpoints(self, limit: int = 10) -> list[dict]:
        """List recent checkpoints."""
        rows = self._conn.execute(
            "SELECT id, data, created_at FROM checkpoints ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": row[0], "data": json.loads(row[1]), "created_at": row[2]} for row in rows]

    # ── Tool Logs ─────────────────────────────────────────────────────────

    def log_tool_call(
        self, tool_name: str, input_data: str, output: str, duration_ms: float = 0
    ) -> None:
        """Log a tool invocation."""
        self._conn.execute(
            "INSERT INTO tool_logs (tool_name, input, output, duration_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (tool_name, input_data, output[:5000], duration_ms, time.time()),
        )
        self._conn.commit()

    def get_tool_logs(self, tool_name: str | None = None, limit: int = 20) -> list[dict]:
        """Get tool execution logs."""
        if tool_name:
            rows = self._conn.execute(
                "SELECT tool_name, input, output, duration_ms, created_at FROM tool_logs "
                "WHERE tool_name = ? ORDER BY id DESC LIMIT ?",
                (tool_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT tool_name, input, output, duration_ms, created_at FROM tool_logs "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
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
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id SERIAL PRIMARY KEY,
                        data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS tool_logs (
                        id SERIAL PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        input TEXT NOT NULL,
                        output TEXT NOT NULL,
                        duration_ms REAL DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            self._pg_tables_created = True
        return pool

    # ── Key-Value Store ───────────────────────────────────────────────────

    async def save(self, key: str, value: Any) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO kv_store (key, value) VALUES ($1, $2) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
                    key,
                    value,
                )
        else:
            self._fallback.save(key, value)

    async def load(self, key: str, default: Any = None) -> Any:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT value FROM kv_store WHERE key = $1", key)
                if row:
                    return row["value"]
                return default
        return self._fallback.load(key, default)

    async def delete(self, key: str) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM kv_store WHERE key = $1", key)
        else:
            self._fallback.delete(key)

    # ── Conversation History ──────────────────────────────────────────────

    async def append_message(self, role: str, content: str, metadata: dict | None = None) -> int:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row_id = await conn.fetchval(
                    "INSERT INTO messages (role, content, metadata) VALUES ($1, $2, $3) RETURNING id",
                    role,
                    content,
                    metadata or {},
                )
                return int(row_id) if row_id else 0
        return self._fallback.append_message(role, content, metadata)

    async def get_messages(self, limit: int = 50, offset: int = 0) -> list[dict]:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, role, content, metadata, extract(epoch from created_at) as created_at "
                    "FROM messages ORDER BY id DESC LIMIT $1 OFFSET $2",
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
        return self._fallback.get_messages(limit, offset)

    async def clear_messages(self) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM messages")
        else:
            self._fallback.clear_messages()

    # ── Checkpoints ───────────────────────────────────────────────────────

    async def checkpoint(self, data: dict) -> int:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row_id = await conn.fetchval(
                    "INSERT INTO checkpoints (data) VALUES ($1) RETURNING id", data
                )
                return int(row_id) if row_id else 0
        return self._fallback.checkpoint(data)

    async def latest_checkpoint(self) -> dict | None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data, extract(epoch from created_at) as created_at "
                    "FROM checkpoints ORDER BY id DESC LIMIT 1"
                )
                if row:
                    return {"data": row["data"], "created_at": row["created_at"]}
                return None
        return self._fallback.latest_checkpoint()

    async def list_checkpoints(self, limit: int = 10) -> list[dict]:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, data, extract(epoch from created_at) as created_at "
                    "FROM checkpoints ORDER BY id DESC LIMIT $1",
                    limit,
                )
                return [
                    {"id": row["id"], "data": row["data"], "created_at": row["created_at"]}
                    for row in rows
                ]
        return self._fallback.list_checkpoints(limit)

    # ── Tool Logs ─────────────────────────────────────────────────────────

    async def log_tool_call(
        self, tool_name: str, input_data: str, output: str, duration_ms: float = 0
    ) -> None:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO tool_logs (tool_name, input, output, duration_ms) VALUES ($1, $2, $3, $4)",
                    tool_name,
                    input_data,
                    output[:5000],
                    duration_ms,
                )
        else:
            self._fallback.log_tool_call(tool_name, input_data, output, duration_ms)

    async def get_tool_logs(self, tool_name: str | None = None, limit: int = 20) -> list[dict]:
        pool = await self._ensure_pg_tables()
        if pool:
            async with pool.acquire() as conn:
                if tool_name:
                    rows = await conn.fetch(
                        "SELECT tool_name, input, output, duration_ms, extract(epoch from created_at) as created_at "
                        "FROM tool_logs WHERE tool_name = $1 ORDER BY id DESC LIMIT $2",
                        tool_name,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT tool_name, input, output, duration_ms, extract(epoch from created_at) as created_at "
                        "FROM tool_logs ORDER BY id DESC LIMIT $1",
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
        return self._fallback.get_tool_logs(tool_name, limit)

    async def close(self) -> None:
        # DBPool handles global connection closure, but we close the fallback
        self._fallback.close()

    @property
    async def stats(self) -> dict:
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
        return self._fallback.stats
