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
