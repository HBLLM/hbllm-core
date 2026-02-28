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

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Lightweight SQLite storage for conversation history.
    """

    def __init__(self, db_path: str | Path = "working_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    domain TEXT,
                    timestamp_iso TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            # Index for fast retrieval of latest turns per tenant+session
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_tenant_session_time 
                ON turns(tenant_id, session_id, timestamp_iso DESC)
            ''')
            # Migrate: add tenant_id column if upgrading from old schema
            try:
                cursor.execute('SELECT tenant_id FROM turns LIMIT 1')
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE turns ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            conn.commit()
            logger.debug("Initialized EpisodicMemory at %s", self.db_path)

    def store_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
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

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO turns (id, tenant_id, session_id, role, content, domain, timestamp_iso, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (turn_id, tenant_id, session_id, role, content, domain, now_iso, meta_str))
            conn.commit()
            
        return turn_id

    def retrieve_recent(self, session_id: str, limit: int = 10, tenant_id: str = "default") -> list[dict[str, Any]]:
        """
        Retrieve the most recent turns for a given tenant's session.
        
        Returns a list of dicts ordered chronologically (oldest first).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM turns 
                WHERE tenant_id = ? AND session_id = ? 
                ORDER BY timestamp_iso DESC 
                LIMIT ?
            ''', (tenant_id, session_id, limit))
            
            rows = cursor.fetchall()

        results = []
        for row in reversed(rows):
            results.append({
                "id": row["id"],
                "tenant_id": row["tenant_id"],
                "role": row["role"],
                "content": row["content"],
                "domain": row["domain"],
                "timestamp": row["timestamp_iso"],
                "metadata": json.loads(row["metadata"]),
            })
            
        return results

    def clear_session(self, session_id: str, tenant_id: str = "default") -> int:
        """Delete all turns for a tenant's session. Returns deleted count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM turns WHERE tenant_id = ? AND session_id = ?', (tenant_id, session_id))
            deleted = cursor.rowcount
            conn.commit()
        return deleted
