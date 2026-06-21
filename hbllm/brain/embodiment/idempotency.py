"""Idempotency Tracking Engine.

Prevents duplicate actions if the cognitive runtime crashes or a delegation loop retries a task.
Generates an idempotency key and checks a lock-table before executing mutating OS actions.

Persistence:
    Uses SQLite for crash-safe idempotency tracking. Keys survive daemon
    restarts, preventing duplicate execution of dangerous actions (e.g.,
    "delete file" or "unlock door") after a crash-recovery cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IdempotencyEngine:
    """Manages idempotency keys to ensure exactly-once execution of mutating actions.

    Keys are persisted to SQLite so they survive daemon restarts.
    Old keys are automatically pruned after a configurable TTL.
    """

    def __init__(
        self,
        db_path: str | Path = "data/idempotency.db",
        key_ttl_hours: float = 24.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.key_ttl_s = key_ttl_hours * 3600
        self._init_db()

    def _init_db(self) -> None:
        """Create the idempotency table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    key TEXT PRIMARY KEY,
                    goal_id TEXT,
                    action TEXT,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'locked'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_idempotency_created
                ON idempotency_keys(created_at)
            """)
        logger.debug("IdempotencyEngine initialized at %s", self.db_path)

    def generate_key(self, goal_id: str, action: str, parameters: dict[str, Any]) -> str:
        """Generate a stable, unique hash for an action payload."""
        payload = json.dumps(
            {"goal_id": goal_id, "action": action, "parameters": parameters}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def check_and_lock(self, idempotency_key: str, goal_id: str = "", action: str = "") -> bool:
        """Check if an action has already been executed. If not, lock it.

        Returns True if safe to execute. Returns False if already executed (duplicate).
        """
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Check if key already exists and is not expired
            row = conn.execute(
                "SELECT created_at, status FROM idempotency_keys WHERE key = ?",
                (idempotency_key,),
            ).fetchone()

            if row is not None:
                created_at, status = row
                age = now - created_at

                if age < self.key_ttl_s:
                    # Key exists and is still valid — duplicate
                    logger.warning(
                        "Idempotency lock triggered for key %.16s... (age=%.0fs, status=%s). "
                        "Skipping duplicate execution.",
                        idempotency_key,
                        age,
                        status,
                    )
                    return False
                else:
                    # Key expired — allow re-execution
                    conn.execute(
                        "DELETE FROM idempotency_keys WHERE key = ?",
                        (idempotency_key,),
                    )

            # Lock the key
            conn.execute(
                "INSERT INTO idempotency_keys (key, goal_id, action, created_at, status) "
                "VALUES (?, ?, ?, ?, 'locked')",
                (idempotency_key, goal_id, action, now),
            )
            conn.commit()

        return True

    def mark_completed(self, idempotency_key: str) -> None:
        """Mark an action as successfully completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE idempotency_keys SET status = 'completed' WHERE key = ?",
                (idempotency_key,),
            )
            conn.commit()

    def clear_lock(self, idempotency_key: str) -> None:
        """Clear a lock if an action failed and needs to be legally retried."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM idempotency_keys WHERE key = ?",
                (idempotency_key,),
            )
            conn.commit()
        logger.debug("Cleared idempotency lock for key %.16s...", idempotency_key)

    def prune_expired(self) -> int:
        """Remove expired keys. Returns count of pruned keys."""
        cutoff = time.time() - self.key_ttl_s
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM idempotency_keys WHERE created_at < ?",
                (cutoff,),
            )
            pruned = cursor.rowcount
            conn.commit()

        if pruned > 0:
            logger.info("Pruned %d expired idempotency keys", pruned)
        return pruned

    def stats(self) -> dict[str, Any]:
        """Get idempotency engine statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM idempotency_keys").fetchone()[0]
            locked = conn.execute(
                "SELECT COUNT(*) FROM idempotency_keys WHERE status = 'locked'"
            ).fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM idempotency_keys WHERE status = 'completed'"
            ).fetchone()[0]

        return {
            "total_keys": total,
            "locked": locked,
            "completed": completed,
            "db_path": str(self.db_path),
        }
