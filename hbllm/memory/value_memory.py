"""
Value / Reward Memory — tracks per-tenant preference signals.

Records positive and negative feedback events keyed by topic/action,
allowing the system to learn what each tenant prefers over time.
This feeds into the RLHF loop and guides future response selection.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hbllm.memory.pool import DatabasePool

logger = logging.getLogger(__name__)


class ValueMemory:
    """
    SQLite-backed preference/reward signal storage.

    Tracks per-tenant reward signals keyed by topic and action,
    using exponential decay for older signals so recent preferences
    carry more weight.
    """

    def __init__(self, db_path: str | Path = "value_memory.db"):
        self.db_path = Path(db_path)
        self.pool = DatabasePool(str(self.db_path))

    async def init_db(self) -> None:
        """Create the rewards table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rewards (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT DEFAULT '',
                    device_id TEXT DEFAULT '',
                    topic TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reward REAL NOT NULL,
                    context TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rewards_tenant_topic
                ON rewards(tenant_id, topic)
            """)
            await conn.commit()

    async def close(self) -> None:
        """Close the persistent database connection."""
        await self.pool.close_all()

    async def record_reward(
        self,
        tenant_id: str,
        topic: str,
        action: str,
        reward: float,
        context: dict[str, Any] | None = None,
        user_id: str = "",
        device_id: str = "",
    ) -> str:
        """
        Record a reward signal for a tenant action.

        Args:
            tenant_id: Tenant providing the signal.
            topic: Category (e.g. "response_style", "domain_preference").
            action: Specific action taken (e.g. "formal_tone", "code_example").
            reward: Signal strength (-1.0 to 1.0).
            context: Optional context dict for why this reward was given.

        Returns:
            The reward record ID.
        """
        reward_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO rewards (id, tenant_id, user_id, device_id, topic, action, reward, context, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    reward_id,
                    tenant_id,
                    user_id,
                    device_id,
                    topic,
                    action,
                    max(-1.0, min(1.0, reward)),  # Clamp to [-1, 1]
                    json.dumps(context or {}),
                    now,
                ),
            )
            await conn.commit()

        logger.debug(
            "Recorded reward for tenant '%s': topic=%s action=%s reward=%.2f",
            tenant_id,
            topic,
            action,
            reward,
        )
        return reward_id

    async def get_preference(
        self, tenant_id: str, topic: str, user_id: str = "", device_id: str = ""
    ) -> dict[str, float]:
        """
        Get aggregated preferences for a topic.

        Returns a dict mapping action → average reward, weighted by
        recency (more recent signals count more).
        """
        async with self.pool.acquire() as conn:
            async with conn.execute(
                """SELECT action, reward, created_at FROM rewards
                   WHERE tenant_id = ? AND user_id = ? AND device_id = ? AND topic = ?
                   ORDER BY created_at DESC""",
                (tenant_id, user_id, device_id, topic),
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return {}

        # Weighted average with exponential decay
        preferences: dict[str, list[float]] = {}
        for i, row in enumerate(rows):
            action = row[0]
            decay = 0.9**i  # More recent = higher weight
            weighted_reward = row[1] * decay
            if action not in preferences:
                preferences[action] = []
            preferences[action].append(weighted_reward)

        return {action: sum(values) / len(values) for action, values in preferences.items()}

    async def get_top_preferences(
        self, tenant_id: str, top_k: int = 5, user_id: str = "", device_id: str = ""
    ) -> list[dict[str, Any]]:
        """
        Get the tenant's strongest preferences across all topics.

        Returns top_k (topic, action) pairs ranked by average reward.
        """
        async with self.pool.acquire() as conn:
            async with conn.execute(
                """SELECT topic, action, AVG(reward) as avg_reward, COUNT(*) as count
                   FROM rewards
                   WHERE tenant_id = ? AND user_id = ? AND device_id = ?
                   GROUP BY topic, action
                   ORDER BY avg_reward DESC
                   LIMIT ?""",
                (tenant_id, user_id, device_id, top_k),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "topic": row[0],
                "action": row[1],
                "avg_reward": round(row[2], 3),
                "count": row[3],
            }
            for row in rows
        ]

    async def get_signal_count(self, tenant_id: str, user_id: str = "", device_id: str = "") -> int:
        """Get total number of reward signals for a tenant/user/device."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM rewards WHERE tenant_id = ? AND user_id = ? AND device_id = ?",
                (tenant_id, user_id, device_id),
            ) as cursor:
                row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear_tenant(self, tenant_id: str, user_id: str = "", device_id: str = "") -> int:
        """Purge all rewards for a tenant/user/device. Returns count of deleted records."""
        async with self.pool.acquire() as conn:
            cursor = await conn.execute(
                "DELETE FROM rewards WHERE tenant_id = ? AND user_id = ? AND device_id = ?",
                (tenant_id, user_id, device_id),
            )
            deleted = cursor.rowcount
            await conn.commit()
        return deleted

    async def get_all_topics(
        self, tenant_id: str, user_id: str = "", device_id: str = ""
    ) -> list[str]:
        """Get all distinct topics that have been recorded for a tenant/user/device."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT DISTINCT topic FROM rewards WHERE tenant_id = ? AND user_id = ? AND device_id = ? ORDER BY topic",
                (tenant_id, user_id, device_id),
            ) as cursor:
                rows = await cursor.fetchall()
        return [row[0] for row in rows]
