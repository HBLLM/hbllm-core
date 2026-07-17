"""
Procedural Memory — stores reusable skills and tool sequences.

Unlike episodic memory (what happened) and semantic memory (facts),
procedural memory stores HOW to do things: multi-step tool sequences,
API call patterns, code generation templates, etc.

Skills are stored per-tenant and retrieved by pattern matching against
a trigger description.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hbllm.memory.interface import MemoryType
from hbllm.memory.pool import DatabasePool
from hbllm.memory.repository import MemoryRepository

logger = logging.getLogger(__name__)


class ProceduralMemory(MemoryRepository):
    """
    SQLite-backed skill/procedure storage.

    Each skill has a name, a trigger pattern (when to use it),
    and a list of steps (how to execute it).
    """

    def __init__(self, db_path: str | Path = "procedural_memory.db"):
        self.db_path = Path(db_path)
        self.pool = DatabasePool(str(self.db_path))

    async def init_db(self) -> None:
        """Create the skills table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    trigger_pattern TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    success_rate REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    source_node TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_tenant
                ON skills(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_name
                ON skills(tenant_id, skill_name)
            """)
            await conn.commit()

    async def close(self) -> None:
        """Close the persistent database connection."""
        await self.pool.close_all()

    async def store_skill(
        self,
        tenant_id: str,
        skill_name: str,
        trigger_pattern: str,
        steps: list[dict[str, Any]],
        source_node: str = "",
    ) -> str:
        """
        Store a new skill or update an existing one.

        Args:
            tenant_id: Tenant owning this skill.
            skill_name: Human-readable skill name.
            trigger_pattern: Description of when this skill applies.
            steps: List of step dicts. Must not be empty.
            source_node: Node that discovered/created the skill.

        Returns:
            The skill ID.

        Raises:
            ValueError: If skill_name is empty or steps is empty.
        """
        if not skill_name or not skill_name.strip():
            raise ValueError("skill_name must not be empty")
        if not steps:
            raise ValueError("steps must not be empty")

        now = datetime.now(timezone.utc).isoformat()

        # Check if skill already exists for this tenant
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT id FROM skills WHERE tenant_id = ? AND skill_name = ?",
                (tenant_id, skill_name),
            ) as cursor:
                existing = await cursor.fetchone()

            if existing:
                # Update existing skill
                await conn.execute(
                    """UPDATE skills
                       SET steps_json = ?, trigger_pattern = ?,
                           source_node = ?, updated_at = ?
                       WHERE id = ?""",
                    (json.dumps(steps), trigger_pattern, source_node, now, existing[0]),
                )
                logger.info("Updated skill '%s' for tenant '%s'", skill_name, tenant_id)
                await conn.commit()
                return str(existing[0])
            else:
                skill_id = uuid.uuid4().hex[:12]
                await conn.execute(
                    """INSERT INTO skills
                       (id, tenant_id, skill_name, trigger_pattern, steps_json,
                        source_node, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        skill_id,
                        tenant_id,
                        skill_name,
                        trigger_pattern,
                        json.dumps(steps),
                        source_node,
                        now,
                        now,
                    ),
                )
                await conn.commit()
                logger.info("Stored new skill '%s' for tenant '%s'", skill_name, tenant_id)
                return skill_id

    async def find_skill(self, tenant_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Find skills matching a query by keyword matching against
        skill name and trigger pattern.
        """
        async with self.pool.acquire() as conn:
            # Simple keyword search — matches against name and trigger pattern
            async with conn.execute(
                """SELECT * FROM skills
                   WHERE tenant_id = ?
                     AND (skill_name LIKE ? OR trigger_pattern LIKE ?)
                   ORDER BY usage_count DESC, success_rate DESC
                   LIMIT ?""",
                (tenant_id, f"%{query}%", f"%{query}%", top_k),
            ) as cursor:
                rows = await cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "skill_name": row[2],
                    "trigger_pattern": row[3],
                    "steps": json.loads(row[4]),
                    "success_rate": row[5],
                    "usage_count": row[6],
                }
                for row in rows
            ]

    async def record_usage(self, skill_id: str, success: bool = True) -> None:
        """Record that a skill was used and whether it succeeded."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT usage_count, success_rate FROM skills WHERE id = ?",
                (skill_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                return

            count = row[0] + 1
            # Exponential moving average for success rate
            old_rate = row[1]
            alpha = 0.3
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_rate

            await conn.execute(
                """UPDATE skills
                   SET usage_count = ?, success_rate = ?, updated_at = ?
                   WHERE id = ?""",
                (count, new_rate, datetime.now(timezone.utc).isoformat(), skill_id),
            )
            await conn.commit()

    async def get_most_used(self, tenant_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Get the most frequently used skills for a tenant."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                """SELECT * FROM skills
                   WHERE tenant_id = ?
                   ORDER BY usage_count DESC
                   LIMIT ?""",
                (tenant_id, top_k),
            ) as cursor:
                rows = await cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "skill_name": row[2],
                    "trigger_pattern": row[3],
                    "steps": json.loads(row[4]),
                    "success_rate": row[5],
                    "usage_count": row[6],
                }
                for row in rows
            ]

    async def delete_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID."""
        async with self.pool.acquire() as conn:
            cursor = await conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            await conn.commit()
            return cursor.rowcount > 0

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert a database row to a skill dictionary."""
        return {
            "id": row[0],
            "tenant_id": row[1],
            "skill_name": row[2],
            "trigger_pattern": row[3],
            "steps": json.loads(row[4]),
            "success_rate": row[5],
            "usage_count": row[6],
        }

    async def get_skill_by_id(self, skill_id: str) -> dict[str, Any] | None:
        """Retrieve a single skill by its ID."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT * FROM skills WHERE id = ?",
                (skill_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_dict(row)

    # ── MemoryRepository interface ───────────────────────────────────
    # Transitional adapters.

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.PROCEDURAL

    async def initialize(self) -> None:
        await self.init_db()

    async def shutdown(self) -> None:
        await self.close()

    async def store(self, content: str, tenant_id: str = "default", **kwargs: Any) -> str:
        """Store content as a procedural skill.

        Keyword Args:
            skill_name: Name of the skill (default: auto-generated).
            trigger_pattern: When to use this skill.
            steps: List of step dicts.
        """
        return await self.store_skill(
            tenant_id=tenant_id,
            skill_name=kwargs.get("skill_name", content[:50]),
            trigger_pattern=kwargs.get("trigger_pattern", content),
            steps=kwargs.get("steps", [{"action": content}]),
        )

    async def retrieve(
        self, memory_id: str, tenant_id: str = "default", **kwargs: Any
    ) -> dict[str, Any] | None:
        """Retrieve a skill by ID."""
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT * FROM skills WHERE id = ? AND tenant_id = ?",
                (memory_id, tenant_id),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    async def search(
        self, query: str, tenant_id: str = "default", **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search skills by trigger pattern."""
        raw_top_k = kwargs.get("top_k")
        if raw_top_k is None:
            raw_top_k = kwargs.get("limit")

        top_k = 3
        if raw_top_k is not None:
            try:
                top_k = int(raw_top_k)
            except (ValueError, TypeError):
                pass

        return await self.find_skill(tenant_id=tenant_id, query=query, top_k=top_k)

    async def stats(self, tenant_id: str = "default") -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM skills WHERE tenant_id = ?",
                (tenant_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return {
            "memory_type": self.memory_type.value,
            "skills": row[0] if row else 0,
        }
