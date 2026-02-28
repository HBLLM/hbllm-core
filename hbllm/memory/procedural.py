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
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProceduralMemory:
    """
    SQLite-backed skill/procedure storage.
    
    Each skill has a name, a trigger pattern (when to use it),
    and a list of steps (how to execute it).
    """

    def __init__(self, db_path: str | Path = "procedural_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the skills table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_tenant 
                ON skills(tenant_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_name 
                ON skills(tenant_id, skill_name)
            """)

    def store_skill(
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
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM skills WHERE tenant_id = ? AND skill_name = ?",
                (tenant_id, skill_name),
            ).fetchone()
            
            if existing:
                # Update existing skill
                conn.execute(
                    """UPDATE skills 
                       SET steps_json = ?, trigger_pattern = ?, 
                           source_node = ?, updated_at = ?
                       WHERE id = ?""",
                    (json.dumps(steps), trigger_pattern, source_node, now, existing[0]),
                )
                logger.info("Updated skill '%s' for tenant '%s'", skill_name, tenant_id)
                return existing[0]
            else:
                skill_id = uuid.uuid4().hex[:12]
                conn.execute(
                    """INSERT INTO skills 
                       (id, tenant_id, skill_name, trigger_pattern, steps_json, 
                        source_node, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (skill_id, tenant_id, skill_name, trigger_pattern,
                     json.dumps(steps), source_node, now, now),
                )
                logger.info("Stored new skill '%s' for tenant '%s'", skill_name, tenant_id)
                return skill_id

    def find_skill(self, tenant_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Find skills matching a query by keyword matching against
        skill name and trigger pattern.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Simple keyword search — matches against name and trigger pattern
            rows = conn.execute(
                """SELECT * FROM skills 
                   WHERE tenant_id = ? 
                     AND (skill_name LIKE ? OR trigger_pattern LIKE ?)
                   ORDER BY usage_count DESC, success_rate DESC
                   LIMIT ?""",
                (tenant_id, f"%{query}%", f"%{query}%", top_k),
            ).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "skill_name": row["skill_name"],
                    "trigger_pattern": row["trigger_pattern"],
                    "steps": json.loads(row["steps_json"]),
                    "success_rate": row["success_rate"],
                    "usage_count": row["usage_count"],
                }
                for row in rows
            ]

    def record_usage(self, skill_id: str, success: bool = True) -> None:
        """Record that a skill was used and whether it succeeded."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT usage_count, success_rate FROM skills WHERE id = ?",
                (skill_id,),
            ).fetchone()
            
            if not row:
                return
            
            count = row[0] + 1
            # Exponential moving average for success rate
            old_rate = row[1]
            alpha = 0.3
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_rate
            
            conn.execute(
                """UPDATE skills 
                   SET usage_count = ?, success_rate = ?, updated_at = ?
                   WHERE id = ?""",
                (count, new_rate, datetime.now(timezone.utc).isoformat(), skill_id),
            )

    def get_most_used(self, tenant_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Get the most frequently used skills for a tenant."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM skills 
                   WHERE tenant_id = ?
                   ORDER BY usage_count DESC
                   LIMIT ?""",
                (tenant_id, top_k),
            ).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "skill_name": row["skill_name"],
                    "trigger_pattern": row["trigger_pattern"],
                    "steps": json.loads(row["steps_json"]),
                    "success_rate": row["success_rate"],
                    "usage_count": row["usage_count"],
                }
                for row in rows
            ]

    def delete_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            return cursor.rowcount > 0

    def get_skill_by_id(self, skill_id: str) -> dict[str, Any] | None:
        """Retrieve a single skill by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (skill_id,),
            ).fetchone()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "tenant_id": row["tenant_id"],
            "skill_name": row["skill_name"],
            "trigger_pattern": row["trigger_pattern"],
            "steps": json.loads(row["steps_json"]),
            "success_rate": row["success_rate"],
            "usage_count": row["usage_count"],
        }
