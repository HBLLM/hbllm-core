"""
Skill Registry — learns, stores, and executes reusable skills from experience.

When an agent successfully completes a task, the skill is extracted
and stored for future reuse. Skills are versioned, rated, and searchable.

Flow:
1. Agent completes task successfully
2. SkillExtractor analyzes the task execution trace
3. Skill is stored with steps, tools, and success criteria
4. Future similar tasks → skill lookup → fast execution
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A learned, reusable skill."""

    skill_id: str
    name: str
    description: str
    category: str  # coding | research | analysis | writing | reasoning
    steps: list[str]  # ordered execution steps
    tools_used: list[str]  # which tools the skill uses
    success_criteria: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    success_rate: float = 1.0
    invocations: int = 0
    avg_latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)


class SkillRegistry:
    """
    Registry of learned skills that grows over time.

    Capabilities:
    - Extract skills from successful task completions
    - Store skills with steps, tools, and examples
    - Search skills by category or similarity
    - Track skill success rates for continuous improvement
    - Version skills as they are refined
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "skill_registry.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    tools_used TEXT NOT NULL,
                    success_criteria TEXT DEFAULT '',
                    examples TEXT DEFAULT '[]',
                    success_rate REAL DEFAULT 1.0,
                    invocations INTEGER DEFAULT 0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_cat ON skills(category)")

    # ─── Skill Extraction ────────────────────────────────────────────

    def extract_and_store(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
        tools_used: list[str],
        success: bool,
        category: str = "general",
    ) -> Skill | None:
        """
        Extract a reusable skill from a successful task execution.

        Args:
            task_description: What the task was
            execution_trace: List of steps taken (action, result pairs)
            tools_used: Which tools were invoked
            success: Whether the task succeeded
            category: Skill category
        """
        if not success:
            return None

        # Extract steps from trace
        steps = []
        for step in execution_trace:
            action = step.get("action", "")
            if action:
                steps.append(action)

        if not steps:
            return None

        skill_id = f"skill_{int(time.time())}_{hash(task_description) % 10000}"
        name = self._generate_skill_name(task_description)

        skill = Skill(
            skill_id=skill_id,
            name=name,
            description=task_description,
            category=category,
            steps=steps,
            tools_used=tools_used,
            success_criteria=f"Successfully completed: {task_description}",
            examples=[{"input": task_description, "steps": steps}],
        )

        self._store(skill)
        logger.info("Extracted skill: %s (%s)", name, skill_id)
        return skill

    def _store(self, skill: Skill) -> None:
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skills
                (skill_id, name, description, category, steps, tools_used,
                 success_criteria, examples, success_rate, invocations,
                 avg_latency_ms, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    skill.skill_id,
                    skill.name,
                    skill.description,
                    skill.category,
                    json.dumps(skill.steps),
                    json.dumps(skill.tools_used),
                    skill.success_criteria,
                    json.dumps(skill.examples),
                    skill.success_rate,
                    skill.invocations,
                    skill.avg_latency_ms,
                    skill.created_at,
                    now,
                ),
            )

    # ─── Skill Lookup ────────────────────────────────────────────────

    def find_skill(self, query: str, category: str | None = None, top_k: int = 5) -> list[Skill]:
        """Find skills matching a query by keyword similarity."""
        with sqlite3.connect(str(self._db_path)) as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM skills WHERE category = ? ORDER BY success_rate DESC, invocations DESC LIMIT ?",
                    (category, top_k * 3),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM skills ORDER BY success_rate DESC, invocations DESC LIMIT ?",
                    (top_k * 3,),
                ).fetchall()

        skills = [self._row_to_skill(r) for r in rows]

        # Score by keyword relevance
        query_words = set(query.lower().split())
        scored = []
        for skill in skills:
            desc_words = set(skill.description.lower().split())
            name_words = set(skill.name.lower().split())
            overlap = len(query_words & (desc_words | name_words))
            scored.append((overlap, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def get_skill(self, skill_id: str) -> Skill | None:
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM skills WHERE skill_id = ?", (skill_id,)).fetchone()
        return self._row_to_skill(row) if row else None

    # ─── Skill Execution Tracking ────────────────────────────────────

    def record_execution(self, skill_id: str, success: bool, latency_ms: float) -> None:
        """Record a skill execution for performance tracking."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT invocations, success_rate, avg_latency_ms FROM skills WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
            if not row:
                return

            invocations = row[0] + 1
            old_rate = row[1]
            old_latency = row[2]

            # Rolling average
            new_rate = (old_rate * row[0] + (1.0 if success else 0.0)) / invocations
            new_latency = (old_latency * row[0] + latency_ms) / invocations

            conn.execute(
                "UPDATE skills SET invocations = ?, success_rate = ?, avg_latency_ms = ?, updated_at = ? "
                "WHERE skill_id = ?",
                (invocations, new_rate, new_latency, time.time(), skill_id),
            )

    # ─── Helpers ─────────────────────────────────────────────────────

    def _generate_skill_name(self, description: str) -> str:
        words = description.split()[:5]
        return " ".join(w.capitalize() for w in words)

    def _row_to_skill(self, row: tuple[Any, ...]) -> Skill:
        return Skill(
            skill_id=row[0],
            name=row[1],
            description=row[2],
            category=row[3],
            steps=json.loads(row[4]),
            tools_used=json.loads(row[5]),
            success_criteria=row[6],
            examples=json.loads(row[7]),
            success_rate=row[8],
            invocations=row[9],
            avg_latency_ms=row[10],
            created_at=row[11],
        )

    def list_skills(self, category: str | None = None, limit: int = 50) -> list[Skill]:
        with sqlite3.connect(str(self._db_path)) as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM skills WHERE category = ? ORDER BY invocations DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM skills ORDER BY invocations DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            cats = conn.execute("SELECT COUNT(DISTINCT category) FROM skills").fetchone()[0]
            avg_sr = conn.execute("SELECT AVG(success_rate) FROM skills").fetchone()[0]
        return {
            "total_skills": total,
            "categories": cats,
            "avg_success_rate": round(avg_sr or 0, 3),
        }
