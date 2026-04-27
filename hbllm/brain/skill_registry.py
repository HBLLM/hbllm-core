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
    version: int = 1
    parent_skill_id: str | None = None
    tokens_used: int = 0
    cost_score: float = 0.0
    failure_types: list[str] = field(default_factory=list)
    confidence_score: float = 0.8
    tenant_id: str = "global"
    source: str = ""  # provenance: "plugin:<name>", "auto-compiled", "user", "graduated"


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
                    updated_at REAL NOT NULL,
                    tenant_id TEXT DEFAULT 'global'
                )
            """)

            # Migration: Add new columns if they don't exist
            cursor = conn.execute("PRAGMA table_info(skills)")
            columns = [col[1] for col in cursor.fetchall()]

            if "version" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN version INTEGER DEFAULT 1")
            if "parent_skill_id" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN parent_skill_id TEXT")
            if "tokens_used" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN tokens_used INTEGER DEFAULT 0")
            if "cost_score" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN cost_score REAL DEFAULT 0.0")
            if "failure_types" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN failure_types TEXT DEFAULT '[]'")
            if "confidence_score" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN confidence_score REAL DEFAULT 0.8")
            if "tenant_id" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN tenant_id TEXT DEFAULT 'global'")
            if "source" not in columns:
                conn.execute("ALTER TABLE skills ADD COLUMN source TEXT DEFAULT ''")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_cat ON skills(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_tenant ON skills(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_source ON skills(source)")

    # ─── Skill Extraction ────────────────────────────────────────────

    def extract_and_store(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
        tools_used: list[str],
        success: bool,
        category: str = "general",
        tenant_id: str = "global",
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
            tenant_id=tenant_id,
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
                 avg_latency_ms, created_at, updated_at,
                 version, parent_skill_id, tokens_used, cost_score, failure_types, confidence_score, tenant_id, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    skill.version,
                    skill.parent_skill_id,
                    skill.tokens_used,
                    skill.cost_score,
                    json.dumps(skill.failure_types),
                    skill.confidence_score,
                    skill.tenant_id,
                    skill.source,
                ),
            )

    # ─── Skill Lookup ────────────────────────────────────────────────

    _SELECT_COLS = (
        "skill_id, name, description, category, steps, tools_used, success_criteria, "
        "examples, success_rate, invocations, avg_latency_ms, created_at, version, "
        "parent_skill_id, tokens_used, cost_score, failure_types, confidence_score, tenant_id, source"
    )

    def find_skill(
        self, query: str, category: str | None = None, top_k: int = 5, tenant_id: str = "global"
    ) -> list[Skill]:
        """Find skills matching a query by keyword similarity."""
        with sqlite3.connect(str(self._db_path)) as conn:
            if category:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM skills WHERE category = ? AND tenant_id IN (?, 'global') ORDER BY confidence_score DESC, success_rate DESC, invocations DESC LIMIT ?",
                    (category, tenant_id, top_k * 3),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM skills WHERE tenant_id IN (?, 'global') ORDER BY confidence_score DESC, success_rate DESC, invocations DESC LIMIT ?",
                    (tenant_id, top_k * 3),
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
            row = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM skills WHERE skill_id = ?", (skill_id,)
            ).fetchone()
        return self._row_to_skill(row) if row else None

    # ─── Skill Execution Tracking ────────────────────────────────────

    def record_execution(
        self,
        skill_id: str,
        success: bool,
        latency_ms: float,
        tokens: int = 0,
        failure_type: str | None = None,
    ) -> None:
        """Record a skill execution for performance tracking."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT invocations, success_rate, avg_latency_ms, tokens_used, confidence_score, failure_types FROM skills WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
            if not row:
                return

            invocations = row[0] + 1
            old_rate = row[1]
            old_latency = row[2]
            old_tokens = row[3]
            old_conf = row[4]
            failures = json.loads(row[5])

            # Rolling average
            new_rate = (old_rate * row[0] + (1.0 if success else 0.0)) / invocations
            new_latency = (old_latency * row[0] + latency_ms) / invocations
            new_tokens = int((old_tokens * row[0] + tokens) / invocations)

            # Confidence update
            new_conf = old_conf
            if success:
                new_conf = min(1.0, old_conf + 0.05)
            else:
                new_conf = max(0.0, old_conf - 0.2)  # penalize failure
                if failure_type and failure_type not in failures:
                    failures.append(failure_type)

            conn.execute(
                "UPDATE skills SET invocations = ?, success_rate = ?, avg_latency_ms = ?, tokens_used = ?, confidence_score = ?, failure_types = ?, updated_at = ? "
                "WHERE skill_id = ?",
                (
                    invocations,
                    new_rate,
                    new_latency,
                    new_tokens,
                    new_conf,
                    json.dumps(failures),
                    time.time(),
                    skill_id,
                ),
            )

    def promote_skill(self, skill_id: str) -> None:
        """Promote a high-confidence skill to the global tier."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE skills SET tenant_id = 'global', updated_at = ? WHERE skill_id = ?",
                (time.time(), skill_id),
            )

    def version_skill(
        self,
        skill_id: str,
        new_steps: list[str],
        test_latency_ms: float = 0.0,
        new_tools: list[str] | None = None,
    ) -> Skill | None:
        """Create a new version of a skill, usually after a failure fix."""
        old_skill = self.get_skill(skill_id)
        if not old_skill:
            return None

        new_skill_id = f"skill_{int(time.time())}_{hash(old_skill.name) % 10000}"

        new_skill = Skill(
            skill_id=new_skill_id,
            name=old_skill.name,
            description=old_skill.description,
            category=old_skill.category,
            steps=new_steps,
            tools_used=new_tools if new_tools is not None else old_skill.tools_used,
            success_criteria=old_skill.success_criteria,
            examples=old_skill.examples,
            success_rate=1.0,
            invocations=1,
            avg_latency_ms=test_latency_ms,
            version=old_skill.version + 1,
            parent_skill_id=old_skill.skill_id,
            tokens_used=old_skill.tokens_used,
            cost_score=old_skill.cost_score,
            confidence_score=0.9,  # new fix starts confident
            tenant_id=old_skill.tenant_id,
        )
        self._store(new_skill)
        logger.info(
            "Versioned skill %s to v%d (%s)", old_skill.name, new_skill.version, new_skill_id
        )
        return new_skill

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
            version=row[12],
            parent_skill_id=row[13],
            tokens_used=row[14],
            cost_score=row[15],
            failure_types=json.loads(row[16]),
            confidence_score=row[17],
            tenant_id=row[18] if len(row) > 18 else "global",
            source=row[19] if len(row) > 19 else "",
        )

    def list_skills(
        self, category: str | None = None, limit: int = 50, tenant_id: str = "global"
    ) -> list[Skill]:
        with sqlite3.connect(str(self._db_path)) as conn:
            if category:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM skills WHERE category = ? AND tenant_id IN (?, 'global') ORDER BY invocations DESC LIMIT ?",
                    (category, tenant_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM skills WHERE tenant_id IN (?, 'global') ORDER BY invocations DESC LIMIT ?",
                    (tenant_id, limit),
                ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    # ─── Skill Removal & Graduation ────────────────────────────────

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID. Returns True if deleted."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute("DELETE FROM skills WHERE skill_id = ?", (skill_id,))
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Removed skill: %s", skill_id)
        return deleted

    def remove_by_source(self, source: str) -> int:
        """
        Remove all skills with a specific source.

        Args:
            source: The source tag (e.g., 'plugin:sentinel-shield')

        Returns:
            Number of skills removed.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute("DELETE FROM skills WHERE source = ?", (source,))
            count = cursor.rowcount
        if count:
            logger.info("Removed %d skills with source '%s'", count, source)
        return count

    def find_by_source(self, source: str) -> list[Skill]:
        """Find all skills from a specific source."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM skills WHERE source = ? ORDER BY invocations DESC",
                (source,),
            ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def graduate_skill(self, skill_id: str) -> bool:
        """
        Graduate a plugin-provided skill to 'graduated' status.

        Graduated skills survive plugin removal — they are considered
        core-learned skills that originated from a plugin but have been
        validated through real usage.

        A skill qualifies for graduation when it has been invoked
        at least once with a reasonable success rate.

        Returns:
            True if the skill was graduated.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE skills SET source = 'graduated', updated_at = ? WHERE skill_id = ?",
                (time.time(), skill_id),
            )
            graduated = cursor.rowcount > 0
        if graduated:
            logger.info("Graduated skill to core: %s", skill_id)
        return graduated

    def graduate_experienced_skills(self, source: str, min_invocations: int = 1) -> list[str]:
        """
        Auto-graduate skills from a source that have been used enough.

        Skills that have been invoked at least ``min_invocations`` times
        are promoted to 'graduated' and will survive source removal.

        Args:
            source: The source to scan (e.g., 'plugin:calendar-sync')
            min_invocations: Minimum usage count to qualify

        Returns:
            List of graduated skill IDs.
        """
        graduated_ids: list[str] = []
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT skill_id FROM skills WHERE source = ? AND invocations >= ?",
                (source, min_invocations),
            ).fetchall()

            for (skill_id,) in rows:
                conn.execute(
                    "UPDATE skills SET source = 'graduated', updated_at = ? WHERE skill_id = ?",
                    (time.time(), skill_id),
                )
                graduated_ids.append(skill_id)

        if graduated_ids:
            logger.info(
                "Auto-graduated %d skills from '%s' (min_invocations=%d)",
                len(graduated_ids), source, min_invocations,
            )
        return graduated_ids

    # ─── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            cats = conn.execute("SELECT COUNT(DISTINCT category) FROM skills").fetchone()[0]
            avg_sr = conn.execute("SELECT AVG(success_rate) FROM skills").fetchone()[0]
            sources = conn.execute(
                "SELECT source, COUNT(*) FROM skills GROUP BY source"
            ).fetchall()
        return {
            "total_skills": total,
            "categories": cats,
            "avg_success_rate": round(avg_sr or 0, 3),
            "by_source": {s: c for s, c in sources},
        }
