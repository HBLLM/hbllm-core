"""
Mechanism Memory System — first-class reusable units of intelligence.

Mechanisms are the true cognitive primitives of HBLLM.  Skills become
"applications of mechanisms" rather than standalone procedures.

Example:
    "Install PostgreSQL", "Install Redis", and "Install Docker"
    all share the same mechanism: "Package Manager Dependency Resolution".

    Instead of recording three separate skills, the system discovers
    the underlying mechanism and reuses it, enabling transfer learning.

Hierarchy:
    abstraction_level 0: Concrete (single-task specific)
    abstraction_level 1: Domain (reusable within a domain)
    abstraction_level 2: Cross-domain (universal cognitive primitive)

Storage:
    SQLite-backed for durability.  Mechanisms are addressed by ID and
    searched by precondition matching or description similarity.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Mechanism:
    """A first-class reusable unit of intelligence.

    Mechanisms capture *why* something works, not just *what* to do.
    They are the atoms of understanding that skills combine.
    """

    id: str
    description: str
    preconditions: list[str]  # When does this mechanism apply?
    process_steps: list[str]  # How does it work?
    expected_outcomes: list[str]  # What should happen?
    confidence: float = 0.8  # Belief confidence (decays, reinforced)
    abstraction_level: int = 0  # 0=concrete, 1=domain, 2=cross-domain
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    is_core: bool = False  # Promoted during sleep consolidation
    domain: str = "general"
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    parent_mechanism_id: str | None = None  # If abstracted from a concrete one
    related_mechanism_ids: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "preconditions": self.preconditions,
            "process_steps": self.process_steps,
            "expected_outcomes": self.expected_outcomes,
            "confidence": self.confidence,
            "abstraction_level": self.abstraction_level,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "is_core": self.is_core,
            "domain": self.domain,
        }


class MechanismStore:
    """SQLite-backed store for first-class mechanisms.

    Mechanisms are the reusable units of intelligence that skills compose.
    This store provides CRUD, search, confidence management, and promotion.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "mechanisms.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mechanisms (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    preconditions TEXT NOT NULL DEFAULT '[]',
                    process_steps TEXT NOT NULL DEFAULT '[]',
                    expected_outcomes TEXT NOT NULL DEFAULT '[]',
                    confidence REAL DEFAULT 0.8,
                    abstraction_level INTEGER DEFAULT 0,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    is_core INTEGER DEFAULT 0,
                    domain TEXT DEFAULT 'general',
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    parent_mechanism_id TEXT,
                    related_mechanism_ids TEXT DEFAULT '[]'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mech_domain ON mechanisms(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mech_core ON mechanisms(is_core)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mech_confidence ON mechanisms(confidence)")

    # ─── Create ──────────────────────────────────────────────────────

    def store(self, mechanism: Mechanism) -> None:
        """Store or update a mechanism."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mechanisms
                (id, description, preconditions, process_steps, expected_outcomes,
                 confidence, abstraction_level, usage_count, success_count, failure_count,
                 is_core, domain, created_at, last_used_at,
                 parent_mechanism_id, related_mechanism_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mechanism.id,
                    mechanism.description,
                    json.dumps(mechanism.preconditions),
                    json.dumps(mechanism.process_steps),
                    json.dumps(mechanism.expected_outcomes),
                    mechanism.confidence,
                    mechanism.abstraction_level,
                    mechanism.usage_count,
                    mechanism.success_count,
                    mechanism.failure_count,
                    1 if mechanism.is_core else 0,
                    mechanism.domain,
                    mechanism.created_at,
                    mechanism.last_used_at,
                    mechanism.parent_mechanism_id,
                    json.dumps(mechanism.related_mechanism_ids),
                ),
            )

    def create(
        self,
        description: str,
        preconditions: list[str],
        process_steps: list[str],
        expected_outcomes: list[str],
        domain: str = "general",
        abstraction_level: int = 0,
        confidence: float = 0.8,
    ) -> Mechanism:
        """Create and store a new mechanism. Returns the created Mechanism."""
        mechanism = Mechanism(
            id=f"mech_{uuid.uuid4().hex[:12]}",
            description=description,
            preconditions=preconditions,
            process_steps=process_steps,
            expected_outcomes=expected_outcomes,
            domain=domain,
            abstraction_level=abstraction_level,
            confidence=confidence,
        )
        self.store(mechanism)
        logger.info(
            "Created mechanism '%s' (domain=%s, level=%d)",
            mechanism.id,
            domain,
            abstraction_level,
        )
        return mechanism

    # ─── Read ────────────────────────────────────────────────────────

    def get(self, mechanism_id: str) -> Mechanism | None:
        """Retrieve a mechanism by ID."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM mechanisms WHERE id = ?", (mechanism_id,)).fetchone()
        return self._row_to_mechanism(row) if row else None

    def find_by_domain(
        self, domain: str, min_confidence: float = 0.0, limit: int = 50
    ) -> list[Mechanism]:
        """Find mechanisms in a specific domain."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM mechanisms WHERE domain = ? AND confidence >= ? "
                "ORDER BY confidence DESC, usage_count DESC LIMIT ?",
                (domain, min_confidence, limit),
            ).fetchall()
        return [self._row_to_mechanism(r) for r in rows]

    def find_by_preconditions(
        self, situation_keywords: list[str], domain: str | None = None, limit: int = 10
    ) -> list[Mechanism]:
        """Find mechanisms whose preconditions match the current situation.

        Uses keyword overlap scoring — lightweight enough for query-time use.
        No LLM needed.
        """
        query = "SELECT * FROM mechanisms WHERE confidence > 0.1"
        params: list[Any] = []
        if domain:
            query += " AND (domain = ? OR domain = 'general')"
            params.append(domain)
        query += " ORDER BY confidence DESC, usage_count DESC LIMIT 200"

        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return []

        situation_set = {w.lower() for w in situation_keywords}
        scored: list[tuple[float, Mechanism]] = []

        for row in rows:
            mechanism = self._row_to_mechanism(row)
            # Score by precondition overlap
            precond_words = {w.lower() for p in mechanism.preconditions for w in p.split()}
            overlap = len(situation_set & precond_words)
            if overlap > 0:
                # Weight by confidence and abstraction level
                score = overlap * mechanism.confidence * (1 + mechanism.abstraction_level * 0.3)
                scored.append((score, mechanism))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def get_core_mechanisms(self) -> list[Mechanism]:
        """Get all promoted core mechanisms (cognitive primitives)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM mechanisms WHERE is_core = 1 ORDER BY usage_count DESC"
            ).fetchall()
        return [self._row_to_mechanism(r) for r in rows]

    def list_all(self, limit: int = 500) -> list[Mechanism]:
        """Get all mechanisms, ordered by confidence descending."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM mechanisms ORDER BY confidence DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_mechanism(r) for r in rows]

    # ─── Update ──────────────────────────────────────────────────────

    def record_usage(self, mechanism_id: str, success: bool) -> None:
        """Record a mechanism usage for tracking."""
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            if success:
                conn.execute(
                    "UPDATE mechanisms SET usage_count = usage_count + 1, "
                    "success_count = success_count + 1, "
                    "confidence = MIN(1.0, confidence + 0.02), "
                    "last_used_at = ? WHERE id = ?",
                    (now, mechanism_id),
                )
            else:
                conn.execute(
                    "UPDATE mechanisms SET usage_count = usage_count + 1, "
                    "failure_count = failure_count + 1, "
                    "confidence = MAX(0.0, confidence - 0.1), "
                    "last_used_at = ? WHERE id = ?",
                    (now, mechanism_id),
                )

    def reinforce(self, mechanism_id: str, confidence_boost: float = 0.05) -> None:
        """Reinforce a mechanism's confidence (e.g. after evidence confirms it)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE mechanisms SET confidence = MIN(1.0, confidence + ?) WHERE id = ?",
                (confidence_boost, mechanism_id),
            )

    def decay_confidence(self, rate: float = 0.01) -> int:
        """Decay all non-core mechanism confidence. Returns count affected."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE mechanisms SET confidence = MAX(0.0, confidence - ?) "
                "WHERE is_core = 0 AND confidence > 0.0",
                (rate,),
            )
            return cursor.rowcount

    def promote_to_core(self, mechanism_id: str) -> bool:
        """Promote a mechanism to core status (cognitive primitive)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE mechanisms SET is_core = 1 WHERE id = ?",
                (mechanism_id,),
            )
            promoted = cursor.rowcount > 0
        if promoted:
            logger.info("Promoted mechanism to core: %s", mechanism_id)
        return promoted

    def find_promotable(
        self, min_usage: int = 10, min_success_rate: float = 0.9
    ) -> list[Mechanism]:
        """Find mechanisms ready for promotion to core status.

        Criteria:
        - Used at least `min_usage` times
        - Success rate >= `min_success_rate`
        - Not already core
        - Confidence >= 0.8
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM mechanisms WHERE is_core = 0 "
                "AND usage_count >= ? AND confidence >= 0.8 "
                "AND (CAST(success_count AS REAL) / MAX(1, success_count + failure_count)) >= ?",
                (min_usage, min_success_rate),
            ).fetchall()
        return [self._row_to_mechanism(r) for r in rows]

    # ─── Delete ──────────────────────────────────────────────────────

    def prune_weak(self, max_confidence: float = 0.1) -> int:
        """Remove mechanisms with very low confidence (failed beliefs)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM mechanisms WHERE confidence <= ? AND is_core = 0",
                (max_confidence,),
            )
            count = cursor.rowcount
        if count:
            logger.info("Pruned %d weak mechanisms (confidence <= %.2f)", count, max_confidence)
        return count

    def get_weak(self, threshold: float = 0.3) -> list[Mechanism]:
        """Find low-confidence mechanisms for review."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM mechanisms WHERE confidence <= ? AND is_core = 0 "
                "ORDER BY confidence ASC LIMIT 50",
                (threshold,),
            ).fetchall()
        return [self._row_to_mechanism(r) for r in rows]

    # ─── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0]
            core = conn.execute("SELECT COUNT(*) FROM mechanisms WHERE is_core = 1").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM mechanisms").fetchone()[0]
            domains = conn.execute(
                "SELECT domain, COUNT(*) FROM mechanisms GROUP BY domain"
            ).fetchall()
            by_level = conn.execute(
                "SELECT abstraction_level, COUNT(*) FROM mechanisms GROUP BY abstraction_level"
            ).fetchall()
        return {
            "total_mechanisms": total,
            "core_mechanisms": core,
            "avg_confidence": round(avg_conf or 0, 3),
            "by_domain": {d: c for d, c in domains},
            "by_abstraction_level": {l: c for l, c in by_level},
        }

    # ─── Helpers ─────────────────────────────────────────────────────

    def _row_to_mechanism(self, row: tuple[Any, ...]) -> Mechanism:
        return Mechanism(
            id=row[0],
            description=row[1],
            preconditions=json.loads(row[2]),
            process_steps=json.loads(row[3]),
            expected_outcomes=json.loads(row[4]),
            confidence=row[5],
            abstraction_level=row[6],
            usage_count=row[7],
            success_count=row[8],
            failure_count=row[9],
            is_core=bool(row[10]),
            domain=row[11],
            created_at=row[12],
            last_used_at=row[13],
            parent_mechanism_id=row[14],
            related_mechanism_ids=json.loads(row[15]) if row[15] else [],
        )
