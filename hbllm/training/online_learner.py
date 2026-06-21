"""Online Learner — continuous model improvement from user interactions.

Implements lightweight online learning techniques:
    1. Few-shot example accumulation (learns from corrections)
    2. Preference tuning (adapts to user style from feedback)
    3. Domain expertise tracking (tracks which domains user uses most)
    4. Tool success rate learning (avoids failing tools)

Does NOT fine-tune model weights. Instead, builds a dynamic prompt
context layer that steers behavior based on accumulated experience.

Architecture:
    - SQLite-backed learning store
    - Records user corrections, feedback, and tool outcomes
    - Generates a "learnings context" block for prompt injection
    - Periodic consolidation into compact preference summaries
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
class LearningEntry:
    """A single learned piece of information."""

    id: int = 0
    tenant_id: str = "default"
    category: str = ""  # "correction", "preference", "tool_outcome", "domain_usage"
    key: str = ""  # Lookup key (e.g., tool name, domain, topic)
    value: str = ""  # The learned information
    confidence: float = 1.0
    reinforcement_count: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class OnlineLearner:
    """Continuous learning engine for user adaptation.

    Learns from:
        - Corrections ("Actually, I meant X not Y")
        - Feedback (thumbs up/down, ratings)
        - Tool outcomes (success/failure rates)
        - Usage patterns (domain frequency)

    Produces:
        - Dynamic prompt context injections
        - Tool selection biases
        - Communication style adjustments

    Usage::

        learner = OnlineLearner(db_path="data/online_learning.db")
        await learner.init_db()

        # Record a correction
        learner.record_correction("user1", "Use Celsius not Fahrenheit")

        # Record tool outcome
        learner.record_tool_outcome("user1", "web_search", success=True)

        # Get learnings for prompt injection
        context = learner.get_learnings_context("user1")
    """

    def __init__(
        self,
        db_path: str | Path = "data/online_learning.db",
        max_entries_per_category: int = 100,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_entries = max_entries_per_category

    async def init_db(self) -> None:
        """Create the learning database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    reinforcement_count INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learnings_tenant_cat
                ON learnings(tenant_id, category)
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_learnings_tenant_cat_key
                ON learnings(tenant_id, category, key)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL,
                    timestamp REAL NOT NULL,
                    error_type TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_outcomes_tenant
                ON tool_outcomes(tenant_id, tool_name)
            """)
            conn.commit()
        finally:
            conn.close()
        logger.debug("OnlineLearner initialized at %s", self.db_path)

    def record_correction(
        self,
        tenant_id: str,
        correction: str,
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a user correction (e.g., 'Use Celsius not Fahrenheit')."""
        key = correction[:100]  # Truncate for key
        self._upsert_learning(
            tenant_id=tenant_id,
            category="correction",
            key=key,
            value=correction,
            metadata={"context": context, **(metadata or {})},
        )

    def record_preference(
        self,
        tenant_id: str,
        preference_key: str,
        preference_value: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a user preference (e.g., 'response_style' → 'concise')."""
        self._upsert_learning(
            tenant_id=tenant_id,
            category="preference",
            key=preference_key,
            value=preference_value,
            metadata=metadata,
        )

    def record_tool_outcome(
        self,
        tenant_id: str,
        tool_name: str,
        success: bool,
        latency_ms: float | None = None,
        error_type: str | None = None,
    ) -> None:
        """Record the outcome of a tool invocation."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO tool_outcomes "
                "(tenant_id, tool_name, success, latency_ms, timestamp, error_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tenant_id, tool_name, int(success), latency_ms, time.time(), error_type),
            )
            conn.commit()
        finally:
            conn.close()

    def record_domain_usage(
        self,
        tenant_id: str,
        domain: str,
    ) -> None:
        """Record usage of a domain (for expertise tracking)."""
        self._upsert_learning(
            tenant_id=tenant_id,
            category="domain_usage",
            key=domain,
            value=domain,
        )

    def get_learnings_context(
        self,
        tenant_id: str,
        max_items: int = 15,
    ) -> str:
        """Generate a prompt-injectable learnings context block.

        Returns a concise text block summarizing what we've learned
        about this user's preferences and corrections.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            lines: list[str] = []

            # Corrections (most reinforced first)
            corrections = conn.execute(
                "SELECT value, reinforcement_count FROM learnings "
                "WHERE tenant_id = ? AND category = 'correction' "
                "ORDER BY reinforcement_count DESC, updated_at DESC LIMIT ?",
                (tenant_id, 5),
            ).fetchall()
            if corrections:
                lines.append("User corrections:")
                for val, count in corrections:
                    lines.append(f"  - {val}" + (f" (×{count})" if count > 1 else ""))

            # Preferences
            prefs = conn.execute(
                "SELECT key, value FROM learnings "
                "WHERE tenant_id = ? AND category = 'preference' "
                "ORDER BY updated_at DESC LIMIT ?",
                (tenant_id, 5),
            ).fetchall()
            if prefs:
                lines.append("User preferences:")
                for key, val in prefs:
                    lines.append(f"  - {key}: {val}")

            # Tool success rates
            tool_stats = conn.execute(
                "SELECT tool_name, "
                "SUM(success) as successes, COUNT(*) as total, "
                "AVG(latency_ms) as avg_latency "
                "FROM tool_outcomes WHERE tenant_id = ? "
                "GROUP BY tool_name HAVING total >= 3 "
                "ORDER BY total DESC LIMIT 5",
                (tenant_id,),
            ).fetchall()
            if tool_stats:
                lines.append("Tool reliability:")
                for name, successes, total, avg_lat in tool_stats:
                    rate = successes / total * 100
                    lat_str = f", avg {avg_lat:.0f}ms" if avg_lat else ""
                    lines.append(f"  - {name}: {rate:.0f}% success ({total} uses{lat_str})")

            return "\n".join(lines) if lines else ""

        finally:
            conn.close()

    def get_tool_success_rate(
        self,
        tenant_id: str,
        tool_name: str,
    ) -> float | None:
        """Get the success rate for a specific tool (0.0-1.0)."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT SUM(success), COUNT(*) FROM tool_outcomes "
                "WHERE tenant_id = ? AND tool_name = ?",
                (tenant_id, tool_name),
            ).fetchone()
            if row and row[1] > 0:
                return row[0] / row[1]
            return None
        finally:
            conn.close()

    def _upsert_learning(
        self,
        tenant_id: str,
        category: str,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a learning entry (reinforcing on conflict)."""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO learnings "
                "(tenant_id, category, key, value, confidence, reinforcement_count, "
                "created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, 1.0, 1, ?, ?, ?) "
                "ON CONFLICT(tenant_id, category, key) DO UPDATE SET "
                "value = excluded.value, "
                "reinforcement_count = reinforcement_count + 1, "
                "confidence = MIN(1.0, confidence + 0.1), "
                "updated_at = excluded.updated_at",
                (
                    tenant_id,
                    category,
                    key,
                    value,
                    now,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self, tenant_id: str = "default") -> dict[str, Any]:
        """Learning statistics for a tenant."""
        conn = sqlite3.connect(self.db_path)
        try:
            by_category = dict(
                conn.execute(
                    "SELECT category, COUNT(*) FROM learnings "
                    "WHERE tenant_id = ? GROUP BY category",
                    (tenant_id,),
                ).fetchall()
            )
            tool_count = conn.execute(
                "SELECT COUNT(DISTINCT tool_name) FROM tool_outcomes WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()[0]
            return {
                "learnings_by_category": by_category,
                "tools_tracked": tool_count,
            }
        finally:
            conn.close()
