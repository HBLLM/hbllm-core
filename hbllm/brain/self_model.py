"""
Self-Model — internal awareness of system capabilities and performance.

Tracks what the system is good at, where it's weak, and uses
this self-knowledge to make better routing and delegation decisions.

Integration:
- Feeds DecisionNode for model/tool selection
- Feeds GoalManager for self-improvement priorities
- Feeds MetaNode for strategic self-reflection
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CapabilityScore:
    """Score for a specific capability."""
    domain: str
    score: float  # 0.0 - 1.0
    sample_count: int
    trend: str  # improving | stable | declining
    last_updated: float


class SelfModel:
    """
    Internal model of system capabilities and performance.

    Tracks:
    - Domain expertise levels (coding, medical, legal, math, etc.)
    - Tool proficiency per tool type
    - Model performance per query category
    - Confidence calibration (predicted vs actual accuracy)
    - Latency profiles per operation type
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "self_model.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS capabilities (
                    domain TEXT PRIMARY KEY,
                    score REAL NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.5,
                    avg_latency_ms REAL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    latency_ms REAL DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_domain ON performance_log(domain)"
            )

    # ─── Recording ───────────────────────────────────────────────────

    def record_outcome(
        self,
        domain: str,
        success: bool,
        confidence: float = 0.5,
        latency_ms: float = 0.0,
    ) -> None:
        """Record an outcome to update capability scores."""
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            # Log event
            conn.execute(
                "INSERT INTO performance_log (domain, success, confidence, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (domain, int(success), confidence, latency_ms, now),
            )

            # Update capability
            row = conn.execute(
                "SELECT score, sample_count, successes, failures, avg_confidence, avg_latency_ms "
                "FROM capabilities WHERE domain = ?",
                (domain,),
            ).fetchone()

            if row:
                n = row[1] + 1
                s = row[2] + (1 if success else 0)
                f = row[3] + (0 if success else 1)
                new_score = s / n
                new_conf = (row[4] * row[1] + confidence) / n
                new_lat = (row[5] * row[1] + latency_ms) / n
                conn.execute(
                    "UPDATE capabilities SET score = ?, sample_count = ?, successes = ?, "
                    "failures = ?, avg_confidence = ?, avg_latency_ms = ?, updated_at = ? "
                    "WHERE domain = ?",
                    (new_score, n, s, f, new_conf, new_lat, now, domain),
                )
            else:
                conn.execute(
                    "INSERT INTO capabilities "
                    "(domain, score, sample_count, successes, failures, "
                    "avg_confidence, avg_latency_ms, created_at, updated_at) "
                    "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
                    (domain, 1.0 if success else 0.0,
                     1 if success else 0, 0 if success else 1,
                     confidence, latency_ms, now, now),
                )

    # ─── Querying ────────────────────────────────────────────────────

    def get_capability(self, domain: str) -> CapabilityScore | None:
        """Get capability score for a domain."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT domain, score, sample_count, updated_at FROM capabilities WHERE domain = ?",
                (domain,),
            ).fetchone()
        if not row:
            return None

        trend = self._compute_trend(domain)
        return CapabilityScore(
            domain=row[0], score=round(row[1], 3),
            sample_count=row[2], trend=trend, last_updated=row[3],
        )

    def get_all_capabilities(self) -> list[CapabilityScore]:
        """Get all capability scores sorted by score."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT domain, score, sample_count, updated_at FROM capabilities ORDER BY score DESC"
            ).fetchall()
        return [
            CapabilityScore(
                domain=r[0], score=round(r[1], 3), sample_count=r[2],
                trend=self._compute_trend(r[0]), last_updated=r[3],
            )
            for r in rows
        ]

    def get_strengths(self, min_score: float = 0.8, min_samples: int = 5) -> list[str]:
        """Get domains where the system excels."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT domain FROM capabilities WHERE score >= ? AND sample_count >= ? ORDER BY score DESC",
                (min_score, min_samples),
            ).fetchall()
        return [r[0] for r in rows]

    def get_weaknesses(self, max_score: float = 0.5, min_samples: int = 5) -> list[str]:
        """Get domains where the system is weak."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT domain FROM capabilities WHERE score <= ? AND sample_count >= ? ORDER BY score ASC",
                (max_score, min_samples),
            ).fetchall()
        return [r[0] for r in rows]

    def should_delegate(self, domain: str, threshold: float = 0.4) -> bool:
        """Check if a domain query should be delegated to a specialist."""
        cap = self.get_capability(domain)
        if cap is None:
            return False  # Unknown domain, try anyway
        return cap.score < threshold

    def recommend_model(self, domain: str) -> str:
        """Recommend which model to use based on domain capability."""
        cap = self.get_capability(domain)
        if cap is None or cap.score > 0.7:
            return "default"  # Use standard model
        if cap.score > 0.4:
            return "large"  # Use more capable model
        return "specialist"  # Use domain-specific model

    # ─── Trend Analysis ──────────────────────────────────────────────

    def _compute_trend(self, domain: str) -> str:
        """Compute performance trend for a domain."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT success FROM performance_log WHERE domain = ? "
                "ORDER BY created_at DESC LIMIT 20",
                (domain,),
            ).fetchall()

        if len(rows) < 6:
            return "stable"

        recent = [r[0] for r in rows[:10]]
        older = [r[0] for r in rows[10:20]]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else recent_avg

        if recent_avg > older_avg + 0.1:
            return "improving"
        if recent_avg < older_avg - 0.1:
            return "declining"
        return "stable"

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregate self-model metrics for GoalManager integration."""
        caps = self.get_all_capabilities()
        if not caps:
            return {}
        return {
            "total_domains": len(caps),
            "avg_score": round(sum(c.score for c in caps) / len(caps), 3),
            "strengths": self.get_strengths(),
            "weaknesses": self.get_weaknesses(),
            "improving": [c.domain for c in caps if c.trend == "improving"],
            "declining": [c.domain for c in caps if c.trend == "declining"],
        }
