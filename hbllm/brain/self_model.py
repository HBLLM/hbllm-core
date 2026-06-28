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

import json
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


@dataclass
class CapabilityProfile:
    capability: str
    confidence: float
    success_rate: float
    avg_cost: dict[str, float]  # avg_tokens, avg_latency_ms
    last_validated: float


@dataclass
class ExperienceRecord:
    capability: str
    executions_count: int
    validation_runs: list[dict[str, Any]]


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

    def _init_db(self) -> None:
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS capability_profiles (
                    capability TEXT PRIMARY KEY,
                    confidence REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_tokens REAL DEFAULT 0.0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    last_validated REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_records (
                    capability TEXT PRIMARY KEY,
                    executions_count INTEGER DEFAULT 0,
                    validation_runs TEXT DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_performance (
                    strategy TEXT,
                    domain TEXT,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    PRIMARY KEY (strategy, domain)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_domain ON performance_log(domain)")

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
                    (
                        domain,
                        1.0 if success else 0.0,
                        1 if success else 0,
                        0 if success else 1,
                        confidence,
                        latency_ms,
                        now,
                        now,
                    ),
                )

    def select_policy(self, domain: str) -> Any:
        """Select an optimized policy for a domain using an epsilon-greedy choice strategy."""
        import random

        from hbllm.brain.cognitive_state import (
            CognitiveBudget,
            CognitivePolicy,
            HierarchicalCognitivePolicy,
        )

        policies = {
            "direct": CognitivePolicy(
                reasoning_strategy="direct",
                simulation_depth=0,
                verification_budget=1,
                retrieval_budget=3,
                planner_type="chain",
                budget=CognitiveBudget(
                    simulation_budget=0, verification_budget=1, planning_budget=15.0
                ),
            ),
            "CoT": CognitivePolicy(
                reasoning_strategy="CoT",
                simulation_depth=1,
                verification_budget=2,
                retrieval_budget=5,
                planner_type="chain",
                budget=CognitiveBudget(
                    simulation_budget=3, verification_budget=2, planning_budget=30.0
                ),
            ),
            "GoT": CognitivePolicy(
                reasoning_strategy="GoT",
                simulation_depth=2,
                verification_budget=4,
                retrieval_budget=10,
                planner_type="graph",
                budget=CognitiveBudget(
                    simulation_budget=10, verification_budget=4, planning_budget=60.0
                ),
            ),
        }

        # Exploration vs Exploitation
        if random.random() < 0.15:
            selected_key = random.choice(list(policies.keys()))
            return HierarchicalCognitivePolicy(global_policy=policies[selected_key])

        # Query past success rates
        best_strategy = "CoT"
        max_rate = -1.0

        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT strategy, successes, failures FROM policy_performance WHERE domain = ?",
                (domain,),
            ).fetchall()

        for row in rows:
            strategy, successes, failures = row
            total = successes + failures
            if total > 0:
                rate = successes / total
                if rate > max_rate:
                    max_rate = rate
                    best_strategy = strategy

        return HierarchicalCognitivePolicy(
            global_policy=policies.get(best_strategy, policies["CoT"])
        )

    def record_policy_outcome(
        self, policy: Any, domain: str, success: bool, cost_metrics: dict[str, float]
    ) -> None:
        """Update policy success/failure counts and average execution latencies."""
        strategy = getattr(policy, "reasoning_strategy", "direct")
        latency = cost_metrics.get("latency_ms", 0.0)

        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT successes, failures, avg_latency_ms FROM policy_performance WHERE strategy = ? AND domain = ?",
                (strategy, domain),
            ).fetchone()

            if row:
                s = row[0] + (1 if success else 0)
                f = row[1] + (0 if success else 1)
                total = s + f
                new_lat = (row[2] * (total - 1) + latency) / total
                conn.execute(
                    "UPDATE policy_performance SET successes = ?, failures = ?, avg_latency_ms = ? WHERE strategy = ? AND domain = ?",
                    (s, f, new_lat, strategy, domain),
                )
            else:
                conn.execute(
                    "INSERT INTO policy_performance (strategy, domain, successes, failures, avg_latency_ms) VALUES (?, ?, ?, ?, ?)",
                    (strategy, domain, 1 if success else 0, 0 if success else 1, latency),
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
            domain=row[0],
            score=round(row[1], 3),
            sample_count=row[2],
            trend=trend,
            last_updated=row[3],
        )

    def get_all_capabilities(self) -> list[CapabilityScore]:
        """Get all capability scores sorted by score."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT domain, score, sample_count, updated_at FROM capabilities ORDER BY score DESC"
            ).fetchall()
        return [
            CapabilityScore(
                domain=r[0],
                score=round(r[1], 3),
                sample_count=r[2],
                trend=self._compute_trend(r[0]),
                last_updated=r[3],
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

    # ─── Competence Tracking (Phase 5) ──────────────────────────────

    def get_capability_profile(self, capability: str) -> CapabilityProfile | None:
        """Get capability profile."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM capability_profiles WHERE capability = ?",
                (capability,),
            ).fetchone()
        if not row:
            return None
        return CapabilityProfile(
            capability=row["capability"],
            confidence=row["confidence"],
            success_rate=row["success_rate"],
            avg_cost={
                "avg_tokens": row["avg_tokens"],
                "avg_latency_ms": row["avg_latency_ms"],
            },
            last_validated=row["last_validated"],
        )

    def get_experience_record(self, capability: str) -> ExperienceRecord | None:
        """Get experience record."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experience_records WHERE capability = ?",
                (capability,),
            ).fetchone()
        if not row:
            return None
        return ExperienceRecord(
            capability=row["capability"],
            executions_count=row["executions_count"],
            validation_runs=json.loads(row["validation_runs"] or "[]"),
        )

    def record_experience(
        self,
        capability: str,
        success: bool,
        tokens_used: int,
        latency_ms: float,
        validation_info: dict[str, Any] | None = None,
    ) -> None:
        """Record capability execution experience and update capability profile."""
        now = time.time()

        # Update experience record
        rec = self.get_experience_record(capability)
        if rec:
            count = rec.executions_count + 1
            runs = rec.validation_runs
            if validation_info:
                runs.append(validation_info)
        else:
            count = 1
            runs = [validation_info] if validation_info else []

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experience_records (capability, executions_count, validation_runs) VALUES (?, ?, ?)",
                (capability, count, json.dumps(runs)),
            )

        # Update capability profile
        prof = self.get_capability_profile(capability)
        if prof:
            new_success_rate = (prof.success_rate * (count - 1) + (1.0 if success else 0.0)) / count
            new_tokens = (prof.avg_cost.get("avg_tokens", 0.0) * (count - 1) + tokens_used) / count
            new_latency = (
                prof.avg_cost.get("avg_latency_ms", 0.0) * (count - 1) + latency_ms
            ) / count
            new_confidence = (prof.confidence * (count - 1) + (1.0 if success else 0.0)) / count
        else:
            new_success_rate = 1.0 if success else 0.0
            new_tokens = float(tokens_used)
            new_latency = latency_ms
            new_confidence = 0.8  # initial baseline confidence

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO capability_profiles (capability, confidence, success_rate, avg_tokens, avg_latency_ms, last_validated) VALUES (?, ?, ?, ?, ?, ?)",
                (capability, new_confidence, new_success_rate, new_tokens, new_latency, now),
            )
