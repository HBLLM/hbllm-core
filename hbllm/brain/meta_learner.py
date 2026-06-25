"""Meta Learner — learns how to learn.

Tracks learning session outcomes per domain, optimizes learning strategies
for cost-efficiency (confidence gained per resource spent), and recommends
the best next action based on historical performance.

Most AI systems learn content. Few learn which learning methods work best.
This module closes that gap.
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


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class LearningSession:
    """Record of a single learning session."""

    session_id: str = ""
    domain: str = ""
    method: str = ""  # "research" | "experiment" | "review" | "external_llm"
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    resource_cost: float = 1.0
    retention_score: float | None = None  # Tested later during sleep
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def confidence_gain(self) -> float:
        return self.confidence_after - self.confidence_before

    @property
    def cost_efficiency(self) -> float:
        """Confidence gained per unit of resource cost."""
        if self.resource_cost <= 0:
            return 0.0
        return self.confidence_gain / self.resource_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "domain": self.domain,
            "method": self.method,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "resource_cost": self.resource_cost,
            "retention_score": self.retention_score,
            "duration_s": self.duration_s,
            "timestamp": self.timestamp,
        }


@dataclass
class LearningStrategy:
    """Optimized learning strategy for a domain."""

    domain: str = ""
    best_source_types: list[str] = field(default_factory=list)
    optimal_depth: str = "intermediate"
    retention_rate: float = 0.5
    experiment_effectiveness: float = 0.5
    preferred_evaluation: str = "prediction"
    total_sessions: int = 0
    avg_confidence_gain: float = 0.0
    avg_cost_per_session: float = 1.0
    cost_efficiency: float = 0.0  # confidence_gain / cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "best_source_types": self.best_source_types,
            "optimal_depth": self.optimal_depth,
            "retention_rate": self.retention_rate,
            "experiment_effectiveness": self.experiment_effectiveness,
            "preferred_evaluation": self.preferred_evaluation,
            "total_sessions": self.total_sessions,
            "avg_confidence_gain": self.avg_confidence_gain,
            "avg_cost_per_session": self.avg_cost_per_session,
            "cost_efficiency": self.cost_efficiency,
        }


# ── Default Strategy ─────────────────────────────────────────────────────────

_DEFAULT_STRATEGY = LearningStrategy(
    domain="default",
    best_source_types=["research", "experiment", "external_llm"],
    optimal_depth="intermediate",
    retention_rate=0.5,
    experiment_effectiveness=0.5,
    preferred_evaluation="prediction",
    total_sessions=0,
    avg_confidence_gain=0.0,
    avg_cost_per_session=1.0,
    cost_efficiency=0.0,
)


# ── Meta Learner ─────────────────────────────────────────────────────────────


class MetaLearner:
    """Learns how to learn — optimizes learning strategies per domain.

    Tracks:
    - Which learning methods work best for each domain
    - Cost-efficiency (confidence gained per resource spent)
    - Retention rates (how well knowledge sticks)
    - Experiment effectiveness

    Recommends:
    - Optimal next action based on current confidence + domain strategy
    - Budget allocation across learning methods
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.data_dir / "meta_learner.db"
        self._init_db()

        # In-memory strategy cache
        self._strategies: dict[str, LearningStrategy] = {}
        self._load_strategies()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    method TEXT NOT NULL,
                    confidence_before REAL,
                    confidence_after REAL,
                    resource_cost REAL DEFAULT 1.0,
                    retention_score REAL,
                    duration_s REAL DEFAULT 0.0,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ls_domain ON learning_sessions(domain)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    domain TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

    def _load_strategies(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute("SELECT * FROM strategies"):
                    data = json.loads(row["data"])
                    strategy = LearningStrategy(
                        **{
                            k: v
                            for k, v in data.items()
                            if k in LearningStrategy.__dataclass_fields__
                        }
                    )
                    self._strategies[strategy.domain] = strategy
        except Exception as e:
            logger.debug("Failed to load strategies: %s", e)

    def _persist_strategy(self, strategy: LearningStrategy) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO strategies
                       (domain, data, updated_at)
                       VALUES (?, ?, ?)""",
                    (strategy.domain, json.dumps(strategy.to_dict()), time.time()),
                )
        except Exception as e:
            logger.warning("Failed to persist strategy: %s", e)

    # ── Core API ─────────────────────────────────────────────────────────

    async def record_session(
        self,
        domain: str,
        method: str,
        confidence_before: float,
        confidence_after: float,
        resource_cost: float = 1.0,
        retention_test_score: float | None = None,
        duration_s: float = 0.0,
    ) -> LearningSession:
        """Record a learning session outcome."""
        session = LearningSession(
            session_id=f"ls_{uuid.uuid4().hex[:12]}",
            domain=domain,
            method=method,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            resource_cost=resource_cost,
            retention_score=retention_test_score,
            duration_s=duration_s,
        )

        # Persist session
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO learning_sessions
                       (session_id, domain, method, confidence_before,
                        confidence_after, resource_cost, retention_score,
                        duration_s, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session.session_id,
                        session.domain,
                        session.method,
                        session.confidence_before,
                        session.confidence_after,
                        session.resource_cost,
                        session.retention_score,
                        session.duration_s,
                        session.timestamp,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist learning session: %s", e)

        # Update strategy for this domain
        self._update_strategy(domain)

        logger.info(
            "Recorded learning session: domain='%s' method='%s' "
            "gain=%.3f cost=%.1f efficiency=%.3f",
            domain,
            method,
            session.confidence_gain,
            session.resource_cost,
            session.cost_efficiency,
        )
        return session

    def get_strategy(self, domain: str) -> LearningStrategy:
        """Get optimal learning strategy for a domain.

        Returns domain-specific strategy if available, otherwise default.
        """
        return self._strategies.get(domain, _DEFAULT_STRATEGY)

    def recommend_next_action(
        self,
        domain: str,
        current_confidence: float,
        remaining_budget: dict[str, int] | None = None,
    ) -> str:
        """Recommend the best next action to learn this domain.

        Optimizes for confidence gained per resource spent.

        Returns: "research" | "experiment" | "review" | "deepen" | "test"
        """
        strategy = self.get_strategy(domain)

        # Adaptive depth based on confidence
        if current_confidence < 0.3:
            # Beginner: focus on research
            return "research"
        elif current_confidence < 0.6:
            # Intermediate: mix of research + experiment
            if strategy.experiment_effectiveness > 0.4:
                return "experiment"
            return "research"
        elif current_confidence < 0.8:
            # Advanced: experimentation + testing
            if strategy.experiment_effectiveness > 0.3:
                return "experiment"
            return "deepen"
        else:
            # Expert: test + review to maintain
            return "test"

    def get_method_effectiveness(self, domain: str) -> dict[str, float]:
        """Get effectiveness of each learning method for a domain."""
        effectiveness: dict[str, list[float]] = {}

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT method, confidence_before, confidence_after, "
                    "resource_cost FROM learning_sessions WHERE domain = ?",
                    (domain,),
                ).fetchall()

                for row in rows:
                    method = row["method"]
                    gain = row["confidence_after"] - row["confidence_before"]
                    cost = row["resource_cost"] or 1.0
                    efficiency = gain / cost

                    if method not in effectiveness:
                        effectiveness[method] = []
                    effectiveness[method].append(efficiency)
        except Exception:
            pass

        return {
            method: sum(vals) / len(vals) if vals else 0.0 for method, vals in effectiveness.items()
        }

    def stats(self) -> dict[str, Any]:
        """Return meta-learner statistics."""
        total_sessions = 0
        total_gain = 0.0
        total_cost = 0.0

        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*), SUM(confidence_after - confidence_before), "
                    "SUM(resource_cost) FROM learning_sessions"
                ).fetchone()
                if row:
                    total_sessions = row[0] or 0
                    total_gain = row[1] or 0.0
                    total_cost = row[2] or 0.0
        except Exception:
            pass

        return {
            "total_sessions": total_sessions,
            "total_confidence_gain": total_gain,
            "total_resource_cost": total_cost,
            "strategies_count": len(self._strategies),
            "overall_efficiency": total_gain / total_cost if total_cost > 0 else 0.0,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _update_strategy(self, domain: str) -> None:
        """Recompute strategy for a domain from session history."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                sessions = conn.execute(
                    "SELECT * FROM learning_sessions WHERE domain = ? "
                    "ORDER BY timestamp DESC LIMIT 50",
                    (domain,),
                ).fetchall()

            if not sessions:
                return

            # Compute method effectiveness
            method_gains: dict[str, list[float]] = {}
            method_costs: dict[str, list[float]] = {}
            retention_scores: list[float] = []
            total_gain = 0.0
            total_cost = 0.0

            for s in sessions:
                method = s["method"]
                gain = s["confidence_after"] - s["confidence_before"]
                cost = s["resource_cost"] or 1.0
                total_gain += gain
                total_cost += cost

                if method not in method_gains:
                    method_gains[method] = []
                    method_costs[method] = []
                method_gains[method].append(gain)
                method_costs[method].append(cost)

                if s["retention_score"] is not None:
                    retention_scores.append(s["retention_score"])

            # Rank methods by cost-efficiency
            method_efficiency: dict[str, float] = {}
            for method in method_gains:
                avg_gain = sum(method_gains[method]) / len(method_gains[method])
                avg_cost = sum(method_costs[method]) / len(method_costs[method])
                method_efficiency[method] = avg_gain / avg_cost if avg_cost > 0 else 0.0

            best_methods = sorted(
                method_efficiency, key=lambda m: method_efficiency[m], reverse=True
            )

            # Compute experiment effectiveness
            exp_gains = method_gains.get("experiment", [])
            experiment_eff = sum(exp_gains) / len(exp_gains) if exp_gains else 0.0

            # Determine optimal depth from confidence distribution
            avg_conf_after = sum(s["confidence_after"] for s in sessions) / len(sessions)
            if avg_conf_after < 0.3:
                depth = "beginner"
            elif avg_conf_after < 0.8:
                depth = "intermediate"
            else:
                depth = "advanced"

            strategy = LearningStrategy(
                domain=domain,
                best_source_types=best_methods[:3],
                optimal_depth=depth,
                retention_rate=(
                    sum(retention_scores) / len(retention_scores) if retention_scores else 0.5
                ),
                experiment_effectiveness=experiment_eff,
                preferred_evaluation="prediction",
                total_sessions=len(sessions),
                avg_confidence_gain=total_gain / len(sessions),
                avg_cost_per_session=total_cost / len(sessions),
                cost_efficiency=total_gain / total_cost if total_cost > 0 else 0.0,
            )

            self._strategies[domain] = strategy
            self._persist_strategy(strategy)

        except Exception as e:
            logger.warning("Failed to update strategy for '%s': %s", domain, e)
