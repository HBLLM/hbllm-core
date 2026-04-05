"""
Cognitive Metrics — live tracking of reasoning quality, hallucination rate, and tool success.

Provides real-time cognitive performance metrics for:
- MetaNode self-reflection
- GoalManager auto-goal generation
- SelfModel capability tracking
- Dashboard monitoring
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
class CognitiveSnapshot:
    """Point-in-time cognitive performance snapshot."""

    timestamp: float
    reasoning_score: float
    hallucination_rate: float
    tool_success_rate: float
    memory_hit_rate: float
    avg_confidence: float
    avg_latency_ms: float
    total_queries: int


class CognitiveMetrics:
    """
    Live cognitive performance tracking.

    Metrics tracked:
    1. Reasoning score — accuracy on complex queries
    2. Hallucination rate — factual errors detected
    3. Tool success rate — tools invoked successfully
    4. Memory hit rate — useful context retrieved
    5. Confidence calibration — predicted vs actual accuracy
    6. Latency — response generation time
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "cognitive_metrics.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    context TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_metric ON events(metric, created_at)"
            )

    # ─── Recording ───────────────────────────────────────────────────

    def record(self, metric: str, value: float, context: str = "") -> None:
        """Record a single metric event."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO events (metric, value, context, created_at) VALUES (?, ?, ?, ?)",
                (metric, value, context, time.time()),
            )

    def record_reasoning(self, score: float, context: str = "") -> None:
        self.record("reasoning_score", score, context)

    def record_hallucination(self, detected: bool, context: str = "") -> None:
        self.record("hallucination", 1.0 if detected else 0.0, context)

    def record_tool_result(self, success: bool, tool: str = "") -> None:
        self.record("tool_success", 1.0 if success else 0.0, tool)

    def record_memory_hit(self, useful: bool, context: str = "") -> None:
        self.record("memory_hit", 1.0 if useful else 0.0, context)

    def record_confidence(self, predicted: float, actual: float) -> None:
        self.record("confidence_error", abs(predicted - actual))

    def record_latency(self, latency_ms: float, operation: str = "") -> None:
        self.record("latency_ms", latency_ms, operation)

    # ─── Querying ────────────────────────────────────────────────────

    def get_metric(self, metric: str, hours: int = 24) -> dict[str, float]:
        """Get stats for a metric over the last N hours."""
        cutoff = time.time() - hours * 3600
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT value FROM events WHERE metric = ? AND created_at >= ?",
                (metric, cutoff),
            ).fetchall()

        values = [r[0] for r in rows]
        if not values:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    def snapshot(self, hours: int = 24) -> CognitiveSnapshot:
        """Get a complete cognitive performance snapshot."""
        reasoning = self.get_metric("reasoning_score", hours)
        hallucination = self.get_metric("hallucination", hours)
        tool = self.get_metric("tool_success", hours)
        memory = self.get_metric("memory_hit", hours)
        confidence = self.get_metric("confidence_error", hours)
        latency = self.get_metric("latency_ms", hours)

        return CognitiveSnapshot(
            timestamp=time.time(),
            reasoning_score=reasoning["avg"],
            hallucination_rate=hallucination["avg"],
            tool_success_rate=tool["avg"],
            memory_hit_rate=memory["avg"],
            avg_confidence=1.0 - confidence["avg"],  # invert error to score
            avg_latency_ms=latency["avg"],
            total_queries=int(reasoning["count"] + tool["count"]),
        )

    def get_trend(self, metric: str, periods: int = 7, period_hours: int = 24) -> list[dict[str, Any]]:
        """Get metric trend over multiple periods."""
        now = time.time()
        trend = []
        for i in range(periods):
            end = now - i * period_hours * 3600
            start = end - period_hours * 3600
            with sqlite3.connect(str(self._db_path)) as conn:
                rows = conn.execute(
                    "SELECT value FROM events WHERE metric = ? AND created_at >= ? AND created_at < ?",
                    (metric, start, end),
                ).fetchall()
            values = [r[0] for r in rows]
            trend.append(
                {
                    "period": i,
                    "avg": round(sum(values) / len(values), 4) if values else 0,
                    "count": len(values),
                }
            )
        trend.reverse()
        return trend

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get all metrics formatted for dashboard/GoalManager."""
        snap = self.snapshot()
        return {
            "reasoning_score": snap.reasoning_score,
            "hallucination_rate": snap.hallucination_rate,
            "tool_success_rate": snap.tool_success_rate,
            "memory_hit_rate": snap.memory_hit_rate,
            "avg_confidence": snap.avg_confidence,
            "avg_latency_ms": snap.avg_latency_ms,
            "total_queries_24h": snap.total_queries,
        }
