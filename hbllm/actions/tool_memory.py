"""
Tool Memory — learns which tools work best for different query types.

Tracks tool usage patterns, success rates, and latency to automatically
optimize tool selection over time. Feeds insights back to RouterNode.
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
class ToolUsageRecord:
    """Record of a single tool invocation."""
    tool_name: str
    query_type: str  # classified query category
    success: bool
    latency_ms: float
    result_quality: float = 0.5  # 0-1 quality rating
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ToolMemory:
    """
    Learns tool preferences from usage patterns.

    Capabilities:
    - Track tool success rates per query type
    - Learn optimal tool ordering for multi-step tasks
    - Discover new tool-query mappings from experience
    - Provide tool recommendations to RouterNode
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "tool_memory.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    result_quality REAL DEFAULT 0.5,
                    context TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_usage_type "
                "ON tool_usage(query_type, tool_name)"
            )

    # ─── Recording ───────────────────────────────────────────────────

    def record(self, record: ToolUsageRecord) -> None:
        """Record a tool usage event."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO tool_usage "
                "(tool_name, query_type, success, latency_ms, result_quality, context, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.tool_name, record.query_type, int(record.success),
                 record.latency_ms, record.result_quality,
                 json.dumps(record.context), record.timestamp),
            )

    def record_sequence(
        self, task_type: str, tools: list[str], success: bool, total_ms: float,
    ) -> None:
        """Record a multi-tool sequence execution."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO tool_sequences (task_type, sequence, success, total_latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_type, json.dumps(tools), int(success), total_ms, time.time()),
            )

    # ─── Recommendations ─────────────────────────────────────────────

    def recommend_tool(self, query_type: str, top_n: int = 3) -> list[dict[str, Any]]:
        """
        Recommend best tools for a query type based on historical performance.

        Returns tools ranked by a composite score of success rate + quality - latency.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("""
                SELECT tool_name,
                       COUNT(*) as uses,
                       AVG(success) as success_rate,
                       AVG(result_quality) as avg_quality,
                       AVG(latency_ms) as avg_latency
                FROM tool_usage
                WHERE query_type = ?
                GROUP BY tool_name
                HAVING uses >= 3
                ORDER BY (AVG(success) * 0.4 + AVG(result_quality) * 0.4 - AVG(latency_ms) / 10000 * 0.2) DESC
                LIMIT ?
            """, (query_type, top_n)).fetchall()

        return [
            {
                "tool": r[0],
                "uses": r[1],
                "success_rate": round(r[2], 3),
                "avg_quality": round(r[3], 3),
                "avg_latency_ms": round(r[4], 1),
                "score": round(r[2] * 0.4 + r[3] * 0.4 - r[4] / 10000 * 0.2, 3),
            }
            for r in rows
        ]

    def recommend_sequence(self, task_type: str) -> list[str] | None:
        """Recommend the best tool sequence for a task type."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("""
                SELECT sequence FROM tool_sequences
                WHERE task_type = ? AND success = 1
                GROUP BY sequence
                ORDER BY COUNT(*) DESC, AVG(total_latency_ms) ASC
                LIMIT 1
            """, (task_type,)).fetchone()

        return json.loads(row[0]) if row else None

    # ─── Discovery ───────────────────────────────────────────────────

    def discover_patterns(self) -> list[dict[str, Any]]:
        """
        Discover query-type → tool patterns from accumulated data.

        Returns strong associations between query types and tools.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("""
                SELECT query_type, tool_name,
                       COUNT(*) as uses,
                       AVG(success) as sr,
                       AVG(result_quality) as quality
                FROM tool_usage
                GROUP BY query_type, tool_name
                HAVING uses >= 5 AND sr > 0.7
                ORDER BY sr * quality DESC
            """).fetchall()

        return [
            {
                "query_type": r[0],
                "best_tool": r[1],
                "uses": r[2],
                "success_rate": round(r[3], 3),
                "quality": round(r[4], 3),
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        """Return tool memory statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM tool_usage").fetchone()[0]
            tools = conn.execute("SELECT COUNT(DISTINCT tool_name) FROM tool_usage").fetchone()[0]
            types = conn.execute("SELECT COUNT(DISTINCT query_type) FROM tool_usage").fetchone()[0]
            seqs = conn.execute("SELECT COUNT(*) FROM tool_sequences").fetchone()[0]
        return {
            "total_usages": total,
            "unique_tools": tools,
            "query_types": types,
            "sequences": seqs,
        }
