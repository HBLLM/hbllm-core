"""Cognitive Observability Layer and Decision Trace Ledger.

Implements the Tiered Observability Logging architecture:
- Tier A: Memory Ring Buffer (fast, volatile)
- Tier B: SQLite Compressed Logs (sampled/high-risk, persistent)
- Tier C: Event-Triggered Full Trace Dumps (on failure, JSON dump)
"""

from __future__ import annotations

import collections
import json
import logging
import sqlite3
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GraphDelta:
    """Represents the difference between the current state and the predicted state."""

    mutated_keys: list[str] = field(default_factory=list)
    added_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)


@dataclass
class UtilityDelta:
    """Represents the risk/utility profile of a scenario."""

    utility_score: float = 0.0
    risk_categories: dict[str, float] = field(default_factory=dict)
    rejected_reason: str | None = None


@dataclass
class DecisionTrace:
    """A compressed record of why the AnticipatoryPlanner made a decision."""

    trace_id: str = field(default_factory=lambda: f"trc_{uuid.uuid4().hex[:12]}")
    goal_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Inputs
    cognitive_pressure: float = 0.0

    # Compressed Options
    # scenario_id -> UtilityDelta
    considered_utilities: dict[str, UtilityDelta] = field(default_factory=dict)

    # Selection
    selected_scenario_id: str = ""
    selected_graph_delta: GraphDelta | None = None

    # Metadata
    is_high_risk: bool = False
    is_failure: bool = False

    def explain_decision(self) -> str:
        """Returns a human-readable explanation of the decision logic."""
        if not self.selected_scenario_id:
            return "No scenario selected."
        utility = self.considered_utilities.get(self.selected_scenario_id)
        reason = (
            utility.rejected_reason if utility and utility.rejected_reason else "Optimal selection."
        )
        return f"Scenario {self.selected_scenario_id} selected. Reasoning: {reason}"


class DecisionTraceLedger:
    """Manages tiered observability storage for cognitive decisions."""

    def __init__(self, data_dir: str | Path = "data", ring_size: int = 100) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Tier A: Memory Ring Buffer
        self.ring_buffer: collections.deque[DecisionTrace] = collections.deque(maxlen=ring_size)

        # Tier B: SQLite
        self.db_path = self.data_dir / "observability.db"
        self._init_db()

        self.total_decisions_seen = 0

    def _init_db(self) -> None:
        """Initialize the SQLite schema for Tier B logs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compressed_traces (
                    trace_id TEXT PRIMARY KEY,
                    goal_id TEXT,
                    timestamp REAL,
                    is_high_risk BOOLEAN,
                    is_failure BOOLEAN,
                    compressed_data BLOB
                )
                """
            )

    def record_decision(self, trace: DecisionTrace) -> None:
        """Record a decision through the tiered storage architecture."""
        self.total_decisions_seen += 1

        # 1. Tier A (Always record to fast ring buffer)
        self.ring_buffer.append(trace)

        # 2. Tier B (Sampled or High Risk)
        # We sample 1 in 10 decisions, UNLESS it's high risk or a failure
        should_persist = (
            trace.is_high_risk or trace.is_failure or (self.total_decisions_seen % 10 == 0)
        )
        if should_persist:
            self._persist_tier_b(trace)

        # 3. Tier C (Full Dumps on Failure)
        if trace.is_failure:
            self._dump_tier_c(trace)

    def _persist_tier_b(self, trace: DecisionTrace) -> None:
        """Compress and save to SQLite."""
        trace_dict = asdict(trace)
        json_str = json.dumps(trace_dict)
        compressed_data = zlib.compress(json_str.encode("utf-8"))

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO compressed_traces
                    (trace_id, goal_id, timestamp, is_high_risk, is_failure, compressed_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        trace.goal_id,
                        trace.timestamp,
                        trace.is_high_risk,
                        trace.is_failure,
                        compressed_data,
                    ),
                )
        except Exception:
            logger.exception("Failed to persist Tier B observability log.")

    def _dump_tier_c(self, trace: DecisionTrace) -> None:
        """Dump an uncompressed full JSON trace to disk for critical failures."""
        dump_dir = self.data_dir / "crash_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)

        dump_file = dump_dir / f"trace_{trace.trace_id}.json"
        try:
            with open(dump_file, "w", encoding="utf-8") as f:
                json.dump(asdict(trace), f, indent=2)
            logger.info("Tier C Dump created: %s", dump_file)
        except Exception:
            logger.exception("Failed to write Tier C crash dump.")

    def get_recent_traces(self) -> list[DecisionTrace]:
        """Fetch Tier A traces."""
        return list(self.ring_buffer)

    def read_tier_b_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Decompress and read a specific trace from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT compressed_data FROM compressed_traces WHERE trace_id = ?", (trace_id,)
                )
                row = cursor.fetchone()
                if row:
                    compressed_data = row[0]
                    json_str = zlib.decompress(compressed_data).decode("utf-8")
                    return json.loads(json_str)
        except Exception:
            logger.exception("Failed to read Tier B observability log.")
        return None

    def explain_decision(self, trace_id: str) -> str:
        """Single-query API: reconstructs the decision reasoning in < 5 seconds."""
        # Check Tier A (Ring Buffer) first
        for trace in self.ring_buffer:
            if trace.trace_id == trace_id:
                return self._generate_explanation(trace)

        # Check Tier B (SQLite)
        raw_trace = self.read_tier_b_trace(trace_id)
        if raw_trace:
            # Reconstruct Dataclass for explanation
            # Simple fallback explanation if reconstruction is complex
            sel_id = raw_trace.get("selected_scenario_id")
            return f"Trace {trace_id} found in Tier B. Selected scenario {sel_id}."

        return f"Trace {trace_id} not found in hot or warm storage."

    def _generate_explanation(self, trace: DecisionTrace) -> str:
        """Generates a human-readable summary of the decision."""
        if not trace.selected_scenario_id:
            return "No scenario was selected. The planner likely aborted or degraded."

        sel_util = trace.considered_utilities.get(trace.selected_scenario_id)
        sel_score = sel_util.utility_score if sel_util else 0.0

        explanation = [
            f"I selected scenario '{trace.selected_scenario_id}' (Utility: {sel_score:.2f})."
        ]

        if trace.selected_graph_delta:
            keys = len(trace.selected_graph_delta.mutated_keys)
            explanation.append(f"It predicted mutations to {keys} state keys.")

        rejected = [uid for uid in trace.considered_utilities if uid != trace.selected_scenario_id]
        if rejected:
            explanation.append(f"I rejected {len(rejected)} alternative(s):")
            for uid in rejected:
                rej_util = trace.considered_utilities[uid]
                reason = rej_util.rejected_reason or "Lower overall utility score."
                explanation.append(f" - {uid} (Utility: {rej_util.utility_score:.2f}): {reason}")

        return "\n".join(explanation)
