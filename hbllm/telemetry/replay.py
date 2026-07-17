"""
Decision Replay Engine — deterministic recording and playback of cognitive decisions.

ADR 002 §5: Replay captures the complete decision context, not just events:
    - Retrieved memory snapshots
    - Active capability selections
    - Planner node choices
    - Simulation outcome vectors
    - Execution results

This enables deterministic debugging: given the same ``DecisionRecord``
sequence, the system produces identical behavior regardless of when
the replay is executed.

Usage::

    from hbllm.telemetry.replay import DecisionReplayEngine, DecisionRecord

    engine = DecisionReplayEngine()

    # Record during live execution
    engine.record(DecisionRecord(
        decision_type="plan_selection",
        input_context={"intent": intent.to_dict()},
        retrieved_memories=[...],
        selected_capabilities=["tool_a", "tool_b"],
        planner_choices={"strategy": "got", "depth": 3},
        simulation_outcomes=[{"safety": 0.95}],
        execution_result={"success": True},
    ))

    # Replay for debugging
    records = engine.get_replay_window(start_time, end_time)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.brain.core.provenance import ProvenanceMetadata

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A complete snapshot of a cognitive decision for deterministic replay.

    Captures everything needed to understand *why* a decision was made:
    not just what happened, but what the system knew, considered, and
    chose from at the time.

    Attributes:
        record_id: Globally unique identifier.
        decision_type: Category of decision (e.g., "plan_selection",
            "tool_routing", "memory_retrieval").
        input_context: The input state that triggered this decision.
        retrieved_memories: Memory snapshots that were retrieved.
        selected_capabilities: Capabilities/tools that were considered.
        planner_choices: Planner decisions and parameters.
        simulation_outcomes: Results from any simulations run.
        execution_result: Final execution output.
        provenance: Causal provenance metadata.
        timestamp: When this decision was made.
    """

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    decision_type: str = ""
    input_context: dict[str, Any] = field(default_factory=dict)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    selected_capabilities: list[str] = field(default_factory=list)
    planner_choices: dict[str, Any] = field(default_factory=dict)
    simulation_outcomes: list[dict[str, Any]] = field(default_factory=list)
    execution_result: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceMetadata | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "record_id": self.record_id,
            "decision_type": self.decision_type,
            "input_context": self.input_context,
            "retrieved_memories": self.retrieved_memories,
            "selected_capabilities": self.selected_capabilities,
            "planner_choices": self.planner_choices,
            "simulation_outcomes": self.simulation_outcomes,
            "execution_result": self.execution_result,
            "timestamp": self.timestamp,
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionRecord:
        prov_data = data.get("provenance")
        prov = ProvenanceMetadata.from_dict(prov_data) if prov_data else None
        return cls(
            record_id=data.get("record_id", uuid.uuid4().hex),
            decision_type=data.get("decision_type", ""),
            input_context=data.get("input_context", {}),
            retrieved_memories=data.get("retrieved_memories", []),
            selected_capabilities=data.get("selected_capabilities", []),
            planner_choices=data.get("planner_choices", {}),
            simulation_outcomes=data.get("simulation_outcomes", []),
            execution_result=data.get("execution_result", {}),
            provenance=prov,
            timestamp=data.get("timestamp", time.time()),
        )


class DecisionReplayEngine:
    """Deterministic decision recording and replay engine.

    Maintains a ring buffer of ``DecisionRecord`` objects and supports
    time-range queries, JSONL export/import, and filtered replay.

    Args:
        max_records: Maximum number of records to retain in memory.
    """

    def __init__(self, max_records: int = 5000) -> None:
        self._records: deque[DecisionRecord] = deque(maxlen=max_records)
        self._max_records = max_records
        self._total_recorded = 0
        self._total_replayed = 0

    def record(self, decision: DecisionRecord) -> None:
        """Record a decision for future replay.

        Args:
            decision: The complete decision snapshot.
        """
        self._records.append(decision)
        self._total_recorded += 1

    def get_replay_window(
        self,
        start_time: float,
        end_time: float,
        decision_type: str | None = None,
    ) -> list[DecisionRecord]:
        """Retrieve decision records within a time window.

        Args:
            start_time: Start of the time range (POSIX timestamp).
            end_time: End of the time range (POSIX timestamp).
            decision_type: Optional filter by decision type.

        Returns:
            List of matching DecisionRecord objects.
        """
        self._total_replayed += 1
        results: list[DecisionRecord] = []
        for record in self._records:
            if start_time <= record.timestamp <= end_time:
                if decision_type and record.decision_type != decision_type:
                    continue
                results.append(record)
        return results

    def get_by_correlation(self, correlation_id: str) -> list[DecisionRecord]:
        """Retrieve all decisions linked by correlation ID.

        Args:
            correlation_id: The correlation/session ID to filter by.

        Returns:
            List of matching DecisionRecord objects.
        """
        return [
            r
            for r in self._records
            if r.provenance and r.provenance.correlation_id == correlation_id
        ]

    def get_causal_chain(self, record_id: str) -> list[DecisionRecord]:
        """Trace the causal chain backward from a given record.

        Follows ``parent_event_id`` links in provenance metadata.

        Args:
            record_id: The starting record's ID.

        Returns:
            Causal chain from root to the given record.
        """
        # Build lookup by event_id
        by_event: dict[str, DecisionRecord] = {}
        for r in self._records:
            if r.provenance:
                by_event[r.provenance.event_id] = r

        chain: list[DecisionRecord] = []
        # Find the starting record
        current = None
        for r in self._records:
            if r.record_id == record_id:
                current = r
                break

        while current is not None:
            chain.append(current)
            if current.provenance and current.provenance.parent_event_id:
                current = by_event.get(current.provenance.parent_event_id)
            else:
                break

        chain.reverse()
        return chain

    def export_jsonl(self, path: str | Path) -> int:
        """Export all records to a JSONL file.

        Args:
            path: File path to write to.

        Returns:
            Number of records exported.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict()) + "\n")
                count += 1
        logger.info("Exported %d decision records to %s", count, path)
        return count

    def import_jsonl(self, path: str | Path) -> int:
        """Import records from a JSONL file.

        Args:
            path: File path to read from.

        Returns:
            Number of records imported.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Replay file not found: %s", path)
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = DecisionRecord.from_dict(json.loads(line))
                    self._records.append(record)
                    count += 1
        self._total_recorded += count
        logger.info("Imported %d decision records from %s", count, path)
        return count

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        return {
            "total_recorded": self._total_recorded,
            "total_replayed": self._total_replayed,
            "current_size": len(self._records),
            "max_records": self._max_records,
        }
