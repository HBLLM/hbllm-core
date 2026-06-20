"""Tests for the Decision Trace Ledger."""

import json
from pathlib import Path

from hbllm.brain.observability.tracer import DecisionTrace, DecisionTraceLedger


def test_decision_trace_ledger(tmp_path: Path):
    ledger = DecisionTraceLedger(data_dir=tmp_path, ring_size=5)

    # Tier A (Ring Buffer)
    for i in range(10):
        trace = DecisionTrace(goal_id=f"goal_{i}")
        ledger.record_decision(trace)

    # Should only keep last 5
    recent = ledger.get_recent_traces()
    assert len(recent) == 5
    assert recent[-1].goal_id == "goal_9"

    # Tier B (Sampled SQLite)
    # The 10th decision was recorded (because total_decisions_seen % 10 == 0)
    assert recent[-1].trace_id is not None
    loaded_trace = ledger.read_tier_b_trace(recent[-1].trace_id)
    assert loaded_trace is not None
    assert loaded_trace["goal_id"] == "goal_9"

    # Tier C (Event-Triggered Full Trace)
    failure_trace = DecisionTrace(goal_id="fail_goal", is_failure=True)
    ledger.record_decision(failure_trace)

    crash_file = tmp_path / "crash_dumps" / f"trace_{failure_trace.trace_id}.json"
    assert crash_file.exists()

    with open(crash_file) as f:
        data = json.load(f)
        assert data["goal_id"] == "fail_goal"
        assert data["is_failure"] is True
