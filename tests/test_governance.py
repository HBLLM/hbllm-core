"""Tests for Cognitive Governance and Epistemic Calibration."""

from __future__ import annotations

import pytest

from hbllm.brain.governance import CognitiveBudget, CognitiveGovernanceEngine


@pytest.fixture
def governance():
    budget = CognitiveBudget(llm_calls_per_hour=5)
    return CognitiveGovernanceEngine(budget=budget)


def test_llm_budget_consumption(governance):
    """Ensure LLM budget hard limits work."""
    for _ in range(5):
        assert governance.consume_llm_call() is True

    # 6th call should fail
    assert governance.consume_llm_call() is False


def test_cognitive_pressure(governance):
    """Ensure cognitive pressure correctly computes graceful degradation."""
    # Low pressure
    pressure = governance.get_cognitive_pressure(memory_pressure=0.2, active_goals=1, queue_depth=5)
    assert pressure < 0.4

    profile = governance.get_degradation_profile(pressure)
    assert profile["max_simulation_depth"] == 3
    assert profile["allow_speculation"] is True

    # High pressure
    pressure = governance.get_cognitive_pressure(
        memory_pressure=0.9, active_goals=15, queue_depth=60
    )
    assert pressure > 0.75

    profile = governance.get_degradation_profile(pressure)
    assert profile["max_simulation_depth"] == 0
    assert profile["allow_speculation"] is False
    assert profile["force_heuristic"] is True
