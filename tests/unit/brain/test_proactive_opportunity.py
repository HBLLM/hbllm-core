"""Unit tests for the Autonomy & Proactive Opportunity framework."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
import pytest

from hbllm.brain.autonomy.autonomy_manager import PresenceMonitor, ProactiveCoordinator
from hbllm.brain.autonomy.opportunity import Opportunity, OpportunityHistory
from hbllm.brain.autonomy.opportunity_source import (
    BatterySource,
    OpportunityScorer,
    SilenceSource,
)
from hbllm.brain.autonomy.presence_state import PresenceState


def test_presence_state_updates():
    """Verify presence state correctly records user input and decays over time."""
    state = PresenceState()
    assert state.engagement_level == 0.5
    assert state.interaction_score == 0.5

    state.update_user_activity(time.time())
    assert state.engagement_level > 0.5
    assert state.interaction_score > 0.5

    old_engagement = state.engagement_level
    state.decay_engagement(10.0, decay_rate=0.01)
    assert state.engagement_level < old_engagement


def test_opportunity_aging():
    """Verify that opportunity aging policies shift priorities dynamically."""
    opp = Opportunity(
        id="opp_1",
        source="test",
        category="silence",
        priority=0.5,
        urgency=0.5,
        confidence=1.0,
        created_at=time.time() - 100.0,
        aging_strategy="escalate",
        aging_rate=0.002,
    )
    now = time.time()
    p = opp.update_priority(now)
    assert p > 0.5
    assert p <= 1.0


def test_silence_source_detection():
    """Verify that silence policies identify candidate opportunities when thresholds are met."""
    policies = {"task": 5.0, "conversation": 10.0, "relationship": 20.0}
    source = SilenceSource(policies=policies)

    presence = PresenceState()
    presence.update_user_activity(time.time() - 6.0)
    presence.interaction_score = 0.5
    presence.engagement_level = 0.2  # Selects task policy (5s threshold)

    opps = source.detect(presence, None)
    assert len(opps) == 1
    assert opps[0].category == "silence"
    assert "task" in opps[0].reason


def test_opportunity_scorer_contextualizes():
    """Verify that the opportunity scorer dampens priority during user stress."""
    opp = Opportunity(
        id="opp_1",
        source="test",
        category="silence",
        priority=0.5,
        urgency=0.5,
        confidence=1.0,
        created_at=time.time(),
    )

    class MockState:
        stress = 0.8
        fatigue = 0.0
        focus_target = ""

    scorer = OpportunityScorer()
    scorer.score(opp, MockState())
    assert opp.priority < 0.5


def test_proactive_coordinator_cooldown():
    """Verify that the global coordinator rate-limits proactive checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "opps.db"
        history = OpportunityHistory(db_path)
        coordinator = ProactiveCoordinator(history, min_gap_seconds=60.0, daily_budget=5)

        opp = Opportunity(
            id="opp_1",
            source="test",
            category="silence",
            priority=0.8,
            urgency=0.8,
            confidence=1.0,
            created_at=time.time(),
        )

        class MockState:
            stress = 0.0
            fatigue = 0.0
            focus_target = ""

        # First trigger should succeed
        winner = coordinator.evaluate_and_route([opp], MockState())
        assert winner is not None
        assert winner.id == "opp_1"

        # Second trigger within cooldown gap should fail
        opp2 = Opportunity(
            id="opp_2",
            source="test",
            category="silence",
            priority=0.9,
            urgency=0.9,
            confidence=1.0,
            created_at=time.time(),
        )
        winner2 = coordinator.evaluate_and_route([opp2], MockState())
        assert winner2 is None

        # Verify history logger wrote both opportunities
        logs = history.get_history()
        assert len(logs) >= 2
