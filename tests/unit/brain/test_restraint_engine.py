"""Tests for RestraintEngine — knowing when NOT to act."""

import pytest

from hbllm.brain.autonomy.restraint import (
    ActionProposal,
    RestraintDecision,
    RestraintEngine,
)


class TestRestraintBasics:
    """Tests for basic restraint decisions."""

    @pytest.fixture
    def engine(self):
        return RestraintEngine(quiet_hours=(3, 4))  # narrow range to avoid test flakiness

    def test_critical_always_approved(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="alarm.trigger",
                category="reflex",
                priority="critical",
                confidence=0.1,  # Even low confidence
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_low_confidence_suppressed(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="suggest.recipe",
                category="suggestion",
                confidence=0.3,
            )
        )
        assert result.decision == RestraintDecision.SUPPRESS
        assert "Confidence" in result.reasons[0]

    def test_normal_confidence_approved(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="light.on",
                category="reflex",
                confidence=0.9,
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_exactly_threshold_passes(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="test",
                category="reflex",
                confidence=0.7,  # Exactly at threshold
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestRejectionBackoff:
    """Tests for exponential backoff on rejections."""

    @pytest.fixture
    def engine(self):
        return RestraintEngine(
            quiet_hours=(3, 4),
            backoff_base_s=10,  # Short for testing
        )

    def test_first_attempt_passes(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="suggest.playlist",
                category="suggestion",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_rejection_triggers_backoff(self, engine):
        engine.record_rejection("t1", "suggest.playlist")
        result = engine.evaluate(
            ActionProposal(
                action_type="suggest.playlist",
                category="suggestion",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.SUPPRESS
        assert "backoff" in result.reasons[0].lower()

    def test_acceptance_reduces_backoff(self, engine):
        engine.record_rejection("t1", "test.action")
        engine.record_acceptance("t1", "test.action")
        # After acceptance, backoff should be cleared
        result = engine.evaluate(
            ActionProposal(
                action_type="test.action",
                category="reflex",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestSuggestionRateLimit:
    """Tests for suggestion frequency limiting."""

    @pytest.fixture
    def engine(self):
        return RestraintEngine(
            quiet_hours=(3, 4),
            max_suggestions_per_hour=2,
        )

    def test_under_limit_passes(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="suggest.1",
                category="suggestion",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_over_limit_deferred(self, engine):
        for i in range(3):
            engine.evaluate(
                ActionProposal(
                    action_type=f"suggest.{i}",
                    category="suggestion",
                    confidence=0.9,
                    tenant_id="t1",
                )
            )
        result = engine.evaluate(
            ActionProposal(
                action_type="suggest.extra",
                category="suggestion",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.DEFER


class TestContextSignals:
    """Tests for social context (meeting, sleeping)."""

    @pytest.fixture
    def engine(self):
        return RestraintEngine(quiet_hours=(3, 4))

    def test_meeting_defers_non_critical(self, engine):
        engine.update_context("t1", in_meeting=True)
        result = engine.evaluate(
            ActionProposal(
                action_type="notify.weather",
                category="proactive",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.DEFER

    def test_meeting_allows_critical(self, engine):
        engine.update_context("t1", in_meeting=True)
        result = engine.evaluate(
            ActionProposal(
                action_type="alarm.fire",
                category="reflex",
                priority="critical",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_sleeping_suppresses(self, engine):
        engine.update_context("t1", is_sleeping=True)
        result = engine.evaluate(
            ActionProposal(
                action_type="suggest.recipe",
                category="suggestion",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.SUPPRESS


class TestIrreversibility:
    """Tests for reversibility checks."""

    @pytest.fixture
    def engine(self):
        return RestraintEngine(quiet_hours=(3, 4))

    def test_irreversible_low_priority_deferred(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="file.delete",
                category="reflex",
                confidence=0.9,
                is_reversible=False,
                priority="normal",
            )
        )
        assert result.decision == RestraintDecision.DEFER
        assert result.alternative_action is not None

    def test_irreversible_high_priority_approved(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="alarm.trigger",
                category="reflex",
                confidence=0.9,
                is_reversible=False,
                priority="high",
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestRestraintStats:
    """Tests for telemetry."""

    def test_stats_tracking(self):
        engine = RestraintEngine(quiet_hours=(3, 4))
        engine.evaluate(
            ActionProposal(
                action_type="test",
                category="reflex",
                confidence=0.9,
            )
        )
        engine.evaluate(
            ActionProposal(
                action_type="test",
                category="suggestion",
                confidence=0.3,
            )
        )
        s = engine.stats()
        assert s["total_evaluated"] == 2
        assert s["decisions"]["approve"] >= 1
        assert s["decisions"]["suppress"] >= 1
