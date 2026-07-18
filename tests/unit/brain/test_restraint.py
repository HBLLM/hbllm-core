"""Tests for RestraintEngine — cognitive safety gate."""

import pytest

from hbllm.brain.autonomy.restraint import (
    ActionProposal,
    RestraintConfig,
    RestraintDecision,
    RestraintEngine,
)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr("hbllm.brain.autonomy.restraint._current_hour", lambda: 12)
    return RestraintEngine(quiet_hours=(2, 3))


class TestConfidenceThreshold:
    def test_low_confidence_suppressed(self, engine):
        result = engine.evaluate(ActionProposal(action_type="test.action", confidence=0.2))
        assert result.decision == RestraintDecision.SUPPRESS

    def test_high_confidence_approved(self, engine):
        result = engine.evaluate(ActionProposal(action_type="test.action", confidence=0.9))
        assert result.decision == RestraintDecision.APPROVE

    def test_threshold_boundary(self, monkeypatch):
        monkeypatch.setattr("hbllm.brain.autonomy.restraint._current_hour", lambda: 12)
        engine = RestraintEngine(
            config=RestraintConfig(min_confidence_approve=0.5),
            quiet_hours=(3, 4),
        )
        below = engine.evaluate(ActionProposal(action_type="a", confidence=0.49))
        assert below.decision == RestraintDecision.SUPPRESS
        above = engine.evaluate(ActionProposal(action_type="b", confidence=0.51))
        assert above.decision == RestraintDecision.APPROVE


class TestReversibility:
    def test_irreversible_low_confidence_deferred(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="dangerous.action",
                confidence=0.9,
                is_reversible=False,
                priority="normal",
            )
        )
        assert result.decision == RestraintDecision.DEFER
        assert result.alternative_action is not None

    def test_irreversible_high_confidence_approved(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="dangerous.action",
                confidence=0.9,
                is_reversible=False,
                priority="high",
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_reversible_medium_confidence_approved(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="safe.action",
                confidence=0.8,
                is_reversible=True,
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestSocialTiming:
    """Quiet hours are set to (3, 4) — only hour 3 is quiet."""

    def test_quiet_hours_suppressed(self, monkeypatch):
        monkeypatch.setattr("hbllm.brain.autonomy.restraint._current_hour", lambda: 23)
        engine = RestraintEngine(quiet_hours=(22, 7))
        result = engine.evaluate(ActionProposal(action_type="remind", confidence=0.8))
        assert result.decision == RestraintDecision.SUPPRESS

    def test_quiet_hours_not_suppressed(self, monkeypatch):
        monkeypatch.setattr("hbllm.brain.autonomy.restraint._current_hour", lambda: 12)
        engine = RestraintEngine(quiet_hours=(22, 7))
        result = engine.evaluate(ActionProposal(action_type="remind", confidence=0.8))
        assert result.decision == RestraintDecision.APPROVE

    def test_high_priority_bypasses_quiet_hours(self):
        engine = RestraintEngine(quiet_hours=(0, 24))  # Always quiet
        result = engine.evaluate(
            ActionProposal(
                action_type="alert",
                confidence=0.8,
                priority="high",
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestCriticalPriority:
    def test_critical_bypasses_all(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="emergency",
                confidence=0.1,
                is_reversible=False,
                priority="critical",
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestContextSignals:
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


class TestRejectionBackoff:
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
        assert "backoff" in result.reasons[0].lower() or "rejected" in result.reasons[0].lower()

    def test_acceptance_clears_backoff(self, engine):
        engine.record_rejection("t1", "test.action")
        engine.record_acceptance("t1", "test.action")
        result = engine.evaluate(
            ActionProposal(
                action_type="test.action",
                category="reflex",
                confidence=0.9,
                tenant_id="t1",
            )
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_no_rejections_approved(self, engine):
        result = engine.evaluate(
            ActionProposal(
                action_type="fresh.action",
                confidence=0.8,
            )
        )
        assert result.decision == RestraintDecision.APPROVE


class TestSuggestionRateLimit:
    def test_over_limit_deferred(self):
        engine = RestraintEngine(
            quiet_hours=(3, 4),
            max_suggestions_per_hour=2,
        )
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


class TestCooldown:
    def test_cooldown_enforced(self):
        engine = RestraintEngine(
            config=RestraintConfig(cooldown_per_action_type_s=300),
            quiet_hours=(3, 4),
        )
        # Force an approval to set the last_action timestamp
        engine.evaluate(ActionProposal(action_type="repeat.action", confidence=0.9))
        result = engine.evaluate(ActionProposal(action_type="repeat.action", confidence=0.9))
        assert result.decision == RestraintDecision.DEFER
        assert result.cooldown_s > 0

    def test_expired_cooldown_approved(self):
        engine = RestraintEngine(
            config=RestraintConfig(cooldown_per_action_type_s=0.0),
            quiet_hours=(3, 4),
        )
        engine.evaluate(ActionProposal(action_type="fast.action", confidence=0.9))
        result = engine.evaluate(ActionProposal(action_type="fast.action", confidence=0.9))
        assert result.decision == RestraintDecision.APPROVE


class TestRateLimiting:
    def test_rate_limit_exceeded(self):
        engine = RestraintEngine(
            config=RestraintConfig(
                max_actions_per_hour=3,
                cooldown_per_action_type_s=0,
            ),
            quiet_hours=(3, 4),
        )
        for i in range(3):
            engine.evaluate(ActionProposal(action_type=f"action_{i}", confidence=0.9))
        result = engine.evaluate(ActionProposal(action_type="new_action", confidence=0.9))
        assert result.decision == RestraintDecision.DEFER


class TestStats:
    def test_stats_tracking(self, engine):
        engine.evaluate(ActionProposal(action_type="a", confidence=0.9))
        engine.evaluate(ActionProposal(action_type="b", confidence=0.1))
        stats = engine.stats()
        assert stats["total_evaluated"] == 2
        assert stats["decisions"]["approve"] >= 1
        assert stats["decisions"]["suppress"] >= 1


class TestRestraintReason:
    def test_to_dict(self, engine):
        result = engine.evaluate(ActionProposal(action_type="test", confidence=0.9))
        d = result.to_dict()
        assert d["decision"] == "approve"
        assert isinstance(d["reasons"], list)
