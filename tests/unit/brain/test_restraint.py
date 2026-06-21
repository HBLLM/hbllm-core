"""Tests for RestraintEngine — cognitive safety gate."""

import pytest

from hbllm.brain.autonomy.restraint import (
    RestraintConfig,
    RestraintDecision,
    RestraintEngine,
)


@pytest.fixture
def engine():
    return RestraintEngine()


class TestConfidenceThreshold:
    def test_low_confidence_suppressed(self, engine):
        result = engine.evaluate("test.action", {"confidence": 0.2})
        assert result.decision == RestraintDecision.SUPPRESS

    def test_high_confidence_approved(self, engine):
        result = engine.evaluate("test.action", {"confidence": 0.9})
        assert result.decision == RestraintDecision.APPROVE

    def test_threshold_boundary(self):
        engine = RestraintEngine(config=RestraintConfig(min_confidence_approve=0.5))
        below = engine.evaluate("a", {"confidence": 0.49})
        assert below.decision == RestraintDecision.SUPPRESS
        above = engine.evaluate("b", {"confidence": 0.51})
        assert above.decision == RestraintDecision.APPROVE


class TestReversibility:
    def test_irreversible_low_confidence_deferred(self, engine):
        result = engine.evaluate(
            "dangerous.action",
            {"confidence": 0.55, "reversible": False},
        )
        assert result.decision == RestraintDecision.DEFER

    def test_irreversible_high_confidence_approved(self, engine):
        result = engine.evaluate(
            "dangerous.action",
            {"confidence": 0.85, "reversible": False},
        )
        assert result.decision == RestraintDecision.APPROVE

    def test_reversible_medium_confidence_approved(self, engine):
        result = engine.evaluate(
            "safe.action",
            {"confidence": 0.55, "reversible": True},
        )
        assert result.decision == RestraintDecision.APPROVE


class TestSocialTiming:
    def test_quiet_hours_suppressed(self, engine):
        result = engine.evaluate("remind", {"confidence": 0.8, "hour": 23})
        assert result.decision == RestraintDecision.SUPPRESS

    def test_early_morning_suppressed(self, engine):
        result = engine.evaluate("remind", {"confidence": 0.8, "hour": 3})
        assert result.decision == RestraintDecision.SUPPRESS

    def test_daytime_approved(self, engine):
        result = engine.evaluate("remind", {"confidence": 0.8, "hour": 14})
        assert result.decision == RestraintDecision.APPROVE

    def test_high_priority_bypasses_quiet_hours(self, engine):
        result = engine.evaluate("alert", {"confidence": 0.8, "hour": 23, "priority": "high"})
        assert result.decision == RestraintDecision.APPROVE


class TestCriticalPriority:
    def test_critical_bypasses_all(self, engine):
        result = engine.evaluate(
            "emergency",
            {"confidence": 0.1, "reversible": False, "priority": "critical", "hour": 3},
        )
        assert result.decision == RestraintDecision.APPROVE


class TestRejectionHistory:
    def test_repeated_rejections_suppress(self):
        engine = RestraintEngine(config=RestraintConfig(max_rejections_before_suppress=2))
        engine.record_rejection("nag")
        engine.record_rejection("nag")
        result = engine.evaluate("nag", {"confidence": 0.9, "hour": 12})
        assert result.decision == RestraintDecision.SUPPRESS
        assert "rejected" in result.reasons[0].lower()

    def test_no_rejections_approved(self, engine):
        result = engine.evaluate("fresh.action", {"confidence": 0.8, "hour": 12})
        assert result.decision == RestraintDecision.APPROVE


class TestCooldown:
    def test_cooldown_enforced(self):
        engine = RestraintEngine(config=RestraintConfig(cooldown_per_action_type_s=300))
        engine.record_approval("repeat.action")
        result = engine.evaluate("repeat.action", {"confidence": 0.9, "hour": 12})
        assert result.decision == RestraintDecision.DEFER
        assert result.cooldown_s > 0

    def test_expired_cooldown_approved(self):
        engine = RestraintEngine(config=RestraintConfig(cooldown_per_action_type_s=0.0))
        engine.record_approval("fast.action")
        result = engine.evaluate("fast.action", {"confidence": 0.9, "hour": 12})
        assert result.decision == RestraintDecision.APPROVE


class TestRateLimiting:
    def test_rate_limit_exceeded(self):
        engine = RestraintEngine(
            config=RestraintConfig(
                max_actions_per_hour=3,
                cooldown_per_action_type_s=0,
            )
        )
        for i in range(3):
            engine.record_approval(f"action_{i}")
        result = engine.evaluate("new_action", {"confidence": 0.9, "hour": 12})
        assert result.decision == RestraintDecision.DEFER


class TestStats:
    def test_stats_tracking(self, engine):
        engine.evaluate("a", {"confidence": 0.9, "hour": 12})
        engine.evaluate("b", {"confidence": 0.1})
        stats = engine.stats()
        assert stats["total_evaluations"] == 2
        assert stats["decisions"]["approve"] >= 1
        assert stats["decisions"]["suppress"] >= 1


class TestRestraintReason:
    def test_to_dict(self, engine):
        result = engine.evaluate("test", {"confidence": 0.9, "hour": 12})
        d = result.to_dict()
        assert d["decision"] == "approve"
        assert isinstance(d["reasons"], list)
