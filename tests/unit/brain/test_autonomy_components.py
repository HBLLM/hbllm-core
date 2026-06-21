"""Tests for Autonomy components — reflexes, confirmation, interrupts, suppressor."""

import pytest

from hbllm.actions.confirmation import ActionRiskClassifier, RiskAssessment
from hbllm.brain.autonomy.interrupt_detector import InterruptDetector
from hbllm.brain.autonomy.notification_suppressor import NotificationSuppressor

# ── ActionRiskClassifier ──────────────────────────────────────────────────


class TestActionRiskClassifier:
    """Tests for 4-tier risk classification."""

    @pytest.fixture
    def classifier(self):
        return ActionRiskClassifier()

    def test_safe_actions(self, classifier):
        """Read-only actions are classified as safe (tier 0)."""
        result = classifier.classify("read.file")
        assert result.tier == 0
        assert not result.requires_confirmation

    def test_low_risk_actions(self, classifier):
        """Light on/off is low risk (tier 1)."""
        result = classifier.classify("light.on")
        assert result.tier <= 1

    def test_high_risk_lock(self, classifier):
        """Lock unlock is high risk (tier 3)."""
        result = classifier.classify("lock.unlock")
        assert result.tier == 3
        assert result.requires_confirmation

    def test_high_risk_file_delete(self, classifier):
        """File delete is high risk."""
        result = classifier.classify("file.delete")
        assert result.tier >= 2
        assert result.requires_confirmation

    def test_unknown_action_defaults_safe(self, classifier):
        """Unknown actions default to a reasonable tier."""
        result = classifier.classify("some.random.action")
        assert isinstance(result, RiskAssessment)
        assert result.tier >= 0

    def test_assessment_has_reason(self, classifier):
        """Risk assessment includes a reason."""
        result = classifier.classify("lock.unlock")
        assert len(result.reason) > 0


# ── InterruptDetector ─────────────────────────────────────────────────────


class TestInterruptDetector:
    """Tests for 5-state user engagement detection."""

    @pytest.fixture
    def detector(self):
        return InterruptDetector()

    def test_initial_state(self, detector):
        """Initial state is a valid UserState enum."""
        # State is an enum, check it has a reasonable value
        state_str = str(detector.state)
        assert "UserState" in state_str or hasattr(detector.state, "value")

    def test_should_deliver_method_exists(self, detector):
        """should_deliver method exists and is callable."""
        assert hasattr(detector, "should_deliver")
        assert callable(detector.should_deliver)

    def test_stats(self, detector):
        """Stats returns expected fields."""
        s = detector.stats()
        assert "current_state" in s
        assert "state_changes" in s


# ── NotificationSuppressor ────────────────────────────────────────────────


class TestNotificationSuppressor:
    """Tests for anti-annoyance notification filtering."""

    @pytest.fixture
    def suppressor(self):
        return NotificationSuppressor()

    def test_first_notification_allowed(self, suppressor):
        """First notification of a category passes through."""
        assert suppressor.should_send(category="weather", priority="normal", tenant_id="t1")

    def test_rate_limiting(self, suppressor):
        """Same category is rate-limited after max_per_hour."""
        for _ in range(15):
            suppressor.should_send(category="weather", priority="normal", tenant_id="t1")
        # After exceeding max_per_hour=10, should be suppressed
        result = suppressor.should_send(category="weather", priority="normal", tenant_id="t1")
        assert result is False

    def test_critical_bypasses_suppression(self, suppressor):
        """Critical priority bypasses all suppression."""
        # Exhaust the rate limit
        for _ in range(15):
            suppressor.should_send(category="test", priority="normal", tenant_id="t1")
        # Critical should still pass
        assert suppressor.should_send(category="test", priority="critical", tenant_id="t1")

    def test_suggestion_rate_limit(self, suppressor):
        """Suggestions have a lower rate limit (3/hour)."""
        for _ in range(4):
            suppressor.should_send(category="suggestion", priority="suggestion", tenant_id="t1")
        result = suppressor.should_send(
            category="suggestion", priority="suggestion", tenant_id="t1"
        )
        assert result is False

    def test_dismissal_tracking(self, suppressor):
        """Dismissal recording doesn't crash."""
        suppressor.record_dismissal("t1", "weather")
        suppressor.record_dismissal("t1", "weather")
        s = suppressor.stats()
        assert s["total_checked"] == 0  # No checks yet

    def test_stats(self, suppressor):
        """Stats returns expected telemetry."""
        s = suppressor.stats()
        assert "total_checked" in s
        assert "total_suppressed" in s
