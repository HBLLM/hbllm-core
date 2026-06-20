"""Tests for SelfModel — internal capability tracking and trend analysis."""

import tempfile

import pytest

from hbllm.brain.self_model import SelfModel


@pytest.fixture
def self_model():
    """Create a SelfModel with a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SelfModel(data_dir=tmpdir)
        yield model


# ── Recording & Querying ─────────────────────────────────────────────────────


class TestSelfModelRecording:
    """Test outcome recording and capability queries."""

    def test_record_first_outcome(self, self_model):
        """First outcome for a domain should create a new capability."""
        self_model.record_outcome(domain="coding", success=True)
        cap = self_model.get_capability("coding")
        assert cap is not None
        assert cap.domain == "coding"
        assert cap.score == 1.0
        assert cap.sample_count == 1

    def test_record_mixed_outcomes(self, self_model):
        """Mixed outcomes should produce a score between 0 and 1."""
        self_model.record_outcome(domain="math", success=True)
        self_model.record_outcome(domain="math", success=True)
        self_model.record_outcome(domain="math", success=False)
        cap = self_model.get_capability("math")
        assert cap is not None
        assert round(cap.score, 2) == 0.67
        assert cap.sample_count == 3

    def test_record_all_failures(self, self_model):
        """All failures should produce a score of 0."""
        self_model.record_outcome(domain="unknown", success=False)
        self_model.record_outcome(domain="unknown", success=False)
        cap = self_model.get_capability("unknown")
        assert cap is not None
        assert cap.score == 0.0

    def test_unknown_domain(self, self_model):
        """Querying an unknown domain should return None."""
        assert self_model.get_capability("nonexistent") is None


# ── Strengths & Weaknesses ───────────────────────────────────────────────────


class TestStrengthsWeaknesses:
    """Test domain classification."""

    def test_get_strengths(self, self_model):
        """Domains with high scores and enough samples should be strengths."""
        for _ in range(10):
            self_model.record_outcome(domain="coding", success=True)
        for _ in range(10):
            self_model.record_outcome(domain="math", success=False)

        strengths = self_model.get_strengths(min_score=0.8, min_samples=5)
        assert "coding" in strengths
        assert "math" not in strengths

    def test_get_weaknesses(self, self_model):
        """Domains with low scores and enough samples should be weaknesses."""
        for _ in range(10):
            self_model.record_outcome(domain="coding", success=True)
        for _ in range(10):
            self_model.record_outcome(domain="medical", success=False)

        weaknesses = self_model.get_weaknesses(max_score=0.5, min_samples=5)
        assert "medical" in weaknesses
        assert "coding" not in weaknesses

    def test_min_samples_filter(self, self_model):
        """Domains with too few samples should not appear in strengths/weaknesses."""
        self_model.record_outcome(domain="rare", success=True)
        strengths = self_model.get_strengths(min_score=0.8, min_samples=5)
        assert "rare" not in strengths


# ── Delegation & Routing ─────────────────────────────────────────────────────


class TestDelegation:
    """Test model recommendation and delegation logic."""

    def test_should_delegate_weak_domain(self, self_model):
        """Weak domains should recommend delegation."""
        for _ in range(5):
            self_model.record_outcome(domain="legal", success=False)
        assert self_model.should_delegate("legal") is True

    def test_should_not_delegate_strong_domain(self, self_model):
        """Strong domains should not recommend delegation."""
        for _ in range(5):
            self_model.record_outcome(domain="coding", success=True)
        assert self_model.should_delegate("coding") is False

    def test_should_not_delegate_unknown(self, self_model):
        """Unknown domains should try anyway (no delegation)."""
        assert self_model.should_delegate("new_domain") is False

    def test_recommend_model_strong(self, self_model):
        """Strong domains should use the default model."""
        for _ in range(5):
            self_model.record_outcome(domain="coding", success=True)
        assert self_model.recommend_model("coding") == "default"

    def test_recommend_model_weak(self, self_model):
        """Weak domains should recommend a specialist model."""
        for _ in range(5):
            self_model.record_outcome(domain="legal", success=False)
        assert self_model.recommend_model("legal") == "specialist"

    def test_recommend_model_medium(self, self_model):
        """Medium domains should recommend a larger model."""
        for i in range(10):
            self_model.record_outcome(domain="writing", success=(i % 2 == 0))
        assert self_model.recommend_model("writing") == "large"


# ── Trend Analysis ───────────────────────────────────────────────────────────


class TestTrends:
    """Test performance trend detection."""

    def test_trend_improving(self, self_model):
        """Recent successes after older failures should show improving."""
        # Older outcomes (failures)
        for _ in range(10):
            self_model.record_outcome(domain="math", success=False)
        # Recent outcomes (successes)
        for _ in range(10):
            self_model.record_outcome(domain="math", success=True)

        cap = self_model.get_capability("math")
        assert cap is not None
        assert cap.trend == "improving"

    def test_trend_declining(self, self_model):
        """Recent failures after older successes should show declining."""
        for _ in range(10):
            self_model.record_outcome(domain="coding", success=True)
        for _ in range(10):
            self_model.record_outcome(domain="coding", success=False)

        cap = self_model.get_capability("coding")
        assert cap is not None
        assert cap.trend == "declining"

    def test_trend_stable_few_samples(self, self_model):
        """Too few samples should show stable."""
        self_model.record_outcome(domain="new", success=True)
        cap = self_model.get_capability("new")
        assert cap is not None
        assert cap.trend == "stable"


# ── Aggregate Metrics ────────────────────────────────────────────────────────


class TestMetrics:
    """Test aggregate metrics for GoalManager integration."""

    def test_empty_metrics(self, self_model):
        """Empty model should return empty metrics."""
        assert self_model.get_metrics() == {}

    def test_populated_metrics(self, self_model):
        """Metrics should include domain counts and lists."""
        for _ in range(10):
            self_model.record_outcome(domain="coding", success=True)
        for _ in range(10):
            self_model.record_outcome(domain="medical", success=False)

        metrics = self_model.get_metrics()
        assert metrics["total_domains"] == 2
        assert 0.0 <= metrics["avg_score"] <= 1.0
        assert "coding" in metrics["strengths"]
        assert "medical" in metrics["weaknesses"]

    def test_get_all_capabilities(self, self_model):
        """All capabilities should be returned sorted by score (desc)."""
        self_model.record_outcome(domain="low", success=False)
        self_model.record_outcome(domain="high", success=True)
        caps = self_model.get_all_capabilities()
        assert len(caps) == 2
        assert caps[0].domain == "high"
        assert caps[1].domain == "low"
