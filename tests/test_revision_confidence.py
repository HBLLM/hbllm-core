"""Tests for RevisionNode and ConfidenceEstimator."""

import pytest
from hbllm.brain.revision_node import RevisionNode
from hbllm.brain.confidence_estimator import ConfidenceEstimator


# ─── ConfidenceEstimator Tests ───────────────────────────────────────────

class TestConfidenceEstimator:
    @pytest.fixture
    def estimator(self):
        return ConfidenceEstimator()

    def test_full_report(self, estimator):
        report = estimator.estimate("What is Python?", "Python is a programming language.")
        assert 0 <= report.overall <= 1
        assert 0 <= report.relevance <= 1
        assert 0 <= report.coherence <= 1

    def test_relevant_response_scores_higher(self, estimator):
        relevant = estimator.score("What is Python?", "Python is a high-level programming language.")
        irrelevant = estimator.score("What is Python?", "The weather today is sunny and warm.")
        assert relevant > irrelevant

    def test_hedging_lowers_confidence(self, estimator):
        confident = estimator.score("What is 2+2?", "The correct answer to your equation is 4.")
        hedging = estimator.score("What is 2+2?", "I think maybe the answer might be 4, possibly.")
        assert confident > hedging

    def test_empty_response_low_score(self, estimator):
        report = estimator.estimate("Explain AI", "No")
        assert report.overall < 0.5
        assert "too_brief" in report.flags

    def test_factual_claims_increase_risk(self, estimator):
        with_claims = estimator.estimate("Tell me about X", "Founded in 1985, the company earned $5.2M in 2020.")
        without_claims = estimator.estimate("Tell me about X", "The company is well known for its services.")
        assert with_claims.factuality_risk > without_claims.factuality_risk

    def test_hallucination_flag(self, estimator):
        report = estimator.estimate(
            "Tell me",
            "Definitely founded in 1847, the 123456789 value is exactly guaranteed 100% proven.",
        )
        assert "high_hallucination_risk" in report.flags


# ─── RevisionNode Tests ──────────────────────────────────────────────────

class TestRevisionNode:
    @pytest.fixture
    def revision_node(self):
        return RevisionNode(confidence_threshold=0.7, max_revisions=3)

    @pytest.mark.asyncio
    async def test_high_confidence_no_revision(self, revision_node):
        result = await revision_node.revise(
            query="What is Python?",
            response="Python is a high-level programming language used for web development and data science.",
        )
        assert result.revision_count == 0
        assert not result.improved

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_revision(self, revision_node):
        result = await revision_node.revise(
            query="Explain quantum computing in detail",
            response="Hmm",  # very short = low confidence
        )
        assert result.revision_count > 0 or len(result.critique_notes) > 0

    @pytest.mark.asyncio
    async def test_stats_tracked(self, revision_node):
        await revision_node.revise("q", "a response with enough words for confidence")
        stats = revision_node.stats()
        assert stats["total_processed"] == 1

    @pytest.mark.asyncio
    async def test_with_custom_critique(self, revision_node):
        async def mock_critique(query, response):
            return {"issues": ["Too vague"], "score": 0.3}

        async def mock_generate(query, feedback):
            return "An improved and detailed response about the topic with relevant information."

        result = await revision_node.revise(
            query="Explain AI",
            response="It's stuff",
            critique_fn=mock_critique,
            generate_fn=mock_generate,
        )
        assert result.improved or result.revision_count > 0
