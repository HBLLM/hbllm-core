"""Tests for the FailureAnalyzer — Root Cause Analysis engine."""

from __future__ import annotations

import pytest

from hbllm.brain.evaluation.failure_analyzer import (
    FailureAnalyzer,
    FailureCategory,
    RootCause,
)


@pytest.fixture
def analyzer():
    return FailureAnalyzer()


class TestAuthFailureDetection:
    """Auth failures should NOT be treated as belief errors."""

    def test_401_unauthorized(self, analyzer):
        root = analyzer.analyze(
            expected="API call returns user data",
            actual="401 Unauthorized",
        )
        assert root.category == FailureCategory.AUTH_FAILURE
        assert not root.is_belief_error
        assert not root.requires_belief_revision

    def test_403_forbidden(self, analyzer):
        root = analyzer.analyze(
            expected="Access to admin panel",
            actual="403 Forbidden - permission denied",
        )
        assert root.category == FailureCategory.AUTH_FAILURE
        assert not root.is_belief_error

    def test_invalid_token(self, analyzer):
        root = analyzer.analyze(
            expected="Authenticated session",
            actual="",
            error_message="Invalid token: jwt expired",
        )
        assert root.category == FailureCategory.AUTH_FAILURE


class TestTimeoutDetection:
    """Timeouts are transient — not belief errors."""

    def test_connection_timeout(self, analyzer):
        root = analyzer.analyze(
            expected="Database query returns results",
            actual="Connection timed out after 30s",
        )
        assert root.category == FailureCategory.TIMEOUT
        assert not root.is_belief_error

    def test_gateway_timeout(self, analyzer):
        root = analyzer.analyze(
            expected="API response",
            actual="504 Gateway Timeout",
        )
        assert root.category == FailureCategory.TIMEOUT


class TestResourceMissing:
    """Missing resources — usually wrong path, not belief error."""

    def test_file_not_found(self, analyzer):
        root = analyzer.analyze(
            expected="Config file exists at /etc/config.yaml",
            actual="FileNotFoundError: No such file or directory",
        )
        assert root.category == FailureCategory.RESOURCE_MISSING
        assert not root.requires_belief_revision

    def test_404_not_found(self, analyzer):
        root = analyzer.analyze(
            expected="API endpoint available",
            actual="404 Not Found",
        )
        assert root.category == FailureCategory.RESOURCE_MISSING

    def test_stale_resource_previously_successful(self, analyzer):
        """If it worked before, it's stale knowledge — a belief error."""
        root = analyzer.analyze(
            expected="API endpoint available",
            actual="404 Not Found",
            context={"previous_success": True},
        )
        assert root.category == FailureCategory.STALE_KNOWLEDGE
        assert root.is_belief_error
        assert root.requires_belief_revision


class TestInputErrors:
    """Input/syntax errors are operational, not belief errors."""

    def test_type_error(self, analyzer):
        root = analyzer.analyze(
            expected="Function processes data",
            actual="TypeError: expected int, got str",
        )
        assert root.category == FailureCategory.WRONG_INPUT
        assert not root.is_belief_error

    def test_syntax_error(self, analyzer):
        root = analyzer.analyze(
            expected="Code executes",
            actual="SyntaxError: unexpected token",
        )
        assert root.category == FailureCategory.WRONG_INPUT

    def test_bad_request(self, analyzer):
        root = analyzer.analyze(
            expected="API accepts request",
            actual="400 Bad Request - malformed JSON",
        )
        assert root.category == FailureCategory.WRONG_INPUT


class TestTrueContradiction:
    """Real belief contradictions should be detected."""

    def test_success_vs_failure(self, analyzer):
        root = analyzer.analyze(
            expected="Command execution success",
            actual="Command execution failure",
        )
        assert root.category == FailureCategory.TRUE_CONTRADICTION
        assert root.is_belief_error
        assert root.requires_belief_revision

    def test_valid_vs_invalid(self, analyzer):
        root = analyzer.analyze(
            expected="Configuration is valid",
            actual="Configuration is invalid",
        )
        assert root.category == FailureCategory.TRUE_CONTRADICTION

    def test_exists_vs_missing(self, analyzer):
        root = analyzer.analyze(
            expected="Service exists and is running",
            actual="Service is missing and stopped",
        )
        assert root.category == FailureCategory.TRUE_CONTRADICTION

    def test_negation_detection(self, analyzer):
        root = analyzer.analyze(
            expected="PostgreSQL accepts connections",
            actual="PostgreSQL does not accept connections",
        )
        assert root.category == FailureCategory.TRUE_CONTRADICTION


class TestUnknownCategory:
    """Ambiguous failures should default to UNKNOWN."""

    def test_unknown_error(self, analyzer):
        root = analyzer.analyze(
            expected="Something happens",
            actual="Something else happened entirely differently",
        )
        assert root.category == FailureCategory.UNKNOWN
        assert not root.requires_belief_revision
        assert root.confidence < 0.5  # Low confidence


class TestRootCauseProperties:
    """Test RootCause dataclass behavior."""

    def test_belief_relevant_categories(self):
        for cat in [
            FailureCategory.TRUE_CONTRADICTION,
            FailureCategory.STALE_KNOWLEDGE,
            FailureCategory.LOGIC_ERROR,
        ]:
            root = RootCause(
                category=cat,
                description="test",
                expected="a",
                actual="b",
            )
            assert root.is_belief_error

    def test_non_belief_categories(self):
        for cat in [
            FailureCategory.AUTH_FAILURE,
            FailureCategory.TIMEOUT,
            FailureCategory.RESOURCE_MISSING,
            FailureCategory.WRONG_INPUT,
            FailureCategory.UNKNOWN,
        ]:
            root = RootCause(
                category=cat,
                description="test",
                expected="a",
                actual="b",
            )
            assert not root.is_belief_error

    def test_to_dict(self, analyzer):
        root = analyzer.analyze(
            expected="test",
            actual="401 Unauthorized",
        )
        d = root.to_dict()
        assert d["category"] == "auth_failure"
        assert "expected" in d
        assert "is_belief_error" in d


class TestAnalyzerStats:
    """Stats tracking."""

    def test_stats_tracking(self, analyzer):
        analyzer.analyze(expected="a", actual="401 Unauthorized")
        analyzer.analyze(expected="b", actual="timeout")
        analyzer.analyze(expected="c", actual="success vs failure")

        stats = analyzer.stats()
        assert stats["total_analyses"] == 3
        assert "auth_failure" in stats["by_category"]
        assert "timeout" in stats["by_category"]


class TestMechanismIdPassthrough:
    """Mechanism IDs should be preserved in root cause."""

    def test_mechanism_ids_preserved(self, analyzer):
        root = analyzer.analyze(
            expected="x",
            actual="401",
            mechanism_ids=["mech_abc", "mech_def"],
        )
        assert root.affected_mechanism_ids == ["mech_abc", "mech_def"]
