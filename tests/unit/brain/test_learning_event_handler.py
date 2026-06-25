"""Tests for the LearningEventHandler — event-driven learning bridge.

Tests cover:
    - Experience events (success/failure) — full belief pipeline
    - Research events (session complete, contradiction discovered) — NO echo
    - Curiosity events (research complete) — mechanism storage
    - Queue processing
    - LearningSubsystem integration
"""

from __future__ import annotations

import pytest

from hbllm.brain.failure_analyzer import FailureAnalyzer
from hbllm.brain.learning_event_handler import LearningEventHandler
from hbllm.brain.learning_subsystem import LearningSubsystem
from hbllm.brain.mechanism_store import MechanismStore
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def mechanism_store(tmp_path):
    return MechanismStore(data_dir=str(tmp_path))


@pytest.fixture
def failure_analyzer():
    return FailureAnalyzer()


@pytest.fixture
def learning_subsystem(mechanism_store, failure_analyzer):
    """Build a LearningSubsystem with only the always-available components."""
    return LearningSubsystem(
        mechanism_store=mechanism_store,
        failure_analyzer=failure_analyzer,
    )


def _make_handler(mechanism_store, failure_analyzer):
    """Create a LearningEventHandler with legacy direct injection."""
    return LearningEventHandler(
        node_id="test_learning",
        mechanism_store=mechanism_store,
        failure_analyzer=failure_analyzer,
    )


def _make_handler_with_subsystem(subsystem):
    """Create a LearningEventHandler with LearningSubsystem."""
    return LearningEventHandler(
        node_id="test_learning",
        learning_subsystem=subsystem,
    )


class TestSuccessHandling:
    """Test learning from successful experiences."""

    @pytest.mark.asyncio
    async def test_success_reinforces_mechanisms(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        mech = mechanism_store.create(
            description="Test mechanism",
            preconditions=["a"],
            process_steps=["b"],
            expected_outcomes=["c"],
            confidence=0.8,
        )

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="learning.experience.success",
            payload={
                "domain": "test",
                "query": "test query",
                "mechanism_ids": [mech.id],
                "execution_trace": [{"step": "a", "result": "ok"}],
            },
        )

        await handler._handle_success(msg)

        updated = mechanism_store.get(mech.id)
        assert updated.usage_count == 1
        assert updated.success_count == 1
        assert updated.confidence > 0.8

    @pytest.mark.asyncio
    async def test_success_queues_model_building(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="learning.experience.success",
            payload={
                "domain": "security",
                "query": "check for sql injection",
                "mechanism_ids": [],
                "execution_trace": [{"step": "scan", "result": "found"}],
            },
        )

        await handler._handle_success(msg)
        assert handler.get_queue_size() == 1
        assert handler._stats["models_queued"] == 1

    @pytest.mark.asyncio
    async def test_success_without_mechanisms(self, mechanism_store, failure_analyzer):
        """Should work fine even without mechanism IDs."""
        handler = _make_handler(mechanism_store, failure_analyzer)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="learning.experience.success",
            payload={
                "domain": "general",
                "query": "hello world",
                "execution_trace": [],
            },
        )

        await handler._handle_success(msg)
        assert handler._stats["successes_processed"] == 1


class TestFailureHandling:
    """Test learning from failed experiences."""

    @pytest.mark.asyncio
    async def test_failure_analyzes_root_cause(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        mech = mechanism_store.create(
            description="Auth mechanism",
            preconditions=["credentials"],
            process_steps=["validate"],
            expected_outcomes=["access"],
            confidence=0.8,
        )

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sil",
            topic="learning.experience.failure",
            payload={
                "domain": "auth",
                "expected": "API returns user data",
                "actual": "401 Unauthorized",
                "error_message": "Unauthorized",
                "mechanism_ids": [mech.id],
            },
        )

        await handler._handle_failure(msg)

        # Auth failure — NOT a belief error, but mechanism still records failure
        updated = mechanism_store.get(mech.id)
        assert updated.usage_count == 1
        assert updated.failure_count == 1
        assert handler._stats["failures_processed"] == 1

    @pytest.mark.asyncio
    async def test_true_contradiction_triggers_revision(self, mechanism_store, failure_analyzer):
        """True contradictions should trigger belief revision attempts."""
        handler = _make_handler(mechanism_store, failure_analyzer)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sil",
            topic="learning.experience.failure",
            payload={
                "expected": "Service is running",
                "actual": "Service is not running",
                "error_message": "Connection refused",
                "mechanism_ids": [],
            },
        )

        await handler._handle_failure(msg)
        # Since we have no belief_engine, it should still process without error
        assert handler._stats["failures_processed"] == 1

    @pytest.mark.asyncio
    async def test_timeout_not_treated_as_belief_error(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="learning.experience.failure",
            payload={
                "expected": "Response",
                "actual": "Connection timed out",
                "error_message": "Timeout after 30s",
                "mechanism_ids": [],
            },
        )

        await handler._handle_failure(msg)
        # Timeout should NOT trigger belief revision
        assert handler._stats["belief_revisions_triggered"] == 0


class TestSessionComplete:
    """Test handling of learning.session.complete — NO belief/meta echo."""

    @pytest.mark.asyncio
    async def test_session_complete_increments_counter(self, learning_subsystem):
        """Session events should be counted but NOT trigger belief updates."""
        handler = _make_handler_with_subsystem(learning_subsystem)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.session.complete",
            payload={
                "session_id": "lg_test123",
                "topic": "SQL Injection",
                "concepts_learned": ["user_input", "query_parsing"],
                "causal_models_built": 2,
                "confidence_before": 0.0,
                "confidence_after": 0.65,
                "experiments_run": 1,
                "contradictions_found": 0,
                "contradictions_resolved": 0,
                "duration_s": 42.0,
                "status": "completed",
            },
        )

        await handler._handle_session_complete(msg)

        assert handler._stats["sessions_received"] == 1
        # Critical: NO belief updates should have happened
        assert handler._stats["belief_revisions_triggered"] == 0
        assert handler._stats["successes_processed"] == 0

    @pytest.mark.asyncio
    async def test_session_complete_no_double_counting(self, learning_subsystem):
        """Verify the echo loop prevention invariant.

        AutonomousLearner already called BeliefRevisionEngine and MetaLearner.
        LearningEventHandler must NOT call them again.
        """
        handler = _make_handler_with_subsystem(learning_subsystem)

        # Send session complete — should NOT touch belief/meta
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.session.complete",
            payload={
                "session_id": "lg_abc",
                "topic": "XSS",
                "concepts_learned": ["dom_manipulation"],
                "causal_models_built": 1,
                "confidence_before": 0.0,
                "confidence_after": 0.5,
            },
        )
        await handler._handle_session_complete(msg)

        # Now send a REAL execution success — this SHOULD touch belief/meta
        success_msg = Message(
            type=MessageType.EVENT,
            source_node_id="workspace",
            topic="learning.experience.success",
            payload={
                "domain": "security",
                "query": "scan for xss",
                "mechanism_ids": [],
                "execution_trace": [],
            },
        )
        await handler._handle_success(success_msg)

        # Only the execution event should count
        assert handler._stats["sessions_received"] == 1
        assert handler._stats["successes_processed"] == 1


class TestContradictionDiscovered:
    """Test handling of learning.contradiction.discovered — log only, NO belief update."""

    @pytest.mark.asyncio
    async def test_contradiction_logged_not_revised(self, learning_subsystem):
        """Research contradictions should be logged but NOT trigger belief revision."""
        handler = _make_handler_with_subsystem(learning_subsystem)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="autonomous_learner",
            topic="learning.contradiction.discovered",
            payload={
                "claim_a": "X causes Y",
                "claim_b": "X does not cause Y",
                "concept": "causality_test",
                "severity": 0.8,
                "source": "autonomous_research",
            },
        )

        await handler._handle_contradiction_discovered(msg)

        assert handler._stats["contradictions_received"] == 1
        # Critical: NO belief revision should happen here
        assert handler._stats["belief_revisions_triggered"] == 0


class TestResearchComplete:
    """Test handling of completed autonomous research."""

    @pytest.mark.asyncio
    async def test_research_stores_mechanisms(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="curiosity",
            topic="curiosity.research.complete",
            payload={
                "domain": "security",
                "source_goal": "Understand XSS vulnerabilities",
                "findings": [{"observation": "XSS exploits user input rendering"}],
                "new_mechanisms": [
                    {
                        "description": "Input Sanitization for HTML",
                        "preconditions": ["user input", "HTML rendering"],
                        "process_steps": ["escape special chars", "validate format"],
                        "expected_outcomes": ["safe HTML output"],
                        "abstraction_level": 1,
                    }
                ],
            },
        )

        await handler._handle_research_complete(msg)

        # Verify mechanism was stored
        security_mechs = mechanism_store.find_by_domain("security")
        assert len(security_mechs) == 1
        assert security_mechs[0].description == "Input Sanitization for HTML"
        assert security_mechs[0].abstraction_level == 1


class TestQueueProcessing:
    """Test background model building queue."""

    @pytest.mark.asyncio
    async def test_empty_queue_returns_zero(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        built = await handler.process_build_queue()
        assert built == 0

    @pytest.mark.asyncio
    async def test_queue_without_builder_returns_zero(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        # Queue something but no builder
        handler._model_build_queue.append({"domain": "test", "trace": [{}]})
        built = await handler.process_build_queue()
        assert built == 0

    @pytest.mark.asyncio
    async def test_stats_action(self, mechanism_store, failure_analyzer):
        handler = _make_handler(mechanism_store, failure_analyzer)
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="learning.stats",
            payload={"action": "stats"},
        )
        resp = await handler.handle_message(msg)
        assert resp is not None
        assert "stats" in resp.payload


class TestLearningSubsystem:
    """Test LearningSubsystem integration."""

    def test_subsystem_summary(self, learning_subsystem):
        summary = learning_subsystem.summary()
        assert summary["mechanism_store"] is True
        assert summary["failure_analyzer"] is True
        assert summary["belief_engine"] is False
        assert summary["has_belief_infrastructure"] is False
        assert summary["has_research_infrastructure"] is False

    def test_handler_uses_subsystem_components(self, learning_subsystem):
        handler = _make_handler_with_subsystem(learning_subsystem)
        assert handler.mechanism_store is learning_subsystem.mechanism_store
        assert handler.failure_analyzer is learning_subsystem.failure_analyzer
        assert handler.belief_engine is None
        assert handler.contradiction_detector is None

    def test_inject_subsystem_replaces(self, mechanism_store, failure_analyzer):
        """inject_subsystem should replace the current subsystem."""
        handler = _make_handler(mechanism_store, failure_analyzer)
        assert handler.mechanism_store is mechanism_store

        new_store = MechanismStore(data_dir="/tmp/test_inject")
        new_subsystem = LearningSubsystem(
            mechanism_store=new_store,
            failure_analyzer=failure_analyzer,
        )
        handler.inject_subsystem(new_subsystem)
        assert handler.mechanism_store is new_store

    def test_stats_include_subsystem_summary(self, learning_subsystem):
        handler = _make_handler_with_subsystem(learning_subsystem)
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="learning.stats",
            payload={"action": "stats"},
        )
        import asyncio

        resp = asyncio.get_event_loop().run_until_complete(handler.handle_message(msg))
        assert "subsystem" in resp.payload
        assert resp.payload["subsystem"]["mechanism_store"] is True
