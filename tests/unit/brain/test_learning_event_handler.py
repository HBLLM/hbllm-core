"""Tests for the LearningEventHandler — event-driven learning bridge."""

from __future__ import annotations

import pytest

from hbllm.brain.failure_analyzer import FailureAnalyzer
from hbllm.brain.learning_event_handler import LearningEventHandler
from hbllm.brain.mechanism_store import MechanismStore
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def mechanism_store(tmp_path):
    return MechanismStore(data_dir=str(tmp_path))


@pytest.fixture
def failure_analyzer():
    return FailureAnalyzer()


def _make_handler(mechanism_store, failure_analyzer):
    """Create a LearningEventHandler without bus registration."""
    return LearningEventHandler(
        node_id="test_learning",
        mechanism_store=mechanism_store,
        failure_analyzer=failure_analyzer,
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
