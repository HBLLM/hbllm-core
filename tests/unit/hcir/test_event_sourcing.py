"""Tests for Phase 1: Semantic Normalizer, Cognitive Journal, and Cognitive Event Log."""

from __future__ import annotations

import time

from hbllm.hcir.cognitive_event_log import CognitiveEventLog
from hbllm.hcir.cognitive_journal import CognitiveEvent, CognitiveJournal
from hbllm.hcir.semantic_normalizer import CognitiveEventKind, SemanticNormalizer
from hbllm.hcir.stores import EventType, GraphEvent, InMemoryEventStore
from hbllm.hcir.types import Provenance

# ═══════════════════════════════════════════════════════════════════════════
# Provenance Extension Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenanceExtensions:
    """Verify Provenance model has traceability + cognitive time fields."""

    def test_provenance_has_traceability_fields(self) -> None:
        p = Provenance(
            created_by="planner_node",
            session_id="session_abc",
            goal_id="goal_123",
            parent_goal_id="goal_parent",
            trace_id="trace_xyz",
            model_used="gemini-2.5-flash",
            reason="User asked a question",
            source_node="router_node",
            source_type="inferred",
        )
        assert p.session_id == "session_abc"
        assert p.goal_id == "goal_123"
        assert p.parent_goal_id == "goal_parent"
        assert p.trace_id == "trace_xyz"
        assert p.model_used == "gemini-2.5-flash"
        assert p.reason == "User asked a question"
        assert p.source_node == "router_node"
        assert p.source_type == "inferred"

    def test_provenance_has_cognitive_time_fields(self) -> None:
        p = Provenance(
            logical_time=42,
            generation=3,
            attention_epoch=7,
            reflection_cycle=2,
        )
        assert p.logical_time == 42
        assert p.generation == 3
        assert p.attention_epoch == 7
        assert p.reflection_cycle == 2

    def test_provenance_defaults_are_backward_compatible(self) -> None:
        """All new fields should have defaults so existing code doesn't break."""
        p = Provenance()
        assert p.session_id == ""
        assert p.goal_id == ""
        assert p.parent_goal_id == ""
        assert p.trace_id == ""
        assert p.model_used == ""
        assert p.reason == ""
        assert p.source_node == ""
        assert p.source_type == ""
        assert p.logical_time == 0
        assert p.generation == 0
        assert p.attention_epoch == 0
        assert p.reflection_cycle == 0
        # Original fields still work
        assert p.created_by == ""
        assert p.simulation_branch == "main"

    def test_provenance_serialization_roundtrip(self) -> None:
        p = Provenance(
            created_by="test",
            session_id="sess_1",
            goal_id="goal_1",
            trace_id="trace_1",
            logical_time=10,
            generation=2,
        )
        d = p.model_dump()
        p2 = Provenance.model_validate(d)
        assert p2.session_id == "sess_1"
        assert p2.logical_time == 10
        assert p2.generation == 2


# ═══════════════════════════════════════════════════════════════════════════
# EventType Extension Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEventTypeExtensions:
    """Verify new cognitive domain EventType entries exist."""

    def test_cognitive_domain_event_types_exist(self) -> None:
        # Perception
        assert EventType.PERCEPTION_RECEIVED == "perception_received"
        # Memory
        assert EventType.MEMORY_STORED == "memory_stored"
        assert EventType.MEMORY_RECALLED == "memory_recalled"
        assert EventType.MEMORY_CONSOLIDATED == "memory_consolidated"
        # Planning
        assert EventType.GOAL_CREATED == "goal_created"
        assert EventType.GOAL_COMPLETED == "goal_completed"
        assert EventType.GOAL_ABANDONED == "goal_abandoned"
        assert EventType.GOAL_BLOCKED == "goal_blocked"
        # Execution
        assert EventType.DECISION_MADE == "decision_made"
        assert EventType.ACTION_PLANNED == "action_planned"
        assert EventType.ACTION_EXECUTED == "action_executed"
        assert EventType.ACTION_RESULT == "action_result"
        assert EventType.CAPABILITY_INVOKED == "capability_invoked"
        # Epistemology
        assert EventType.BELIEF_UPDATED == "belief_updated"
        assert EventType.BELIEF_REVISED == "belief_revised"
        assert EventType.PREDICTION_MADE == "prediction_made"
        assert EventType.PREDICTION_VERIFIED == "prediction_verified"
        assert EventType.PREDICTION_ERROR == "prediction_error"
        # Governance
        assert EventType.GOVERNANCE_EVALUATED == "governance_evaluated"
        assert EventType.GOVERNANCE_BLOCKED == "governance_blocked"
        # Learning
        assert EventType.SKILL_LEARNED == "skill_learned"
        assert EventType.LEARNING_EVENT == "learning_event"
        # Cognitive state
        assert EventType.COGNITIVE_STATE_CHANGED == "cognitive_state_changed"
        assert EventType.ATTENTION_SHIFTED == "attention_shifted"
        # Transaction compensation
        assert EventType.TRANSACTION_COMPENSATED == "transaction_compensated"

    def test_graph_event_has_traceability_fields(self) -> None:
        ge = GraphEvent(
            sequence=1,
            event_type=EventType.GOAL_CREATED,
            timestamp=time.time(),
            author="planner",
            session_id="sess_1",
            goal_id="goal_1",
            trace_id="trace_1",
            logical_time=5,
            generation=1,
        )
        assert ge.session_id == "sess_1"
        assert ge.goal_id == "goal_1"
        assert ge.trace_id == "trace_1"
        assert ge.logical_time == 5
        assert ge.generation == 1

    def test_graph_event_traceability_defaults(self) -> None:
        ge = GraphEvent(
            sequence=1,
            event_type=EventType.NODE_ADDED,
            timestamp=time.time(),
            author="system",
        )
        assert ge.session_id == ""
        assert ge.goal_id == ""
        assert ge.trace_id == ""
        assert ge.logical_time == 0
        assert ge.generation == 0


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Normalizer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticNormalizer:
    """Verify the SemanticNormalizer maps raw events to canonical kinds."""

    def setup_method(self) -> None:
        self.normalizer = SemanticNormalizer()

    def test_exact_topic_match(self) -> None:
        assert self.normalizer.normalize("memory.store") == CognitiveEventKind.MEMORY_STORED
        assert self.normalizer.normalize("memory.search") == CognitiveEventKind.MEMORY_RECALLED
        assert self.normalizer.normalize("decision.made") == CognitiveEventKind.DECISION_MADE

    def test_prefix_match(self) -> None:
        # "perception.vision" matches "perception.vision"
        assert (
            self.normalizer.normalize("perception.vision") == CognitiveEventKind.PERCEPTION_RECEIVED
        )
        assert (
            self.normalizer.normalize("perception.audio") == CognitiveEventKind.PERCEPTION_RECEIVED
        )

    def test_unrecognized_topic_returns_none(self) -> None:
        assert self.normalizer.normalize("unknown.topic") is None

    def test_alias_from_message_data(self) -> None:
        """Aliases resolve from message.data['event_name']."""

        class MockMessage:
            type = "query"
            data = {"event_name": "GoalCreated"}

        kind = self.normalizer.normalize("some.topic", MockMessage())
        assert kind == CognitiveEventKind.GOAL_CREATED

    def test_multiple_aliases_normalize_to_same_kind(self) -> None:
        """Different names for the same concept all normalize identically."""

        class Msg1:
            type = "query"
            data = {"event_name": "GoalCreated"}

        class Msg2:
            type = "query"
            data = {"event_name": "NewGoal"}

        class Msg3:
            type = "query"
            data = {"event_name": "IntentGoal"}

        class Msg4:
            type = "query"
            data = {"event_name": "GeneratedGoal"}

        for msg in [Msg1(), Msg2(), Msg3(), Msg4()]:
            assert self.normalizer.normalize("custom.topic", msg) == CognitiveEventKind.GOAL_CREATED

    def test_register_custom_topic(self) -> None:
        self.normalizer.register_topic(
            "my_plugin.detected", CognitiveEventKind.OBSERVATION_RECEIVED
        )
        assert (
            self.normalizer.normalize("my_plugin.detected")
            == CognitiveEventKind.OBSERVATION_RECEIVED
        )

    def test_register_custom_alias(self) -> None:
        self.normalizer.register_alias("MyCustomGoal", CognitiveEventKind.GOAL_CREATED)

        class Msg:
            type = "query"
            data = {"event_name": "MyCustomGoal"}

        assert self.normalizer.normalize("custom.topic", Msg()) == CognitiveEventKind.GOAL_CREATED


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Journal Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveJournal:
    """Verify the CognitiveJournal records and replays events correctly."""

    def setup_method(self) -> None:
        self.store = InMemoryEventStore()
        self.journal = CognitiveJournal(self.store)

    def test_record_returns_sequence_number(self) -> None:
        event = CognitiveEvent(
            kind=CognitiveEventKind.GOAL_CREATED,
            author="planner",
            trace_id="trace_1",
        )
        seq = self.journal.record(event)
        assert seq == 1

    def test_sequential_recording(self) -> None:
        for i in range(5):
            event = CognitiveEvent(
                kind=CognitiveEventKind.MEMORY_STORED,
                author=f"node_{i}",
            )
            seq = self.journal.record(event)
            assert seq == i + 1
        assert self.journal.latest_sequence == 5

    def test_logical_clock_advances(self) -> None:
        event = CognitiveEvent(kind=CognitiveEventKind.GOAL_CREATED, author="test")
        self.journal.record(event)
        assert self.journal.logical_clock == 1

        self.journal.record(event)
        assert self.journal.logical_clock == 2

    def test_logical_clock_respects_remote_time(self) -> None:
        event = CognitiveEvent(
            kind=CognitiveEventKind.GOAL_CREATED,
            author="remote",
            logical_time=100,
        )
        self.journal.record(event)
        assert self.journal.logical_clock == 101

    def test_replay_returns_all_events(self) -> None:
        for kind in [
            CognitiveEventKind.GOAL_CREATED,
            CognitiveEventKind.DECISION_MADE,
            CognitiveEventKind.ACTION_EXECUTED,
        ]:
            self.journal.record(CognitiveEvent(kind=kind, author="test"))

        events = list(self.journal.replay())
        assert len(events) == 3
        assert events[0].kind == CognitiveEventKind.GOAL_CREATED
        assert events[1].kind == CognitiveEventKind.DECISION_MADE
        assert events[2].kind == CognitiveEventKind.ACTION_EXECUTED

    def test_replay_with_range(self) -> None:
        for i in range(10):
            self.journal.record(
                CognitiveEvent(kind=CognitiveEventKind.MEMORY_STORED, author=f"n{i}")
            )

        events = list(self.journal.replay(from_seq=5, to_seq=7))
        assert len(events) == 3  # seq 5, 6, 7

    def test_replay_by_trace(self) -> None:
        self.journal.record(
            CognitiveEvent(
                kind=CognitiveEventKind.GOAL_CREATED,
                author="test",
                trace_id="trace_A",
            )
        )
        self.journal.record(
            CognitiveEvent(
                kind=CognitiveEventKind.DECISION_MADE,
                author="test",
                trace_id="trace_B",
            )
        )
        self.journal.record(
            CognitiveEvent(
                kind=CognitiveEventKind.ACTION_EXECUTED,
                author="test",
                trace_id="trace_A",
            )
        )

        trace_a = self.journal.replay_by_trace("trace_A")
        assert len(trace_a) == 2
        assert trace_a[0].kind == CognitiveEventKind.GOAL_CREATED
        assert trace_a[1].kind == CognitiveEventKind.ACTION_EXECUTED

    def test_event_provenance_survives_roundtrip(self) -> None:
        event = CognitiveEvent(
            kind=CognitiveEventKind.SKILL_LEARNED,
            author="learner_node",
            tenant_id="tenant_42",
            session_id="sess_1",
            goal_id="goal_abc",
            trace_id="trace_xyz",
            model_used="gemini-2.5-flash",
            confidence=0.95,
            reason="User explicitly requested learning",
            source_node="experience_node",
            data={"skill_name": "weather_lookup"},
        )
        self.journal.record(event)

        replayed = list(self.journal.replay())[0]
        assert replayed.kind == CognitiveEventKind.SKILL_LEARNED
        assert replayed.author == "learner_node"
        assert replayed.tenant_id == "tenant_42"
        assert replayed.session_id == "sess_1"
        assert replayed.goal_id == "goal_abc"
        assert replayed.trace_id == "trace_xyz"
        assert replayed.model_used == "gemini-2.5-flash"
        assert replayed.confidence == 0.95
        assert replayed.reason == "User explicitly requested learning"

    def test_count(self) -> None:
        assert self.journal.count() == 0
        self.journal.record(CognitiveEvent(kind=CognitiveEventKind.GOAL_CREATED, author="test"))
        assert self.journal.count() == 1


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Event Log Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveEventLog:
    """Verify the CognitiveEventLog filters and indexes events correctly."""

    def setup_method(self) -> None:
        self.store = InMemoryEventStore()
        self.log = CognitiveEventLog(self.store)

    def test_significant_event_is_recorded(self) -> None:
        event = CognitiveEvent(
            kind=CognitiveEventKind.GOAL_CREATED,
            author="planner",
            trace_id="trace_1",
        )
        assert self.log.record_if_significant(event) is True
        assert self.log.latest_sequence == 1

    def test_insignificant_event_is_filtered(self) -> None:
        event = CognitiveEvent(
            kind=CognitiveEventKind.ROUTING_DECIDED,
            author="router",
        )
        assert self.log.record_if_significant(event) is False
        assert self.log.latest_sequence == 0

    def test_memory_recalled_is_filtered(self) -> None:
        """MEMORY_RECALLED is not in the significant events set (it's read-only)."""
        event = CognitiveEvent(
            kind=CognitiveEventKind.MEMORY_RECALLED,
            author="memory",
        )
        assert self.log.record_if_significant(event) is False

    def test_query_by_trace(self) -> None:
        for kind in [
            CognitiveEventKind.GOAL_CREATED,
            CognitiveEventKind.DECISION_MADE,
            CognitiveEventKind.ACTION_EXECUTED,
        ]:
            self.log.record_if_significant(
                CognitiveEvent(kind=kind, author="test", trace_id="trace_A")
            )

        self.log.record_if_significant(
            CognitiveEvent(
                kind=CognitiveEventKind.MEMORY_STORED,
                author="test",
                trace_id="trace_B",
            )
        )

        results = self.log.query_by_trace("trace_A")
        assert len(results) == 3
        assert all(r.trace_id == "trace_A" for r in results)

    def test_query_by_goal(self) -> None:
        self.log.record_if_significant(
            CognitiveEvent(
                kind=CognitiveEventKind.GOAL_CREATED,
                author="planner",
                goal_id="goal_123",
            )
        )
        self.log.record_if_significant(
            CognitiveEvent(
                kind=CognitiveEventKind.ACTION_PLANNED,
                author="planner",
                goal_id="goal_123",
            )
        )
        self.log.record_if_significant(
            CognitiveEvent(
                kind=CognitiveEventKind.GOAL_COMPLETED,
                author="planner",
                goal_id="goal_456",
            )
        )

        results = self.log.query_by_goal("goal_123")
        assert len(results) == 2

    def test_query_by_session(self) -> None:
        self.log.record_if_significant(
            CognitiveEvent(
                kind=CognitiveEventKind.GOAL_CREATED,
                author="test",
                session_id="sess_A",
            )
        )
        self.log.record_if_significant(
            CognitiveEvent(
                kind=CognitiveEventKind.DECISION_MADE,
                author="test",
                session_id="sess_A",
            )
        )

        results = self.log.query_by_session("sess_A")
        assert len(results) == 2

    def test_query_by_kind(self) -> None:
        for _ in range(5):
            self.log.record_if_significant(
                CognitiveEvent(kind=CognitiveEventKind.BELIEF_UPDATED, author="test")
            )
        self.log.record_if_significant(
            CognitiveEvent(kind=CognitiveEventKind.GOAL_CREATED, author="test")
        )

        beliefs = self.log.query_by_kind(CognitiveEventKind.BELIEF_UPDATED)
        assert len(beliefs) == 5
        goals = self.log.query_by_kind(CognitiveEventKind.GOAL_CREATED)
        assert len(goals) == 1

    def test_query_by_kind_with_limit(self) -> None:
        for _ in range(20):
            self.log.record_if_significant(
                CognitiveEvent(kind=CognitiveEventKind.MEMORY_STORED, author="test")
            )

        recent = self.log.query_by_kind(CognitiveEventKind.MEMORY_STORED, limit=5)
        assert len(recent) == 5

    def test_replay(self) -> None:
        for kind in [
            CognitiveEventKind.GOAL_CREATED,
            CognitiveEventKind.DECISION_MADE,
            CognitiveEventKind.SKILL_LEARNED,
        ]:
            self.log.record_if_significant(CognitiveEvent(kind=kind, author="test"))

        events = list(self.log.replay())
        assert len(events) == 3
        assert events[0].kind == CognitiveEventKind.GOAL_CREATED
        assert events[2].kind == CognitiveEventKind.SKILL_LEARNED

    def test_add_significant_kind(self) -> None:
        """After adding ROUTING_DECIDED to significant, it should be recorded."""
        event = CognitiveEvent(kind=CognitiveEventKind.ROUTING_DECIDED, author="test")
        assert self.log.record_if_significant(event) is False

        self.log.add_significant_kind(CognitiveEventKind.ROUTING_DECIDED)
        assert self.log.record_if_significant(event) is True

    def test_remove_significant_kind(self) -> None:
        self.log.remove_significant_kind(CognitiveEventKind.GOAL_CREATED)
        event = CognitiveEvent(kind=CognitiveEventKind.GOAL_CREATED, author="test")
        assert self.log.record_if_significant(event) is False

    def test_empty_queries_return_empty_lists(self) -> None:
        assert self.log.query_by_trace("nonexistent") == []
        assert self.log.query_by_goal("nonexistent") == []
        assert self.log.query_by_session("nonexistent") == []
        assert self.log.query_by_kind(CognitiveEventKind.GOAL_CREATED) == []
