"""
Unit tests for ADR 002 — Operational Architecture Implementation.

Tests all 7 operational architecture components:
    1. Universal Causal Provenance
    2. Lightweight Language-Independent Intent
    3. Tripartite Executive Controllers
    4. Resource-Budget Priority Scheduler
    5. Observability & Decision Replay
    6. Generalized Forecasting Interface
    7. Ephemeral DigitalTwin
"""

from __future__ import annotations

import time
from typing import Any

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Provenance Tests
# ═══════════════════════════════════════════════════════════════════════════
from hbllm.brain.core.provenance import ProvenanceMetadata, VerificationState


class TestProvenance:
    """Test universal causal provenance primitives."""

    def test_create_root_provenance(self) -> None:
        prov = ProvenanceMetadata.create(
            source="perception.audio_in",
            confidence=0.92,
            expiry_seconds=300.0,
        )
        assert prov.source == "perception.audio_in"
        assert prov.confidence == 0.92
        assert prov.parent_event_id is None
        assert prov.expiry is not None
        assert prov.verification_state == VerificationState.UNVERIFIED
        assert len(prov.event_id) == 32  # UUID4 hex

    def test_derive_child_provenance(self) -> None:
        parent = ProvenanceMetadata.create(
            source="perception.audio_in",
            confidence=0.9,
            correlation_id="session_abc",
        )
        child = ProvenanceMetadata.derive(
            parent=parent,
            source="brain.snn.comprehension",
            confidence=0.85,
        )
        assert child.parent_event_id == parent.event_id
        assert child.correlation_id == parent.correlation_id
        assert child.source == "brain.snn.comprehension"
        assert child.confidence == 0.85

    def test_expiry_check(self) -> None:
        prov = ProvenanceMetadata.create(
            source="test",
            expiry_seconds=-1.0,  # Already expired
        )
        assert prov.is_expired is True
        assert prov.effective_verification == VerificationState.STALE

    def test_trustworthiness_check(self) -> None:
        prov = ProvenanceMetadata.create(
            source="test",
            confidence=0.8,
            verification_state=VerificationState.VERIFIED,
        )
        assert prov.is_trustworthy is True

        low_conf = ProvenanceMetadata.create(
            source="test",
            confidence=0.2,
            verification_state=VerificationState.VERIFIED,
        )
        assert low_conf.is_trustworthy is False

    def test_serialization_roundtrip(self) -> None:
        prov = ProvenanceMetadata.create(
            source="test.node",
            confidence=0.75,
            expiry_seconds=60.0,
        )
        d = prov.to_dict()
        restored = ProvenanceMetadata.from_dict(d)
        assert restored.event_id == prov.event_id
        assert restored.source == prov.source
        assert restored.confidence == prov.confidence

    def test_causal_chain(self) -> None:
        root = ProvenanceMetadata.create(source="root")
        child1 = ProvenanceMetadata.derive(parent=root, source="child1")
        child2 = ProvenanceMetadata.derive(parent=child1, source="child2")
        assert child2.parent_event_id == child1.event_id
        assert child1.parent_event_id == root.event_id
        assert root.parent_event_id is None


# ═══════════════════════════════════════════════════════════════════════════
# Step 1b: CognitiveEvent with Provenance
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.core.cognitive_event import CognitiveEvent, CognitiveEventType


class TestCognitiveEventProvenance:
    """Test provenance integration in CognitiveEvent."""

    def test_event_with_provenance(self) -> None:
        prov = ProvenanceMetadata.create(source="test_node")
        event = CognitiveEvent(
            type=CognitiveEventType.USER_SPOKE,
            source_node="perception",
            provenance=prov,
        )
        assert event.provenance is not None
        assert event.provenance.source == "test_node"

    def test_event_without_provenance_backward_compat(self) -> None:
        event = CognitiveEvent(
            type=CognitiveEventType.USER_SPOKE,
            source_node="perception",
        )
        assert event.provenance is None

    def test_with_saliency_preserves_provenance(self) -> None:
        prov = ProvenanceMetadata.create(source="test")
        event = CognitiveEvent(
            type=CognitiveEventType.ATTENTION_SPIKE,
            source_node="snn",
            provenance=prov,
        )
        updated = event.with_saliency(0.95)
        assert updated.provenance is not None
        assert updated.provenance.event_id == prov.event_id

    def test_to_dict_includes_provenance(self) -> None:
        prov = ProvenanceMetadata.create(source="test")
        event = CognitiveEvent(
            type=CognitiveEventType.MEMORY_UPDATED,
            source_node="memory",
            provenance=prov,
        )
        d = event.to_dict()
        assert "provenance" in d
        assert d["provenance"]["source"] == "test"


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Intent Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.control.intent import Intent, IntentStatus, IntentType


class TestIntent:
    """Test lightweight language-independent intent abstraction."""

    def test_create_intent(self) -> None:
        intent = Intent.create(
            intent_type=IntentType.QUERY,
            semantic_target="weather forecast",
            parameters={"location": "Colombo"},
            source_text="What's the weather?",
            source_node="perception.audio_in",
        )
        assert intent.intent_type == IntentType.QUERY
        assert intent.semantic_target == "weather forecast"
        assert intent.parameters == {"location": "Colombo"}
        assert intent.status == IntentStatus.PENDING
        assert intent.provenance is not None
        assert intent.provenance.source == "perception.audio_in"

    def test_lifecycle_transitions(self) -> None:
        intent = Intent.create(
            intent_type=IntentType.COMMAND,
            semantic_target="send email",
        )
        assert intent.status == IntentStatus.PENDING

        active = intent.activate()
        assert active.status == IntentStatus.ACTIVE

        completed = active.complete()
        assert completed.status == IntentStatus.COMPLETED

    def test_fail_transition(self) -> None:
        intent = Intent.create(
            intent_type=IntentType.QUERY,
            semantic_target="unknown",
        )
        failed = intent.fail()
        assert failed.status == IntentStatus.FAILED

    def test_serialization_roundtrip(self) -> None:
        intent = Intent.create(
            intent_type=IntentType.CONVERSATION,
            semantic_target="greet user",
            parameters={"tone": "friendly"},
        )
        d = intent.to_dict()
        restored = Intent.from_dict(d)
        assert restored.intent_id == intent.intent_id
        assert restored.intent_type == IntentType.CONVERSATION
        assert restored.semantic_target == "greet user"

    def test_intent_type_coverage(self) -> None:
        """All intent types should be constructible."""
        for it in IntentType:
            intent = Intent.create(
                intent_type=it,
                semantic_target=f"test_{it.value}",
            )
            assert intent.intent_type == it


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Tripartite Executive Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.control.deliberative_controller import DeliberativeController
from hbllm.brain.control.reactive_controller import (
    ReactiveController,
    ReflexResult,
    ReflexType,
)
from hbllm.brain.control.reflective_controller import (
    ReflectionType,
    ReflectiveController,
)


class TestReactiveController:
    """Test sub-10ms reflex handler."""

    @pytest.mark.asyncio
    async def test_no_reflex_passthrough(self) -> None:
        controller = ReactiveController()
        event = CognitiveEvent(
            type=CognitiveEventType.USER_SPOKE,
            source_node="test",
        )
        result = await controller.process(event)
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_registered_reflex_fires(self) -> None:
        controller = ReactiveController()

        async def wake_handler(event: CognitiveEvent) -> ReflexResult:
            return ReflexResult(handled=True, action_taken="wake_word_detected")

        controller.register_reflex(
            CognitiveEventType.ATTENTION_SPIKE,
            ReflexType.WAKE_WORD,
            wake_handler,
        )
        event = CognitiveEvent(
            type=CognitiveEventType.ATTENTION_SPIKE,
            source_node="perception",
        )
        result = await controller.process(event)
        assert result.handled is True
        assert result.reflex_type == ReflexType.WAKE_WORD
        assert result.provenance is not None

    def test_is_reactive_event(self) -> None:
        controller = ReactiveController()

        async def dummy(e: CognitiveEvent) -> ReflexResult:
            return ReflexResult(handled=True)

        controller.register_reflex(
            CognitiveEventType.ATTENTION_SPIKE,
            ReflexType.SAFETY_INTERRUPT,
            dummy,
        )
        reactive = CognitiveEvent(
            type=CognitiveEventType.ATTENTION_SPIKE,
            source_node="test",
        )
        non_reactive = CognitiveEvent(
            type=CognitiveEventType.MEMORY_UPDATED,
            source_node="test",
        )
        assert controller.is_reactive_event(reactive) is True
        assert controller.is_reactive_event(non_reactive) is False


class TestDeliberativeController:
    """Test multi-step planning controller."""

    @pytest.mark.asyncio
    async def test_basic_deliberation(self) -> None:
        controller = DeliberativeController()
        intent = Intent.create(
            intent_type=IntentType.QUERY,
            semantic_target="weather forecast",
        )
        result = await controller.deliberate(intent)
        assert result.success is True
        assert result.plan is not None
        assert result.plan.intent_id == intent.intent_id
        assert len(result.reasoning_trace) > 0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_deliberation_produces_provenance(self) -> None:
        controller = DeliberativeController()
        intent = Intent.create(
            intent_type=IntentType.COMMAND,
            semantic_target="send notification",
        )
        result = await controller.deliberate(intent)
        assert result.provenance is not None
        assert result.provenance.source == "deliberative_controller"


class TestReflectiveController:
    """Test post-execution evaluation controller."""

    @pytest.mark.asyncio
    async def test_basic_reflection(self) -> None:
        controller = ReflectiveController()
        result = await controller.reflect(
            task_id="task_001",
            outcome={"success": True, "tools_used": ["tool_a"]},
            reasoning_trace=[{"step": "extract_goal", "result": "test"}],
        )
        assert result.total_score > 0
        assert len(result.events) >= 2  # outcome + consolidation
        assert result.events[0].reflection_type == ReflectionType.OUTCOME_EVALUATION

    @pytest.mark.asyncio
    async def test_reflection_emits_consolidation(self) -> None:
        controller = ReflectiveController()
        result = await controller.reflect(
            task_id="task_002",
            outcome={"success": True},
            reasoning_trace=[{"step": "plan"}],
        )
        types = [e.reflection_type for e in result.events]
        assert ReflectionType.MEMORY_CONSOLIDATION in types

    @pytest.mark.asyncio
    async def test_compaction_request_after_threshold(self) -> None:
        controller = ReflectiveController(compaction_threshold=2)
        await controller.reflect("t1", {"success": True})
        result = await controller.reflect("t2", {"success": True})
        types = [e.reflection_type for e in result.events]
        assert ReflectionType.COMPACTION_REQUEST in types


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Cognitive Scheduler Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.control.cognitive_scheduler import (
    CognitiveScheduler,
    ResourceBudget,
    TaskPriority,
)


class TestCognitiveScheduler:
    """Test budget-aware priority scheduler."""

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        scheduler = CognitiveScheduler()

        async def noop() -> str:
            return "done"

        scheduler.submit("bg_task", TaskPriority.BACKGROUND, noop())
        scheduler.submit("user_task", TaskPriority.USER_INTERACTIVE, noop())

        # First task executed should be USER_INTERACTIVE
        result = await scheduler.run_next()
        assert result is not None
        assert result.name == "user_task"

    @pytest.mark.asyncio
    async def test_resource_budget_enforcement(self) -> None:
        scheduler = CognitiveScheduler(
            global_budget=ResourceBudget(cpu_shares=0.5, ram_mb=1024, attention_units=1.0),
        )

        async def noop() -> str:
            return "done"

        # Submit a task that takes all CPU budget
        scheduler.submit(
            "heavy_task",
            TaskPriority.LATENCY_SENSITIVE,
            noop(),
            budget=ResourceBudget(cpu_shares=0.5),
        )
        result = await scheduler.run_next()
        assert result is not None
        assert result.name == "heavy_task"

    @pytest.mark.asyncio
    async def test_task_cancellation(self) -> None:
        scheduler = CognitiveScheduler()

        async def noop() -> str:
            return "done"

        task_id = scheduler.submit("cancel_me", TaskPriority.BACKGROUND, noop())
        assert scheduler.pending_count() == 1
        assert scheduler.cancel(task_id) is True
        assert scheduler.pending_count() == 0

    def test_stats(self) -> None:
        scheduler = CognitiveScheduler()
        stats = scheduler.stats()
        assert "total_submitted" in stats
        assert "queue_by_priority" in stats
        assert "allocated_resources" in stats


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Telemetry & Replay Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.telemetry.replay import DecisionRecord, DecisionReplayEngine
from hbllm.telemetry.timeline import CognitiveTimeline


class TestCognitiveTimeline:
    """Test high-resolution event timeline."""

    def test_record_and_query(self) -> None:
        timeline = CognitiveTimeline(max_entries=100)
        timeline.record("perception", "audio_received", {"text": "hello"})
        timeline.record("brain.snn", "saliency_scored", {"score": 0.9})

        recent = timeline.query(last_n=10)
        assert len(recent) == 2
        assert recent[0].subsystem == "perception"
        assert recent[1].subsystem == "brain.snn"

    def test_subsystem_filter(self) -> None:
        timeline = CognitiveTimeline()
        timeline.record("perception", "event_a")
        timeline.record("memory", "event_b")
        timeline.record("perception", "event_c")

        filtered = timeline.query(subsystem="perception")
        assert len(filtered) == 2

    def test_time_range_export(self) -> None:
        timeline = CognitiveTimeline()
        t1 = time.time()
        timeline.record("test", "event_1")
        t2 = time.time()

        exported = timeline.export_range(t1 - 1, t2 + 1)
        assert len(exported) == 1


class TestDecisionReplayEngine:
    """Test deterministic decision replay."""

    def test_record_and_retrieve(self) -> None:
        engine = DecisionReplayEngine()
        record = DecisionRecord(
            decision_type="plan_selection",
            input_context={"intent": "test"},
            retrieved_memories=[{"content": "fact_1"}],
            selected_capabilities=["tool_a"],
            planner_choices={"strategy": "got"},
            execution_result={"success": True},
        )
        engine.record(record)

        window = engine.get_replay_window(record.timestamp - 1, record.timestamp + 1)
        assert len(window) == 1
        assert window[0].decision_type == "plan_selection"

    def test_correlation_filter(self) -> None:
        engine = DecisionReplayEngine()
        prov = ProvenanceMetadata.create(source="test", correlation_id="session_xyz")
        engine.record(
            DecisionRecord(
                decision_type="test",
                provenance=prov,
            )
        )
        engine.record(
            DecisionRecord(
                decision_type="other",
                provenance=ProvenanceMetadata.create(source="test", correlation_id="other_session"),
            )
        )

        results = engine.get_by_correlation("session_xyz")
        assert len(results) == 1

    def test_causal_chain_tracing(self) -> None:
        engine = DecisionReplayEngine()
        root_prov = ProvenanceMetadata.create(source="root")
        child_prov = ProvenanceMetadata.derive(parent=root_prov, source="child")

        r1 = DecisionRecord(decision_type="root", provenance=root_prov)
        r2 = DecisionRecord(decision_type="child", provenance=child_prov)
        engine.record(r1)
        engine.record(r2)

        chain = engine.get_causal_chain(r2.record_id)
        assert len(chain) == 2
        assert chain[0].decision_type == "root"
        assert chain[1].decision_type == "child"

    def test_export_import_jsonl(self, tmp_path: Any) -> None:
        engine = DecisionReplayEngine()
        engine.record(DecisionRecord(decision_type="test_a"))
        engine.record(DecisionRecord(decision_type="test_b"))

        path = tmp_path / "replay.jsonl"
        exported = engine.export_jsonl(path)
        assert exported == 2

        engine2 = DecisionReplayEngine()
        imported = engine2.import_jsonl(path)
        assert imported == 2


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: WorldState Forecasting Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.perception.world_state import WorldStateEngine


class TestWorldStateForecasting:
    """Test generalized forecasting interface."""

    @pytest.mark.asyncio
    async def test_predict(self) -> None:
        engine = WorldStateEngine()
        prediction = await engine.predict(horizon_seconds=60.0)
        assert prediction.confidence > 0
        assert prediction.horizon_seconds == 60.0
        assert "predicted_at" in prediction.predicted_state

    @pytest.mark.asyncio
    async def test_simulate(self) -> None:
        engine = WorldStateEngine()
        outcome = await engine.simulate(
            [
                {"type": "query", "target": "weather"},
                {"type": "execute", "target": "script.py"},
            ]
        )
        assert outcome.action_count == 2
        assert outcome.risk_score > 0  # "execute" is risky
        assert len(outcome.projected_changes) == 2

    @pytest.mark.asyncio
    async def test_estimate_uncertainty(self) -> None:
        engine = WorldStateEngine()
        metrics = await engine.estimate_uncertainty()
        assert metrics.aggregate_uncertainty > 0
        # All sections should be stale (no data received)
        assert len(metrics.stale_sections) == 6


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: DigitalTwin Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.self_model.digital_twin import DigitalTwin


class TestDigitalTwin:
    """Test ephemeral operational state manager."""

    def test_creation(self) -> None:
        twin = DigitalTwin(system_id="test-node-01")
        assert twin.system_id == "test-node-01"
        assert twin.started_at > 0

    def test_hardware_update(self) -> None:
        twin = DigitalTwin()
        twin.update_hardware(cpu_percent=45.0, ram_mb_used=1200)
        assert twin.hardware.cpu_percent == 45.0
        assert twin.hardware.ram_mb_used == 1200

    def test_goal_lifecycle(self) -> None:
        twin = DigitalTwin()
        twin.register_active_goal("g1", "Answer user query")
        assert twin.active_goal_count == 1
        twin.complete_goal("g1")
        assert twin.active_goal_count == 0

    def test_plugin_management(self) -> None:
        twin = DigitalTwin()
        twin.register_plugin("temporal-reasoning", version="1.2.0")
        assert "temporal-reasoning" in twin.snapshot()["plugins"]
        twin.unregister_plugin("temporal-reasoning")
        assert "temporal-reasoning" not in twin.snapshot()["plugins"]

    def test_full_snapshot(self) -> None:
        twin = DigitalTwin(system_id="snapshot-test")
        twin.update_hardware(cpu_percent=30.0)
        twin.register_active_goal("g1", "Test goal")
        twin.register_plugin("test-plugin")
        twin.register_device("sensor-01", {"type": "temperature"})
        twin.register_task("task-01", {"name": "bg_compaction"})

        snap = twin.snapshot()
        assert snap["system_id"] == "snapshot-test"
        assert snap["hardware"]["cpu_percent"] == 30.0
        assert snap["active_goal_count"] == 1
        assert snap["plugin_count"] == 1
        assert snap["device_count"] == 1
        assert snap["task_count"] == 1
        assert "platform" in snap
        assert snap["uptime_seconds"] >= 0

    def test_reset_clears_state(self) -> None:
        twin = DigitalTwin()
        twin.register_active_goal("g1", "goal")
        twin.register_plugin("p1")
        twin.reset()
        assert twin.active_goal_count == 0
        assert twin.snapshot()["plugin_count"] == 0
