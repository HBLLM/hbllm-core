"""Tests for Phase 2: Tiered Workspace, Task Frames, and Bus Bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from hbllm.hcir.adapters.hcir_bus_bridge import HCIRBusBridge
from hbllm.hcir.cognitive_event_log import CognitiveEventLog
from hbllm.hcir.cognitive_journal import CognitiveJournal
from hbllm.hcir.graph import GoalNode, HCIRNodeType, ObservationNode
from hbllm.hcir.query import GraphQuery
from hbllm.hcir.semantic_normalizer import SemanticNormalizer
from hbllm.hcir.stores import InMemoryEventStore
from hbllm.hcir.workspace_tiers import (
    TaskFrame,
    TieredWorkspace,
    WorkingWorkspace,
    WorkspaceTier,
)

# ═══════════════════════════════════════════════════════════════════════════
# Task Frame Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskFrame:
    """Verify TaskFrame lifecycle."""

    def test_create_frame(self) -> None:
        frame = TaskFrame(goal_id="goal_abc")
        assert frame.is_active
        assert frame.goal_id == "goal_abc"
        assert frame.status == "active"
        assert frame.workspace is not None

    def test_close_frame(self) -> None:
        frame = TaskFrame(goal_id="goal_abc")
        frame.close("completed")
        assert not frame.is_active
        assert frame.status == "completed"
        assert frame.closed_at is not None

    def test_close_with_abandonment(self) -> None:
        frame = TaskFrame(goal_id="goal_abc")
        frame.close("abandoned")
        assert frame.status == "abandoned"

    def test_close_with_eviction(self) -> None:
        frame = TaskFrame(goal_id="goal_abc")
        frame.close("evicted")
        assert frame.status == "evicted"

    def test_frame_workspace_is_usable(self) -> None:
        frame = TaskFrame(goal_id="goal_abc")
        node = GoalNode(id="g1", description="Test goal")
        frame.workspace.add_node(node)
        assert frame.workspace.get_node("g1") is not None


# ═══════════════════════════════════════════════════════════════════════════
# Working Workspace Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkingWorkspace:
    """Verify WorkingWorkspace manages task frames correctly."""

    def test_create_frame(self) -> None:
        ww = WorkingWorkspace()
        frame = ww.create_frame("goal_1")
        assert frame.is_active
        assert len(ww.active_frames) == 1

    def test_multiple_frames(self) -> None:
        ww = WorkingWorkspace()
        ww.create_frame("goal_1")
        ww.create_frame("goal_2")
        assert len(ww.active_frames) == 2

    def test_close_frame(self) -> None:
        ww = WorkingWorkspace()
        frame = ww.create_frame("goal_1")
        ww.close_frame(frame.frame_id, "completed")
        assert len(ww.active_frames) == 0
        assert len(ww.all_frames) == 1

    def test_get_frame_by_goal(self) -> None:
        ww = WorkingWorkspace()
        ww.create_frame("goal_1")
        ww.create_frame("goal_2")
        found = ww.get_frame_by_goal("goal_2")
        assert found is not None
        assert found.goal_id == "goal_2"

    def test_eviction_on_max_frames(self) -> None:
        ww = WorkingWorkspace(max_frames=3)
        f1 = ww.create_frame("goal_1")
        f1.close("completed")
        ww.create_frame("goal_2")
        ww.create_frame("goal_3")
        # This should evict f1 (inactive)
        ww.create_frame("goal_4")
        assert len(ww.all_frames) == 3

    def test_get_node_across_frames(self) -> None:
        ww = WorkingWorkspace()
        _f1 = ww.create_frame("goal_1")
        f2 = ww.create_frame("goal_2")
        node = GoalNode(id="g_target", description="Find me")
        f2.workspace.add_node(node)
        found = ww.get_node_across_frames("g_target")
        assert found is not None
        assert found.id == "g_target"


# ═══════════════════════════════════════════════════════════════════════════
# Tiered Workspace Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTieredWorkspace:
    """Verify TieredWorkspace promotion, demotion, and cross-tier queries."""

    def setup_method(self) -> None:
        self.tiered = TieredWorkspace()

    def test_create_task_frame(self) -> None:
        frame = self.tiered.create_task_frame("goal_1")
        assert frame.is_active
        assert frame.goal_id == "goal_1"

    def test_close_task_frame(self) -> None:
        frame = self.tiered.create_task_frame("goal_1")
        self.tiered.close_task_frame(frame.frame_id, "completed")
        assert not frame.is_active

    def test_get_tier(self) -> None:
        assert self.tiered.get_tier(WorkspaceTier.BRAIN) is self.tiered.brain
        assert self.tiered.get_tier(WorkspaceTier.PERSISTENT) is self.tiered.persistent
        assert self.tiered.get_tier(WorkspaceTier.META) is self.tiered.meta
        assert self.tiered.get_tier(WorkspaceTier.WORKING) is None

    def test_promote_brain_to_persistent(self) -> None:
        node = GoalNode(id="g_promote", description="Promotable goal")
        self.tiered.brain.add_node(node)

        result = self.tiered.promote("g_promote", WorkspaceTier.BRAIN, WorkspaceTier.PERSISTENT)
        assert result is True

        # Node should exist in persistent
        persistent_node = self.tiered.persistent.get_node("g_promote")
        assert persistent_node is not None
        assert persistent_node.id == "g_promote"

        # Original should still exist in brain (copy, not move)
        assert self.tiered.brain.get_node("g_promote") is not None

    def test_promote_nonexistent_node(self) -> None:
        result = self.tiered.promote("nonexistent", WorkspaceTier.BRAIN, WorkspaceTier.PERSISTENT)
        assert result is False

    def test_demote_persistent_to_brain(self) -> None:
        node = ObservationNode(id="obs_1", sensor_source="test")
        self.tiered.persistent.add_node(node)

        result = self.tiered.demote("obs_1", WorkspaceTier.PERSISTENT, WorkspaceTier.BRAIN)
        assert result is True

        # Should be in brain, removed from persistent
        assert self.tiered.brain.get_node("obs_1") is not None
        assert self.tiered.persistent.get_node("obs_1") is None

    def test_query_across_tiers(self) -> None:
        # Add goals to different tiers
        self.tiered.brain.add_node(GoalNode(id="g_brain", description="Brain goal"))
        self.tiered.persistent.add_node(GoalNode(id="g_persist", description="Persistent goal"))

        frame = self.tiered.create_task_frame("g_working")
        frame.workspace.add_node(GoalNode(id="g_working", description="Working goal"))

        query = GraphQuery(node_type=HCIRNodeType.GOAL)
        result = self.tiered.query_across_tiers(query)
        assert result.total_matches == 3

    def test_query_specific_tiers(self) -> None:
        self.tiered.brain.add_node(GoalNode(id="g_brain", description="Brain goal"))
        self.tiered.persistent.add_node(GoalNode(id="g_persist", description="Persistent"))

        query = GraphQuery(node_type=HCIRNodeType.GOAL)
        result = self.tiered.query_across_tiers(query, tiers=[WorkspaceTier.BRAIN])
        assert result.total_matches == 1
        assert result.nodes[0].id == "g_brain"

    def test_archive_brain(self) -> None:
        self.tiered.brain.add_node(GoalNode(id="g1", description="Goal 1"))
        self.tiered.brain.add_node(GoalNode(id="g2", description="Goal 2"))

        count = self.tiered.archive_brain()
        assert count == 2

        # Nodes should be in persistent
        assert self.tiered.persistent.get_node("g1") is not None
        assert self.tiered.persistent.get_node("g2") is not None

    def test_snapshot(self) -> None:
        self.tiered.brain.add_node(GoalNode(id="g1", description="Snap"))
        snap = self.tiered.snapshot(WorkspaceTier.BRAIN)
        assert snap is not None
        assert snap.branch == "brain"

    def test_auto_snapshot_on_commits(self) -> None:
        tiered = TieredWorkspace(snapshot_interval=3)
        initial_version = tiered.persistent.current_version

        tiered.notify_commit()
        tiered.notify_commit()
        # Third commit should trigger auto-snapshot
        tiered.notify_commit()

        assert tiered.persistent.current_version > initial_version


# ═══════════════════════════════════════════════════════════════════════════
# Bus Bridge Tests
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MockMessage:
    """Minimal mock of hbllm.network.messages.Message for testing."""

    id: str = "msg_test_001"
    type: str = "event"
    source_node_id: str = "test_node"
    tenant_id: str = "default"
    topic: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class MockBus:
    """Minimal async bus mock for testing the bridge."""

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}
        self._subscriptions: list[Any] = []

    async def subscribe(self, topic: str, handler: Any, tenant_id: str | None = None) -> Any:
        self._handlers[topic] = handler

        class Sub:
            pass

        sub = Sub()
        self._subscriptions.append(sub)
        return sub

    async def unsubscribe(self, sub: Any) -> None:
        pass

    async def simulate_event(self, topic: str, message: Any) -> None:
        """Simulate a bus event by calling the handler directly."""
        handler = self._handlers.get(topic)
        if handler:
            await handler(message)


class TestHCIRBusBridge:
    """Verify the bus bridge projects events correctly."""

    def setup_method(self) -> None:
        self.bus = MockBus()
        self.normalizer = SemanticNormalizer()
        self.journal = CognitiveJournal(InMemoryEventStore())
        self.event_log = CognitiveEventLog(InMemoryEventStore())
        self.workspace = TieredWorkspace()
        self.bridge = HCIRBusBridge(
            bus=cast(Any, self.bus),
            normalizer=self.normalizer,
            journal=self.journal,
            event_log=self.event_log,
            tiered_workspace=self.workspace,
            tx_manager=None,  # Direct mode for testing
        )

    @pytest.mark.asyncio
    async def test_start_subscribes_to_topics(self) -> None:
        await self.bridge.start()
        assert self.bridge.is_running
        assert len(self.bus._handlers) > 0

    @pytest.mark.asyncio
    async def test_stop_clears_subscriptions(self) -> None:
        await self.bridge.start()
        await self.bridge.stop()
        assert not self.bridge.is_running

    @pytest.mark.asyncio
    async def test_memory_store_event_projects_to_persistent(self) -> None:
        await self.bridge.start()

        msg = MockMessage(
            topic="memory.store",
            source_node_id="memory_node",
            data={"summary": "User likes coffee"},
        )
        await self.bus.simulate_event("memory.store", msg)

        assert self.bridge.events_processed == 1
        assert self.bridge.events_projected == 1
        assert self.journal.count() == 1

    @pytest.mark.asyncio
    async def test_goal_created_event_projects_to_working(self) -> None:
        await self.bridge.start()

        msg = MockMessage(
            topic="planning.goal_created",
            source_node_id="planner_node",
            data={"description": "Plan dinner", "goal_id": "goal_dinner"},
        )
        await self.bus.simulate_event("planning.goal_created", msg)

        assert self.bridge.events_processed == 1
        assert self.bridge.events_projected == 1
        # Should have created a task frame
        assert len(self.workspace.working.active_frames) > 0

    @pytest.mark.asyncio
    async def test_cognitive_state_event_projects_to_brain(self) -> None:
        await self.bridge.start()

        msg = MockMessage(
            topic="cognitive_state.updated",
            source_node_id="state_reducer",
            data={"curiosity": 0.8, "focus": 0.9},
        )
        await self.bus.simulate_event("cognitive_state.updated", msg)

        assert self.bridge.events_processed == 1
        # Should be in brain workspace
        nodes = self.workspace.brain.graph.nodes_by_type(HCIRNodeType.OBSERVATION)
        assert len(nodes) == 1

    @pytest.mark.asyncio
    async def test_unrecognized_event_still_journaled(self) -> None:
        await self.bridge.start()

        msg = MockMessage(
            topic="unknown.custom.event",
            source_node_id="plugin",
        )
        # Call handler directly since the bus wouldn't dispatch
        # an unsubscribed topic
        await self.bridge._on_event(cast(Any, msg))

        # Unrecognized → still journaled as observation
        assert self.journal.count() == 1
        assert self.bridge.events_processed == 1

    @pytest.mark.asyncio
    async def test_multiple_events_sequence(self) -> None:
        await self.bridge.start()

        events = [
            ("planning.goal_created", {"description": "Find weather", "goal_id": "g1"}),
            ("decision.made", {"choice": "use_api"}),
            ("action.executed", {"intent": "call_weather_api"}),
            ("memory.store", {"summary": "Weather is sunny"}),
        ]

        for topic, data in events:
            msg = MockMessage(topic=topic, source_node_id="test", data=data)
            await self.bus.simulate_event(topic, msg)

        assert self.bridge.events_processed == 4
        assert self.bridge.events_projected == 4
        assert self.journal.count() == 4

    @pytest.mark.asyncio
    async def test_event_provenance_is_set(self) -> None:
        await self.bridge.start()

        msg = MockMessage(
            topic="memory.store",
            source_node_id="memory_node",
            tenant_id="tenant_42",
            data={"session_id": "sess_1", "goal_id": "g1"},
        )
        await self.bus.simulate_event("memory.store", msg)

        # Check the journal event has provenance
        events = list(self.journal.replay())
        assert len(events) == 1
        assert events[0].tenant_id == "tenant_42"
        assert events[0].author == "memory_node"
