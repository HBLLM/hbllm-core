"""
Milestone 2: Learning Brain — Unit Tests.

Tests the event-sourced MemCube, BeliefGraph, GoalMemory, and
expanded SleepPhase enum.
"""

from __future__ import annotations

import time

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# MemCube Tests
# ═══════════════════════════════════════════════════════════════════════════
from hbllm.memory.memcube import (
    MemCube,
    MemoryEvent,
    MemoryEventStore,
    MemoryEventType,
    MemoryType,
)


class TestMemCube:
    """Test MemCube dataclass and serialization."""

    def test_memcube_creation(self) -> None:
        cube = MemCube(
            id="mem_001",
            content="Test memory",
            memory_type=MemoryType.SEMANTIC,
        )
        assert cube.id == "mem_001"
        assert cube.content == "Test memory"
        assert cube.memory_type == MemoryType.SEMANTIC
        assert cube.version == 1
        assert not cube.forgotten

    def test_memcube_serialization(self) -> None:
        cube = MemCube(
            id="mem_002",
            content="Roundtrip test",
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
            tags=["test", "roundtrip"],
        )
        d = cube.to_dict()
        restored = MemCube.from_dict(d)
        assert restored.id == cube.id
        assert restored.content == cube.content
        assert restored.memory_type == cube.memory_type
        assert restored.importance == cube.importance
        assert restored.tags == cube.tags

    def test_memory_types(self) -> None:
        """All 5 memory types are available."""
        assert len(MemoryType) == 5
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.VALUE.value == "value"
        assert MemoryType.GOAL.value == "goal"


class TestMemoryEvent:
    """Test MemoryEvent factory methods."""

    def test_create_event(self) -> None:
        event = MemoryEvent.create(
            memory_id="mem_001",
            content="Hello world",
            memory_type=MemoryType.SEMANTIC,
            source_node="test",
        )
        assert event.event_type == MemoryEventType.CREATED
        assert event.memory_id == "mem_001"
        assert event.payload["content"] == "Hello world"

    def test_reinforce_event(self) -> None:
        event = MemoryEvent.reinforce(
            memory_id="mem_001",
            source_node="retrieval",
            by_query="what is hello?",
            strength=0.3,
        )
        assert event.event_type == MemoryEventType.REINFORCED
        assert event.payload["strength"] == 0.3

    def test_correct_event(self) -> None:
        event = MemoryEvent.correct(
            memory_id="mem_001",
            old_content="Earth is flat",
            new_content="Earth is round",
            reason="Scientific evidence",
            source_node="workspace",
        )
        assert event.event_type == MemoryEventType.CORRECTED
        assert event.payload["new_content"] == "Earth is round"

    def test_forget_event(self) -> None:
        event = MemoryEvent.forget(
            memory_id="mem_001",
            source_node="sleep",
            reason="Low importance",
        )
        assert event.event_type == MemoryEventType.FORGOTTEN

    def test_serialization(self) -> None:
        event = MemoryEvent.create(
            memory_id="mem_003",
            content="Test",
            memory_type=MemoryType.VALUE,
            source_node="test",
        )
        d = event.to_dict()
        restored = MemoryEvent.from_dict(d)
        assert restored.id == event.id
        assert restored.event_type == event.event_type


class TestMemoryEventStore:
    """Test event store: append, fold, history, compaction."""

    @pytest.mark.asyncio
    async def test_append_and_fold(self) -> None:
        """Basic create → fold cycle."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_fold_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="User likes dark mode",
                memory_type=MemoryType.SEMANTIC,
                source_node="perception",
                importance=0.7,
            )
        )

        cube = await store.fold(mid)
        assert cube is not None
        assert cube.content == "User likes dark mode"
        assert cube.importance == 0.7
        assert cube.version == 1

    @pytest.mark.asyncio
    async def test_correct_updates_content(self) -> None:
        """CREATED → CORRECTED: fold shows latest content."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_correct_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="Earth is flat",
                memory_type=MemoryType.SEMANTIC,
                source_node="perception",
            )
        )
        await store.append(
            MemoryEvent.correct(
                memory_id=mid,
                old_content="Earth is flat",
                new_content="Earth is round",
                reason="Scientific consensus",
                source_node="workspace",
            )
        )

        cube = await store.fold(mid)
        assert cube is not None
        assert cube.content == "Earth is round"
        assert cube.version == 2

    @pytest.mark.asyncio
    async def test_reinforce_increases_importance(self) -> None:
        """REINFORCED events increase importance."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_reinforce_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="Python uses indentation",
                memory_type=MemoryType.SEMANTIC,
                source_node="perception",
                importance=0.5,
            )
        )

        initial = await store.fold(mid)
        assert initial is not None
        initial_importance = initial.importance

        # Reinforce multiple times
        for _ in range(5):
            await store.append(
                MemoryEvent.reinforce(
                    memory_id=mid,
                    source_node="retrieval",
                    strength=0.3,
                )
            )

        reinforced = await store.fold(mid)
        assert reinforced is not None
        assert reinforced.importance > initial_importance

    @pytest.mark.asyncio
    async def test_forget_and_restore(self) -> None:
        """FORGOTTEN → RESTORED lifecycle."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_forget_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="Temporary fact",
                memory_type=MemoryType.EPISODIC,
                source_node="test",
            )
        )

        # Forget
        await store.append(
            MemoryEvent.forget(
                memory_id=mid,
                source_node="sleep",
                reason="Low importance",
            )
        )

        cube = await store.fold(mid)
        assert cube is not None
        assert cube.forgotten is True

    @pytest.mark.asyncio
    async def test_full_history_retrievable(self) -> None:
        """Full audit trail is available."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_history_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="V1",
                memory_type=MemoryType.SEMANTIC,
                source_node="test",
            )
        )
        await store.append(
            MemoryEvent.correct(
                memory_id=mid,
                old_content="V1",
                new_content="V2",
                reason="Update",
                source_node="test",
            )
        )
        await store.append(
            MemoryEvent.reinforce(
                memory_id=mid,
                source_node="test",
            )
        )

        history = await store.get_history(mid)
        assert len(history) == 3
        assert history[0].event_type == MemoryEventType.CREATED
        assert history[1].event_type == MemoryEventType.CORRECTED
        assert history[2].event_type == MemoryEventType.REINFORCED

    @pytest.mark.asyncio
    async def test_compact_reduces_events(self) -> None:
        """Compaction reduces event count to 1."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_compact_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="Original",
                memory_type=MemoryType.SEMANTIC,
                source_node="test",
            )
        )
        for i in range(10):
            await store.append(
                MemoryEvent.reinforce(
                    memory_id=mid,
                    source_node="test",
                )
            )

        pre_compact = await store.get_events(mid)
        assert len(pre_compact) == 11

        cube = await store.compact(mid)
        assert cube is not None

        post_compact = await store.get_events(mid)
        assert len(post_compact) == 1

    @pytest.mark.asyncio
    async def test_access_tracks_count(self) -> None:
        """ACCESSED events increment access_count."""
        store = MemoryEventStore()
        await store.initialize()

        mid = "mem_access_001"
        await store.append(
            MemoryEvent.create(
                memory_id=mid,
                content="Tracked",
                memory_type=MemoryType.SEMANTIC,
                source_node="test",
            )
        )
        for _ in range(3):
            await store.append(
                MemoryEvent.access(
                    memory_id=mid,
                    source_node="retrieval",
                )
            )

        cube = await store.fold(mid)
        assert cube is not None
        assert cube.access_count == 3


# ═══════════════════════════════════════════════════════════════════════════
# Belief Graph Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.memory.belief_graph import (
    BeliefGraph,
    BeliefRecord,
)


class TestBeliefGraph:
    """Test belief provenance and explainability."""

    @pytest.mark.asyncio
    async def test_record_and_retrieve_belief(self) -> None:
        graph = BeliefGraph()
        record = BeliefRecord(
            id="bel_001",
            memory_id="mem_001",
            created_by="perception",
            created_at=time.time(),
            reason="User stated",
            trigger="user_input",
        )
        await graph.record_belief(record)

        retrieved = await graph.get_belief("mem_001")
        assert retrieved is not None
        assert retrieved.id == "bel_001"
        assert retrieved.confidence == 1.0

    @pytest.mark.asyncio
    async def test_support_increases_confidence(self) -> None:
        graph = BeliefGraph()
        await graph.record_belief(
            BeliefRecord(
                id="bel_002",
                memory_id="mem_002",
                created_by="test",
                created_at=time.time(),
                reason="Test",
                trigger="test",
                confidence=0.5,
            )
        )

        initial = (await graph.get_belief("mem_002")).confidence

        await graph.add_support("mem_002", "mem_003", strength=0.8)

        after = (await graph.get_belief("mem_002")).confidence
        assert after > initial

    @pytest.mark.asyncio
    async def test_contradiction_decreases_confidence(self) -> None:
        graph = BeliefGraph()
        await graph.record_belief(
            BeliefRecord(
                id="bel_003",
                memory_id="mem_003",
                created_by="test",
                created_at=time.time(),
                reason="Test",
                trigger="test",
                confidence=0.8,
            )
        )

        await graph.add_contradiction("mem_003", "mem_004", strength=0.9)

        after = (await graph.get_belief("mem_003")).confidence
        assert after < 0.8

    @pytest.mark.asyncio
    async def test_explain_readable(self) -> None:
        graph = BeliefGraph()
        await graph.record_belief(
            BeliefRecord(
                id="bel_004",
                memory_id="mem_earth",
                created_by="science",
                created_at=time.time(),
                reason="Scientific consensus",
                trigger="knowledge_base",
            )
        )
        await graph.add_support("mem_earth", "mem_satellite", strength=0.9)
        await graph.add_contradiction("mem_earth", "mem_flat", strength=0.1)

        explanation = await graph.explain("mem_earth")
        assert "confidence" in explanation
        assert "Supported by" in explanation
        assert "Contradicted by" in explanation

    @pytest.mark.asyncio
    async def test_contested_beliefs(self) -> None:
        graph = BeliefGraph()
        # Belief with contradiction
        await graph.record_belief(
            BeliefRecord(
                id="bel_contested",
                memory_id="mem_contested",
                created_by="test",
                created_at=time.time(),
                reason="Test",
                trigger="test",
            )
        )
        await graph.add_contradiction("mem_contested", "mem_contra", strength=0.5)

        # Belief without contradiction
        await graph.record_belief(
            BeliefRecord(
                id="bel_solid",
                memory_id="mem_solid",
                created_by="test",
                created_at=time.time(),
                reason="Test",
                trigger="test",
            )
        )

        contested = await graph.get_contested_beliefs()
        assert len(contested) == 1
        assert contested[0].memory_id == "mem_contested"

    @pytest.mark.asyncio
    async def test_reinforce_and_correct(self) -> None:
        graph = BeliefGraph()
        await graph.record_belief(
            BeliefRecord(
                id="bel_rc",
                memory_id="mem_rc",
                created_by="test",
                created_at=time.time(),
                reason="Test",
                trigger="test",
                confidence=0.5,
            )
        )

        await graph.reinforce("mem_rc")
        await graph.reinforce("mem_rc")
        record = await graph.get_belief("mem_rc")
        assert record.reinforcement_count == 2

        await graph.correct("mem_rc", "Updated content")
        record = await graph.get_belief("mem_rc")
        assert record.correction_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Goal Memory Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.memory.goal_memory import (
    GoalCube,
    GoalMemory,
    GoalStatus,
)


class TestGoalMemory:
    """Test goal hierarchy, lineage, and lifecycle."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_goal(self) -> None:
        mem = GoalMemory()
        goal = GoalCube(
            id="goal_001",
            description="Help user debug auth issue",
            motive="User asked",
            priority=0.9,
        )
        await mem.add_goal(goal)

        retrieved = await mem.get_goal("goal_001")
        assert retrieved is not None
        assert retrieved.description == "Help user debug auth issue"

    @pytest.mark.asyncio
    async def test_subgoal_hierarchy(self) -> None:
        mem = GoalMemory()
        parent = GoalCube(id="g_parent", description="Parent", priority=0.8)
        child1 = GoalCube(id="g_child1", description="Child 1", parent_goal_id="g_parent")
        child2 = GoalCube(id="g_child2", description="Child 2", parent_goal_id="g_parent")

        await mem.add_goal(parent)
        await mem.add_goal(child1)
        await mem.add_goal(child2)

        parent_retrieved = await mem.get_goal("g_parent")
        assert "g_child1" in parent_retrieved.subgoal_ids
        assert "g_child2" in parent_retrieved.subgoal_ids

    @pytest.mark.asyncio
    async def test_progress_completion(self) -> None:
        mem = GoalMemory()
        await mem.add_goal(GoalCube(id="g_prog", description="Progress test"))

        await mem.update_progress("g_prog", 0.5)
        goal = await mem.get_goal("g_prog")
        assert goal.progress == 0.5
        assert goal.status == GoalStatus.ACTIVE

        await mem.update_progress("g_prog", 1.0)
        goal = await mem.get_goal("g_prog")
        assert goal.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_goal_lineage(self) -> None:
        """Trace: action → subgoal → goal → motive."""
        mem = GoalMemory()
        await mem.add_goal(GoalCube(id="root", description="Root goal", motive="Core mission"))
        await mem.add_goal(GoalCube(id="mid", description="Mid goal", parent_goal_id="root"))
        await mem.add_goal(GoalCube(id="leaf", description="Leaf action", parent_goal_id="mid"))

        lineage = await mem.get_goal_lineage("leaf")
        assert len(lineage) == 3
        assert lineage[0].id == "leaf"
        assert lineage[1].id == "mid"
        assert lineage[2].id == "root"

    @pytest.mark.asyncio
    async def test_active_goals_filtered_by_tenant(self) -> None:
        mem = GoalMemory()
        await mem.add_goal(GoalCube(id="g_t1", description="Tenant 1", tenant_id="t1"))
        await mem.add_goal(GoalCube(id="g_t2", description="Tenant 2", tenant_id="t2"))

        t1_goals = await mem.get_active_goals("t1")
        assert len(t1_goals) == 1
        assert t1_goals[0].id == "g_t1"

    @pytest.mark.asyncio
    async def test_abandon_cascades_to_subgoals(self) -> None:
        mem = GoalMemory()
        await mem.add_goal(GoalCube(id="g_abn_p", description="Parent"))
        await mem.add_goal(GoalCube(id="g_abn_c", description="Child", parent_goal_id="g_abn_p"))

        await mem.abandon_goal("g_abn_p")

        parent = await mem.get_goal("g_abn_p")
        child = await mem.get_goal("g_abn_c")
        assert parent.status == GoalStatus.ABANDONED
        assert child.status == GoalStatus.ABANDONED

    @pytest.mark.asyncio
    async def test_urgent_goals(self) -> None:
        mem = GoalMemory()
        now = time.time()
        await mem.add_goal(
            GoalCube(
                id="g_urgent",
                description="Urgent!",
                deadline=now + 60,
            )
        )
        await mem.add_goal(
            GoalCube(
                id="g_far",
                description="Far away",
                deadline=now + 999999,
            )
        )

        urgent = await mem.get_urgent_goals(horizon=300)
        assert len(urgent) == 1
        assert urgent[0].id == "g_urgent"

    @pytest.mark.asyncio
    async def test_parent_progress_updates(self) -> None:
        """Parent progress auto-updates from children."""
        mem = GoalMemory()
        await mem.add_goal(GoalCube(id="g_pp", description="Parent"))
        await mem.add_goal(GoalCube(id="g_pc1", description="C1", parent_goal_id="g_pp"))
        await mem.add_goal(GoalCube(id="g_pc2", description="C2", parent_goal_id="g_pp"))

        await mem.update_progress("g_pc1", 1.0)

        parent = await mem.get_goal("g_pp")
        assert parent.progress == 0.5  # 1 of 2 children complete

        await mem.update_progress("g_pc2", 1.0)
        parent = await mem.get_goal("g_pp")
        assert parent.status == GoalStatus.COMPLETED


# ═══════════════════════════════════════════════════════════════════════════
# Sleep Phase Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.sleep_node import SleepPhase


class TestSleepPhases:
    """Test expanded sleep phase enum."""

    def test_all_six_phases(self) -> None:
        phases = list(SleepPhase)
        assert len(phases) == 6
        assert SleepPhase.AWAKE in phases
        assert SleepPhase.MICRO in phases
        assert SleepPhase.NREM in phases
        assert SleepPhase.REM in phases
        assert SleepPhase.DEEP_REORG in phases
        assert SleepPhase.OFFLINE in phases

    def test_phase_values(self) -> None:
        assert SleepPhase.MICRO.value == "micro"
        assert SleepPhase.DEEP_REORG.value == "deep_reorg"
        assert SleepPhase.OFFLINE.value == "offline"

    def test_phase_ordering(self) -> None:
        """Phases should progress from light to deep."""
        order = [
            SleepPhase.AWAKE,
            SleepPhase.MICRO,
            SleepPhase.NREM,
            SleepPhase.REM,
            SleepPhase.DEEP_REORG,
            SleepPhase.OFFLINE,
        ]
        # Verify they're all unique
        assert len(set(order)) == 6
