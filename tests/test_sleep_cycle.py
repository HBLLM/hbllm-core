import asyncio

import pytest

from hbllm.brain.sleep_node import SleepCycleNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


# Mock Memory Node to simulate retrieval
class MockMemoryNode:
    def __init__(self, bus):
        self.bus = bus
        self.store_calls = []

    async def start(self):
        await self.bus.subscribe("memory.retrieve_recent", self.handle_retrieve)
        await self.bus.subscribe("memory.store", self.handle_store)
        await self.bus.subscribe("system.sleep.dpo_trigger", self.handle_dpo_trigger)

    async def handle_retrieve(self, msg: Message):
        # Return 5 dummy turns to trigger the Sleep Node compression
        turns = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        return msg.create_response({"turns": turns})

    async def handle_store(self, msg: Message):
        self.store_calls.append(msg)
        return None

    async def handle_dpo_trigger(self, msg: Message):
        # Immediately fire the learning_update event so sleep cycle doesn't hang
        update_msg = Message(
            type=MessageType.EVENT,
            source_node_id="mock_learner",
            topic="system.learning_update",
            payload={"status": "complete"},
        )
        await self.bus.publish("system.learning_update", update_msg)
        return None


@pytest.fixture
async def simulated_sleep_env():
    bus = InProcessBus()
    await bus.start()

    memory = MockMemoryNode(bus)
    await memory.start()

    # Very short timeout for fast tests
    sleep_node = SleepCycleNode(node_id="sleep_01", idle_timeout_seconds=0.5)
    await sleep_node.start(bus)

    yield bus, sleep_node, memory

    await sleep_node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_sleep_cycle_triggers_on_idle(simulated_sleep_env):
    bus, sleep_node, memory = simulated_sleep_env

    # Assert it starts awake
    assert not sleep_node.is_sleeping

    # Let it idle past the 0.5s timeout + 0.5s loop interval
    await asyncio.sleep(1.2)

    # Should now be asleep (in consolidation mode) or have finished sleeping
    # Since we mocked memory, the compression should execute quickly and reset flag
    assert len(memory.store_calls) > 0

    store_msg = memory.store_calls[0]
    payload = store_msg.payload

    assert payload["role"] == "system"
    assert "CONSOLIDATED MEMORY" in payload["content"]


@pytest.mark.asyncio
async def test_sleep_cycle_interrupted_by_user(simulated_sleep_env):
    bus, sleep_node, memory = simulated_sleep_env

    # Wait half the timeout
    await asyncio.sleep(0.3)

    # User sends a query, which should reset the idle timer
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="router.query",
        payload={"text": "Hello"},
    )
    await bus.publish("router.query", msg)

    # Wait another half timeout (total 0.6s since start)
    await asyncio.sleep(0.3)

    # Because of the interruption, it should NOT have triggered sleep
    assert not sleep_node.is_sleeping
    assert len(memory.store_calls) == 0

    # Wait a full idle timeout from the interrupt + loop interval
    await asyncio.sleep(1.2)

    # Now it should trigger
    assert len(memory.store_calls) > 0


@pytest.mark.asyncio
async def test_sleep_phase_transitions(simulated_sleep_env):
    bus, sleep_node, memory = simulated_sleep_env
    from hbllm.brain.sleep_node import SleepPhase

    # Initially awake
    assert sleep_node.current_phase == SleepPhase.AWAKE
    assert not sleep_node.is_sleeping

    # Explicitly force cycle entry
    await sleep_node._enter_sleep_cycle()

    # The cycle method tracks internal transitions.
    # At completion, it resets to AWAKE
    assert sleep_node.current_phase == SleepPhase.AWAKE
    assert not sleep_node.is_sleeping


# ── Gap Closer Tests ─────────────────────────────────────────────────────


class MockMemoryNodeWithTemporal:
    """Mock memory node that returns turns containing relative temporal references."""

    def __init__(self, bus):
        self.bus = bus
        self.store_calls = []

    async def start(self):
        await self.bus.subscribe("memory.retrieve_recent", self.handle_retrieve)
        await self.bus.subscribe("memory.store", self.handle_store)
        await self.bus.subscribe("system.sleep.dpo_trigger", self.handle_dpo_trigger)

    async def handle_retrieve(self, msg: Message):
        turns = [
            {
                "role": "user",
                "content": "I discussed this yesterday with John",
                "timestamp": 1746608400.0,  # 2025-05-07T09:00:00Z
            },
            {"role": "assistant", "content": "Sure, I remember."},
            {"role": "user", "content": "We also talked about it last week"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": "Regular message without temporal refs"},
        ]
        return msg.create_response({"turns": turns})

    async def handle_store(self, msg: Message):
        self.store_calls.append(msg)
        return None

    async def handle_dpo_trigger(self, msg: Message):
        update_msg = Message(
            type=MessageType.EVENT,
            source_node_id="mock_learner",
            topic="system.learning_update",
            payload={"status": "complete"},
        )
        await self.bus.publish("system.learning_update", update_msg)
        return None


class MockKnowledgeNode:
    """Mock knowledge graph node that returns entities with contradictory relations."""

    def __init__(self, bus):
        self.bus = bus
        self.prune_calls = []

    async def start(self):
        await self.bus.subscribe("knowledge.query", self.handle_query)

    async def handle_query(self, msg: Message):
        action = msg.payload.get("action", "")

        if action == "all_entities":
            return msg.create_response(
                {
                    "entities": [
                        {"label": "user", "type": "person", "id": "e1"},
                        {"label": "dark mode", "type": "preference", "id": "e2"},
                        {"label": "light mode", "type": "preference", "id": "e3"},
                    ]
                }
            )

        if action == "all_relations":
            return msg.create_response(
                {
                    "relations": [
                        {
                            "source_id": "e1",
                            "target_id": "e2",
                            "relation_type": "prefers",
                            "created_at": 1000.0,
                        },
                        {
                            "source_id": "e1",
                            "target_id": "e3",
                            "relation_type": "prefers",
                            "created_at": 2000.0,  # Newer — should be kept
                        },
                    ]
                }
            )

        if action == "remove_relation":
            self.prune_calls.append(msg.payload)
            return None

        return msg.create_response({})


@pytest.fixture
async def temporal_sleep_env():
    bus = InProcessBus()
    await bus.start()

    memory = MockMemoryNodeWithTemporal(bus)
    await memory.start()

    sleep_node = SleepCycleNode(node_id="sleep_temp", idle_timeout_seconds=0.0)
    await sleep_node.start(bus)

    yield bus, sleep_node, memory

    await sleep_node.stop()
    await bus.stop()


@pytest.fixture
async def contradiction_sleep_env():
    bus = InProcessBus()
    await bus.start()

    memory = MockMemoryNode(bus)
    await memory.start()

    kg = MockKnowledgeNode(bus)
    await kg.start()

    sleep_node = SleepCycleNode(node_id="sleep_kg", idle_timeout_seconds=0.0)
    await sleep_node.start(bus)

    yield bus, sleep_node, memory, kg

    await sleep_node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_temporal_normalization(temporal_sleep_env):
    """Verify relative temporal refs ('yesterday', 'last week') are normalized."""
    _bus, sleep_node, memory = temporal_sleep_env

    count = await sleep_node._normalize_temporal_references()

    # Let the bus dispatch loop deliver the stored messages
    await asyncio.sleep(0.3)

    # Should have normalized at least 2 refs ("yesterday" and "last week")
    assert count >= 2

    # Verify the normalized entries were stored
    normalized_stores = [m for m in memory.store_calls if m.payload.get("normalized") is True]
    assert len(normalized_stores) >= 1

    # Check that absolute dates were injected
    for store in normalized_stores:
        content = store.payload.get("content", "")
        # The content should now contain parenthesized absolute dates
        assert "(" in content and ")" in content


@pytest.mark.asyncio
async def test_contradiction_resolution(contradiction_sleep_env):
    """Verify conflicting KG facts are detected and pruned (keep newest)."""
    _bus, sleep_node, _memory, kg = contradiction_sleep_env

    resolved = await sleep_node._resolve_contradictions()

    # Let the bus dispatch loop deliver the prune messages
    await asyncio.sleep(0.3)

    # Should have resolved 1 contradiction (the older 'prefers' relation)
    assert resolved == 1

    # Verify the prune message was sent for the older relation
    assert len(kg.prune_calls) == 1
    pruned = kg.prune_calls[0]
    assert pruned["target_id"] == "e2"  # dark mode (older, created_at=1000)
    assert pruned["relation_type"] == "prefers"


@pytest.mark.asyncio
async def test_dream_journal_generated(temporal_sleep_env):
    """Verify sleep cycle generates a human-readable dream journal."""
    _bus, sleep_node, memory = temporal_sleep_env

    report = {
        "memories_consolidated": 5,
        "contradictions_resolved": 1,
        "temporal_refs_normalized": 3,
        "training_ran": True,
        "skills_optimized": 1,
        "goals_replayed": 2,
        "duration_seconds": 4.2,
    }

    journal = await sleep_node._generate_dream_journal(report)

    # Let the bus dispatch loop deliver the store message
    await asyncio.sleep(0.3)

    # Verify journal content
    assert "Dream Journal" in journal
    assert "5 recent" in journal  # memories
    assert "3 relative time" in journal  # temporal
    assert "1 conflicting" in journal  # contradictions
    assert "Neural Plasticity" in journal  # DPO
    assert "2 knowledge gap" in journal  # curiosity

    # Verify it was stored in memory
    journal_stores = [
        m for m in memory.store_calls if m.payload.get("session_id") == "dream_journal"
    ]
    assert len(journal_stores) == 1
    assert "Dream Journal" in journal_stores[0].payload["content"]


@pytest.mark.asyncio
async def test_manual_trigger(temporal_sleep_env):
    """Verify system.sleep.force triggers a consolidation cycle."""
    bus, sleep_node, memory = temporal_sleep_env

    # Send a manual force trigger
    force_msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="system.sleep.force",
        payload={},
    )
    response = await bus.request("system.sleep.force", force_msg, timeout=1.0)

    # Should acknowledge the trigger
    assert response.payload.get("status") == "consolidation_started"

    # Wait for the sleep cycle to complete
    await asyncio.sleep(2.0)

    # Sleep should have completed and produced stores
    assert len(memory.store_calls) > 0
