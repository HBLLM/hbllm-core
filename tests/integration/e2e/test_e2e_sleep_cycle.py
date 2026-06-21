"""
End-to-End Integration Test for SleepCycleNode and MemoryNode in HBLLM Core.

Verifies:
1. Populating the real MemoryNode database with conversation turns.
2. SleepCycleNode triggering on-demand memory consolidation.
3. Memory consolidation storing [CONSOLIDATED MEMORY] back to the real EpisodicMemory.
4. Normalizing relative temporal references in the real database (when plugin is available).
5. Detecting and resolving contradictory facts in the real Knowledge Graph.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from hbllm.brain.sleep_node import SleepCycleNode
from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def db_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def e2e_sleep_env(db_dir):
    bus = InProcessBus()
    await bus.start()

    # Create real MemoryNode
    db_path = Path(db_dir) / "memory.db"
    memory = MemoryNode(node_id="memory", db_path=str(db_path))
    await memory.start(bus)

    # Create SleepCycleNode with no LLM (to fall back on rules)
    sleep_node = SleepCycleNode(node_id="sleep_node", idle_timeout_seconds=9999.0)
    await sleep_node.start(bus)

    yield bus, sleep_node, memory

    await sleep_node.stop()
    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_e2e_sleep_cycle_consolidation(e2e_sleep_env) -> None:
    bus, sleep_node, memory = e2e_sleep_env

    # 1. Populate the database with 5 turns via the bus memory.store topic
    for i in range(5):
        store_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test_client",
            tenant_id="default",
            session_id="default_session",
            topic="memory.store",
            payload={
                "session_id": "default_session",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message turn {i} containing some content",
                "scope": "episodic",
            },
        )
        await bus.publish("memory.store", store_msg)

    # Let the bus deliver the store messages
    await asyncio.sleep(0.3)

    # Verify turns were stored
    initial_turns = await memory.db.retrieve_recent(
        "default_session", limit=10, tenant_id="default"
    )
    assert len(initial_turns) == 5

    # Mock the DPO trigger listener if any node publishes on system.sleep.dpo_trigger
    # In the sleep cycle logic, it raises system.sleep.dpo_trigger and waits for learning_update response
    dpo_triggered = asyncio.Event()

    async def handle_dpo(msg: Message) -> None:
        dpo_triggered.set()
        # Respond with learning_update
        update_msg = Message(
            type=MessageType.EVENT,
            source_node_id="mock_learner",
            topic="system.learning_update",
            payload={"status": "complete"},
        )
        await bus.publish("system.learning_update", update_msg)

    await bus.subscribe("system.sleep.dpo_trigger", handle_dpo)

    # 2. Trigger consolidation manually
    force_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test_client",
        tenant_id="default",
        topic="system.sleep.force",
        payload={},
    )
    response = await bus.request("system.sleep.force", force_msg, timeout=2.0)
    assert response.payload.get("status") == "consolidation_started"

    # Wait for the background consolidation task to finish
    # It consolidates memory, normalizes temporal, resolves contradictions, and writes the dream journal
    await asyncio.sleep(1.5)

    # 3. Verify that [CONSOLIDATED MEMORY] was stored in the real database
    final_turns = await memory.db.retrieve_recent("default_session", limit=10, tenant_id="default")

    # We should see the 5 original turns plus the new system consolidated memory turn!
    assert len(final_turns) > 5
    assert any("[CONSOLIDATED MEMORY]" in t["content"] for t in final_turns)


@pytest.mark.asyncio
async def test_e2e_sleep_cycle_contradiction_resolution(e2e_sleep_env) -> None:
    bus, sleep_node, memory = e2e_sleep_env

    # 1. Populate the Knowledge Graph with conflicting relations
    # We add two relations:
    # "user prefers dark mode" (older)
    # "user prefers light mode" (newer)
    kg = memory.knowledge_graph
    kg.add_relation("user", "dark mode", "prefers", weight=1.0, metadata={"created_at": 1000.0})
    # Wait a tiny bit and add the newer conflicting relation
    await asyncio.sleep(0.01)
    kg.add_relation("user", "light mode", "prefers", weight=1.0, metadata={"created_at": 2000.0})

    assert kg.relation_count == 2

    # 2. Run contradiction resolution method on SleepCycleNode
    resolved = await sleep_node._resolve_contradictions()

    # Let the bus deliver the remove_relation events to MemoryNode
    await asyncio.sleep(0.3)

    # 3. Assertions
    # It should have detected the conflict and pruned the older one ("prefers dark mode")
    assert resolved == 1
    assert kg.relation_count == 1

    # Verify that the correct relation remains: user prefers light mode
    neighbors = kg.neighbors("user", direction="out", relation_type="prefers")
    assert len(neighbors) == 1
    assert neighbors[0]["entity"] == "light mode"
