import asyncio
import os
import tempfile

import pytest

from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
async def memory_env():
    bus = InProcessBus()
    await bus.start()

    # Use a temp DB for isolation
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    mem = MemoryNode(node_id="mem_01", db_path=tmp.name)
    await mem.start(bus)

    yield bus, mem, tmp.name

    await mem.stop()
    await bus.stop()
    os.unlink(tmp.name)


@pytest.mark.asyncio
async def test_tenant_isolation_in_memory(memory_env):
    """Verify that different tenants cannot see each other's conversations."""
    bus, mem, _ = memory_env

    # Tenant A stores a message
    store_a = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenant_A",
        topic="memory.store",
        payload={
            "session_id": "session_1",
            "tenant_id": "tenant_A",
            "role": "user",
            "content": "Hello from Tenant A",
        },
    )
    await bus.publish("memory.store", store_a)
    await asyncio.sleep(0.1)

    # Tenant B stores a message in the SAME session_id
    store_b = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenant_B",
        topic="memory.store",
        payload={
            "session_id": "session_1",
            "tenant_id": "tenant_B",
            "role": "user",
            "content": "Hello from Tenant B",
        },
    )
    await bus.publish("memory.store", store_b)
    await asyncio.sleep(0.1)

    # Retrieve for Tenant A — should only see Tenant A's message
    turns_a = await mem.db.retrieve_recent("session_1", limit=10, tenant_id="tenant_A")
    assert len(turns_a) == 1
    assert turns_a[0]["content"] == "Hello from Tenant A"
    assert turns_a[0]["tenant_id"] == "tenant_A"

    # Retrieve for Tenant B — should only see Tenant B's message
    turns_b = await mem.db.retrieve_recent("session_1", limit=10, tenant_id="tenant_B")
    assert len(turns_b) == 1
    assert turns_b[0]["content"] == "Hello from Tenant B"
    assert turns_b[0]["tenant_id"] == "tenant_B"


@pytest.mark.asyncio
async def test_memory_clear_respects_tenant(memory_env):
    """Verify that clearing a session only removes the correct tenant's data."""
    bus, mem, _ = memory_env

    # Store for both tenants
    for tid in ("t1", "t2"):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id=tid,
            topic="memory.store",
            payload={
                "session_id": "shared_session",
                "tenant_id": tid,
                "role": "user",
                "content": f"Message from {tid}",
            },
        )
        await bus.publish("memory.store", msg)

    await asyncio.sleep(0.1)

    # Clear only t1
    deleted = await mem.db.clear_session("shared_session", tenant_id="t1")
    assert deleted == 1

    # t2's data should still be there
    turns = await mem.db.retrieve_recent("shared_session", limit=10, tenant_id="t2")
    assert len(turns) == 1
    assert turns[0]["content"] == "Message from t2"


@pytest.mark.asyncio
async def test_knowledge_graph_tenant_isolation(memory_env):
    """Verify that reflection entities are isolated per tenant in KnowledgeGraph."""
    bus, mem, _ = memory_env

    # Tenant A publishes reflection message
    ref_a = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenant_A",
        topic="system.reflection",
        payload={
            "content": "Tenant A reflections",
            "entities": [{"label": "Apple", "type": "fruit"}],
            "rules": [],
        },
    )
    await bus.publish("system.reflection", ref_a)
    await asyncio.sleep(0.1)

    # Tenant B publishes reflection message
    ref_b = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenant_B",
        topic="system.reflection",
        payload={
            "content": "Tenant B reflections",
            "entities": [{"label": "Banana", "type": "fruit"}],
            "rules": [],
        },
    )
    await bus.publish("system.reflection", ref_b)
    await asyncio.sleep(0.1)

    # Directly check memory node's isolated knowledge graphs
    kg_a = mem._get_kg("tenant_A")
    kg_b = mem._get_kg("tenant_B")

    # Verify Tenant A has apple but not banana
    assert any(e.label == "apple" for e in kg_a._entities.values())
    assert not any(e.label == "banana" for e in kg_a._entities.values())

    # Verify Tenant B has banana but not apple
    assert any(e.label == "banana" for e in kg_b._entities.values())
    assert not any(e.label == "apple" for e in kg_b._entities.values())

    # Query via handle_knowledge_query for Tenant A
    query_msg_a = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenant_A",
        topic="knowledge.query",
        payload={"action": "all_entities"},
    )
    resp_a = await mem.handle_knowledge_query(query_msg_a)
    assert resp_a is not None
    entities_a = resp_a.payload["entities"]
    assert any(e["label"] == "apple" for e in entities_a)
    assert not any(e["label"] == "banana" for e in entities_a)

    # Query via handle_knowledge_query for Tenant B
    query_msg_b = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenant_B",
        topic="knowledge.query",
        payload={"action": "all_entities"},
    )
    resp_b = await mem.handle_knowledge_query(query_msg_b)
    assert resp_b is not None
    entities_b = resp_b.payload["entities"]
    assert any(e["label"] == "banana" for e in entities_b)
    assert not any(e["label"] == "apple" for e in entities_b)
