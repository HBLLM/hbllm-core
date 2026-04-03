"""
End-to-End Integration Test — verifies nodes process messages on the bus,
memory stores/retrieves, and persistence survives restarts.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


class MockLLM:
    """Minimal LLM that returns deterministic responses."""

    def generate(self, prompt, **kwargs):
        return "This is a test response from the mock LLM."

    async def async_generate(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


@pytest.fixture
def tmp_db_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_memory_store_and_retrieve(tmp_db_dir):
    """MemoryNode stores and retrieves conversation turns via the bus."""
    bus = InProcessBus()
    await bus.start()

    db_path = Path(tmp_db_dir) / "memory.db"
    memory = MemoryNode(node_id="memory", db_path=str(db_path))
    await memory.start(bus)

    # Store user message
    store_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        session_id="s1",
        topic="memory.store",
        payload={"session_id": "s1", "tenant_id": "t1", "role": "user", "content": "Hello world"},
    )
    await bus.publish("memory.store", store_msg)
    await asyncio.sleep(0.3)

    turns = memory.db.retrieve_recent("s1", limit=5, tenant_id="t1")
    assert len(turns) >= 1
    assert any(t["content"] == "Hello world" for t in turns)

    # Store assistant reply
    store_msg2 = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        session_id="s1",
        topic="memory.store",
        payload={"session_id": "s1", "tenant_id": "t1", "role": "assistant", "content": "Hi!"},
    )
    await bus.publish("memory.store", store_msg2)
    await asyncio.sleep(0.3)

    turns = memory.db.retrieve_recent("s1", limit=10, tenant_id="t1")
    assert len(turns) >= 2

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_memory_multi_tenant_isolation(tmp_db_dir):
    """Different tenants have isolated memory."""
    bus = InProcessBus()
    await bus.start()

    db_path = Path(tmp_db_dir) / "memory.db"
    memory = MemoryNode(node_id="memory", db_path=str(db_path))
    await memory.start(bus)

    # Store for tenant A
    msg_a = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenantA",
        topic="memory.store",
        payload={
            "session_id": "s1",
            "tenant_id": "tenantA",
            "role": "user",
            "content": "Tenant A data",
        },
    )
    await bus.publish("memory.store", msg_a)

    # Store for tenant B
    msg_b = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="tenantB",
        topic="memory.store",
        payload={
            "session_id": "s1",
            "tenant_id": "tenantB",
            "role": "user",
            "content": "Tenant B data",
        },
    )
    await bus.publish("memory.store", msg_b)
    await asyncio.sleep(0.3)

    turns_a = memory.db.retrieve_recent("s1", tenant_id="tenantA")
    turns_b = memory.db.retrieve_recent("s1", tenant_id="tenantB")
    assert len(turns_a) == 1
    assert turns_a[0]["content"] == "Tenant A data"
    assert len(turns_b) == 1
    assert turns_b[0]["content"] == "Tenant B data"

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_bus_message_delivery():
    """Messages published on the bus reach subscribed handlers."""
    bus = InProcessBus()
    await bus.start()

    received = []

    async def handler(msg: Message):
        received.append(msg)

    await bus.subscribe("test.topic", handler)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.topic",
        payload={"data": "hello"},
    )
    await bus.publish("test.topic", msg)
    await asyncio.sleep(0.3)

    assert len(received) == 1
    assert received[0].payload["data"] == "hello"

    await bus.stop()


@pytest.mark.asyncio
async def test_knowledge_graph_via_bus(tmp_db_dir):
    """KnowledgeGraph queries work through the bus via MemoryNode."""
    bus = InProcessBus()
    await bus.start()

    db_path = Path(tmp_db_dir) / "memory.db"
    memory = MemoryNode(node_id="memory", db_path=str(db_path))
    await memory.start(bus)

    # Directly add to knowledge graph
    memory.knowledge_graph.add_relation("python", "programming language", "is_a")
    memory.knowledge_graph.add_relation("python", "indentation", "uses")

    assert memory.knowledge_graph.entity_count >= 3
    assert memory.knowledge_graph.relation_count >= 2

    neighbors = memory.knowledge_graph.neighbors("python", direction="out")
    assert len(neighbors) == 2

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_memory_persistence_round_trip(tmp_db_dir):
    """MemoryNode persists KG and semantic data on stop, reloads on restart."""
    bus = InProcessBus()
    await bus.start()

    db_path = Path(tmp_db_dir) / "memory.db"
    memory = MemoryNode(node_id="memory", db_path=str(db_path))
    await memory.start(bus)

    # Add data
    memory.knowledge_graph.add_relation("neural network", "machine learning", "is_a")
    memory.semantic_db.store("Neural networks are powerful.")

    # Stop persists data
    await memory.stop()
    await bus.stop()

    # Verify files
    assert (Path(tmp_db_dir) / "knowledge_graph.json").exists()
    assert (Path(tmp_db_dir) / "semantic" / "documents.json").exists()

    # Restart — should auto-load
    bus2 = InProcessBus()
    await bus2.start()
    memory2 = MemoryNode(node_id="memory2", db_path=str(db_path))
    await memory2.start(bus2)

    assert memory2.knowledge_graph.entity_count >= 2
    assert memory2.knowledge_graph.relation_count >= 1
    assert memory2.semantic_db.count >= 1

    await memory2.stop()
    await bus2.stop()


@pytest.mark.asyncio
async def test_semantic_search_via_memory(tmp_db_dir):
    """SemanticMemory search works through MemoryNode."""
    bus = InProcessBus()
    await bus.start()

    db_path = Path(tmp_db_dir) / "memory.db"
    memory = MemoryNode(node_id="memory", db_path=str(db_path))
    await memory.start(bus)

    memory.semantic_db.store("Python programming tutorial", {"domain": "tech"})
    memory.semantic_db.store("Chocolate cake recipe", {"domain": "food"})

    assert memory.semantic_db.count == 2

    # Search
    results = memory.semantic_db.search("Python coding", top_k=1)
    assert len(results) >= 1

    await memory.stop()
    await bus.stop()
