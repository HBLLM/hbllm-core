"""
Tests for GraphRAG Hierarchical Clustering during Sleep Cycles.
"""

import asyncio

import pytest

from hbllm.brain.sleep_node import SleepCycleNode
from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


class MockLLM:
    """Mock LLM to simulate the GraphRAG clustering instruction."""
    def __init__(self):
        self.called = False

    async def generate_json(self, prompt: str, **kwargs) -> dict:
        self.called = True
        return {
            "communities": [
                {
                    "name": "Machine Learning Algorithms",
                    "summary": "Various algorithms used in ML systems.",
                    "members": ["attention", "softmax", "transformers"]
                },
                {
                    "name": "Hardware",
                    "summary": "Physical compute units.",
                    "members": ["gpu", "tpu"]
                }
            ]
        }


@pytest.mark.asyncio
async def test_graphrag_hierarchical_clustering():
    """
    Test Phase 11 feature:
    Validates that the SleepNode queries memory for leaf nodes, uses the LLM
    to group them into Communities, and writes them back to the MemoryNode.
    """
    bus = InProcessBus()
    await bus.start()

    # 1. Setup Memory Node with some dummy leaf entities already parsed
    memory = MemoryNode("memory", ":memory:")
    await memory.start(bus)

    # Pre-populate leaf nodes
    memory.knowledge_graph.add_entity(label="attention", entity_type="concept")
    memory.knowledge_graph.add_entity(label="softmax", entity_type="concept")
    memory.knowledge_graph.add_entity(label="transformers", entity_type="concept")
    memory.knowledge_graph.add_entity(label="gpu", entity_type="concept")
    memory.knowledge_graph.add_entity(label="tpu", entity_type="concept")

    # Mock learner to immediately complete phase 2 so test doesn't hang
    async def mock_learner(msg):
        await bus.publish("system.learning_update", Message(
            type=MessageType.EVENT,
            source_node_id="mock_learner",
            topic="system.learning_update",
            payload={"status": "complete"}
        ))
    await bus.subscribe("system.sleep.dpo_trigger", mock_learner)

    # 2. Setup Sleep Node with our mock LLM
    mock_llm = MockLLM()
    sleep_node = SleepCycleNode("sleep", idle_timeout_seconds=0.1, llm=mock_llm)
    await sleep_node.start(bus)

    try:
        assert memory.knowledge_graph.entity_count == 5

        # Wait for the sleep node to notice idleness and trigger a deep sleep cycle
        # We need to wait enough time for the monitor loop to trigger, fetch memories,
        # fetch entities, and process LLM
        await asyncio.sleep(0.5)

        assert mock_llm.called, "The SleepNode did not attempt to cluster entities via LLM."

        # Memory Graph should now have 2 new community nodes (Total: 7)
        assert memory.knowledge_graph.entity_count == 7

        # 5 leaves point to Comm 1/2 = 5 relations
        assert memory.knowledge_graph.relation_count == 5

        ml_comm = memory.knowledge_graph.get_entity("Machine Learning Algorithms")
        assert ml_comm is not None
        assert ml_comm.entity_type == "community"
        assert ml_comm.attributes["is_macro_node"] is True

        hw_comm = memory.knowledge_graph.get_entity("Hardware")
        assert hw_comm is not None
        assert hw_comm.entity_type == "community"

        # Verify relations
        ml_neighbors = memory.knowledge_graph.neighbors("Machine Learning Algorithms", direction="in", relation_type="member_of")
        ml_member_labels = {rel["entity"] for rel in ml_neighbors}
        assert ml_member_labels == {"attention", "softmax", "transformers"}

        hw_neighbors = memory.knowledge_graph.neighbors("Hardware", direction="in", relation_type="member_of")
        hw_member_labels = {rel["entity"] for rel in hw_neighbors}
        assert hw_member_labels == {"gpu", "tpu"}
    finally:
        await sleep_node.stop()
        await memory.stop()
        await bus.stop()
