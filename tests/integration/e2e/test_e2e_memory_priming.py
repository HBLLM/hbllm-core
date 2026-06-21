"""
End-to-End Integration Test for SNN Priming and Latent Memory Clusters in HBLLM Core.

Verifies:
1. Document storage preserves original domains while registering dynamic cluster IDs.
2. User queries published to `router.query` stimulate the WorkingMemoryPrimer in the MemoryNode.
3. RAG search queries published to `memory.search` apply SNN primed boosts to bias search scores.
4. User feedback published to `memory.feedback` updates cluster statistics and reinforces Hebbian synaptic weights.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest_asyncio.fixture
async def memory_bus_system(monkeypatch):
    """Boot a mock InProcessBus and MemoryNode with an in-memory database."""
    bus = InProcessBus()
    await bus.start()

    memory = MemoryNode(node_id="memory", db_path=":memory:")
    await memory.start(bus)

    # Patch ServiceRegistry.verify_message to bypass cryptographic validation in tests
    import hbllm.network.registry

    async def mock_verify_message(*args, **kwargs):
        return True

    monkeypatch.setattr(
        hbllm.network.registry.ServiceRegistry, "verify_message", mock_verify_message
    )

    yield bus, memory

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_e2e_memory_priming_and_feedback(memory_bus_system):
    bus, memory = memory_bus_system

    # 1. Store documents with user-supplied metadata domains
    doc_phys = memory.semantic_db.store(
        "Introduction to string theory and quantum loop gravity.",
        metadata={"domain": "physics"},
        tenant_id="default",
        user_id="default",
        device_id="default",
    )
    doc_math = memory.semantic_db.store(
        "Solving differential equations using calculus principles.",
        metadata={"domain": "math"},
        tenant_id="default",
        user_id="default",
        device_id="default",
    )

    # Verify that original domains are preserved (not overwritten by cluster manager)
    memory.semantic_db._flush_tfidf()
    assert memory.semantic_db.documents[doc_phys]["metadata"]["domain"] == "physics"
    assert memory.semantic_db.documents[doc_math]["metadata"]["domain"] == "math"

    # Verify they were assigned to dynamic clusters in LatentClusterManager
    c_phys = memory.semantic_db.cluster_manager.cluster_assignments[doc_phys]
    c_math = memory.semantic_db.cluster_manager.cluster_assignments[doc_math]
    assert c_phys is not None
    assert c_math is not None
    assert c_phys != c_math

    # 2. Simulate user query arriving on the bus, triggering text-based priming
    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test_client",
        tenant_id="default",
        topic="router.query",
        payload={"text": "Solving equations using math and calculus"},
    )
    await bus.publish("router.query", query_msg)

    # Wait for the async task callback to process the priming stimulation
    await asyncio.sleep(0.2)

    # The SNN primer should have been stimulated for the "math" category
    boosts = memory.primer.get_boosts()
    assert boosts["math"] > 0.0
    assert boosts["physics"] == 0.0

    # 3. Perform a memory search via the bus topic `memory.search`
    # We query for "theory". Normally, the physics doc matches "theory" better,
    # but since "math" is primed, the math document should receive the primed retrieval boost
    # and outrank the physics document.
    search_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test_client",
        tenant_id="default",
        topic="memory.search",
        payload={"query_text": "solving gravity", "top_k": 2},
    )

    response = await bus.request("memory.search", search_msg, timeout=3.0)
    results = response.payload["results"]

    # Verify that search returned results, and because of math priming, doc_math is boosted to the top!
    assert len(results) >= 2
    assert results[0]["id"] == doc_math

    # Verify that the activation count for doc_math's cluster was incremented
    assert memory.semantic_db.cluster_manager.cluster_stats[c_math]["activation_count"] == 1

    # 4. Simulate positive feedback on the retrieval output to verify Hebbian learning
    feedback_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test_client",
        tenant_id="default",
        topic="memory.feedback",
        payload={"note_id": doc_math, "useful": True},
    )

    feedback_response = await bus.request("memory.feedback", feedback_msg, timeout=3.0)
    assert feedback_response.payload["status"] == "updated"

    # Verify that the success rate of the dynamic cluster was updated in stats
    assert memory.semantic_db.cluster_manager.cluster_stats[c_math]["positive_feedback_count"] == 1
    assert memory.semantic_db.cluster_manager.cluster_stats[c_math]["success_rate"] == 1.0

    # Verify that Hebbian Plasticity updated synaptic weights from prime_cat "math" to "math"
    # and also to the dynamic cluster of doc_math!
    weights = memory.semantic_db.synaptic_weights
    math_weights = weights.get("math", {})
    assert math_weights.get("math") > 0.95
    assert math_weights.get(f"cluster_{c_math}") > 0.0
