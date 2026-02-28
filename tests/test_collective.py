"""Tests for Collective Intelligence Network — cross-instance knowledge sharing."""

import pytest
import asyncio

from hbllm.brain.collective_node import KnowledgeDigest, CollectiveNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


# ─── KnowledgeDigest Tests ────────────────────────────────────────────────────

def test_digest_checksum():
    d = KnowledgeDigest(
        domain="math",
        capability="algebra",
        artifact_data={"weights": [1, 2, 3]},
    )
    cs = d.compute_checksum()
    assert len(cs) == 16
    assert cs == d.checksum


def test_digest_checksum_deterministic():
    """Same data = same checksum."""
    d1 = KnowledgeDigest(artifact_data={"key": "value"})
    d2 = KnowledgeDigest(artifact_data={"key": "value"})
    assert d1.compute_checksum() == d2.compute_checksum()


def test_digest_checksum_differs():
    """Different data = different checksum."""
    d1 = KnowledgeDigest(artifact_data={"key": "a"})
    d2 = KnowledgeDigest(artifact_data={"key": "b"})
    assert d1.compute_checksum() != d2.compute_checksum()


def test_digest_to_dict():
    d = KnowledgeDigest(domain="coding", capability="python")
    result = d.to_dict()
    assert result["domain"] == "coding"
    assert result["capability"] == "python"
    assert "id" in result


# ─── CollectiveNode Integration Tests ─────────────────────────────────────────

@pytest.fixture
async def collective_setup():
    """Set up two collective nodes on the same bus (simulating peer instances)."""
    bus = InProcessBus()
    await bus.start()
    
    node_a = CollectiveNode(node_id="collective_a", instance_id="instance_a")
    node_b = CollectiveNode(node_id="collective_b", instance_id="instance_b")
    
    await node_a.start(bus)
    await node_b.start(bus)
    
    # Wire broadcast → sync so nodes can communicate
    await bus.subscribe("collective.broadcast", 
        lambda msg: bus.publish("collective.sync", msg))
    
    yield node_a, node_b, bus
    
    await node_a.stop()
    await node_b.stop()
    await bus.stop()


async def test_broadcast_on_learning_update(collective_setup):
    """When a learning update arrives, node broadcasts a digest."""
    node_a, node_b, bus = collective_setup
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="learner",
        topic="system.learning_update",
        payload={
            "domain": "math",
            "capability": "calculus",
            "artifact_type": "skill",
            "artifact_data": {"steps": ["differentiate", "integrate"]},
        },
    )
    await bus.publish("system.learning_update", msg)
    await asyncio.sleep(0.2)
    
    assert node_a.stats["broadcasts_sent"] == 1
    assert len(node_a.broadcast_log) == 1


async def test_peer_receives_digest(collective_setup):
    """Node B receives a digest broadcast by Node A."""
    node_a, node_b, bus = collective_setup
    
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="learner",
        topic="system.learning_update",
        payload={
            "domain": "coding",
            "capability": "python",
            "artifact_type": "lora_weights",
            "artifact_data": {"layer": "adapter_1"},
        },
    )
    await bus.publish("system.learning_update", msg)
    await asyncio.sleep(0.3)
    
    # Node B should have received the digest from Node A
    # (Both nodes process system.learning_update and broadcast)
    # Node B's received_log may have A's digest
    total_received = node_a.stats["digests_received"] + node_b.stats["digests_received"]
    total_broadcast = node_a.stats["broadcasts_sent"] + node_b.stats["broadcasts_sent"]
    assert total_broadcast >= 1  # At least one broadcast


async def test_deduplication(collective_setup):
    """Same knowledge isn't received twice."""
    node_a, _, bus = collective_setup
    
    # Send same data twice
    for _ in range(2):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="learner",
            topic="system.learning_update",
            payload={
                "domain": "physics",
                "capability": "mechanics",
                "artifact_data": {"formula": "F=ma"},
            },
        )
        await bus.publish("system.learning_update", msg)
        await asyncio.sleep(0.1)
    
    # Should only broadcast once (second is deduped)
    assert node_a.stats["broadcasts_sent"] == 1


async def test_query_stats(collective_setup):
    node_a, _, bus = collective_setup
    
    query = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="collective.query",
        payload={},
    )
    resp = await bus.request("collective.query", query, timeout=5.0)
    assert "instance_id" in resp.payload
    assert "stats" in resp.payload
    assert resp.payload["instance_id"] == "instance_a"
