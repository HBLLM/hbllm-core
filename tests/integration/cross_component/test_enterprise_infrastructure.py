"""
Tests for Enterprise Infrastructure Hardening:
- Replay Protection
- Dead Letter Queue (DLQ)
- Key Revocation
"""

import asyncio

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import DeviceTier, NodeInfo, NodeType
from hbllm.network.registry import ServiceRegistry
from hbllm.security.identity import NodeIdentity
from hbllm.security.trust import TrustInterceptor


@pytest.fixture
async def secure_bus():
    registry = ServiceRegistry()
    await registry.start()
    bus = InProcessBus()
    interceptor = TrustInterceptor(registry=registry, bus=bus)
    bus.add_interceptor(interceptor)
    await bus.start()
    yield bus, registry, interceptor
    await bus.stop()
    await registry.stop()


@pytest.mark.asyncio
async def test_replay_protection(secure_bus):
    bus, registry, interceptor = secure_bus

    # Register a mock node
    identity = NodeIdentity.generate()
    node_info = NodeInfo(
        node_id="test_node",
        node_type=NodeType.DOMAIN_MODULE,
        public_key=identity.public_key_b64,
        device_tier=DeviceTier.SERVER,
        scopes=["public"],
    )
    registry._nodes["test_node"] = node_info

    msg = Message(
        type=MessageType.EVENT,
        topic="public.test",
        source_node_id="test_node",
        payload={"data": "original"},
    )
    msg.signature = identity.sign(msg.signable_data)

    received_msgs = []

    async def capture_msg(m):
        received_msgs.append(m)

    await bus.subscribe("public.test", capture_msg)

    dlq_msgs = []

    async def capture_dlq(m):
        dlq_msgs.append(m)

    await bus.subscribe("system.dlq", capture_dlq)

    # Publish the message the first time
    await bus.publish("public.test", msg)
    await asyncio.sleep(0.5)

    assert len(received_msgs) == 1
    assert len(dlq_msgs) == 0

    # Replay the exact same message
    await bus.publish("public.test", msg)
    await asyncio.sleep(0.5)

    # Should NOT be delivered again
    assert len(received_msgs) == 1

    # Should be forwarded to DLQ (1 detailed, 1 generic from bus)
    assert len(dlq_msgs) == 2
    assert dlq_msgs[0].payload["dlq_reason"] == "replay_attack"
    assert dlq_msgs[0].payload["original_topic"] == "public.test"


@pytest.mark.asyncio
async def test_key_revocation_blocks_messages(secure_bus):
    bus, registry, interceptor = secure_bus

    # Register a mock node
    identity = NodeIdentity.generate()
    node_info = NodeInfo(
        node_id="hacked_node",
        node_type=NodeType.DOMAIN_MODULE,
        public_key=identity.public_key_b64,
        device_tier=DeviceTier.SERVER,
        scopes=["public"],
    )
    registry._nodes["hacked_node"] = node_info

    # Revoke the node
    await registry.revoke_node("hacked_node", reason="Compromised credentials")

    msg = Message(
        type=MessageType.EVENT,
        topic="public.test",
        source_node_id="hacked_node",
        payload={"data": "malicious payload"},
    )
    msg.signature = identity.sign(msg.signable_data)

    received_msgs = []

    async def capture_msg(m):
        received_msgs.append(m)

    await bus.subscribe("public.test", capture_msg)

    dlq_msgs = []

    async def capture_dlq(m):
        dlq_msgs.append(m)

    await bus.subscribe("system.dlq", capture_dlq)

    # Publish from revoked node
    await bus.publish("public.test", msg)
    await asyncio.sleep(0.5)

    # Should NOT be delivered
    assert len(received_msgs) == 0

    # Should be forwarded to DLQ
    assert len(dlq_msgs) == 2
    assert dlq_msgs[0].payload["dlq_reason"] == "node_revoked"


@pytest.mark.asyncio
async def test_invalid_signature_to_dlq(secure_bus):
    bus, registry, interceptor = secure_bus

    # Register a mock node
    identity = NodeIdentity.generate()
    node_info = NodeInfo(
        node_id="legit_node",
        node_type=NodeType.DOMAIN_MODULE,
        public_key=identity.public_key_b64,
        device_tier=DeviceTier.SERVER,
        scopes=["public"],
    )
    registry._nodes["legit_node"] = node_info

    msg = Message(
        type=MessageType.EVENT,
        topic="public.test",
        source_node_id="legit_node",
        payload={"data": "forged payload"},
    )
    # Intentionally bad signature
    msg.signature = "invalid_base64_signature_string=="

    received_msgs = []

    async def capture_msg(m):
        received_msgs.append(m)

    await bus.subscribe("public.test", capture_msg)

    dlq_msgs = []

    async def capture_dlq(m):
        dlq_msgs.append(m)

    await bus.subscribe("system.dlq", capture_dlq)

    # Publish forged message
    await bus.publish("public.test", msg)
    await asyncio.sleep(0.5)

    # Should NOT be delivered
    assert len(received_msgs) == 0

    # Should be forwarded to DLQ
    assert len(dlq_msgs) == 2
    assert dlq_msgs[0].payload["dlq_reason"] == "invalid_signature"
