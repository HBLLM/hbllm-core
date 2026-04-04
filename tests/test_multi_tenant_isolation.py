import asyncio

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.mark.asyncio
async def test_multi_tenant_isolation():
    bus = InProcessBus()
    await bus.start()

    received_a = []
    received_b = []

    async def handler_a(msg: Message) -> None:
        received_a.append(msg)

    async def handler_b(msg: Message) -> None:
        received_b.append(msg)

    await bus.subscribe("test.topic", handler_a, tenant_id="tenant_A")
    await bus.subscribe("test.topic", handler_b, tenant_id="tenant_B")

    msg_a = Message(
        type=MessageType.EVENT, source_node_id="test", topic="test.topic", tenant_id="tenant_A"
    )
    msg_b = Message(
        type=MessageType.EVENT, source_node_id="test", topic="test.topic", tenant_id="tenant_B"
    )

    await bus.publish("test.topic", msg_a)
    await bus.publish("test.topic", msg_b)

    await asyncio.sleep(0.1)

    assert len(received_a) == 1
    assert received_a[0].tenant_id == "tenant_A"
    assert len(received_b) == 1
    assert received_b[0].tenant_id == "tenant_B"
    await bus.stop()
