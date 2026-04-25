import pytest
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

from plugins.hbllm_temporal import TemporalReasoningNode

@pytest.mark.asyncio
async def test_temporal_node_remind():
    bus = InProcessBus()
    await bus.start()
    node = TemporalReasoningNode(node_id="test_temporal")
    await node.start(bus)

    published_messages = []
    
    async def capture(msg: Message) -> None:
        published_messages.append(msg)

    await bus.subscribe("query.temporal.response", capture)

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="query.temporal",
        payload={"text": "remind me to buy milk in 10 minutes"}
    )
    
    await bus.publish("query.temporal", msg)
    
    import asyncio
    await asyncio.sleep(0.1)
    
    assert len(published_messages) == 1
    reply_msg = published_messages[0]
    assert "I'll remind you" in reply_msg.payload["text"]
    
    await bus.stop()

@pytest.mark.asyncio
async def test_temporal_node_past():
    bus = InProcessBus()
    await bus.start()
    node = TemporalReasoningNode(node_id="test_temporal")
    await node.start(bus)

    published_messages = []
    
    async def capture(msg: Message) -> None:
        published_messages.append(msg)

    await bus.subscribe("query.temporal.response", capture)

    msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="query.temporal",
        payload={"text": "what happened before the party"}
    )
    
    await bus.publish("query.temporal", msg)
    
    import asyncio
    await asyncio.sleep(0.1)
    
    assert len(published_messages) == 1
    reply_msg = published_messages[0]
    assert "episodic memory" in reply_msg.payload["text"]
    
    await bus.stop()
