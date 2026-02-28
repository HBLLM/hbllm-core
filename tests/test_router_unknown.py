import asyncio
import logging
import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.brain.router_node import RouterNode
from tests.mock_llm import MockLLM

@pytest.mark.asyncio
async def test_router_unknown_topic():
    bus = InProcessBus()
    await bus.start()
    
    mock_llm = MockLLM()
    router = RouterNode(node_id="test_router", llm=mock_llm)
    router.spawn_trigger_count = 2
    
    await router.start(bus)
    await asyncio.sleep(0.1)
    
    spawn_events = []
    
    async def spawn_handler(msg: Message) -> Message | None:
        spawn_events.append(msg)
        return None
        
    await bus.subscribe("system.spawn", spawn_handler)
    
    # 1. Ask a general query (should route, no spawn)
    msg1 = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="router.query",
        payload=QueryPayload(text="Hello world").model_dump()
    )
    resp1 = await router.handle_message(msg1)
    assert resp1 is None
    assert len(spawn_events) == 0
    
    # 2. Ask an unknown topic query (MockLLM returns low confidence for "biology")
    msg2 = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="router.query",
        payload=QueryPayload(text="Tell me about marine biology.").model_dump()
    )
    resp2 = await router.handle_message(msg2)
    assert resp2 is None
    assert router.unknown_counts["general_unknown"] == 1
    assert len(spawn_events) == 0
    
    # 3. Ask a SECOND unknown topic query (Should trigger spawn)
    msg3 = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="router.query",
        payload=QueryPayload(text="Explain quantum biology to me.").model_dump()
    )
    resp3 = await router.handle_message(msg3)
    
    assert resp3.type == MessageType.RESPONSE
    assert "spawning" in resp3.payload["text"]
    
    await asyncio.sleep(0.1)
    
    assert len(spawn_events) == 1
    spawn_payload = spawn_events[0].payload
    assert spawn_payload["trigger_query"] == "Explain quantum biology to me."
    
    assert router.unknown_counts["general_unknown"] == 0
    
    await router.stop()
    await bus.stop()
