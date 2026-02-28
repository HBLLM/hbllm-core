import asyncio
import uuid
import pytest
from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.network.redis_bus import RedisBus
import redis.asyncio as redis

@pytest.fixture
async def redis_bus():
    bus = RedisBus("redis://localhost:6379")
    try:
        # Check if Redis is alive
        client = redis.from_url("redis://localhost:6379")
        await client.ping()
        await client.aclose()
    except redis.ConnectionError:
        pytest.skip("Redis server not running on localhost:6379")
        
    await bus.start()
    yield bus
    await bus.stop()

@pytest.mark.asyncio
async def test_redis_bus_pubsub(redis_bus: RedisBus):
    received = []
    
    async def handler(msg: Message):
        received.append(msg)
        return None
        
    sub = await redis_bus.subscribe("test.topic", handler)
    
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test_node",
        topic="test.topic",
        payload=QueryPayload(text="Hello Redis!").model_dump()
    )
    
    await redis_bus.publish("test.topic", msg)
    await asyncio.sleep(0.1) # Wait for network dispatch
    
    assert len(received) == 1
    assert received[0].payload["text"] == "Hello Redis!"
    
    await redis_bus.unsubscribe(sub)

@pytest.mark.asyncio
async def test_redis_bus_request_response(redis_bus: RedisBus):
    async def echo_handler(msg: Message):
        # Create correlated response
        return msg.create_response({"echo": msg.payload["text"]})
        
    sub = await redis_bus.subscribe("test.echo", echo_handler)
    
    req = Message(
        type=MessageType.QUERY,
        source_node_id="client_node",
        topic="test.echo",
        payload=QueryPayload(text="Ping").model_dump()
    )
    
    response = await redis_bus.request("test.echo", req, timeout=2.0)
    
    assert response is not None
    assert response.correlation_id == req.id
    assert response.payload["echo"] == "Ping"
    
    await redis_bus.unsubscribe(sub)
