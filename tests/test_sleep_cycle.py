import pytest
import asyncio
import time

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.sleep_node import SleepCycleNode

# Mock Memory Node to simulate retrieval
class MockMemoryNode:
    def __init__(self, bus):
        self.bus = bus
        self.store_calls = []
    
    async def start(self):
        await self.bus.subscribe("memory.retrieve_recent", self.handle_retrieve)
        await self.bus.subscribe("memory.store", self.handle_store)
        
    async def handle_retrieve(self, msg: Message):
        # Return 5 dummy turns to trigger the Sleep Node compression
        turns = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        return msg.create_response({"turns": turns})
        
    async def handle_store(self, msg: Message):
        self.store_calls.append(msg)
        return None

@pytest.fixture
async def simulated_sleep_env():
    bus = InProcessBus()
    await bus.start()
    
    memory = MockMemoryNode(bus)
    await memory.start()
    
    # Very short timeout for fast tests
    sleep_node = SleepCycleNode(node_id="sleep_01", idle_timeout_seconds=0.5)
    await sleep_node.start(bus)
    
    yield bus, sleep_node, memory
    
    await sleep_node.stop()
    await bus.stop()

@pytest.mark.asyncio
async def test_sleep_cycle_triggers_on_idle(simulated_sleep_env):
    bus, sleep_node, memory = simulated_sleep_env
    
    # Assert it starts awake
    assert not sleep_node.is_sleeping
    
    # Let it idle past the 0.5s timeout + 0.5s loop interval
    await asyncio.sleep(1.2)
    
    # Should now be asleep (in consolidation mode) or have finished sleeping
    # Since we mocked memory, the compression should execute quickly and reset flag
    assert len(memory.store_calls) > 0
    
    store_msg = memory.store_calls[0]
    payload = store_msg.payload
    
    assert payload["role"] == "system"
    assert "CONSOLIDATED MEMORY" in payload["content"]

@pytest.mark.asyncio
async def test_sleep_cycle_interrupted_by_user(simulated_sleep_env):
    bus, sleep_node, memory = simulated_sleep_env
    
    # Wait half the timeout
    await asyncio.sleep(0.3)
    
    # User sends a query, which should reset the idle timer
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="user",
        topic="router.query",
        payload={"text": "Hello"}
    )
    await bus.publish("router.query", msg)
    
    # Wait another half timeout (total 0.6s since start)
    await asyncio.sleep(0.3)
    
    # Because of the interruption, it should NOT have triggered sleep
    assert not sleep_node.is_sleeping
    assert len(memory.store_calls) == 0
    
    # Wait a full idle timeout from the interrupt + loop interval
    await asyncio.sleep(1.2)
    
    # Now it should trigger
    assert len(memory.store_calls) > 0
