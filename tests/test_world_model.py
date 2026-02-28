import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.world_model_node import WorldModelNode

@pytest.fixture
async def simulated_bus():
    bus = InProcessBus()
    await bus.start()
    world_model = WorldModelNode(node_id="world_model_01")
    await world_model.start(bus)
    
    yield bus, world_model
    
    await world_model.stop()
    await bus.stop()

@pytest.mark.asyncio
async def test_world_model_safe_code(simulated_bus):
    bus, world_model = simulated_bus
    
    responses = []
    async def thought_listener(msg):
        responses.append(msg)
        return None
    await bus.subscribe("workspace.thought", thought_listener)
    
    # Send a safe piece of code
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="workspace.simulate",
        payload={
            "action_type": "execute_python",
            "content": "def add(a, b): return a + b\nprint(add(2, 2))"
        }
    )
    
    await bus.publish("workspace.simulate", msg)
    await asyncio.sleep(0.1)
    
    assert len(responses) == 1
    resp = responses[0]
    payload = resp.payload
    
    assert payload["type"] == "simulation_result"
    assert payload["prediction"] == "SUCCESS"
    assert "AST check passed" in payload["content"]
    assert payload["confidence"] == 1.0

@pytest.mark.asyncio
async def test_world_model_dangerous_import(simulated_bus):
    bus, world_model = simulated_bus
    
    responses = []
    async def thought_listener(msg):
        responses.append(msg)
        return None
    await bus.subscribe("workspace.thought", thought_listener)
    
    # Send dangerous code
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="workspace.simulate",
        payload={
            "action_type": "execute_python",
            "content": "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"
        }
    )
    
    await bus.publish("workspace.simulate", msg)
    await asyncio.sleep(0.1)
    
    assert len(responses) == 1
    resp = responses[0]
    payload = resp.payload
    
    assert payload["type"] == "simulation_result"
    assert payload["prediction"] == "FAILURE"
    assert "blocked by sandbox safety policies" in payload["content"]

@pytest.mark.asyncio
async def test_world_model_syntax_error(simulated_bus):
    bus, world_model = simulated_bus
    
    responses = []
    async def thought_listener(msg):
        responses.append(msg)
        return None
    await bus.subscribe("workspace.thought", thought_listener)
    
    # Send malformed code
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_01",
        topic="workspace.simulate",
        payload={
            "action_type": "execute_python",
            "content": "def bad_func(:\n  return 'oops'"
        }
    )
    
    await bus.publish("workspace.simulate", msg)
    await asyncio.sleep(0.1)
    
    assert len(responses) == 1
    resp = responses[0]
    payload = resp.payload
    
    assert payload["type"] == "simulation_result"
    assert payload["prediction"] == "FAILURE"
    assert "SyntaxError" in payload["content"]
