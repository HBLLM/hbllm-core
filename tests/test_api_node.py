import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.actions.api_node import ApiNode
from tests.mock_llm import MockLLM

@pytest.fixture
async def api_env():
    bus = InProcessBus()
    await bus.start()
    
    mock_llm = MockLLM()
    api_node = ApiNode("api_node_01", llm=mock_llm)
    await api_node.start(bus)
    
    yield bus, api_node
    
    await api_node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_api_node_schema_synthesis(api_env):
    bus, api_node = api_env
    
    caught_thoughts = []
    
    async def thought_catcher(msg: Message):
        caught_thoughts.append(msg)
        
    await bus.subscribe("workspace.thought", thought_catcher)
    
    # Send a schema generation query
    query_msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_mock",
        topic="module.evaluate",
        payload={
            "text": "Generate an OpenAPI schema for a weather API endpoint",
            "domain_hint": "api_synth"
        }
    )
    
    await bus.publish("module.evaluate", query_msg)
    await asyncio.sleep(0.5)
    
    assert len(caught_thoughts) == 1
    thought = caught_thoughts[0]
    
    assert thought.payload["type"] == "api_synthesis"
    assert thought.payload["confidence"] == 0.90
    assert thought.payload["content"]  # Has content from MockLLM


@pytest.mark.asyncio
async def test_api_node_ignores_non_api_prompts(api_env):
    bus, api_node = api_env
    
    caught_thoughts = []
    async def thought_catcher(msg: Message):
        caught_thoughts.append(msg)
        
    await bus.subscribe("workspace.thought", thought_catcher)
    
    query_msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace_mock",
        topic="module.evaluate",
        payload={
            "text": "What is the meaning of life?",
            "domain_hint": "general"
        }
    )
    
    await bus.publish("module.evaluate", query_msg)
    await asyncio.sleep(0.1)
    
    # ApiNode should ignore it entirely
    assert len(caught_thoughts) == 0
