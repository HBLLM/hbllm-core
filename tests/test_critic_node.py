import pytest
import asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.workspace_node import WorkspaceNode
from hbllm.brain.critic_node import CriticNode
from tests.mock_llm import MockLLM


# Mock Intuition Engine for generation
class MockIntuitionNode:
    def __init__(self, bus):
        self.bus = bus
        self.evals_received = []
        
    async def start(self):
        await self.bus.subscribe("module.evaluate", self.handle_evaluate)
        
    async def handle_evaluate(self, msg: Message):
        self.evals_received.append(msg)
        
        text = msg.payload.get("text", "")
        if "CRITICAL FEEDBACK" in text:
            await self.propose("I am a helpful assistant and the sky is blue.", msg.correlation_id)
        else:
            await self.propose("As an AI language model, I don't know the answer.", msg.correlation_id)
            
    async def propose(self, content, corr_id):
        thought = Message(
            type=MessageType.EVENT,
            source_node_id="intuition_01",
            topic="workspace.thought",
            payload={"type": "intuition", "confidence": 0.9, "content": content},
            correlation_id=corr_id
        )
        await self.bus.publish("workspace.thought", thought)


@pytest.fixture
async def critic_env():
    bus = InProcessBus()
    await bus.start()
    
    workspace = WorkspaceNode("workspace_01")
    await workspace.start(bus)
    
    mock_llm = MockLLM()
    critic = CriticNode("critic_01", llm=mock_llm)
    await critic.start(bus)
    
    intuition = MockIntuitionNode(bus)
    await intuition.start()
    
    yield bus, workspace, critic, intuition
    
    await workspace.stop()
    await critic.stop()
    await bus.stop()

@pytest.mark.asyncio
async def test_critic_active_halting_and_backtracking(critic_env):
    bus, workspace, critic, intuition = critic_env
    
    # 1. Provide an initial query to the workspace
    query = Message(
        type=MessageType.EVENT,
        source_node_id="router",
        topic="workspace.update",
        payload={"text": "What color is the sky?"},
        correlation_id="test_halt_001"
    )
    
    await bus.publish("workspace.update", query)
    
    # 2. Wait for the pipeline to settle.
    await asyncio.sleep(0.5) 
    
    # Intuition should have been called TWICE (once for initial, once for forced backtrack)
    assert len(intuition.evals_received) == 2
    
    second_eval = intuition.evals_received[1]
    
    # The second eval must contain the critic's backtracking context
    assert "CRITICAL FEEDBACK" in second_eval.payload.get("text", "")
    
    # Assert the flawed thought was removed from the blackboard
    board = workspace.blackboards.get("test_halt_001")
    if board:
        for t in board["thoughts"]:
            assert "As an AI language model" not in t["content"]
