import pytest
import asyncio
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.actions.execution_node import ExecutionNode
from hbllm.brain.workspace_node import WorkspaceNode

@pytest.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()

@pytest.mark.asyncio
async def test_workspace_executes_python_and_commits(bus):
    exec_node = ExecutionNode(node_id="exec_1", timeout=1.0)
    workspace = WorkspaceNode(node_id="workspace_1", thinking_deadline=0.5)
    
    await exec_node.start(bus)
    await workspace.start(bus)
    
    decision_received = []
    async def mock_decision(msg: Message) -> None:
        decision_received.append(msg.payload)
        
    await bus.subscribe("decision.evaluate", mock_decision)
    
    try:
        update_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.update",
            payload={"text": "Write code to add 2+2"},
        )
        await bus.publish("workspace.update", update_msg)
        await asyncio.sleep(0.1)
        
        corr_id = list(workspace.blackboards.keys())[0]
        
        thought_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.thought",
            payload={
                "type": "intuition",
                "confidence": 0.9,
                "content": "Here is the code: ```python\nprint(2+2)\n```"
            },
            correlation_id=corr_id
        )
        await bus.publish("workspace.thought", thought_msg)
        
        # Wait for deadline to expire + execution to finish
        await asyncio.sleep(1.0)
        
        assert len(decision_received) == 1
        decision = decision_received[0]["selected_thought"]
        assert "execution_output" in decision
        assert decision["execution_output"] == "4"
        
    finally:
        await exec_node.stop()
        await workspace.stop()
        await asyncio.sleep(0.1)

@pytest.mark.asyncio
async def test_workspace_fails_bad_python_and_monologues(bus):
    exec_node = ExecutionNode(node_id="exec_2", timeout=1.0)
    workspace = WorkspaceNode(node_id="workspace_2", thinking_deadline=0.5)
    
    await exec_node.start(bus)
    await workspace.start(bus)
    
    monologue_received = []
    async def mock_monologue(msg: Message) -> None:
        monologue_received.append(msg.payload)
        
    await bus.subscribe("module.evaluate", mock_monologue)
    
    try:
        update_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.update",
            payload={"text": "Write bad code"},
        )
        await bus.publish("workspace.update", update_msg)
        await asyncio.sleep(0.1)
        
        corr_id = list(workspace.blackboards.keys())[0]
        
        # Clear the initial broadcast from workspace.update
        monologue_received.clear()
        
        thought_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.thought",
            payload={
                "type": "intuition",
                "confidence": 0.9,
                "content": "Bad code: ```python\nprint(1/0)\n```"
            },
            correlation_id=corr_id
        )
        await bus.publish("workspace.thought", thought_msg)
        
        await asyncio.sleep(1.0)
        
        # Should have broadcast a failure monologue
        assert len(monologue_received) == 1
        assert "CRITICAL SYSTEM ERROR" in monologue_received[0]["text"]
        assert "ZeroDivisionError" in monologue_received[0]["text"]
        
    finally:
        await exec_node.stop()
        await workspace.stop()
        await asyncio.sleep(0.1)
