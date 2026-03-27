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

from hbllm.brain.learner_node import LearnerNode

@pytest.mark.asyncio
async def test_autonomous_learning(bus):
    exec_node = ExecutionNode(node_id="exec_3", timeout=1.0)
    workspace = WorkspaceNode(node_id="workspace_3", thinking_deadline=1.0)
    
    # Batch size 2 prevents it from actually triggering torch logic
    learner = LearnerNode(node_id="learner_1", batch_size=2, model=None, tokenizer=None)
    
    await exec_node.start(bus)
    await workspace.start(bus)
    await learner.start(bus)
    
    try:
        # 1. Provide the initial query
        update_msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.update",
            payload={"text": "Calculate fibonacci 5", "message_id": "test_query_123"},
        )
        await bus.publish("workspace.update", update_msg)
        await asyncio.sleep(0.1)
        
        corr_id = list(workspace.blackboards.keys())[0]
        
        # 2. First attempt fails (Logic Error / Crash)
        bad_thought = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.thought",
            payload={
                "type": "intuition",
                "confidence": 0.9,
                "content": "Let's divide by zero: ```python\nprint(1/0)\n```"
            },
            correlation_id=corr_id
        )
        await bus.publish("workspace.thought", bad_thought)
        
        # Wait up to 3 seconds for the negative pair to be registered by the learner
        for _ in range(30):
            if "Calculate fibonacci 5" in learner.pending_pairs:
                if learner.pending_pairs["Calculate fibonacci 5"]["rejected"]:
                    break
            await asyncio.sleep(0.1)
        
        # Check that Learner has the negative pair
        assert "Calculate fibonacci 5" in learner.pending_pairs
        assert learner.pending_pairs["Calculate fibonacci 5"]["rejected"] is not None
        assert learner.pending_pairs["Calculate fibonacci 5"]["chosen"] is None
        
        # 3. Second attempt succeeds
        good_thought = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="workspace.thought",
            payload={
                "type": "intuition",
                "confidence": 0.95,
                "content": "Let's do it safely: ```python\nprint(5)\n```"
            },
            correlation_id=corr_id
        )
        await bus.publish("workspace.thought", good_thought)
        
        # Wait up to 3 seconds for the positive pair to be stitched and removed
        for _ in range(30):
            if "Calculate fibonacci 5" not in learner.pending_pairs:
                break
            await asyncio.sleep(0.1)
        
        # Both pairs should now be stitched into perfect contrastive DPO batch
        assert "Calculate fibonacci 5" not in learner.pending_pairs
        assert len(learner.paired_buffer) == 1
        
        paired = learner.paired_buffer[0]
        assert paired[0] == "Calculate fibonacci 5"
        assert "print(5)" in paired[1] # Chosen
        assert "print(1/0)" in paired[2] # Rejected
        
    finally:
        await exec_node.stop()
        await workspace.stop()
        await learner.stop()
        await asyncio.sleep(0.1)
