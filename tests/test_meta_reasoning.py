import asyncio
import os
import json
import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType, FeedbackPayload
from hbllm.brain.meta_node import MetaReasoningNode

@pytest.mark.asyncio
async def test_meta_reasoning_node_triggers_improvement(tmp_path):
    bus = InProcessBus()
    await bus.start()
    
    meta_node = MetaReasoningNode(node_id="meta_test")
    # Redirect reflection directory to pytest temp dir
    meta_node.reflection_dir = str(tmp_path / "reflection")
    os.makedirs(meta_node.reflection_dir, exist_ok=True)
    meta_node.weakness_threshold = 2 # lower for test
    
    await meta_node.start(bus)
    await asyncio.sleep(0.1) # settling time
    
    improve_events = []
    async def improve_handler(msg: Message) -> Message | None:
        improve_events.append(msg)
        return None
        
    await bus.subscribe("system.improve", improve_handler)
    
    # 1. Send negative feedback for coding
    msg1 = Message(
        type=MessageType.FEEDBACK,
        source_node_id="user",
        topic="system.feedback",
        payload=FeedbackPayload(
            message_id="msg1",
            rating=-1,
            prompt="Write a rust script",
            response="Sorry I cannot do that",
            module_id="coding"
        ).model_dump()
    )
    await bus.publish("system.feedback", msg1)
    await asyncio.sleep(0.1)
    
    # Should not trigger yet
    assert len(improve_events) == 0
    assert len(meta_node.negative_feedback_buffer["coding"]) == 1
    
    # 2. Send positive feedback (should be ignored)
    msg2 = Message(
        type=MessageType.FEEDBACK,
        source_node_id="user",
        topic="system.feedback",
        payload=FeedbackPayload(
            message_id="msg2",
            rating=1,
            prompt="Write a python script",
            response="print('hello')",
            module_id="coding"
        ).model_dump()
    )
    await bus.publish("system.feedback", msg2)
    await asyncio.sleep(0.1)
    
    # Buffer should still be 1 (positive feedback ignored by meta reasoner)
    assert len(improve_events) == 0
    assert len(meta_node.negative_feedback_buffer["coding"]) == 1

    # 3. Send second negative feedback (should trigger threshold = 2)
    msg3 = Message(
        type=MessageType.FEEDBACK,
        source_node_id="user",
        topic="system.feedback",
        payload=FeedbackPayload(
            message_id="msg3",
            rating=-1,
            prompt="Fix this python bug",
            response="I don't know",
            module_id="coding"
        ).model_dump()
    )
    await bus.publish("system.feedback", msg3)
    
    # Let the file IO and pub/sub complete
    await asyncio.sleep(0.2)
    
    assert len(improve_events) == 1
    payload = improve_events[0].payload
    
    assert payload["domain"] == "coding"
    assert "reasoning" in payload
    dataset_path = payload["dataset_path"]
    
    # Verify file was written properly
    assert os.path.exists(dataset_path)
    with open(dataset_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        sample1 = json.loads(lines[0])
        sample2 = json.loads(lines[1])
        assert sample1["domain"] == "coding"
        assert sample1["instruction"] == "Write a rust script"
        assert sample2["instruction"] == "Fix this python bug"
        
    # Verify buffer cleared
    assert len(meta_node.negative_feedback_buffer["coding"]) == 0
    
    await meta_node.stop()
    await bus.stop()
