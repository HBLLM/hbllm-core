import asyncio

import pytest
from hbllm_assistant.drivers.terminal_node import TerminalExecutionNode

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.mark.asyncio
async def test_terminal_node_execution():
    bus = InProcessBus()
    await bus.start()
    node = TerminalExecutionNode(node_id="test_terminal")
    await node.run(bus)

    # We will subscribe to sensory.input to capture the feedback from the terminal
    captured_messages = []

    async def capture(msg: Message):
        captured_messages.append(msg)

    await bus.subscribe("sensory.input", capture)

    # Ask the terminal to run a bash command
    action_msg = Message(
        topic="action.execute",
        type=MessageType.EVENT,
        source_node_id="brain",
        payload={"target": "terminal", "command": "echo 'hello_world'"},
    )

    await bus.publish("action.execute", action_msg)

    # Allow some time for subprocess to run
    await asyncio.sleep(0.5)

    # Verify the terminal output successfully routed back to the brain
    assert len(captured_messages) == 1
    reply = captured_messages[0]

    assert reply.source_node_id == "test_terminal"
    assert reply.payload["exit_code"] == 0
    assert "hello_world" in reply.payload["text"]
    await bus.stop()
