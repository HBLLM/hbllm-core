import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from hbllm_assistant.drivers.telegram_node import TelegramInterfaceNode

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.mark.asyncio
async def test_telegram_node_routing_to_bus():
    bus = InProcessBus()
    await bus.start()
    node = TelegramInterfaceNode(node_id="test_telegram")
    await node.run(bus)

    # We subscribe to router.query to verify that incoming telegram messages reach the brain
    captured = []

    async def capture(msg: Message):
        captured.append(msg)

    await bus.subscribe("router.query", capture)

    # Simulate an incoming message from the telegram chat
    await node.publish_incoming("Please run tests.", chat_id=123)

    # Yield for asyncio loop
    await asyncio.sleep(0.1)

    assert len(captured) == 1
    assert captured[0].payload["text"] == "Please run tests."
    assert captured[0].payload["chat_id"] == 123
    assert captured[0].source_node_id == "test_telegram"
    await bus.stop()


@pytest.mark.asyncio
async def test_telegram_node_routing_from_bus():
    bus = InProcessBus()
    await bus.start()
    node = TelegramInterfaceNode(node_id="test_telegram")

    # We mock out the actual telegram python-bot bot execution
    node.bot_app = MagicMock()
    node.bot_app.bot = AsyncMock()

    await node.run(bus)

    # Send a reply from the brain to the telegram node via sensory.output
    reply_msg = Message(
        topic="sensory.output",
        type=MessageType.EVENT,
        source_node_id="brain",
        payload={"target": "telegram", "text": "Tests passed!", "chat_id": 123},
    )

    await bus.publish("sensory.output", reply_msg)

    await asyncio.sleep(0.1)

    # Assert the bot.send_message was invoked with the exact text and chat_id
    node.bot_app.bot.send_message.assert_called_once_with(chat_id=123, text="Tests passed!")
    await bus.stop()
