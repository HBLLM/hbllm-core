import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe


class DummyPlugin(HBLLMPlugin):
    def __init__(self, node_id="test_plug"):
        super().__init__(node_id=node_id)
        self.received = False

    @subscribe("test.topic")
    async def handle_test_topic(self, message: Message) -> None:
        self.received = True


@pytest.mark.asyncio
async def test_plugin_auto_subscribe():
    bus = InProcessBus()
    await bus.start()

    plugin = DummyPlugin()
    await plugin.start(bus)

    # Send message to the bus on 'test.topic'
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test_harness",
        topic="test.topic",
        payload={"k": "v"},
    )

    await bus.publish("test.topic", msg)

    # Allow async queue to process
    import asyncio

    await asyncio.sleep(0.1)

    assert plugin.received is True, "Plugin did not automatically receive @subscribe message"
    await plugin.stop()
    await bus.stop()
