import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from plugins.hbllm_emotion import EmotionNode


@pytest.mark.asyncio
async def test_emotion_node_happy():
    bus = InProcessBus()
    await bus.start()
    node = EmotionNode(node_id="test_emotion")
    await node.start(bus)

    published_messages = []

    async def capture(msg: Message) -> None:
        published_messages.append(msg)

    await bus.subscribe("identity.emotion", capture)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="user",
        topic="sensory.input",
        payload={"text": "I am so happy and awesome today!"},
    )
    await bus.publish("sensory.input", msg)

    import asyncio

    await asyncio.sleep(0.1)

    assert len(published_messages) == 1
    emotion_msg = published_messages[0]
    assert emotion_msg.topic == "identity.emotion"
    assert emotion_msg.payload["emotion_label"] == "happy"
    assert emotion_msg.payload["valence"] > 0

    await bus.stop()


@pytest.mark.asyncio
async def test_emotion_node_sad():
    bus = InProcessBus()
    await bus.start()
    node = EmotionNode(node_id="test_emotion")
    await node.start(bus)

    published_messages = []

    async def capture(msg: Message) -> None:
        published_messages.append(msg)

    await bus.subscribe("identity.emotion", capture)

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="user",
        topic="sensory.input",
        payload={"text": "This is terrible and bad"},
    )
    await bus.publish("sensory.input", msg)

    import asyncio

    await asyncio.sleep(0.1)

    assert len(published_messages) == 1
    emotion_msg = published_messages[0]
    assert emotion_msg.payload["emotion_label"] == "sad"
    assert emotion_msg.payload["valence"] < 0

    await bus.stop()
