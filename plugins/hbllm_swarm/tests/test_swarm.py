import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from plugins.hbllm_swarm import SwarmManagerNode


@pytest.mark.asyncio
async def test_swarm_manager_node():
    bus = InProcessBus()
    await bus.start()
    node = SwarmManagerNode(node_id="test_swarm")
    await node.start(bus)

    published_messages = []

    async def capture(msg: Message) -> None:
        published_messages.append(msg)

    # We expect the swarm to publish back to task_aggregate
    await bus.subscribe("task_aggregate", capture)

    # Dispatch a swarm task
    sub_tasks = [{"name": "research_topic_a"}, {"name": "research_topic_b"}]

    msg = Message(
        type=MessageType.TASK_DECOMPOSE,
        source_node_id="planner",
        topic="task_decompose",
        payload={"sub_tasks": sub_tasks},
    )

    await bus.publish("task_decompose", msg)

    import asyncio

    # Wait for the mock sub-agents to complete (they sleep for 1.0s)
    await asyncio.sleep(1.2)

    assert len(published_messages) == 1
    reply_msg = published_messages[0]

    results = reply_msg.payload.get("results", [])
    assert len(results) == 2

    # Assert that both tasks completed
    task_names = [res["task"] for res in results]
    assert "research_topic_a" in task_names
    assert "research_topic_b" in task_names

    await bus.stop()
