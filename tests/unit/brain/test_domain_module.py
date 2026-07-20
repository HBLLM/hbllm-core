from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.modules.base_module import DomainModuleNode
from hbllm.network.messages import Message, MessageType


class MockLLM:
    def __init__(self):
        self.generate = AsyncMock(return_value="Direct test output")


@pytest.mark.asyncio
async def test_domain_module_direct_query():
    bus = MagicMock()
    bus.subscribe = AsyncMock(return_value=MagicMock())
    bus.publish = AsyncMock()

    mock_llm = MockLLM()
    node = DomainModuleNode(
        node_id="domain_general",
        domain_name="general",
        llm=mock_llm,
    )
    # 1. Test startup subscriptions
    await node.start(bus)
    assert bus.subscribe.call_count == 2
    bus.subscribe.assert_any_call("module.evaluate", node.handle_message)
    bus.subscribe.assert_any_call("domain.general.query", node.handle_message)

    # 2. Test handle_message with module.evaluate (should publish workspace.thought and return None)
    evaluate_msg = Message(
        type=MessageType.QUERY,
        source_node_id="workspace",
        topic="module.evaluate",
        payload={
            "text": "Hello world",
            "domain_hint": "general",
        },
        correlation_id="corr-1",
    )

    resp = await node.handle_message(evaluate_msg)
    assert resp is None
    bus.publish.assert_called_once()
    published_msg = bus.publish.call_args[0][1]
    assert published_msg.topic == "workspace.thought"
    assert published_msg.payload["content"] == "Direct test output"
    assert published_msg.payload["type"] == "intuition_general"

    # Reset mock
    bus.publish.reset_mock()

    # 3. Test handle_message with direct domain.general.query (should return RESPONSE Message directly)
    direct_msg = Message(
        type=MessageType.QUERY,
        source_node_id="pipeline",
        topic="domain.general.query",
        payload={
            "text": "Direct query hello",
        },
        correlation_id="corr-2",
    )

    resp = await node.handle_message(direct_msg)
    assert resp is not None
    assert resp.type == MessageType.RESPONSE
    assert resp.topic == "domain.general.query.response"
    assert resp.payload["text"] == "Direct test output"
    assert resp.correlation_id == direct_msg.id
    bus.publish.assert_not_called()
