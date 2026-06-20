import asyncio

import pytest

from hbllm.brain.sentinel_node import SentinelNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


class MockPolicyEngine:
    def evaluate(self, text, tenant_id, context):
        class Result:
            def __init__(self, passed):
                self.passed = passed
                self.violations = ["blocked mock"] if not passed else []
                self.applied_policies = []
                self.warnings = []

        if "bad" in text:
            return Result(False)
        return Result(True)


@pytest.mark.asyncio
async def test_proactive_policy():
    bus = InProcessBus()
    await bus.start()

    sentinel = SentinelNode(node_id="sentinel", policy_engine=MockPolicyEngine())
    await sentinel.start(bus)

    events = []
    errors = []

    async def handle_event(msg):
        events.append(msg)

    async def handle_error(msg):
        errors.append(msg)

    await bus.subscribe("test.query", handle_event)
    await bus.subscribe("test.query.error", handle_error)

    good_msg = Message(
        type=MessageType.QUERY, source_node_id="test", topic="test.query", payload={"text": "good"}
    )
    bad_msg = Message(
        type=MessageType.QUERY, source_node_id="test", topic="test.query", payload={"text": "bad"}
    )

    await bus.publish("test.query", good_msg)
    await asyncio.sleep(0.1)

    assert len(events) == 1
    assert events[0].is_security_cleared is True

    await bus.publish("test.query", bad_msg)
    await asyncio.sleep(0.1)

    assert len(events) == 1  # unchanged because message was blocked
    assert len(errors) > 0  # At least one error message generated
    assert errors[0].type == MessageType.ERROR

    await sentinel.stop()
    await bus.stop()
