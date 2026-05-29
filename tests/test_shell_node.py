"""
Unit tests for HostShellNode.
"""

from __future__ import annotations

import asyncio

import pytest

from hbllm.actions.shell_node import HostShellNode
from hbllm.brain.policy_engine import Policy, PolicyAction, PolicyEngine, PolicyType
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
async def bus():
    bus = InProcessBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
class TestHostShellNode:
    async def test_successful_execution(self, bus):
        node = HostShellNode(
            node_id="test_shell",
            require_manual_approval=False,
        )
        await node.start(bus)

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="action.execute_shell",
            payload={"command": "echo 'hello world'"},
        )

        resp = await bus.request("action.execute_shell", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert resp.payload["output"] == "hello world"
        assert resp.payload["exit_code"] == 0

        await node.stop()

    async def test_failure_execution(self, bus):
        node = HostShellNode(
            node_id="test_shell",
            require_manual_approval=False,
        )
        await node.start(bus)

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="action.execute_shell",
            payload={"command": "false"},
        )

        resp = await bus.request("action.execute_shell", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "FAILURE"
        assert resp.payload["exit_code"] != 0

        await node.stop()

    async def test_blocklist_security(self, bus):
        node = HostShellNode(
            node_id="test_shell",
            require_manual_approval=False,
        )
        await node.start(bus)

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="action.execute_shell",
            payload={"command": "rm -rf /"},
        )

        resp = await bus.request("action.execute_shell", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "rejected" in resp.payload["error"]

        await node.stop()

    async def test_policy_engine_blocking(self, bus):
        # Setup policy engine to block commands containing "secret"
        pe = PolicyEngine()
        policy = Policy(
            name="no_secrets",
            type=PolicyType.DENY,
            action=PolicyAction.BLOCK,
            pattern="secret",
            description="Do not access secrets",
        )
        pe.add_policy(policy)

        node = HostShellNode(
            node_id="test_shell",
            require_manual_approval=False,
            policy_engine=pe,
        )
        await node.start(bus)

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="action.execute_shell",
            payload={"command": "echo secret_value"},
        )

        resp = await bus.request("action.execute_shell", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "governance policy" in resp.payload["error"]

        await node.stop()

    async def test_manual_approval_rejection(self, bus):
        node = HostShellNode(
            node_id="test_shell",
            require_manual_approval=True,
        )

        # Mock approval prompt to immediately reject
        async def mock_approval(cmd):
            return False

        node._get_interactive_approval = mock_approval
        await node.start(bus)

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="action.execute_shell",
            payload={"command": "echo 'hello'"},
        )

        resp = await bus.request("action.execute_shell", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "rejected by user" in resp.payload["error"]

        await node.stop()
