"""Tests for the hardened TerminalExecutionNode plugin."""

from __future__ import annotations

import asyncio

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from plugins.hbllm_terminal import TerminalExecutionNode


class _AllowShellPermissions:
    async def check(self, actor: str, capability: str) -> bool:
        return capability == "shell"


class _DenyAllPermissions:
    async def check(self, actor: str, capability: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_terminal_node_execution(tmp_path):
    bus = InProcessBus()
    await bus.start()
    node = TerminalExecutionNode(
        node_id="test_terminal",
        permission_engine=_AllowShellPermissions(),
        workspace_dir=tmp_path,
    )
    await node.run(bus)

    captured_messages: list[Message] = []

    async def capture(msg: Message) -> None:
        captured_messages.append(msg)

    await bus.subscribe("sensory.input", capture)

    action_msg = Message(
        topic="action.execute",
        type=MessageType.EVENT,
        source_node_id="brain",
        payload={"target": "terminal", "command": "echo hello_world"},
    )

    await bus.publish("action.execute", action_msg)
    for _ in range(40):
        if captured_messages:
            break
        await asyncio.sleep(0.05)

    assert len(captured_messages) == 1
    reply = captured_messages[0]
    assert reply.source_node_id == "test_terminal"
    assert reply.payload["exit_code"] == 0
    assert "hello_world" in reply.payload["text"]
    await bus.stop()


@pytest.mark.asyncio
async def test_terminal_node_fails_closed_without_permissions_and_blocks_interpreters(tmp_path):
    bus = InProcessBus()
    await bus.start()
    captured: list[Message] = []

    async def capture(msg: Message) -> None:
        captured.append(msg)

    await bus.subscribe("sensory.input", capture)

    # 1. No PermissionEngine -> denied
    no_perm_node = TerminalExecutionNode(node_id="no_perm", workspace_dir=tmp_path)
    await no_perm_node.run(bus)
    await bus.publish(
        "action.execute",
        Message(
            topic="action.execute",
            type=MessageType.EVENT,
            source_node_id="brain",
            payload={"target": "terminal", "command": "echo should_not_run"},
        ),
    )
    await asyncio.sleep(0.15)
    assert captured == []

    # 2. Disallowed interpreter (python3, git, bash) and outside-workspace path -> blocked
    allow_node = TerminalExecutionNode(
        node_id="allow_perm",
        permission_engine=_AllowShellPermissions(),
        workspace_dir=tmp_path,
    )
    await allow_node.run(bus)
    for blocked_cmd in (
        "python3 -c 'print(1)'",
        "python -V",
        "pytest --version",
        "git status",
        "cat /etc/passwd",
        "ls ../",
    ):
        await bus.publish(
            "action.execute",
            Message(
                topic="action.execute",
                type=MessageType.EVENT,
                source_node_id="brain",
                payload={"target": "terminal", "command": blocked_cmd},
            ),
        )
    await asyncio.sleep(0.2)
    assert captured == []
    await bus.stop()
