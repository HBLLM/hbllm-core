"""
Unit tests for CompilerVerification plugin (hardened execution & PermissionEngine gating).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Add the compiler-verification plugin path to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler_verification import CompilerVerification

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


class _AllowAllPermissions:
    async def check(self, actor: str, capability: str) -> bool:
        return True


class _DenyAllPermissions:
    async def check(self, actor: str, capability: str) -> bool:
        return False


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def compiler_plugin(bus, tmp_path):
    (tmp_path / "test.py").write_text("print('ok')\n", encoding="utf-8")
    plugin = CompilerVerification(
        node_id="test_compiler",
        workspace_dir=str(tmp_path),
        permission_engine=_AllowAllPermissions(),
    )
    await plugin.start(bus)
    yield plugin
    await plugin.stop()


@pytest.mark.asyncio
class TestCompilerVerificationPlugin:
    async def test_verify_success(self, compiler_plugin, bus):
        compiler_plugin.commands = {".py": "echo passed {filepath}"}

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="workspace.compile.verify",
            payload={"filepath": "test.py"},
        )

        resp = await bus.request("workspace.compile.verify", query, timeout=10.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert resp.payload["exit_code"] == 0
        assert "passed test.py" in resp.payload["output"]

    async def test_verify_failure_publishes_thought(self, compiler_plugin, bus):
        compiler_plugin.commands = {".py": "false"}

        thoughts = []

        async def thought_handler(msg):
            thoughts.append(msg)

        await bus.subscribe("workspace.thought", thought_handler)

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="workspace.compile.verify",
            payload={"filepath": "test.py"},
            correlation_id="corr_1234",
        )

        resp = await bus.request("workspace.compile.verify", query, timeout=10.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "FAILURE"
        assert resp.payload["exit_code"] != 0

        await asyncio.sleep(0.05)

        assert len(thoughts) == 1
        thought = thoughts[0]
        assert thought.topic == "workspace.thought"
        assert thought.correlation_id == "corr_1234"
        assert thought.payload["type"] == "simulation_result"
        assert thought.payload["prediction"] == "FAILURE"
        assert "Compiler/Test failed" in thought.payload["content"]

    async def test_denies_without_permission_engine(self, bus, tmp_path):
        plugin = CompilerVerification(node_id="unauth_compiler", workspace_dir=str(tmp_path))
        await plugin.start(bus)
        try:
            query = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="workspace.compile.verify",
                payload={"filepath": "test.py"},
            )
            resp = await bus.request("workspace.compile.verify", query, timeout=2.0)
            assert resp.type == MessageType.ERROR
            assert "PermissionEngine" in resp.payload["error"]
        finally:
            await plugin.stop()

    async def test_denies_when_permission_engine_rejects(self, bus, tmp_path):
        plugin = CompilerVerification(
            node_id="denied_compiler",
            workspace_dir=str(tmp_path),
            permission_engine=_DenyAllPermissions(),
        )
        await plugin.start(bus)
        try:
            query = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="workspace.compile.verify",
                payload={"filepath": "test.py"},
            )
            resp = await bus.request("workspace.compile.verify", query, timeout=2.0)
            assert resp.type == MessageType.ERROR
            assert "denied" in resp.payload["error"].lower()
        finally:
            await plugin.stop()

    async def test_rejects_command_override(self, compiler_plugin, bus):
        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="workspace.compile.verify",
            payload={"filepath": "test.py", "command": "echo hacked"},
        )
        resp = await bus.request("workspace.compile.verify", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "command" in resp.payload["error"].lower()

    async def test_rejects_path_traversal_and_shell_metacharacters(self, compiler_plugin, bus):
        compiler_plugin.commands = {".py": "echo {filepath}"}
        for bad_path in ("../../etc/passwd", "/etc/passwd", "test.py; rm -rf /", "a$(id).py"):
            query = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="workspace.compile.verify",
                payload={"filepath": bad_path},
            )
            resp = await bus.request("workspace.compile.verify", query, timeout=2.0)
            assert resp.type == MessageType.ERROR, f"Expected error for {bad_path!r}"
