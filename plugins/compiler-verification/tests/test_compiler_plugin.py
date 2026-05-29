"""
Unit tests for CompilerVerification plugin.
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


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def compiler_plugin(bus):
    plugin = CompilerVerification(node_id="test_compiler")
    await plugin.start(bus)
    yield plugin
    await plugin.stop()


@pytest.mark.asyncio
class TestCompilerVerificationPlugin:
    async def test_verify_success(self, compiler_plugin, bus):
        # Override commands to map python to a command that always passes
        compiler_plugin.commands = {".py": "echo 'all tests passed'"}

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="workspace.compile.verify",
            payload={"filepath": "test.py"},
        )

        resp = await bus.request("workspace.compile.verify", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert resp.payload["exit_code"] == 0
        assert "all tests passed" in resp.payload["output"]

    async def test_verify_failure_publishes_thought(self, compiler_plugin, bus):
        # Override commands to map python to a command that always fails
        compiler_plugin.commands = {".py": "false"}

        # Track thoughts published on the bus
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

        resp = await bus.request("workspace.compile.verify", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "FAILURE"
        assert resp.payload["exit_code"] != 0

        # Wait a brief moment for the async publish to register
        await asyncio.sleep(0.05)

        # Confirm a failure thought was published to the workspace
        assert len(thoughts) == 1
        thought = thoughts[0]
        assert thought.topic == "workspace.thought"
        assert thought.correlation_id == "corr_1234"
        assert thought.payload["type"] == "simulation_result"
        assert thought.payload["prediction"] == "FAILURE"
        assert "Compiler/Test failed" in thought.payload["content"]
