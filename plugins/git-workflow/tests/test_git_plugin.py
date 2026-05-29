"""
Unit tests for GitWorkflow plugin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Add the git-workflow plugin path to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from git_workflow import GitWorkflow

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def git_plugin(bus):
    plugin = GitWorkflow(node_id="test_git")
    await plugin.start(bus)
    yield plugin
    await plugin.stop()


@pytest.mark.asyncio
class TestGitWorkflowPlugin:
    async def test_git_status(self, git_plugin, bus):
        async def mock_run(args):
            return (0, "On branch main\nnothing to commit, working tree clean", "")

        git_plugin._run_git = mock_run

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="git.status",
        )
        resp = await bus.request("git.status", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert "working tree clean" in resp.payload["output"]

    async def test_git_diff(self, git_plugin, bus):
        async def mock_run(args):
            return (0, "diff --git a/a.py b/a.py\n+new line", "")

        git_plugin._run_git = mock_run

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="git.diff",
        )
        resp = await bus.request("git.diff", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert "+new line" in resp.payload["output"]

    async def test_git_commit(self, git_plugin, bus):
        called_args = []

        async def mock_run(args):
            called_args.append(args)
            return (0, "[main a1b2c3d] test commit", "")

        git_plugin._run_git = mock_run

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="git.commit",
            payload={"message": "feat: test"},
        )
        resp = await bus.request("git.commit", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert len(called_args) == 2
        assert called_args[0] == ["add", "-A"]
        assert called_args[1] == ["commit", "-m", "feat: test"]

    async def test_git_branch(self, git_plugin, bus):
        called_args = []

        async def mock_run(args):
            called_args.append(args)
            return (0, "Switched to a new branch 'feature-x'", "")

        git_plugin._run_git = mock_run

        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="git.branch",
            payload={"branch_name": "feature-x", "create": True},
        )
        resp = await bus.request("git.branch", query, timeout=2.0)
        assert resp.type == MessageType.RESPONSE
        assert resp.payload["status"] == "SUCCESS"
        assert called_args[0] == ["checkout", "-b", "feature-x"]
