"""
Git Workflow Plugin — exposes git actions to the MessageBus.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from hbllm.network.messages import Message
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger(__name__)


class GitWorkflow(HBLLMPlugin):
    """
    Plugin that exposes Git operations to the HBLLM MessageBus.
    """

    def __init__(self, node_id: str = "git_workflow", workspace_dir: str | None = None) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["git_status", "git_diff", "git_commit", "git_branch", "git_push"],
        )
        self.workspace_dir = os.path.abspath(workspace_dir or os.getcwd())

    async def _run_git(self, args: list[str]) -> tuple[int, str, str]:
        """Helper to run a git command in the workspace directory."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            return (
                proc.returncode if proc.returncode is not None else -1,
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            )
        except Exception as e:
            logger.exception("Failed to execute git command: %s", args)
            return (-1, "", str(e))

    @subscribe("git.status")
    async def on_git_status(self, message: Message) -> Message:
        """Get git status of the workspace."""
        code, out, err = await self._run_git(["status"])
        if code == 0:
            return message.create_response({"status": "SUCCESS", "output": out})
        return message.create_error(f"Git status failed: {err}")

    @subscribe("git.diff")
    async def on_git_diff(self, message: Message) -> Message:
        """Get git diff of the workspace."""
        code, out, err = await self._run_git(["diff"])
        if code == 0:
            return message.create_response({"status": "SUCCESS", "output": out})
        return message.create_error(f"Git diff failed: {err}")

    @subscribe("git.commit")
    async def on_git_commit(self, message: Message) -> Message:
        """Stage all changes and commit them with a message."""
        commit_msg = message.payload.get("message")
        if not commit_msg:
            return message.create_error("Missing 'message' in git.commit payload")

        # 1. Stage all changes
        code_add, _, err_add = await self._run_git(["add", "-A"])
        if code_add != 0:
            return message.create_error(f"Git add failed: {err_add}")

        # 2. Commit changes
        code_commit, out_commit, err_commit = await self._run_git(["commit", "-m", commit_msg])
        if (
            code_commit == 0
            or "nothing to commit" in out_commit
            or "no changes added to commit" in out_commit
        ):
            return message.create_response(
                {"status": "SUCCESS", "output": out_commit or "Nothing to commit"}
            )
        return message.create_error(f"Git commit failed: {err_commit}")

    @subscribe("git.branch")
    async def on_git_branch(self, message: Message) -> Message:
        """Create or switch to a branch."""
        branch_name = message.payload.get("branch_name")
        create = message.payload.get("create", True)

        if not branch_name:
            return message.create_error("Missing 'branch_name' in git.branch payload")

        args = ["checkout", "-b" if create else "", branch_name]
        args = [arg for arg in args if arg]

        code, out, err = await self._run_git(args)
        if code == 0 or "Already on" in err or "Switched to" in err:
            return message.create_response({"status": "SUCCESS", "output": out or err})
        return message.create_error(f"Git branch switch failed: {err}")

    @subscribe("git.push")
    async def on_git_push(self, message: Message) -> Message:
        """Push a branch to remote origin."""
        branch_name = message.payload.get("branch_name")
        if not branch_name:
            return message.create_error("Missing 'branch_name' in git.push payload")

        code, out, err = await self._run_git(["push", "origin", branch_name])
        if code == 0:
            return message.create_response({"status": "SUCCESS", "output": out or err})
        return message.create_error(f"Git push failed: {err}")
