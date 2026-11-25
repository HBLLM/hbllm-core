"""Sandboxed Terminal Plugin — permission-gated, allowlisted command execution."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)

# Restricted read-only diagnostic executable allowlist; python/pytest/git are excluded
DEFAULT_ALLOWED_EXECUTABLES = frozenset(
    {
        "ls",
        "pwd",
        "echo",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "rg",
    }
)
DEFAULT_TIMEOUT_SECONDS = 30.0
_FORBIDDEN_CONTROL_CHARS = frozenset({"\x00", "\n", "\r"})
_PATH_BOUND_EXECUTABLES = frozenset({"ls", "cat", "head", "tail", "wc"})


class TerminalExecutionNode:
    """
    OS Execution driver that safely executes allowlisted commands via subprocess
    exec (never shell=True), gated by PermissionEngine and confined to workspace_dir.
    """

    def __init__(
        self,
        node_id: str = "assistant_terminal",
        permission_engine: Any | None = None,
        allowed_executables: frozenset[str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        workspace_dir: str | Path | None = None,
    ) -> None:
        self.node_id = node_id
        self.bus: InProcessBus | None = None
        self.permission_engine = permission_engine
        self.allowed_executables = allowed_executables or DEFAULT_ALLOWED_EXECUTABLES
        self.timeout_seconds = timeout_seconds
        self.workspace_dir = Path(workspace_dir or Path.cwd()).resolve()

    async def run(self, bus: InProcessBus) -> None:
        """Register this driver onto the core MessageBus."""
        self.bus = bus
        logger.info("[%s] Initializing sandboxed Terminal OS driver...", self.node_id)
        await bus.subscribe("action.execute", self._handle_execution)

    def _validate_path_confinement(self, argv: list[str]) -> bool:
        """Ensure path arguments do not escape workspace_dir or reference home/system paths."""
        exe = argv[0]
        args = argv[1:]
        if exe == "echo" or exe == "pwd":
            return True
        for idx, arg in enumerate(args):
            if arg.startswith("-"):
                continue
            looks_like_path = (
                exe in _PATH_BOUND_EXECUTABLES
                or "/" in arg
                or ".." in arg
                or arg.startswith("~")
                or (exe in ("grep", "rg") and idx > 0)
            )
            if not looks_like_path:
                continue
            if arg.startswith("~"):
                return False
            resolved = (self.workspace_dir / arg).resolve()
            if not resolved.is_relative_to(self.workspace_dir):
                return False
        return True

    async def _handle_execution(self, message: Message) -> None:
        """Validate permissions and execute allowlisted commands via exec without shell."""
        if message.payload.get("target") != "terminal":
            return
        command = message.payload.get("command")
        if not command or not isinstance(command, str):
            return
        if any(ch in _FORBIDDEN_CONTROL_CHARS for ch in command):
            logger.warning("[%s] Rejected command with forbidden control chars", self.node_id)
            return

        # Fail closed unless PermissionEngine explicitly grants 'shell'
        if self.permission_engine is None:
            logger.warning(
                "[%s] Denied terminal command: no PermissionEngine attached", self.node_id
            )
            return
        granted = await self.permission_engine.check("terminal_node", "shell")
        if not granted:
            logger.warning("[%s] PermissionEngine denied 'shell' for terminal_node", self.node_id)
            return

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            logger.warning("[%s] Invalid command syntax: %s", self.node_id, exc)
            return
        if not argv or argv[0] not in self.allowed_executables:
            logger.warning("[%s] Blocked non-allowlisted executable: %s", self.node_id, argv[:1])
            return
        if not self._validate_path_confinement(argv):
            logger.warning("[%s] Blocked path escaping workspace: %s", self.node_id, argv)
            return

        clean_env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        logger.info("[%s] Executing allowlisted command: %s", self.node_id, argv[0])
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir),
                env=clean_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout_seconds
                )
            except (TimeoutError, asyncio.TimeoutError):
                process.kill()
                await process.wait()
                logger.error(
                    "[%s] Command timed out after %.1fs", self.node_id, self.timeout_seconds
                )
                return

            stdout = stdout_bytes.decode(errors="replace").strip()
            stderr = stderr_bytes.decode(errors="replace").strip()
            exit_code = process.returncode

            output_text = (
                f"Command: {argv[0]}\nExit Code: {exit_code}\nStdout:\n{stdout}\nStderr:\n{stderr}"
            )
            reply = Message(
                topic="sensory.input",
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                payload={"text": output_text, "exit_code": exit_code},
            )
            if self.bus:
                await self.bus.publish("sensory.input", reply)
        except Exception as e:
            logger.error("[%s] Failed to execute: %s", self.node_id, e)


__plugin__ = {
    "name": "terminal_node",
    "version": "0.3.0",
    "description": "Permission-gated terminal command execution (confined, no shell=True).",
}


async def register(
    bus: Any, registry: Any = None, permission_engine: Any | None = None
) -> TerminalExecutionNode:
    node = TerminalExecutionNode(node_id="desktop_terminal", permission_engine=permission_engine)
    await node.run(bus)
    return node
