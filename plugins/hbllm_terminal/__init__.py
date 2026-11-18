"""Sandboxed Terminal Execution Plugin (PermissionEngine-gated, no shell=True)."""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger("hbllm_assistant.terminal")

# Strict default allowlist of read-only / safe diagnostic executables
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
        "git",
        "python",
        "python3",
        "pytest",
    }
)
DEFAULT_TIMEOUT_SECONDS = 30.0


class TerminalExecutionNode:
    """Permission-gated terminal driver using exec (never shell=True) with bounded timeout."""

    def __init__(
        self,
        node_id: str = "assistant_terminal",
        permission_engine: Any | None = None,
        allowed_executables: frozenset[str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.node_id = node_id
        self.bus: InProcessBus | None = None
        self.permission_engine = permission_engine
        self.allowed_executables = allowed_executables or DEFAULT_ALLOWED_EXECUTABLES
        self.timeout_seconds = timeout_seconds

    async def run(self, bus: InProcessBus) -> None:
        """Register this driver onto the core MessageBus."""
        self.bus = bus
        logger.info("[%s] Initializing sandboxed Terminal OS driver...", self.node_id)
        await bus.subscribe("action.execute", self._handle_execution)

    async def _handle_execution(self, message: Message) -> None:
        """Validate permissions and execute allowlisted commands via exec without shell."""
        if message.payload.get("target") != "terminal":
            return
        command = message.payload.get("command")
        if not command or not isinstance(command, str):
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

        logger.info("[%s] Executing allowlisted command: %s", self.node_id, argv[0])
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout_seconds
                )
            except TimeoutError:
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
    "version": "0.2.0",
    "description": "Permission-gated terminal command execution (no shell=True).",
}


async def register(
    bus: Any, registry: Any = None, permission_engine: Any | None = None
) -> TerminalExecutionNode:
    node = TerminalExecutionNode(node_id="desktop_terminal", permission_engine=permission_engine)
    await node.run(bus)
    return node
