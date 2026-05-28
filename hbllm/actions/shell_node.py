"""
Host Shell Action Node.

Executes shell commands locally in the workspace directory.
Restricted by command blacklisting, standard regex policy rules,
and optional interactive console verification.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

# Basic safety blocklist for commands that must never be run
COMMAND_BLOCKLIST: list[str] = [
    r"\brm\s+-rf\s+(/|\*|\$|~)",     # Destructive deletions
    r"\bchmod\s+.*777\b",            # Unsafe file permissions
    r"\bcurl\s+.*\bsh\b",            # Piping network scripts directly to shell
    r"\bwget\s+.*\bsh\b",            # Piping network scripts directly to shell
    r"\bcat\s+.*\.env\b",            # Leaking secrets
    r"\bcat\s+/etc/passwd\b",        # Leaking system accounts
    r"\bdd\s+if=",                   # Low-level disk formatting
    r"\bmkfs\b",                     # Low-level disk formatting
    r"\breboot\b",                   # OS control
    r"\bshutdown\b",                 # OS control
]


class HostShellNode(Node):
    """
    Action node that allows executing shell commands on the local machine.
    """

    def __init__(
        self,
        node_id: str,
        workspace_dir: str | None = None,
        require_manual_approval: bool = True,
        policy_engine: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.ACTION,
            capabilities=["execute_shell"],
        )
        self.workspace_dir = os.path.abspath(workspace_dir or os.getcwd())
        self.require_manual_approval = require_manual_approval
        self.policy_engine = policy_engine
        self.timeout = timeout

    async def on_start(self) -> None:
        logger.info("Starting HostShellNode (manual approval=%s)", self.require_manual_approval)
        await self.bus.subscribe("action.execute_shell", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping HostShellNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        command = message.payload.get("command", "").strip()
        if not command:
            return message.create_error("No 'command' provided for execution.")

        # ── 1. Security Check: Blocklist Regexes ──
        for pattern in COMMAND_BLOCKLIST:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning("HostShellNode blocked command due to blocklist rule: %s", command)
                return message.create_error(
                    f"Command rejected: matches blocked pattern '{pattern}'."
                )

        # ── 2. Security Check: Policy Engine (if active) ──
        if self.policy_engine:
            res = self.policy_engine.evaluate(command, tenant_id=message.tenant_id or "default", domain="shell")
            if not res.passed:
                violations = "; ".join(res.violations)
                logger.warning("HostShellNode blocked command by Policy Engine: %s (%s)", command, violations)
                return message.create_error(f"Command rejected by governance policy: {violations}")

        # ── 3. Manual Interactive Approval (if configured) ──
        if self.require_manual_approval:
            approved = await self._get_interactive_approval(command)
            if not approved:
                logger.warning("HostShellNode command rejected by user: %s", command)
                return message.create_error("Command execution rejected by user.")

        # ── 4. Execution ──
        result = await self._run_command(command)
        return message.create_response(result)

    async def _get_interactive_approval(self, command: str) -> bool:
        """Prompt console user for permission to execute command."""
        print(f"\n⚠️  [HostShellNode] AGENT REQUESTS COMMAND EXECUTION:\n   $ {command}", flush=True)
        if not sys.stdin.isatty():
            print("❌ Stdin is not a TTY. Autorejecting in headless/non-interactive mode.", flush=True)
            return False

        try:
            loop = asyncio.get_running_loop()
            # Read input asynchronously using default executor to avoid blocking the event loop
            ans = await loop.run_in_executor(None, input, "Approve execution? [y/N]: ")
            return ans.strip().lower() in ("y", "yes")
        except Exception as e:
            logger.error("Failed to read user approval input: %s", e)
            return False

    async def _run_command(self, command: str) -> dict[str, Any]:
        """Run the command in a subprocess with timeout limits."""
        logger.info("Executing shell command: %s", command)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except (TimeoutError, asyncio.TimeoutError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.communicate()
                return {
                    "status": "TIMEOUT",
                    "output": f"Command timed out after {self.timeout} seconds.",
                    "error": "TimeoutError",
                    "exit_code": -1,
                }

            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode if proc.returncode is not None else -1

            if exit_code == 0:
                return {
                    "status": "SUCCESS",
                    "output": out_str,
                    "error": err_str,
                    "exit_code": exit_code,
                }
            else:
                return {
                    "status": "FAILURE",
                    "output": out_str,
                    "error": err_str,
                    "exit_code": exit_code,
                }

        except Exception as e:
            logger.exception("Error executing shell command")
            return {
                "status": "ERROR",
                "output": "",
                "error": str(e),
                "exit_code": -1,
            }
