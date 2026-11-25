"""
Compiler Verification Plugin — Deterministic Execution Verification

Runs external compilers, linters, or test suites (e.g., pytest, cargo check,
tsc, mypy) on workspace files without using shell=True, enforcing PermissionEngine
authorization, workspace path confinement, and subprocess kill-on-timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger(__name__)

# Default file extension to compile/test commands
DEFAULT_COMMANDS: dict[str, str] = {
    ".py": "venv/bin/pytest {filepath} -q --tb=short",
    ".rs": "cargo test",
    ".js": "npm test",
    ".ts": "npm test",
    ".go": "go test ./...",
    ".c": "make",
    ".cpp": "make",
}

_FORBIDDEN_SHELL_CHARS = frozenset(
    {";", "&", "|", "$", "`", "<", ">", "(", ")", "\n", "\r", "\x00"}
)


class CompilerVerification(HBLLMPlugin):
    """Subscribes to ``workspace.compile.verify`` requests and executes local compilers/tests."""

    def __init__(
        self,
        node_id: str = "compiler_verification",
        workspace_dir: str | None = None,
        config_path: str | None = None,
        permission_engine: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["workspace_compile_verify"],
        )
        self.workspace_dir = str(Path(workspace_dir or os.getcwd()).resolve())
        self.commands = DEFAULT_COMMANDS.copy()
        self.permission_engine = permission_engine
        self.timeout_seconds = timeout_seconds

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    custom_cmds = json.load(f)
                    self.commands.update(custom_cmds)
            except Exception as e:
                logger.warning("Failed to load compiler commands config: %s", e)

    async def on_start(self) -> None:
        await super().on_start()
        logger.info("CompilerVerification plugin started (workspace=%s)", self.workspace_dir)

    async def on_stop(self) -> None:
        await super().on_stop()
        logger.info("CompilerVerification plugin stopped")

    def _confine_filepath(self, raw_filepath: str) -> tuple[str | None, str | None]:
        """Validate *raw_filepath* has no shell metacharacters and stays within workspace_dir."""
        if not raw_filepath or not isinstance(raw_filepath, str):
            return None, "Invalid 'filepath' in verification request."
        if any(ch in _FORBIDDEN_SHELL_CHARS for ch in raw_filepath):
            return None, "Forbidden characters in 'filepath'."
        ws_root = Path(self.workspace_dir).resolve()
        candidate = (ws_root / raw_filepath).resolve()
        if not candidate.is_relative_to(ws_root):
            return None, f"Path '{raw_filepath}' escapes workspace directory."
        return str(candidate.relative_to(ws_root)), None

    @subscribe("workspace.compile.verify")
    async def on_verify(self, message: Message) -> Message:
        """Run compilation/test checks for a confined workspace file."""
        # 0. Fail closed unless PermissionEngine authorizes subprocess execution
        if self.permission_engine is None:
            return message.create_error(
                "CompilerVerification denied: no PermissionEngine attached."
            )
        if not await self.permission_engine.check(self.node_id, "subprocess"):
            return message.create_error(
                f"CompilerVerification denied: permission 'subprocess' not granted for '{self.node_id}'."
            )

        if message.payload.get("command"):
            return message.create_error(
                "Arbitrary 'command' overrides are disabled for security; configure extension commands instead."
            )

        filepath = message.payload.get("filepath", "")
        if not filepath:
            return message.create_error("Missing 'filepath' in verification request.")

        safe_relpath, path_err = self._confine_filepath(str(filepath))
        if path_err or safe_relpath is None:
            return message.create_error(path_err or "Invalid filepath.")

        ext = os.path.splitext(safe_relpath)[1].lower()
        cmd_template = self.commands.get(ext)
        if not cmd_template:
            return message.create_error(f"No compiler command registered for extension '{ext}'")
        if any(ch in _FORBIDDEN_SHELL_CHARS for ch in cmd_template):
            return message.create_error(
                "Configured command template contains forbidden characters."
            )

        try:
            tokens = shlex.split(cmd_template)
        except ValueError as exc:
            return message.create_error(f"Invalid command template: {exc}")
        if not tokens:
            return message.create_error("Empty compiler command template.")

        argv = [
            safe_relpath if tok == "{filepath}" else tok.replace("{filepath}", safe_relpath)
            for tok in tokens
        ]

        logger.info("CompilerVerification running argv: %s", argv)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except (TimeoutError, asyncio.TimeoutError):
                proc.kill()
                await proc.wait()
                return message.create_error(
                    f"Verification timed out after {self.timeout_seconds:.1f}s"
                )

            exit_code = proc.returncode if proc.returncode is not None else -1
            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()

            status = "SUCCESS" if exit_code == 0 else "FAILURE"
            output_msg = out_str or err_str

            if status == "FAILURE":
                logger.warning("Verification failed. Publishing failure thought to blackboard.")
                thought_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=message.session_id,
                    topic="workspace.thought",
                    payload={
                        "type": "simulation_result",
                        "confidence": 1.0,
                        "prediction": "FAILURE",
                        "content": f"Compiler/Test failed under command: '{' '.join(argv)}'. Output:\n{output_msg}",
                    },
                    correlation_id=message.correlation_id or message.id,
                )
                await self.bus.publish("workspace.thought", thought_msg)

            return message.create_response(
                {
                    "status": status,
                    "exit_code": exit_code,
                    "output": out_str,
                    "error": err_str,
                }
            )

        except Exception as e:
            logger.exception("Verification process error")
            return message.create_error(f"Verification process failed: {e}")
