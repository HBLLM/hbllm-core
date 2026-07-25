"""
Compiler Verification Plugin — runs compilers/tests and feeds results to workspace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

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


class CompilerVerification(HBLLMPlugin):
    """
    Plugin that compiles code or runs test suites dynamically based on file extensions.
    """

    def __init__(
        self,
        node_id: str = "compiler_verification",
        workspace_dir: str | None = None,
        config_path: str | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["workspace_compile_verify"],
        )
        self.workspace_dir = os.path.abspath(workspace_dir or os.getcwd())
        self.commands = dict(DEFAULT_COMMANDS)

        # Load custom command mappings if config_path exists
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                    if isinstance(custom_config, dict):
                        self.commands.update(custom_config)
                        logger.info(
                            "Loaded custom compiler mappings: %s", list(custom_config.keys())
                        )
            except Exception as e:
                logger.error("Failed to load compiler config from %s: %s", config_path, e)

    @subscribe("workspace.compile.verify")
    async def on_verify(self, message: Message) -> Message:
        """Run compilation/test checks for a file."""
        filepath = message.payload.get("filepath", "")
        cmd_override = message.payload.get("command", "")

        # 1. Resolve command
        cmd = ""
        if cmd_override:
            cmd = cmd_override
        elif filepath:
            ext = os.path.splitext(filepath)[1].lower()
            cmd_template = self.commands.get(ext)
            if cmd_template:
                cmd = cmd_template.format(filepath=filepath)
            else:
                return message.create_error(f"No compiler command registered for extension '{ext}'")
        else:
            return message.create_error(
                "Missing both 'filepath' and 'command' in verification request."
            )

        # 2. Run shell command
        logger.info("CompilerVerification running command: %s", cmd)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            exit_code = proc.returncode if proc.returncode is not None else -1
            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()

            status = "SUCCESS" if exit_code == 0 else "FAILURE"
            output_msg = out_str or err_str

            # 3. Publish DPO simulation thought if validation failed
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
                        "content": f"Compiler/Test failed under command: '{cmd}'. Output:\n{output_msg}",
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
