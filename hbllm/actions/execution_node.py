"""
Execution Node for Tool Calling.

Provides a sandboxed execution environment with resource limits
to run dynamically generated code and return the stdout/stderr to the cognitive loop.

Sandbox features:
  - Memory limit (256 MB via ulimit on POSIX)
  - CPU time limit (10s)
  - Isolated Python (--isolated, no site-packages)
  - Restricted environment variables
  - Temporary directory isolation
  - Output truncation (max 10KB)
"""

import asyncio
import logging
import os
import platform
import subprocess
import sys
import tempfile

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

# Sandbox settings
MAX_MEMORY_MB = 256
MAX_CPU_SECONDS = 10
MAX_OUTPUT_BYTES = 10_240  # 10 KB output limit
TIMEOUT_SECONDS = 15.0


class ExecutionNode(Node):
    """
    Acts as the 'hands' of the Modular Brain, allowing it to verify its own logic
    by running synthesized Python scripts in a sandboxed environment.
    """
    def __init__(self, node_id: str, sandbox_enabled: bool = True):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN,
            capabilities=["code_execution", "tool_calling"],
        )
        self.topic_sub = "task.execute.python"
        self.sandbox_enabled = sandbox_enabled

    async def on_start(self) -> None:
        logger.info("Starting ExecutionNode (sandbox=%s)", self.sandbox_enabled)
        await self.bus.subscribe(self.topic_sub, self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping ExecutionNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        code = message.payload.get("code")
        if not code:
            return message.create_error("No 'code' provided in payload.")

        try:
            result = await asyncio.to_thread(self._run_code, code)
            return message.create_response({
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "truncated": result.get("truncated", False),
                "domain": "execution",
            })
        except Exception as e:
            logger.error("Execution error: %s", e)
            return message.create_error(f"Execution failure: {e}")

    def _run_code(self, code: str) -> dict[str, str | bool]:
        """Run code in a sandboxed subprocess with resource limits."""
        tmpdir = tempfile.mkdtemp(prefix="hbllm_exec_")
        script_path = os.path.join(tmpdir, "script.py")
        
        with open(script_path, "w") as f:
            f.write(code)
        
        try:
            # Build command with sandbox flags
            cmd = self._build_command(script_path)
            env = self._build_env(tmpdir)
            
            # On POSIX, use preexec_fn to set resource limits
            preexec = None
            if self.sandbox_enabled and platform.system() != "Windows":
                preexec = self._set_resource_limits
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env=env,
                cwd=tmpdir,
                preexec_fn=preexec,
            )
            
            stdout = result.stdout
            stderr = result.stderr
            truncated = False
            
            # Truncate output to prevent memory bombs
            if len(stdout) > MAX_OUTPUT_BYTES:
                stdout = stdout[:MAX_OUTPUT_BYTES] + "\n... [output truncated]"
                truncated = True
            if len(stderr) > MAX_OUTPUT_BYTES:
                stderr = stderr[:MAX_OUTPUT_BYTES] + "\n... [output truncated]"
                truncated = True
            
            return {"stdout": stdout, "stderr": stderr, "truncated": truncated}
            
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                "stderr": f"TimeoutExpired: Code took longer than {TIMEOUT_SECONDS}s to execute.",
                "truncated": False,
            }
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _build_command(self, script_path: str) -> list[str]:
        """Build the Python invocation command with isolation flags."""
        cmd = [sys.executable]
        if self.sandbox_enabled:
            # -I = isolated mode (no user site-packages, ignores PYTHON* env vars)
            # -S = don't import site module (no .pth processing)
            cmd.extend(["-I", "-S"])
        cmd.append(script_path)
        return cmd

    def _build_env(self, tmpdir: str) -> dict[str, str]:
        """Build a minimal, restricted environment for the subprocess."""
        if not self.sandbox_enabled:
            return dict(os.environ)
        
        # Minimal environment — only essentials
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
            "LANG": "en_US.UTF-8",
        }
        return env

    @staticmethod
    def _set_resource_limits() -> None:
        """Set POSIX resource limits for the child process."""
        try:
            import resource
            # Memory limit (256 MB)
            mem_bytes = MAX_MEMORY_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
            # Max file size (10 MB — prevent disk bombs)
            max_file = 10 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file, max_file))
            # No child processes
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ImportError, ValueError, OSError) as e:
            # resource module unavailable on some platforms
            logging.getLogger(__name__).debug("Could not set resource limits: %s", e)

