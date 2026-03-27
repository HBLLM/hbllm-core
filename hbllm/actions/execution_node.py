"""
Sandboxed Execution Node.

Receives Python code, executes it in a secure subprocess (with timeout), 
and returns the deterministic output or traceback as a hard reward signal.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
import tempfile
from typing import Any

from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class ExecutionNode(Node):
    """Executes code securely to provide deterministic ground-truth verification."""

    def __init__(self, node_id: str, timeout: float = 3.0):
        # We will set node_type to CORE since ACTION doesn't exist
        super().__init__(node_id=node_id, node_type=NodeType.CORE)
        self.timeout = timeout

    async def on_start(self) -> None:
        logger.info("Starting ExecutionNode")
        await self.bus.subscribe("action.execute_code", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping ExecutionNode")

    async def handle_message(self, message: Message) -> Message | None:
        """Handle execution requests."""
        if message.type != MessageType.QUERY:
            return None
            
        code = message.payload.get("code", "")
        if not code:
            # Try to extract code blocks if just text is passed
            text = message.payload.get("text", "")
            match = re.search(r"```python\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                return message.create_error("No Python code provided for execution.")
                
        # Run code in an isolated subprocess
        result = await self._execute_python(code)
        return message.create_response(result)
        
    async def _execute_python(self, code: str) -> dict[str, Any]:
        """Write to temp file and run."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as temp_script:
            temp_script.write(code)
            temp_script.flush()
            
            try:
                # Run the script with asyncio.create_subprocess_exec
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, temp_script.name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return {
                        "status": "FAILURE",
                        "output": f"Execution timed out after {self.timeout} seconds.",
                        "error": "TimeoutError"
                    }
                
                output = stdout.decode().strip()
                error = stderr.decode().strip()
                await proc.wait()  # Ensure process transport closes cleanly
                
                if proc.returncode == 0:
                    return {
                        "status": "SUCCESS",
                        "output": output,
                        "error": error
                    }
                else:
                    return {
                        "status": "FAILURE",
                        "output": output,
                        "error": error
                    }
            except Exception as e:
                return {
                    "status": "FAILURE",
                    "output": "",
                    "error": str(e)
                }
