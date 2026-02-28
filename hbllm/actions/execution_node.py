"""
Execution Node for Tool Calling.

Provides a sandboxed execution environment (currently relying on Python subprocess) 
to run dynamically generated code and return the stdout/stderr to the cognitive loop.
"""

import asyncio
import logging
import subprocess
import tempfile
import os

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class ExecutionNode(Node):
    """
    Acts as the 'hands' of the Modular Brain, allowing it to verify its own logic
    by running synthesized Python scripts and capturing the outputs.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN, capabilities=["code_execution", "tool_calling"])
        self.topic_sub = "task.execute.python"

    async def on_start(self) -> None:
        logger.info("Starting ExecutionNode")
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
            return message.create_response({"stdout": result["stdout"], "stderr": result["stderr"], "domain": "execution"})
        except Exception as e:
            logger.error("Execution error: %s", e)
            return message.create_error(f"Execution failure: {e}")

    def _run_code(self, code: str) -> dict[str, str]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
            
        try:
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=10.0
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout.decode() if e.stdout else "",
                "stderr": "TimeoutExpired: Code took longer than 10.0s to execute."
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
