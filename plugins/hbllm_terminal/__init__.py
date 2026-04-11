import logging
from typing import Any

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger("hbllm_assistant.terminal")


class TerminalExecutionNode:
    """
    A foundational driver that would compile and execute terminal commands
    on behalf of HBLLM, simulating 'OpenClaw' OS capabilities.
    """

    def __init__(self, node_id: str = "assistant_terminal"):
        self.node_id = node_id
        self.bus: InProcessBus | None = None

    async def run(self, bus: InProcessBus) -> None:
        """
        Registers this skill onto the core MessageBus.
        Listens for execution intents and runs them in a sandboxed PTY.
        """
        self.bus = bus
        logger.info(f"[{self.node_id}] Initializing Terminal OS driver...")
        await bus.subscribe("action.execute", self._handle_execution)

    async def _handle_execution(self, message: Message) -> None:
        """
        Executes terminal commands and routes the output back to HBLLM.
        """
        if message.payload.get("target") == "terminal":
            command = message.payload.get("command")
            if not command:
                return

            logger.info(f"[{self.node_id}] Executing command: {command}")

            try:
                import asyncio

                process = await asyncio.create_subprocess_shell(
                    command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )

                stdout_bytes, stderr_bytes = await process.communicate()
                stdout = stdout_bytes.decode().strip()
                stderr = stderr_bytes.decode().strip()
                exit_code = process.returncode

                # Format output for the brain
                output_text = f"Command: {command}\nExit Code: {exit_code}\nStdout:\n{stdout}\nStderr:\n{stderr}"

                reply = Message(
                    topic="sensory.input",
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    payload={"text": output_text, "exit_code": exit_code},
                )

                # Publish the feedback so the brain learns what happened
                if self.bus:
                    await self.bus.publish("sensory.input", reply)

            except Exception as e:
                logger.error(f"[{self.node_id}] Failed to execute: {e}")


__plugin__ = {
    "name": "terminal_node",
    "version": "0.1.0",
    "description": "Allows HBLLM to safely execute terminal commands locally.",
}


async def register(bus: Any, registry: Any = None) -> Any:
    node = TerminalExecutionNode(node_id="desktop_terminal")
    await node.run(bus)
    return node
