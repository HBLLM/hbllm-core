"""
Tool Router Node — Generic multiplexer for agentic tool calls.

Listens for `action.tool_call` requests containing a generic XML-extracted
tool name and arguments. Routes the payload to the correct specialized
action node (e.g., McpClientNode, ExecutionNode) over the MessageBus,
and returns the resulting observation.
"""

import json
import logging

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class ToolRouterNode(Node):
    """
    Subscribes to `action.tool_call`.
    Inspects the `tool_name`, decides the internal bus topic (e.g., `mcp.file_tools.list_dir`
    or `action.execute_code`), forwards the request, and normalizes the response.
    """

    def __init__(self, node_id: str):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["tool_router"],
        )

    async def on_start(self) -> None:
        logger.info("Starting ToolRouterNode")
        await self.bus.subscribe("action.tool_call", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping ToolRouterNode")

    async def handle_message(self, message: Message) -> Message | None:
        """
        Route a generic tool call request to the specific capability.
        """
        if message.type != MessageType.QUERY:
            return None

        tool_name = message.payload.get("tool_name", "")
        tool_args_raw = message.payload.get("arguments", "{}")

        try:
            tool_args = (
                json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
            )
        except json.JSONDecodeError:
            logger.warning("ToolRouterNode received invalid JSON arguments for tool %s", tool_name)
            return message.create_response({"status": "ERROR", "error": "Invalid JSON arguments"})

        logger.info("ToolRouter processing call for tool: %s", tool_name)

        target_topic = ""
        target_payload = {}

        # 1. Native Execution Node Routing
        if tool_name == "execute_python":
            target_topic = "action.execute_code"
            # Some LLMs nest code in "code" param, some just write it as text. We handle standard JSON.
            target_payload = {"code": tool_args.get("code", "")}
        else:
            # 2. Generic Tool Routing (e.g. MCP)
            target_topic = f"action.tool.{tool_name}"
            target_payload = {"tool_name": tool_name, "arguments": tool_args}

        req = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic=target_topic,
            payload=target_payload,
        )

        try:
            # Execute tool across the bus
            resp = await self.request(target_topic, req, timeout=30.0)
            return message.create_response(resp.payload)
        except TimeoutError:
            return message.create_error(f"Tool execution for '{tool_name}' timed out.")
        except Exception as e:
            logger.error("Error executing tool %s: %s", tool_name, e)
            return message.create_error(f"Execution error: {str(e)}")
