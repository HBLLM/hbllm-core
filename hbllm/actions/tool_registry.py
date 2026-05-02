"""
Tool Registry — direct tool invocation pattern for HBLLM.

Provides:
  - ToolResult: standard result dataclass
  - ToolRegistry: register and invoke tools by name
  - @tool decorator: register functions as tools via decorator
  - ToolNode: bus-based Node wrapper for registered tools
  - create_tool_from_code: dynamic tool creation from code strings
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Result from a tool invocation."""

    tool: str
    success: bool
    output: str
    error: str = ""
    duration_ms: float = 0.0


# ── Tool Registry (Direct Invocation) ──────────────────────────────────────────


class ToolRegistry:
    """Registry of available tools for direct invocation (non-bus pattern).

    Supports runtime tool lifecycle management:
    - Register / unregister tools dynamically
    - Mark tools as temporarily unavailable (e.g., service down, rate-limited)
    - Filter tool lists by availability for LLM decision-making
    - Publish lifecycle events on the bus (if connected)
    """

    def __init__(self, bus: Any = None) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._availability: dict[str, bool] = {}
        self._unavailable_reasons: dict[str, str] = {}
        self._bus = bus  # Optional MessageBus for lifecycle events

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[ToolResult]],
        parameters: dict[str, str] | None = None,
    ) -> None:
        """Register a tool. If a tool with the same name exists, it is replaced."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters or {},
        }
        self._availability[name] = True
        self._unavailable_reasons.pop(name, None)
        logger.info("Tool registered: %s", name)
        self._fire_event("registered", name)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found and removed."""
        removed = self._tools.pop(name, None)
        self._availability.pop(name, None)
        self._unavailable_reasons.pop(name, None)
        if removed:
            logger.info("Tool unregistered: %s", name)
            self._fire_event("unregistered", name)
            return True
        return False

    # ── Availability ──────────────────────────────────────────────────

    def set_availability(self, name: str, available: bool, reason: str = "") -> None:
        """Mark a tool as available or unavailable without removing it.

        Use this for temporary outages (service down, rate-limited, etc.).
        The tool definition is preserved so it can be restored instantly.
        """
        if name not in self._tools:
            logger.warning("Cannot set availability for unknown tool: %s", name)
            return
        was_available = self._availability.get(name, True)
        self._availability[name] = available
        if available:
            self._unavailable_reasons.pop(name, None)
        else:
            self._unavailable_reasons[name] = reason
        # Only fire event on state change
        if was_available and not available:
            logger.warning("Tool marked unavailable: %s (%s)", name, reason)
            self._fire_event("unavailable", name, {"reason": reason})
        elif not was_available and available:
            logger.info("Tool restored: %s", name)
            self._fire_event("available", name)

    def get_availability(self, name: str) -> dict[str, Any]:
        """Check a tool's availability status."""
        if name not in self._tools:
            return {"registered": False, "available": False, "reason": "Tool not registered"}
        return {
            "registered": True,
            "available": self._availability.get(name, True),
            "reason": self._unavailable_reasons.get(name, ""),
        }

    def available_tools(self) -> list[str]:
        """Return names of all currently available (registered + online) tools."""
        return [name for name, info in self._tools.items() if self._availability.get(name, True)]

    # ── Invocation ────────────────────────────────────────────────────

    async def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """Invoke a tool by name. Checks availability before execution."""
        if name not in self._tools:
            return ToolResult(tool=name, success=False, output="", error=f"Unknown tool: {name}")
        if not self._availability.get(name, True):
            reason = self._unavailable_reasons.get(name, "temporarily offline")
            return ToolResult(
                tool=name,
                success=False,
                output="",
                error=f"Tool '{name}' is currently unavailable: {reason}",
            )
        start = time.monotonic()
        try:
            result = await self._tools[name]["handler"](**kwargs)
            result.duration_ms = (time.monotonic() - start) * 1000
            return result
        except Exception as e:
            return ToolResult(
                tool=name,
                success=False,
                output="",
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    # ── Listing ───────────────────────────────────────────────────────

    def list_tools(self, available_only: bool = False) -> list[dict[str, Any]]:
        """List registered tools with availability status.

        Args:
            available_only: If True, only return tools that are currently available.
                This should be used when building LLM prompts so the model
                never plans steps using offline tools.
        """
        result = []
        for t in self._tools.values():
            name = t["name"]
            available = self._availability.get(name, True)
            if available_only and not available:
                continue
            entry: dict[str, Any] = {
                "name": name,
                "description": t["description"],
                "parameters": t["parameters"],
                "available": available,
            }
            if not available:
                entry["unavailable_reason"] = self._unavailable_reasons.get(name, "")
            result.append(entry)
        return result

    # ── Bus Events ────────────────────────────────────────────────────

    def _fire_event(
        self, event_type: str, tool_name: str, extra: dict[str, Any] | None = None
    ) -> None:
        """Fire a tool lifecycle event on the bus (non-blocking, best-effort)."""
        if self._bus is None:
            return
        import asyncio

        payload: dict[str, Any] = {"tool_name": tool_name, "event": event_type}
        if extra:
            payload.update(extra)
        try:
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="tool_registry",
                topic=f"system.tool.{event_type}",
                payload=payload,
            )
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._bus.publish(f"system.tool.{event_type}", msg))
            else:
                loop.run_until_complete(self._bus.publish(f"system.tool.{event_type}", msg))
        except Exception as e:
            logger.debug("Failed to publish tool event: %s", e)


# ── @tool Decorator ────────────────────────────────────────────────────────────

# Registry of all tools explicitly collected by the @tool decorator
_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def tool(name: str = "", description: str = ""):
    """
    Decorator to register a python function as an HBLLM Tool.

    Extracts the docstring and type hints to create a tool schema,
    and stores the callable so the runtime can instantiate a node for it.
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or inspect.getdoc(func) or f"Execute {tool_name}"

        # Simplified schema extraction for agentic reasoning
        sig = inspect.signature(func)
        parameters = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = "string"
            if param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"

            parameters[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "name": tool_name,
            "description": tool_desc,
            "parameters": {"type": "object", "properties": parameters, "required": required},
        }

        _TOOL_REGISTRY[tool_name] = {
            "schema": schema,
            "callable": func,
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_tool_registry() -> dict[str, dict[str, Any]]:
    """Return the global decorator-based tool registry."""
    return _TOOL_REGISTRY


# ── ToolNode (Bus-Based Wrapper) ───────────────────────────────────────────────


class ToolNode(Node):
    """
    Wraps a python function registered via @tool and exposes it to the HBLLM bus.
    """

    def __init__(self, tool_name: str, func: Callable, tenant_id: str):
        super().__init__(node_id=f"tool_{tool_name}_{tenant_id}", node_type=NodeType.ACTION)
        self.tool_name = tool_name
        self.func = func
        self.topic = f"action.tool.{self.tool_name}"

    async def on_start(self) -> None:
        logger.info(f"Registering Tool: {self.tool_name} on topic {self.topic}")
        await self.bus.subscribe(self.topic, self.handle_message)

    async def on_stop(self) -> None:
        pass

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        arguments = message.payload.get("arguments", {})
        logger.info(f"Executing tool {self.tool_name} with {arguments}")

        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**arguments)
            else:
                result = self.func(**arguments)

            return message.create_response({"status": "SUCCESS", "output": result})
        except Exception as e:
            logger.error(f"Tool {self.tool_name} failed: {e}")
            return message.create_error(f"Tool execution failed: {str(e)}")


# ── Dynamic Tool Creation ──────────────────────────────────────────────────────


def create_tool_from_code(code_string: str, function_name: str) -> Callable:
    """
    Dynamically executes a code string and extracts a specific function by name.
    Used for registering induced skills at runtime.
    """
    # Create a clean namespace
    namespace = {}

    # Execute the code — SkillInductionNode already performed AST safety checks
    exec(code_string, {"__builtins__": {}}, namespace)

    func = namespace.get(function_name)
    if not func or not callable(func):
        raise ValueError(f"Function '{function_name}' not found in induced code.")

    return func
