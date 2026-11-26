"""Actions — executable effector nodes (code execution, browser, API, tools)."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ApiNode": ("hbllm.actions.api_node", "ApiNode"),
    "BrowserNode": ("hbllm.actions.browser_node", "BrowserNode"),
    "CausalObstacleResolver": ("hbllm.actions.causal_planner", "CausalObstacleResolver"),
    "TopologicalFrontierNavigator": (
        "hbllm.actions.causal_planner",
        "TopologicalFrontierNavigator",
    ),
    "ComplexityDetector": ("hbllm.actions.complexity", "ComplexityDetector"),
    "ExecutionNode": ("hbllm.actions.execution_node", "ExecutionNode"),
    "FuzzyNode": ("hbllm.actions.fuzzy_node", "FuzzyNode"),
    "LogicNode": ("hbllm.actions.logic_node", "LogicNode"),
    "CancelTaskTool": ("hbllm.actions.scheduler_tools", "CancelTaskTool"),
    "ScheduleEventTool": ("hbllm.actions.scheduler_tools", "ScheduleEventTool"),
    "ScheduleRecurringTool": ("hbllm.actions.scheduler_tools", "ScheduleRecurringTool"),
    "ToolNode": ("hbllm.actions.tool_registry", "ToolNode"),
    "ToolRegistry": ("hbllm.actions.tool_registry", "ToolRegistry"),
    "ToolResult": ("hbllm.actions.tool_registry", "ToolResult"),
    "tool": ("hbllm.actions.tool_registry", "tool"),
    "ToolRouterNode": ("hbllm.actions.tool_router", "ToolRouterNode"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
