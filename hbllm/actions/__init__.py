"""Actions — executable effector nodes (code execution, browser, API, tools)."""

from __future__ import annotations

from pathlib import Path

# Extend package search path to include experiments/legacy_actions
_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "legacy_actions"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))

from hbllm.actions.api_node import ApiNode
from hbllm.actions.browser_node import BrowserNode
from hbllm.actions.causal_planner import (
    CausalObstacleResolver,
    TopologicalFrontierNavigator,
)
from hbllm.actions.complexity import ComplexityDetector
from hbllm.actions.execution_node import ExecutionNode
from hbllm.actions.fuzzy_node import FuzzyNode
from hbllm.actions.logic_node import LogicNode
from hbllm.actions.scheduler_tools import (
    CancelTaskTool,
    ScheduleEventTool,
    ScheduleRecurringTool,
)
from hbllm.actions.tool_registry import ToolNode, ToolRegistry, ToolResult, tool
from hbllm.actions.tool_router import ToolRouterNode

__all__ = [
    "ApiNode",
    "BrowserNode",
    "CausalObstacleResolver",
    "ComplexityDetector",
    "ExecutionNode",
    "FuzzyNode",
    "LogicNode",
    "TopologicalFrontierNavigator",
    "ToolNode",
    "ToolRegistry",
    "ToolResult",
    "ToolRouterNode",
    "ScheduleEventTool",
    "ScheduleRecurringTool",
    "CancelTaskTool",
    "tool",
]
