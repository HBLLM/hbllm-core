"""
Structured Action Schema for the Decision Layer.

Defines typed action objects that replace ad-hoc string/regex matching
in the DecisionNode routing logic. The ActionPlanner produces ActionPlan
instances; the DecisionNode dispatches based on ActionType.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    """All action types the DecisionNode can dispatch."""

    TEXT_RESPONSE = "text_response"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    AUDIO_OUTPUT = "audio_output"
    API_CALL = "api_call"
    IOT_COMMAND = "iot_command"
    MCP_TOOL = "mcp_tool"
    CLARIFY = "clarify"  # confidence too low — ask the user to clarify
    SHELL_EXECUTION = "shell_execution"


class RiskLevel(Enum):
    """Risk tiers that determine which safety checks are applied."""

    LOW = "low"  # PolicyEngine only (text, audio)
    MEDIUM = "medium"  # PolicyEngine + keyword blocklist (search, API)
    HIGH = "high"  # PolicyEngine + LLM safety classifier (code, IoT, MCP)


# Default risk mappings per action type
_ACTION_RISK: dict[ActionType, RiskLevel] = {
    ActionType.TEXT_RESPONSE: RiskLevel.LOW,
    ActionType.AUDIO_OUTPUT: RiskLevel.LOW,
    ActionType.CLARIFY: RiskLevel.LOW,
    ActionType.WEB_SEARCH: RiskLevel.MEDIUM,
    ActionType.API_CALL: RiskLevel.MEDIUM,
    ActionType.CODE_EXECUTION: RiskLevel.HIGH,
    ActionType.IOT_COMMAND: RiskLevel.HIGH,
    ActionType.MCP_TOOL: RiskLevel.HIGH,
    ActionType.SHELL_EXECUTION: RiskLevel.HIGH,
}


@dataclass
class ActionPlan:
    """A structured, typed description of what the DecisionNode should do."""

    action_type: ActionType
    content: str  # primary payload — text, extracted code, search query, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_level(self) -> RiskLevel:
        """Derive risk level from the action type."""
        return _ACTION_RISK.get(self.action_type, RiskLevel.HIGH)

    @property
    def requires_safety_llm(self) -> bool:
        """Whether this action warrants an LLM safety classification call."""
        return self.risk_level == RiskLevel.HIGH
