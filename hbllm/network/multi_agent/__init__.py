"""Multi-Agent package init."""

from hbllm.network.multi_agent.coordinator import MultiAgentCoordinator
from hbllm.network.multi_agent.protocol import (
    AgentCapability,
    AgentIdentity,
    AgentMessage,
    AgentMessageType,
    DelegationTask,
)

__all__ = [
    "AgentCapability",
    "AgentIdentity",
    "AgentMessage",
    "AgentMessageType",
    "DelegationTask",
    "MultiAgentCoordinator",
]
