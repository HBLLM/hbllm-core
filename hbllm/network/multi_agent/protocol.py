"""Multi-Agent Protocol — inter-agent communication specification.

Defines the wire format for communication between HBLLM instances
(or compatible agents) in a multi-agent network.

Protocol Messages:
    DISCOVER     — "I exist, here are my capabilities"
    DELEGATE     — "Please handle this task"
    RESULT       — "Here's the result of the delegated task"
    NEGOTIATE    — "Can you do X? What would it cost?"
    HEARTBEAT    — "I'm still alive"
    CAPABILITY   — "My capabilities have changed"
    CONSENSUS    — "I propose we agree on X"

Transport-agnostic — works over MessageBus, HTTP, MQTT, or WebSocket.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentMessageType(str, Enum):
    """Types of inter-agent messages."""

    DISCOVER = "discover"
    DELEGATE = "delegate"
    RESULT = "result"
    NEGOTIATE = "negotiate"
    HEARTBEAT = "heartbeat"
    CAPABILITY = "capability"
    CONSENSUS = "consensus"
    ERROR = "error"


class AgentCapability(str, Enum):
    """Advertised agent capabilities."""

    REASONING = "reasoning"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    IOT_CONTROL = "iot_control"
    VISION = "vision"
    SPEECH = "speech"
    MEMORY = "memory"
    PLANNING = "planning"
    TRANSLATION = "translation"
    MATH = "math"
    CREATIVE = "creative"
    DATA_ANALYSIS = "data_analysis"


@dataclass
class AgentIdentity:
    """An agent's identity and capabilities."""

    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "HBLLM Agent"
    version: str = "1.0.0"
    capabilities: list[str] = field(default_factory=list)
    load: float = 0.0  # 0.0-1.0, current workload
    max_concurrent: int = 5
    endpoint: str = ""  # How to reach this agent
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "load": self.load,
            "max_concurrent": self.max_concurrent,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        }


@dataclass
class AgentMessage:
    """A message in the multi-agent protocol."""

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    message_type: AgentMessageType = AgentMessageType.HEARTBEAT
    sender_id: str = ""
    recipient_id: str = ""  # Empty = broadcast
    correlation_id: str = ""  # Links request → response
    timestamp: float = field(default_factory=time.time)
    ttl_s: float = 300.0  # Time-to-live
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0=normal, 1=high, 2=critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "ttl_s": self.ttl_s,
            "payload": self.payload,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        return cls(
            message_id=data.get("message_id", uuid.uuid4().hex),
            message_type=AgentMessageType(data.get("message_type", "heartbeat")),
            sender_id=data.get("sender_id", ""),
            recipient_id=data.get("recipient_id", ""),
            correlation_id=data.get("correlation_id", ""),
            timestamp=data.get("timestamp", time.time()),
            ttl_s=data.get("ttl_s", 300.0),
            payload=data.get("payload", {}),
            priority=data.get("priority", 0),
        )

    def create_reply(
        self,
        message_type: AgentMessageType,
        payload: dict[str, Any],
    ) -> AgentMessage:
        """Create a reply to this message."""
        return AgentMessage(
            message_type=message_type,
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            correlation_id=self.message_id,
            payload=payload,
        )

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_s


@dataclass
class DelegationTask:
    """A task delegated to another agent."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    input_data: dict[str, Any] = field(default_factory=dict)
    max_duration_s: float = 300.0
    priority: int = 0
    delegated_to: str = ""
    status: str = "pending"  # pending, accepted, running, completed, failed
    result: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "required_capabilities": self.required_capabilities,
            "input_data": self.input_data,
            "max_duration_s": self.max_duration_s,
            "priority": self.priority,
            "status": self.status,
        }
