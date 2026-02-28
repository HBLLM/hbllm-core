"""
Message types for inter-node communication.

All communication between nodes happens via strongly-typed messages.
This ensures consistency whether the bus is in-process or distributed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be sent between nodes."""

    # Queries & responses
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    EVENT = "event"

    # Routing
    ROUTE_REQUEST = "route_request"
    ROUTE_DECISION = "route_decision"

    # Task planning
    TASK_DECOMPOSE = "task_decompose"
    TASK_RESULT = "task_result"
    TASK_AGGREGATE = "task_aggregate"

    # Memory operations
    MEMORY_STORE = "memory_store"
    MEMORY_SEARCH = "memory_search"
    MEMORY_RESULT = "memory_result"

    # Health & lifecycle
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    NODE_REGISTERED = "node_registered"
    NODE_DEREGISTERED = "node_deregistered"

    # Learning
    FEEDBACK = "feedback"
    LEARNING_UPDATE = "learning_update"
    # Self-Expansion
    SPAWN_REQUEST = "spawn_request"
    SPAWN_COMPLETE = "spawn_complete"
    
    # AGI / Self-Improvement
    SYSTEM_IMPROVE = "system_improve"


class Priority(int, Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Message(BaseModel):
    """Base message for all inter-node communication."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    source_node_id: str
    target_node_id: str | None = None  # None = broadcast
    tenant_id: str = "default"         # Phase 9.5: Multi-tenant isolation
    session_id: str = "default"        # Phase 9.5: Session correlation
    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None  # Links request → response
    ttl_seconds: float | None = None  # Time-to-live

    def create_response(
        self,
        payload: dict[str, Any],
        msg_type: MessageType = MessageType.RESPONSE,
    ) -> Message:
        """Create a response message correlated to this message."""
        return Message(
            type=msg_type,
            source_node_id=self.target_node_id or "system",
            target_node_id=self.source_node_id,
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            topic=f"{self.topic}.response",
            payload=payload,
            correlation_id=self.id,
        )

    def create_error(self, error: str, code: str = "UNKNOWN") -> Message:
        """Create an error response."""
        return Message(
            type=MessageType.ERROR,
            source_node_id=self.target_node_id or "system",
            target_node_id=self.source_node_id,
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            topic=f"{self.topic}.error",
            payload={"error": error, "code": code},
            correlation_id=self.id,
        )


# ----- Domain-specific message payloads -----


class QueryPayload(BaseModel):
    """Payload for user queries routed to domain modules."""

    text: str
    context: list[dict[str, Any]] = Field(default_factory=list)  # Conversation history
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteDecisionPayload(BaseModel):
    """Payload for routing decisions from the cognitive router."""

    target_modules: list[str]
    confidence_scores: dict[str, float]
    detected_intent: str
    requires_planning: bool = False
    sub_tasks: list[dict[str, Any]] = Field(default_factory=list)


class MemorySearchPayload(BaseModel):
    """Payload for memory search requests."""

    query_text: str | None = None
    embedding: list[float] | None = None
    memory_type: str = "semantic"  # semantic, episodic, procedural
    top_k: int = 5
    domain_filter: str | None = None


class FeedbackPayload(BaseModel):
    """Payload for user feedback on responses."""

    message_id: str
    rating: int  # -1 (bad), 0 (neutral), 1 (good)
    prompt: str | None = None
    response: str | None = None
    comment: str | None = None
    module_id: str | None = None


class HeartbeatPayload(BaseModel):
    """Payload for health check heartbeats."""

    node_id: str
    status: str = "healthy"
    uptime_seconds: float = 0.0
    capabilities: list[str] = Field(default_factory=list)
    load: float = 0.0  # 0.0 - 1.0

class SpawnRequestPayload(BaseModel):
    """Payload to request the creation of a new domain module."""
    
    topic: str
    trigger_query: str
    confidence_score: float

class SystemImprovePayload(BaseModel):
    """Payload to trigger offline self-improvement on a weak domain."""
    
    domain: str
    reasoning: str
    dataset_path: str
