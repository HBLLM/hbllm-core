"""
Transport Protocol — The abstract contract for all communication primitives.

A Transport is a "dumb pipe": it sends bytes, receives bytes, and reports
metrics. It does NOT understand routing, capabilities, or cognition.

Concrete implementations: InProcessTransport, WebSocketTransport,
RedisTransport, WebRTCTransport.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from hbllm.network.messages import Message

logger = logging.getLogger(__name__)

# Type alias for message handlers at the transport level
TransportHandler = Callable[[Message], Coroutine[Any, Any, Message | None]]


class TransportState(StrEnum):
    """Lifecycle state of a transport."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"  # Connected but experiencing issues
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"


class TransportMetrics(BaseModel):
    """Raw metrics reported by a transport. Used by the RIL for scoring."""

    transport_id: str = ""
    transport_type: str = ""  # "inprocess", "websocket", "redis", "webrtc"
    state: TransportState = TransportState.DISCONNECTED

    # Latency tracking (milliseconds)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    _latency_samples: list[float] = []

    # Throughput
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    # Reliability
    send_errors: int = 0
    reconnections: int = 0
    last_send_at: float = 0.0
    last_receive_at: float = 0.0
    uptime_start: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample and recompute averages."""
        samples = self._latency_samples
        samples.append(latency_ms)
        # Keep a rolling window of 100 samples
        if len(samples) > 100:
            samples.pop(0)
        self.avg_latency_ms = sum(samples) / len(samples)
        self.min_latency_ms = min(samples)
        self.max_latency_ms = max(samples)

    def record_send(self, byte_count: int = 0) -> None:
        self.messages_sent += 1
        self.bytes_sent += byte_count
        self.last_send_at = time.monotonic()

    def record_receive(self, byte_count: int = 0) -> None:
        self.messages_received += 1
        self.bytes_received += byte_count
        self.last_receive_at = time.monotonic()

    def record_drop(self) -> None:
        self.messages_dropped += 1

    def record_error(self) -> None:
        self.send_errors += 1

    @property
    def uptime_seconds(self) -> float:
        if self.uptime_start == 0.0:
            return 0.0
        return time.monotonic() - self.uptime_start

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction of total sends."""
        total = self.messages_sent + self.send_errors
        if total == 0:
            return 0.0
        return self.send_errors / total


class Transport(ABC):
    """
    Abstract base class for all HBLLM transport implementations.

    A transport is a "dumb pipe". It:
      - Sends messages to a destination.
      - Receives messages and dispatches to a handler.
      - Reports raw metrics (latency, errors, throughput).

    A transport MUST NOT:
      - Decide routing (that's the RIL's job).
      - Understand capabilities (that's the Discovery layer).
      - Interpret cognition (that's the Cognition layer).
    """

    def __init__(self, transport_id: str, transport_type: str) -> None:
        self.transport_id = transport_id
        self.transport_type = transport_type
        self.metrics = TransportMetrics(
            transport_id=transport_id,
            transport_type=transport_type,
        )
        self._state = TransportState.DISCONNECTED
        self._message_handler: TransportHandler | None = None

    @property
    def state(self) -> TransportState:
        return self._state

    def set_handler(self, handler: TransportHandler) -> None:
        """Set the callback invoked when this transport receives a message."""
        self._message_handler = handler

    @abstractmethod
    async def start(self) -> None:
        """Start the transport and establish connections."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport and clean up resources."""
        ...

    @abstractmethod
    async def send(self, topic: str, message: Message) -> None:
        """Send a message to the given topic via this transport."""
        ...

    @abstractmethod
    async def send_request(
        self, topic: str, message: Message, timeout: float = 30.0
    ) -> Message:
        """Send a request and wait for a correlated response."""
        ...

    @abstractmethod
    def has_subscribers(self, topic: str) -> bool:
        """Check if this transport has local subscribers for a topic."""
        ...

    def get_metrics(self) -> TransportMetrics:
        """Return the current metrics snapshot."""
        self.metrics.state = self._state
        return self.metrics

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.transport_id} state={self._state}>"
