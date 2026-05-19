"""
Transport Layer — Dumb pipes for HBLLM network communication.

This package provides the Transport protocol and concrete implementations.
Transports ONLY handle sending/receiving messages and reporting raw metrics.
They do NOT understand routing, capabilities, or cognition.
"""

from hbllm.network.transports.base import Transport, TransportMetrics, TransportState

__all__ = [
    "Transport",
    "TransportMetrics",
    "TransportState",
    "InProcessTransport",
    "WebSocketTransport",
    "RedisTransport",
    "WebRTCTransport",
]


def __getattr__(name: str):  # noqa: N807
    """Lazy imports for concrete transports to avoid heavy deps at import time."""
    if name == "InProcessTransport":
        from hbllm.network.transports.inprocess import InProcessTransport

        return InProcessTransport
    if name == "WebSocketTransport":
        from hbllm.network.transports.websocket import WebSocketTransport

        return WebSocketTransport
    if name == "RedisTransport":
        from hbllm.network.transports.redis import RedisTransport

        return RedisTransport
    if name == "WebRTCTransport":
        from hbllm.network.transports.webrtc import WebRTCTransport

        return WebRTCTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
