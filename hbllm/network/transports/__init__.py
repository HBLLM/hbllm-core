"""
Transport Layer — Dumb pipes for HBLLM network communication.

This package provides the Transport protocol and concrete implementations.
Transports ONLY handle sending/receiving messages and reporting raw metrics.
They do NOT understand routing, capabilities, or cognition.
"""

from hbllm.network.transports.base import Transport, TransportMetrics, TransportState

__all__ = ["Transport", "TransportMetrics", "TransportState"]
