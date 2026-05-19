"""Network layer — distributed-ready communication for HBLLM nodes."""

from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.cognition_router import CognitionRouter
from hbllm.network.discovery import CapabilityRegistry
from hbllm.network.node_state import NodeRole, NodeStateEngine
from hbllm.network.routing import ExecutionContext, RoutingIntelligenceLayer
from hbllm.network.transports import Transport, TransportMetrics, TransportState

__all__ = [
    "InProcessBus",
    "MessageBus",
    "CognitionRouter",
    "Transport",
    "TransportMetrics",
    "TransportState",
    "ExecutionContext",
    "RoutingIntelligenceLayer",
    "NodeStateEngine",
    "NodeRole",
    "CapabilityRegistry",
]
