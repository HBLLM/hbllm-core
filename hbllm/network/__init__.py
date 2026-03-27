"""Network layer — distributed-ready communication for HBLLM nodes."""

from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.cognition_router import CognitionRouter

__all__ = ["InProcessBus", "MessageBus", "CognitionRouter"]
