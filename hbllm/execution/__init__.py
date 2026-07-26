"""
HBLLM Execution OS — Unified execution kernel.

The Execution OS sits between the Cognitive OS (HCIR) and execution
backends (LLMs, vision models, audio, tools, code, robotics).

Architecture:
    HCIR → ExecutionOrchestrator → ExecutionBus → RuntimeRegistry → Provider

The LLM is just another execution backend. The architecture is centered
on cognition and execution, not on any particular model technology.
"""

from hbllm.execution.events import ExecutionEvent, ExecutionEventData
from hbllm.execution.manifest import ExecutionManifest
from hbllm.execution.payload import (
    ExecutionPayload,
    PayloadAttachment,
    PayloadAudio,
    PayloadDocument,
    PayloadImage,
    PayloadMessage,
    PayloadTool,
)
from hbllm.execution.plan import (
    ExecutionConstraints,
    ExecutionPlan,
    ExecutionRequest,
    TaskType,
)
from hbllm.execution.result import (
    ExecutionMetrics,
    ExecutionResult,
    ProviderMetadata,
    TokenUsage,
)
from hbllm.execution.state import ExecutionState, ExecutionStatus

__all__ = [
    # Plan
    "TaskType",
    "ExecutionConstraints",
    "ExecutionRequest",
    "ExecutionPlan",
    # Payload
    "ExecutionPayload",
    "PayloadMessage",
    "PayloadImage",
    "PayloadAudio",
    "PayloadDocument",
    "PayloadTool",
    "PayloadAttachment",
    # Result
    "ExecutionResult",
    "TokenUsage",
    "ExecutionMetrics",
    "ProviderMetadata",
    # Events
    "ExecutionEvent",
    "ExecutionEventData",
    # State
    "ExecutionState",
    "ExecutionStatus",
    # Manifest
    "ExecutionManifest",
]
