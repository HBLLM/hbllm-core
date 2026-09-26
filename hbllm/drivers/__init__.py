"""HBLLM Unified Driver Management Layer.

Provides standardized interfaces for external devices, environments, and peripherals:
Input -> Processing (HCIR Cognition) -> Output -> Feedback

Supports both single-driver (legacy) and concurrent multi-driver (MIMO) modes.
"""

from __future__ import annotations

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
    DriverModality,
)
from hbllm.drivers.cognitive_blackbox import (
    AgentPhase,
    AgentState,
    CarryingState,
    CognitiveBlackbox,
)
from hbllm.drivers.manager import DriverManager

__all__ = [
    # --- Driver Abstractions ---
    "BaseDriver",
    "DriverAction",
    "DriverCapability",
    "DriverFeedback",
    "DriverInput",
    "DriverModality",
    # --- Driver Manager ---
    "DriverManager",
    # --- Cognitive Blackbox (MIMO) ---
    "CognitiveBlackbox",
    "AgentState",
    "AgentPhase",
    "CarryingState",
]
