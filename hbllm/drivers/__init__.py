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
    DriverStreamType,
    SynapticDeviceDescriptor,
)
from hbllm.drivers.cognitive_blackbox import (
    AgentPhase,
    AgentState,
    CarryingState,
    CognitiveBlackbox,
)
from hbllm.drivers.discovery import DeviceDiscoveryEngine
from hbllm.drivers.manager import DriverManager
from hbllm.drivers.network_driver import NetworkDeviceDriver
from hbllm.drivers.node import DriverManagerNode
from hbllm.drivers.serial_driver import SerialDeviceDriver

__all__ = [
    # --- Driver Abstractions ---
    "BaseDriver",
    "DriverAction",
    "DriverCapability",
    "DriverFeedback",
    "DriverInput",
    "DriverModality",
    "DriverStreamType",
    "SynapticDeviceDescriptor",
    # --- Drivers ---
    "SerialDeviceDriver",
    "NetworkDeviceDriver",
    # --- Driver Manager & Discovery ---
    "DriverManager",
    "DriverManagerNode",
    "DeviceDiscoveryEngine",
    # --- Cognitive Blackbox (MIMO) ---
    "CognitiveBlackbox",
    "AgentState",
    "AgentPhase",
    "CarryingState",
]
