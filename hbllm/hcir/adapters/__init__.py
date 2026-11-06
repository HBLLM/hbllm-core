"""HCIR Integration Adapters — bridges between existing HBLLM subsystems and HCIR."""

from __future__ import annotations

from hbllm.hcir.adapters.actuator_adapter import ActuatorCapabilityAdapter, ActuatorCommand
from hbllm.hcir.adapters.cognitive_state_adapter import CognitiveStateAdapter
from hbllm.hcir.adapters.hcir_bus_bridge import HCIRBusBridge
from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend, MigrationPhase
from hbllm.hcir.adapters.memory_adapter import MemoryAdapter
from hbllm.hcir.adapters.memory_consistency import (
    MemoryConsistencyChecker,
    MemoryFallbackEvent,
    MemoryIdentityBridge,
    MemoryMigrationReceipt,
)
from hbllm.hcir.adapters.memory_migration_proxy import MemoryMigrationProxy
from hbllm.hcir.adapters.node_adapter import NodeAdapter
from hbllm.hcir.adapters.sensor_adapter import SensorCapabilityAdapter, SensorReading
from hbllm.hcir.adapters.snn_adapter import (
    SNNCapabilityExecutor,
    SNNPopulationConfig,
    SpikeTrain,
)
from hbllm.hcir.adapters.telemetry_adapter import (
    TelemetryStreamAdapter,
    TelemetryWindowSummary,
)

__all__ = [
    # Actuation & Sensing
    "ActuatorCapabilityAdapter",
    "ActuatorCommand",
    "SensorCapabilityAdapter",
    "SensorReading",
    "TelemetryStreamAdapter",
    "TelemetryWindowSummary",
    # Bus & Nodes
    "HCIRBusBridge",
    "NodeAdapter",
    "CognitiveStateAdapter",
    # Memory Systems
    "HCIRMemoryBackend",
    "MigrationPhase",
    "MemoryAdapter",
    "MemoryConsistencyChecker",
    "MemoryIdentityBridge",
    "MemoryMigrationReceipt",
    "MemoryFallbackEvent",
    "MemoryMigrationProxy",
    # Neuromorphic SNN
    "SNNCapabilityExecutor",
    "SNNPopulationConfig",
    "SpikeTrain",
]
