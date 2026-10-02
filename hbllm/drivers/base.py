"""Base Driver abstractions and protocols for HBLLM device and environment management.

Drivers are thin I/O adapters that connect external environments to the HBLLM core.
They provide raw observations and accept action commands — nothing more.

All cognition, learning, planning, and interpretation happens inside the core
(CognitiveBlackbox). Drivers have ZERO knowledge of HCIR internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DriverStreamType(StrEnum):
    """Execution and observation streaming paradigm of the driver."""

    POLL = "poll"  # Discrete step-based polling (e.g. ARC, turn-based games)
    INTERRUPT_EVENT = (
        "interrupt_event"  # Asynchronous interrupt events (e.g. door opened, key pressed)
    )
    CONTINUOUS_STREAM = (
        "continuous_stream"  # Real-time continuous stream (e.g. camera, audio PCM, IMU)
    )


class DriverCapability(StrEnum):
    """Capabilities supported by connected drivers."""

    DISCRETE_ACTIONS = "discrete_actions"
    CONTINUOUS_ACTIONS = "continuous_actions"
    SPATIAL_2D = "spatial_2d"
    SPATIAL_3D = "spatial_3d"
    STREAMING_OBSERVATIONS = "streaming_observations"
    STEP_BASED_EXECUTION = "step_based_execution"
    ASYNC_EVENT_DRIVEN = "async_event_driven"
    STATE_RESTORATION = "state_restoration"


class DriverModality(StrEnum):
    """Sensory modality of the driver's observations."""

    GRID_2D = "grid_2d"  # ARC-AGI, Sokoban, grid worlds
    IMAGE = "image"  # Camera, screenshots
    AUDIO = "audio"  # Microphone, voice
    TEXT = "text"  # Terminal, chat
    PROPRIOCEPTION = "proprioception"  # Robot joint states
    STRUCTURED = "structured"  # JSON/API responses
    EVENT_STREAM = "event_stream"  # Real-time discrete/continuous events


@dataclass
class SynapticDeviceDescriptor:
    """Standardized 'Cognitive USB' descriptor for plug-and-play peripherals.

    Exposes peripheral capabilities, modalities, sample rates, and action schemas
    so the HBLLM Brain can dynamically mount and route inputs/outputs without restarting.
    """

    device_id: str
    vendor_id: str = ""
    product_id: str = ""
    version: str = "1.0.0"
    device_type: str = "bidirectional"  # "afferent" | "efferent" | "bidirectional"
    modalities: list[DriverModality] = field(default_factory=list)
    stream_type: DriverStreamType = DriverStreamType.POLL
    sample_rate_hz: float = 0.0  # 0.0 for event/poll, >0 for continuous stream
    salience_base_weight: float = 0.5  # Base priority in AttentionSystem (0.0 to 1.0)
    action_schema: list[dict[str, Any]] = field(
        default_factory=list
    )  # Declared actions & parameter specs
    is_reversible: bool = True  # Whether actions can be safely undone via rollback
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "version": self.version,
            "device_type": self.device_type,
            "modalities": [
                m.value if isinstance(m, DriverModality) else str(m) for m in self.modalities
            ],
            "stream_type": (
                self.stream_type.value
                if isinstance(self.stream_type, DriverStreamType)
                else str(self.stream_type)
            ),
            "sample_rate_hz": self.sample_rate_hz,
            "salience_base_weight": self.salience_base_weight,
            "action_schema": self.action_schema,
            "is_reversible": self.is_reversible,
            "metadata": self.metadata,
        }


@dataclass
class DriverInput:
    """Standardized representation of raw input signals from a connected device/environment."""

    raw_data: Any
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    state_id: str = ""
    source_id: str = ""  # Which driver produced this observation
    modality: DriverModality = DriverModality.STRUCTURED  # Sensory modality


@dataclass
class DriverAction:
    """Standardized representation of an executable action for a connected device."""

    action_id: Any
    semantic_intent: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class DriverFeedback:
    """Standardized feedback from the connected device after an action is dispatched."""

    success: bool
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    causal_delta: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None


class BaseDriver(ABC):
    """Abstract Base Driver interface for all external devices, environments, and runtimes.

    Drivers are pure I/O adapters. They know HOW to talk to an environment
    but have ZERO knowledge of HCIR internals (no SpatialEntity, no EntityRole,
    no workspace state). All interpretation and learning happens inside core.

    Canonical cycle (orchestrated by DriverManager):
    1. Input: get_inputs() -> DriverInput (raw observations)
    2. Perception: get_perception_data() -> dict (raw structured perception for blackbox)
    3. Action Vocabulary: get_action_list() -> list[DriverAction]
    4. Output Dispatch: send_output() -> Any
    5. Feedback Observation: process_feedback() -> DriverFeedback
    """

    def __init__(
        self,
        name: str,
        capabilities: set[DriverCapability] | None = None,
        descriptor: SynapticDeviceDescriptor | None = None,
    ) -> None:
        self.name: str = name
        self.capabilities: set[DriverCapability] = capabilities or set()
        self.is_connected: bool = False
        self._target: Any = None
        if descriptor is not None:
            self.descriptor = descriptor
        else:
            stream_type = (
                DriverStreamType.CONTINUOUS_STREAM
                if DriverCapability.STREAMING_OBSERVATIONS in self.capabilities
                else (
                    DriverStreamType.INTERRUPT_EVENT
                    if DriverCapability.ASYNC_EVENT_DRIVEN in self.capabilities
                    else DriverStreamType.POLL
                )
            )
            self.descriptor = SynapticDeviceDescriptor(
                device_id=self.name,
                stream_type=stream_type,
            )

    @abstractmethod
    def connect(self, target: Any) -> bool:
        """Establish a connection to the target device or environment."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection to the target device or environment."""
        ...

    @abstractmethod
    def get_inputs(self) -> DriverInput:
        """Retrieve current raw observation signals from the connected device/environment."""
        ...

    @abstractmethod
    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        """Query currently available action vocabulary / capabilities from the device."""
        ...

    @abstractmethod
    def send_output(self, action: DriverAction) -> Any:
        """Dispatch a concrete action command to the connected device/environment."""
        ...

    @abstractmethod
    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        """Normalize raw device output/signals into standardized DriverFeedback."""
        ...

    async def connect_async(self, target: Any) -> bool:
        """Establish connection asynchronously (defaults to synchronous connect)."""
        return self.connect(target)

    async def disconnect_async(self) -> None:
        """Tear down connection asynchronously (defaults to synchronous disconnect)."""
        self.disconnect()

    async def get_inputs_async(self) -> DriverInput:
        """Retrieve inputs asynchronously (defaults to synchronous get_inputs)."""
        return self.get_inputs()

    async def handle_action(self, action: DriverAction) -> DriverFeedback:
        """Execute an action asynchronously and return standardized feedback."""
        raw_result = self.send_output(action)
        return self.process_feedback(raw_result)

    async def stream_inputs(self) -> Any:
        """Yield streaming observations if supported.

        By default, yields the current input if connected.
        Drivers with continuous streams or event-driven sources can override this.
        """
        if self.is_connected:
            yield self.get_inputs()

    def get_perception_data(self, inputs: DriverInput) -> dict[str, Any]:
        """Return raw structured perception data for the core to interpret.

        This replaces lift_to_hcir(). Drivers return raw structured data
        (e.g., segmented objects, grid topology) and the core blackbox
        does all interpretation, entity classification, and learning.

        Override in domain-specific drivers. Default returns empty dict.
        """
        return {}

    def resolve_action(
        self,
        intent: Any,
        available_actions: list[DriverAction],
        context: dict[str, Any] | None = None,
    ) -> DriverAction | None:
        """Dynamically resolve an action matching a requested cognitive intent or plan step.

        Drivers can override this method to provide domain-specific action resolution.
        The default implementation matches requested intent against DriverAction.semantic_intent.
        """
        if not available_actions:
            return None
        target_intent = str(intent).strip().lower()
        for a in available_actions:
            if a.semantic_intent and a.semantic_intent.strip().lower() == target_intent:
                return a
        for a in available_actions:
            if a.semantic_intent:
                norm = a.semantic_intent.strip().lower()
                if target_intent in norm or norm in target_intent:
                    return a
        return None
