"""Base Driver abstractions and protocols for HBLLM device and environment management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.hcir.spatial_planner import SpatialEntity


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


@dataclass
class DriverInput:
    """Standardized representation of raw input signals from a connected device/environment."""

    raw_data: Any
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    state_id: str = ""


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

    Standardizes the canonical cycle:
    1. Input: get_inputs() -> DriverInput
    2. Sensory Lifting: lift_to_hcir() -> tuple[list[SpatialEntity], set[tuple[int, int]]]
    3. Action Vocabulary: get_action_list() -> list[DriverAction]
    4. Output Dispatch: send_output() -> bool
    5. Feedback Observation: process_feedback() -> DriverFeedback
    """

    def __init__(self, name: str, capabilities: set[DriverCapability] | None = None) -> None:
        self.name: str = name
        self.capabilities: set[DriverCapability] = capabilities or set()
        self.is_connected: bool = False
        self._target: Any = None

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
    def send_output(self, action: DriverAction) -> bool:
        """Dispatch a concrete action command to the connected device/environment."""
        ...

    @abstractmethod
    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        """Normalize raw device output/signals into standardized DriverFeedback."""
        ...

    def lift_to_hcir(self, inputs: DriverInput) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
        """Lift raw sensory inputs into domain-neutral HCIR entities and impassable barriers.

        Override in perceptual/sensory drivers. Default implementation returns empty sets.
        """
        return [], set()
