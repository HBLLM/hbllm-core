"""
World Model Registry — Tracks Predictor Models, Model Versions, & Validation Lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelLifecycleState(str, Enum):
    """Lifecycle status of a world model."""

    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RETIRED = "retired"


@dataclass
class WorldModelDescriptor:
    """Descriptor defining a registered predictor model."""

    model_id: str
    model_version: str
    domain: str
    supported_horizons_ms: list[int]
    status: ModelLifecycleState = ModelLifecycleState.ACTIVE
    required_capabilities: list[str] = field(default_factory=list)


class WorldModelRegistry:
    """Registry tracking registered world models and model versions."""

    def __init__(self) -> None:
        self._models: dict[str, WorldModelDescriptor] = {}

    def register_model(self, descriptor: WorldModelDescriptor) -> None:
        """Register or update a world model descriptor."""
        self._models[descriptor.model_id] = descriptor
        logger.info(
            "WorldModelRegistry registered model '%s' version '%s' [%s]",
            descriptor.model_id,
            descriptor.model_version,
            descriptor.status.value,
        )

    def get_model(self, model_id: str) -> WorldModelDescriptor | None:
        """Retrieve model descriptor by ID."""
        return self._models.get(model_id)

    def list_active_models_for_domain(self, domain: str) -> list[WorldModelDescriptor]:
        """Retrieve active models matching a given domain."""
        return [
            m
            for m in self._models.values()
            if m.domain == domain and m.status == ModelLifecycleState.ACTIVE
        ]
