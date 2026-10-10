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

    @classmethod
    def create_default(cls) -> WorldModelRegistry:
        """Create a registry pre-populated with default core world predictors."""
        reg = cls()
        reg.register_model(
            WorldModelDescriptor(
                model_id="whole_grid_v1",
                model_version="1.0.0",
                domain="discrete_2d_grid",
                supported_horizons_ms=[1000, 60000],
                status=ModelLifecycleState.ACTIVE,
                required_capabilities=["W083_whole_grid", "raster_rollout"],
            )
        )
        reg.register_model(
            WorldModelDescriptor(
                model_id="physics_v1",
                model_version="1.0.0",
                domain="continuous_physics",
                supported_horizons_ms=[1000, 60000],
                status=ModelLifecycleState.ACTIVE,
                required_capabilities=["kinematics", "collision"],
            )
        )
        return reg
