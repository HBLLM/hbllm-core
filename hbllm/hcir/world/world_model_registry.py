"""World Model Registry — Tracks Predictor Models, Model Versions, & Validation Lifecycle.

Tracks registered predictor models and lifecycle states, lifted to declarative
CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class ModelLifecycleState(StrEnum):
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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export model descriptor as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"model:{self.model_id}",
            operator="==",
            value=self.status.value,
            relation=self.domain,
            target_entity=self.model_version,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering model descriptor."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"model_{self.model_id}",
                node_data={
                    "model_id": self.model_id,
                    "model_version": self.model_version,
                    "domain": self.domain,
                    "supported_horizons_ms": list(self.supported_horizons_ms),
                    "status": self.status.value,
                    "required_capabilities": list(self.required_capabilities),
                },
            )
        ]


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

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all registered models as declarative causal predicates."""
        return [m.to_causal_predicate() for m in self._models.values()]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting model descriptors."""
        ops: list[TransactionOperation] = []
        for m in self._models.values():
            ops.extend(m.to_transaction_operations())
        return ops

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
