"""
Digital Twin Registry — Empirical Physical Telemetry & Environment State Container.

Enforces strict cognitive separation: DigitalTwinRegistry ONLY records empirical physical telemetry.
Cognitive inferences and predictions cannot directly mutate DigitalTwinRegistry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """Live telemetry state of physical environment."""

    name: str = "default_env"
    status: str = "nominal"
    last_updated: float = field(default_factory=time.time)


@dataclass
class PhysicalEntity:
    """Live state of physical device or asset."""

    entity_id: str
    entity_name: str
    status: str = "active"
    telemetry: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class DigitalTwinRegistry:
    """Registry managing empirical physical state digital twins.

    Usage::

        twin = DigitalTwinRegistry(world_id="factory_a")
        twin.sync_sensor_telemetry("sensor_temp", 42.5)
        snapshot = twin.create_snapshot()
    """

    def __init__(self, world_id: str = "default_world") -> None:
        self.world_id = world_id
        self._environment = EnvironmentState()
        self._variables: dict[str, Any] = {}
        self._entities: dict[str, PhysicalEntity] = {}

    def sync_sensor_telemetry(
        self,
        variable_name: str,
        value: Any,
        source: str = "sensor",
    ) -> None:
        """Sync empirical sensor observation into DigitalTwinRegistry.

        Only empirical hardware or verified sensor telemetry may invoke this method.
        """
        self._variables[variable_name] = value
        self._environment.last_updated = time.time()
        logger.debug(
            "DigitalTwin [%s] synced sensor variable '%s' = %s (source=%s)",
            self.world_id,
            variable_name,
            value,
            source,
        )

    def register_entity(
        self, entity_id: str, entity_name: str, status: str = "active"
    ) -> PhysicalEntity:
        """Register or update a physical entity digital twin."""
        entity = PhysicalEntity(entity_id=entity_id, entity_name=entity_name, status=status)
        self._entities[entity_id] = entity
        return entity

    def get_variable(self, variable_name: str) -> Any | None:
        """Retrieve current value of an empirical variable."""
        return self._variables.get(variable_name)

    def get_entity(self, entity_id: str) -> PhysicalEntity | None:
        """Retrieve physical entity by ID."""
        return self._entities.get(entity_id)

    def create_snapshot(self) -> WorldStateSnapshot:
        """Create an immutable snapshot of current empirical digital twin state."""
        entity_states = {e_id: e.status for e_id, e in self._entities.items()}
        return WorldStateSnapshot(
            world_id=self.world_id,
            timestamp=time.time(),
            variables=dict(self._variables),
            entity_states=entity_states,
        )
