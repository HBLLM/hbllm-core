"""Digital Twin Registry — Empirical Physical Telemetry & Environment State Container.

Enforces strict cognitive separation: DigitalTwinRegistry ONLY records empirical physical telemetry.
Cognitive inferences and predictions cannot directly mutate DigitalTwinRegistry.
Proposals pass through TransactionManager or are lifted to declarative CausalPredicates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export physical entity state as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"entity:{self.entity_id}",
            operator="==",
            value=self.status,
            relation=self.entity_name,
            target_entity=self.entity_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering physical entity state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"phys_{self.entity_id}",
                node_data={
                    "entity_id": self.entity_id,
                    "entity_name": self.entity_name,
                    "status": self.status,
                    "telemetry": dict(self.telemetry),
                    "last_updated": self.last_updated,
                },
            )
        ]


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
        variables = dict(self._variables)
        if self._entities and "spatial_entities" not in variables:
            variables["spatial_entities"] = {
                e_id: {"name": e.entity_name, "status": e.status, **e.telemetry}
                for e_id, e in self._entities.items()
            }
        return WorldStateSnapshot(
            world_id=self.world_id,
            timestamp=time.time(),
            variables=variables,
            entity_states=entity_states,
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export empirical variables and entities as declarative causal predicates."""
        preds: list[CausalPredicate] = []
        for var_name, var_val in self._variables.items():
            if isinstance(var_val, (int, float, str, bool)):
                preds.append(
                    CausalPredicate(
                        variable=var_name,
                        operator="==",
                        value=var_val,
                        relation="telemetry",
                        target_entity=self.world_id,
                    )
                )
        for ent in self._entities.values():
            preds.append(ent.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting physical entity states."""
        ops: list[TransactionOperation] = []
        for ent in self._entities.values():
            ops.extend(ent.to_transaction_operations())
        return ops
