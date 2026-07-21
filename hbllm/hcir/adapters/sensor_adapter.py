"""
Sensor Capability Adapter — ingests physical sensor and API telemetry into HCIR.

Converts physical sensor telemetry streams into typed HCIR graph nodes:

    Physical Sensor / API Telemetry
                  │
        SensorCapabilityAdapter
                  ├── Raw Reading → ObservationNode
                  ├── Telemetry Value → WorldVariableNode
                  └── Threshold Variance → WorldKernel Surprise Check
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import NodeLifecycle, ObservationNode, WorldVariableNode
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope, UncertaintyVector

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """A telemetry reading from a physical sensor or API endpoint."""

    sensor_id: str
    sensor_type: str
    variable_name: str
    value: Any
    unit: str = ""
    confidence: float = 0.95
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SensorCapabilityAdapter:
    """Ingests sensor readings into HCIR ObservationNodes & WorldVariableNodes.

    Usage::

        adapter = SensorCapabilityAdapter(services)
        reading = SensorReading(
            sensor_id="temp_sensor_01",
            sensor_type="thermocouple",
            variable_name="temperature",
            value=26.4,
            unit="Celsius",
        )
        tx = adapter.ingest_reading(reading)
    """

    def __init__(self, services: KernelServices) -> None:
        self._services = services
        self._history: list[SensorReading] = []

    def ingest_reading(
        self,
        reading: SensorReading,
        author: str = "sensor_adapter",
        tenant_id: str = "default",
    ) -> HCIRTransaction | None:
        """Process sensor reading and commit ObservationNode & WorldVariableNode to graph."""
        self._history.append(reading)

        obs_id = f"obs_{reading.sensor_id}_{int(reading.timestamp)}"
        var_id = f"wv_{reading.variable_name}"

        # 1. Construct raw ObservationNode
        obs_node = ObservationNode(
            id=obs_id,
            sensor_source=f"{reading.sensor_type}:{reading.sensor_id}",
            payload={
                "variable": reading.variable_name,
                "value": reading.value,
                "unit": reading.unit,
                "metadata": reading.metadata,
            },
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=reading.confidence),
            provenance=Provenance(created_by=author, source_type="observed"),
            scope=Scope(tenant_id=tenant_id),
            tags=["sensor_telemetry", reading.sensor_type, reading.variable_name],
        )

        # 2. Construct or update WorldVariableNode
        wv_node = WorldVariableNode(
            id=var_id,
            variable_name=reading.variable_name,
            value=reading.value,
            unit=reading.unit,
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=reading.confidence),
            provenance=Provenance(created_by=author, source_type="observed"),
            scope=Scope(tenant_id=tenant_id),
            tags=["world_variable", reading.variable_name],
        )

        # 3. Submit transaction via TransactionManager
        tx = HCIRTransaction(
            author=author,
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_id=obs_node.id,
                    node_data=obs_node.model_dump(),
                ),
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=wv_node.id,
                    node_data=wv_node.model_dump(),
                ),
            ],
            provenance=Provenance(created_by=author, source_type="observed"),
        )

        res = self._services.transaction_manager.commit(tx)
        if res.is_committed:
            logger.info("Ingested sensor reading '%s' = %s %s", reading.variable_name, reading.value, reading.unit)
            return tx

        logger.warning("Sensor reading ingestion rejected for sensor '%s'", reading.sensor_id)
        return None
