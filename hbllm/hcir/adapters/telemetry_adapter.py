"""
Telemetry Stream Adapter — sliding window aggregation into EnvironmentStateNodes.

Aggregates real-time time-series telemetry streams:

    Telemetry Stream (SensorReading, SensorReading, ...)
                  │
        TelemetryStreamAdapter
                  ├── Sliding Window Aggregation (mean, min, max)
                  └── EnvironmentStateNode Snapshot Update
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from hbllm.hcir.adapters.sensor_adapter import SensorReading
from hbllm.hcir.graph import EnvironmentStateNode, NodeLifecycle
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)


@dataclass
class TelemetryWindowSummary:
    """Aggregated statistics for a single telemetry variable over a window."""

    variable_name: str
    count: int = 0
    mean_value: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")


class TelemetryStreamAdapter:
    """Sliding window aggregator for real-time sensor telemetry.

    Usage::

        adapter = TelemetryStreamAdapter(services, environment_name="Greenhouse_Alpha")
        adapter.push_reading(r1)
        adapter.push_reading(r2)
        tx = adapter.flush_window()
    """

    def __init__(
        self,
        services: KernelServices,
        environment_name: str = "default_environment",
        window_size: int = 10,
    ) -> None:
        self._services = services
        self._environment_name = environment_name
        self._window_size = window_size
        self._buffer: list[SensorReading] = []

    def push_reading(self, reading: SensorReading) -> None:
        """Buffer a telemetry reading into the current sliding window."""
        self._buffer.append(reading)
        if len(self._buffer) >= self._window_size:
            self.flush_window()

    def flush_window(self, author: str = "telemetry_adapter", tenant_id: str = "default") -> HCIRTransaction | None:
        """Compute aggregated window statistics and commit EnvironmentStateNode."""
        if not self._buffer:
            return None

        by_var: dict[str, list[float]] = defaultdict(list)
        for r in self._buffer:
            if isinstance(r.value, (int, float)):
                by_var[r.variable_name].append(float(r.value))

        summaries: dict[str, dict[str, float]] = {}
        for var_name, values in by_var.items():
            if values:
                summaries[var_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        env_id = f"env_state_{self._environment_name}"
        env_node = EnvironmentStateNode(
            id=env_id,
            environment_name=self._environment_name,
            active_variables=list(summaries.keys()),
            overall_status="nominal",
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["telemetry_aggregated", self._environment_name],
        )

        tx = HCIRTransaction(
            author=author,
            operations=[
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=env_node.id,
                    node_data=env_node.model_dump(),
                )
            ],
        )

        res = self._services.transaction_manager.commit(tx)
        self._buffer.clear()

        if res.is_committed:
            logger.info("Flushed telemetry window for environment '%s' (%d variables)", self._environment_name, len(summaries))
            return tx
        return None
