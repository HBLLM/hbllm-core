"""Telemetry Sensor Adapter — concrete UnifiedPerceptionProvider for system & hardware telemetry.

Collects CPU, memory, battery, temperature, and environmental telemetry
and emits structured observations for the EvidenceNormalizer.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import Any

from hbllm.runtime.providers.capability import ProviderCapability
from hbllm.runtime.providers.perception import UnifiedPerceptionProvider

logger = logging.getLogger(__name__)


class TelemetrySensorAdapter:
    """Concrete perception adapter for hardware, system, and IoT telemetry.

    Conforms to ``UnifiedPerceptionProvider``.
    Emits raw sensor observation dictionaries that ``EvidenceNormalizer.normalize_sensor()``
    converts to canonical ``PerceptualEvidenceNode`` objects.

    Usage::

        adapter = TelemetrySensorAdapter()
        telemetry_readings = await adapter.observe()
    """

    def __init__(
        self,
        provider_id: str = "system_telemetry",
    ) -> None:
        self._provider_id = provider_id

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for telemetry sensing."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="perception",
            capabilities=["telemetry", "hardware_monitor", "battery_status"],
            modalities=["sensor"],
            latency_profile="very_low",
            quality_profile="high",
            memory_requirement_mb=10,
            hardware_requirements=["cpu"],
            requires_network=False,
        )

    async def initialize(self) -> None:
        """Initialize telemetry collectors."""
        logger.info("Initialized TelemetrySensorAdapter (%s)", self._provider_id)

    async def shutdown(self) -> None:
        """Release telemetry resources."""
        logger.info("Shutdown TelemetrySensorAdapter (%s)", self._provider_id)

    async def observe(self, input_data: Any = None) -> list[dict[str, Any]]:
        """Produce typed sensor telemetry readings.

        Args:
            input_data: Optional specific sensor query parameter.

        Returns:
            List of telemetry dictionaries with sensor_id, predicate, value, value_type.
        """
        readings: list[dict[str, Any]] = []
        now = time.time()

        try:
            # 1. System load / platform
            readings.append(
                {
                    "sensor_id": f"host_{platform.node() or 'local'}",
                    "predicate": "system_os",
                    "value": f"{platform.system()} {platform.release()}",
                    "value_type": "string",
                    "confidence": 1.0,
                    "observed_at": now,
                }
            )

            # 2. CPU load average (if supported)
            if hasattr(os, "getloadavg"):
                load1, load5, load15 = os.getloadavg()
                readings.append(
                    {
                        "sensor_id": "cpu_load_monitor",
                        "predicate": "load_average",
                        "value": [float(load1), float(load5), float(load15)],
                        "value_type": "vector3",
                        "confidence": 1.0,
                        "observed_at": now,
                    }
                )

            return readings
        except Exception as e:
            logger.error("TelemetrySensorAdapter observation failed: %s", e)
            return []
