"""
Migration Metrics & Telemetry — HCIR §10.

Records fallback metrics, fallback reasons (HCIR_TIMEOUT, UNAVAILABLE), and
tracks the Cognitive Authority Metric (verifying 100% decision authority in HCIR mode).
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode

logger = logging.getLogger(__name__)


class MigrationMetricRecord(BaseModel):
    """Telemetry record for a capability execution or fallback event."""

    capability_name: str
    migration_mode: MigrationMode
    selected_backend: str
    fallback_reason: str | None = None
    elapsed_ms: int = 0
    timestamp: float = Field(default_factory=time.time)


class MigrationMetrics:
    """Telemetry recorder for HCIR kernel migration metrics and authority tracking."""

    def __init__(self) -> None:
        self._records: list[MigrationMetricRecord] = []
        self._hcir_decisions: int = 0
        self._legacy_decisions: int = 0

    def record_execution(
        self,
        capability_name: str,
        mode: MigrationMode,
        backend: str,
        fallback_reason: str | None = None,
        elapsed_ms: int = 0,
    ) -> MigrationMetricRecord:
        """Record a capability execution event."""
        rec = MigrationMetricRecord(
            capability_name=capability_name,
            migration_mode=mode,
            selected_backend=backend,
            fallback_reason=fallback_reason,
            elapsed_ms=elapsed_ms,
        )
        self._records.append(rec)
        if backend == "hcir":
            self._hcir_decisions += 1
        else:
            self._legacy_decisions += 1
        logger.debug(
            "MigrationMetrics recorded: capability=%s, backend=%s", capability_name, backend
        )
        return rec

    def get_cognitive_authority_metric(self) -> float:
        """Calculate Cognitive Authority Metric (percentage of decisions made by HCIR)."""
        total = self._hcir_decisions + self._legacy_decisions
        if total == 0:
            return 100.0
        return (self._hcir_decisions / total) * 100.0

    def get_fallback_count(self) -> int:
        return sum(1 for r in self._records if r.fallback_reason is not None)
