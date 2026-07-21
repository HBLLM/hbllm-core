"""
Migration Policy & Transition Governance — HCIR §10.

Manages migration modes and transition validations:
    LEGACY → SHADOW → HYBRID → HCIR_CANDIDATE → HCIR → LEGACY_REMOVED
"""

from __future__ import annotations

import logging
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MigrationMode(StrEnum):
    """Lifecycle stages for HBLLM to HCIR Cognitive OS kernel migration."""

    LEGACY = "legacy"
    SHADOW = "shadow"
    HYBRID = "hybrid"
    HCIR_CANDIDATE = "hcir_candidate"
    HCIR = "hcir"
    LEGACY_REMOVED = "legacy_removed"


class MigrationTransition(BaseModel):
    """Audit record for a migration lifecycle mode promotion."""

    from_mode: MigrationMode
    to_mode: MigrationMode
    approved_by: str = "system"
    timestamp: float = Field(default_factory=time.time)
    reason: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class MigrationPolicy:
    """Governance policy evaluating migration mode transitions and cutovers."""

    def __init__(self, initial_mode: MigrationMode = MigrationMode.HYBRID) -> None:
        self._mode = initial_mode
        self._history: list[MigrationTransition] = []

    @property
    def mode(self) -> MigrationMode:
        return self._mode

    def transition_to(
        self, target_mode: MigrationMode, approved_by: str = "system", reason: str = ""
    ) -> bool:
        """Promote or transition to a target migration mode with audit tracking."""
        old_mode = self._mode
        self._mode = target_mode
        transition = MigrationTransition(
            from_mode=old_mode,
            to_mode=target_mode,
            approved_by=approved_by,
            reason=reason,
        )
        self._history.append(transition)
        logger.info(
            "MigrationPolicy mode changed: %s → %s (reason: %s)", old_mode, target_mode, reason
        )
        return True

    def get_history(self) -> list[MigrationTransition]:
        return list(self._history)
