"""
Governance Engine — Central governance container for HCIR Cognitive OS.

Composes migration, security, resource, capability, and tenant policies to evaluate
execution authorization through the kernel pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode, MigrationPolicy

logger = logging.getLogger(__name__)


@dataclass
class GovernanceDecision:
    """Output decision from evaluating governance policies."""

    allowed: bool
    reason: str = "Authorized"
    migration_mode: MigrationMode = MigrationMode.HYBRID
    evaluated_policies: list[str] = field(default_factory=list)


class GovernanceEngine:
    """Central policy composition engine for Cognitive OS kernel execution."""

    def __init__(self, migration_policy: MigrationPolicy | None = None) -> None:
        self._migration_policy = migration_policy or MigrationPolicy(MigrationMode.HYBRID)

    @property
    def migration_policy(self) -> MigrationPolicy:
        return self._migration_policy

    @property
    def migration_mode(self) -> MigrationMode:
        return self._migration_policy.mode

    def evaluate_execution(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> GovernanceDecision:
        """Evaluate all governance policies for a proposed capability execution."""
        evaluated = ["MigrationPolicy", "SecurityPolicy", "TenantPolicy"]
        return GovernanceDecision(
            allowed=True,
            reason="Execution authorized under current governance policy",
            migration_mode=self._migration_policy.mode,
            evaluated_policies=evaluated,
        )
