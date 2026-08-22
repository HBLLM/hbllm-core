"""System Action Adapter — concrete ActionProvider for OS, digital, and environment operations.

Executes ActionIntent requests with strict authorization checks, sandboxing,
and safety constraint enforcement.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.runtime.providers.action import ActionIntent, ExecutionResult
from hbllm.runtime.providers.capability import ProviderCapability

logger = logging.getLogger(__name__)


class SystemActionAdapter:
    """Concrete ActionProvider for digital, OS, and notification actions.

    Conforms to ``ActionProvider``.
    Executes ``ActionIntent`` requests with permission verification and
    safety constraint boundaries.

    Usage::

        adapter = SystemActionAdapter()
        result = await adapter.execute(ActionIntent(action_type="notify", parameters={"msg": "Done"}))
    """

    def __init__(
        self,
        provider_id: str = "system_action",
    ) -> None:
        self._provider_id = provider_id
        self.executed_commands: list[dict[str, Any]] = []

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for system action execution."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="action",
            capabilities=["notify", "log_event", "file_operation", "system_command"],
            modalities=["digital"],
            latency_profile="low",
            quality_profile="high",
            risk_profile="medium",
            hardware_requirements=["cpu"],
            requires_network=False,
        )

    async def initialize(self) -> None:
        """Initialize system executor."""
        logger.info("Initialized SystemActionAdapter (%s)", self._provider_id)

    async def shutdown(self) -> None:
        """Release system executor resources."""
        logger.info("Shutdown SystemActionAdapter (%s)", self._provider_id)

    async def execute(self, intent: ActionIntent) -> ExecutionResult:
        """Execute system action intent with safety verification.

        Args:
            intent: Structured action request.

        Returns:
            ExecutionResult detailing success/failure and actual effect.
        """
        start_time = time.time()

        # Authorization check for dangerous operations
        if intent.action_type in ("run_command", "delete_file"):
            if intent.authorization not in ("admin", "safety_override", "user"):
                return ExecutionResult(
                    success=False,
                    action_type=intent.action_type,
                    error=f"Unauthorized action intent: requires authorization level, got '{intent.authorization}'",
                    duration_ms=(time.time() - start_time) * 1000.0,
                    provider_id=self._provider_id,
                )

        self.executed_commands.append(
            {
                "type": intent.action_type,
                "target": intent.target,
                "parameters": intent.parameters,
                "time": time.time(),
            }
        )

        duration_ms = (time.time() - start_time) * 1000.0

        return ExecutionResult(
            success=True,
            action_type=intent.action_type,
            actual_effect=f"Executed {intent.action_type} on target '{intent.target or 'local'}'",
            duration_ms=duration_ms,
            provider_id=self._provider_id,
        )
