"""Intervention and Reversibility Model.

Provides the API for humans to pause, stop, or override the autonomy core,
and implements Undo Semantics (Reversibility) for mutating actions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReversibilityPolicy:
    """Defines how to undo or compensate for a mutating action."""

    action_name: str
    rollback_handler: Callable[..., Coroutine[Any, Any, bool]] | None = None
    compensation_handler: Callable[..., Coroutine[Any, Any, bool]] | None = None

    async def rollback(self, **kwargs: Any) -> bool:
        """Attempt to restore the previous state perfectly."""
        if self.rollback_handler:
            logger.info("Executing Rollback for %s", self.action_name)
            return await self.rollback_handler(**kwargs)
        return False

    async def compensate(self, **kwargs: Any) -> bool:
        """Attempt to compensate (e.g., send a correction email) if rollback is impossible."""
        if self.compensation_handler:
            logger.info("Executing Compensation for %s", self.action_name)
            return await self.compensation_handler(**kwargs)
        return False


class InterventionAPI:
    """The control surface for human overrides."""

    def __init__(self, autonomy_core: Any) -> None:
        # Expected to hold a reference to AutonomyCore
        self.core = autonomy_core
        self.reversibility_registry: dict[str, ReversibilityPolicy] = {}

    def register_reversibility(self, policy: ReversibilityPolicy) -> None:
        """Register undo semantics for a specific action."""
        self.reversibility_registry[policy.action_name] = policy

    def pause(self) -> None:
        """Instantly suspends the AutonomyCore tick loop."""
        logger.warning("HUMAN INTERVENTION: Pausing AutonomyCore.")
        if hasattr(self.core, "pause"):
            self.core.pause()

    def resume(self) -> None:
        """Resumes the AutonomyCore."""
        logger.info("HUMAN INTERVENTION: Resuming AutonomyCore.")
        if hasattr(self.core, "resume"):
            self.core.resume()

    def stop(self, flush_queue: bool = True) -> None:
        """Hard-cancels the current active TaskGraph."""
        logger.critical("HUMAN INTERVENTION: Hard Stop Triggered.")
        if hasattr(self.core, "stop"):
            self.core.stop(flush_queue=flush_queue)

    def override_utility(self, new_weights: dict[str, float]) -> None:
        """Dynamically alter the ValueArbitrator weights (e.g. ignore battery)."""
        logger.warning("HUMAN INTERVENTION: Overriding dynamic utility weights: %s", new_weights)
        # Assuming core.planner.value_system is accessible
        # Implementation left to the value system integration
        pass

    async def attempt_undo(self, action_name: str, **kwargs: Any) -> bool:
        """Attempt to reverse a recently executed action."""
        policy = self.reversibility_registry.get(action_name)
        if not policy:
            logger.error("No reversibility policy registered for %s", action_name)
            return False

        # Try perfect rollback first
        success = await policy.rollback(**kwargs)
        if success:
            return True

        # Fallback to compensation
        return await policy.compensate(**kwargs)
