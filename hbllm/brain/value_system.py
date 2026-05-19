"""Dynamic Value Arbitration and Policy-Driven Utility."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class UtilityPolicy(ABC):
    """Base class for dynamic utility weighting policies."""

    @abstractmethod
    def apply_modifiers(self, base_utility: float, context: dict[str, Any]) -> float:
        """Modify base utility according to contextual rules."""
        pass


class ResourceConservationPolicy(UtilityPolicy):
    """Multiplies utility of resource-saving actions when resources are low."""

    def apply_modifiers(self, base_utility: float, context: dict[str, Any]) -> float:
        action_type = context.get("action_type", "")
        battery = context.get("system_battery", 100)

        if action_type == "conserve_resource" and battery < 20:
            # High multiplier when battery is critical
            multiplier = 1.0 + (5.0 * (20 - battery) / 20.0)
            return base_utility * multiplier

        return base_utility


class UrgencyOverridePolicy(UtilityPolicy):
    """Spikes utility for highly urgent user actions, overriding other constraints."""

    def apply_modifiers(self, base_utility: float, context: dict[str, Any]) -> float:
        urgency = context.get("urgency_weight", 0.0)
        if urgency > 0.8:
            return base_utility * (1.0 + (urgency * 10.0))

        return base_utility


class InterruptionPenaltyPolicy(UtilityPolicy):
    """Penalizes actions that interrupt the user during focused work."""

    def apply_modifiers(self, base_utility: float, context: dict[str, Any]) -> float:
        user_focused = context.get("user_is_focused", False)
        action_interrupts = context.get("action_interrupts_user", False)

        if user_focused and action_interrupts:
            # 8x penalty applied as a division
            return base_utility / 8.0

        return base_utility


class DynamicValueArbitrator:
    """Arbitrates utility conflicts using active policies."""

    def __init__(self, policies: list[UtilityPolicy] | None = None) -> None:
        self.policies = policies or [
            ResourceConservationPolicy(),
            UrgencyOverridePolicy(),
            InterruptionPenaltyPolicy()
        ]

    def compute_utility(self, base_utility: float, context: dict[str, Any]) -> float:
        """Calculate dynamic utility based on policies."""
        utility = base_utility

        for policy in self.policies:
            utility = policy.apply_modifiers(utility, context)

        # Optional: user preference baseline weight
        user_preference = context.get("user_preference_weight", 1.0)
        utility *= user_preference

        return utility
