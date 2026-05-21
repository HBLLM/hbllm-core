"""Human Cognitive Load Model.

Tracks the human's attention budget and interruption fatigue to ensure
the system stays quiet when the user is exhausted or focused.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HumanAttentionState:
    """The estimated cognitive state of the human user."""

    attention_budget: float = 100.0  # Max 100
    approval_fatigue: float = 0.0  # 0.0 to 1.0 (1.0 = completely fatigued)
    focus_mode_active: bool = False  # e.g., in a meeting or deep work
    last_interruption_time: float = 0.0


class HumanAttentionModel:
    """Calculates and manages the user's interruption tolerance."""

    def __init__(self) -> None:
        self.state = HumanAttentionState()

    def record_interruption(self, severity: float = 0.1) -> None:
        """Called whenever the system prompts the user (e.g., Explanation-First approval)."""
        now = time.time()

        # If interrupted too soon after the last one, fatigue spikes
        time_since_last = now - self.state.last_interruption_time
        if time_since_last < 300:  # 5 minutes
            fatigue_spike = 0.2
        else:
            fatigue_spike = 0.05

        self.state.approval_fatigue = min(
            1.0, self.state.approval_fatigue + fatigue_spike + severity
        )
        self.state.attention_budget = max(0.0, self.state.attention_budget - (10.0 * severity))
        self.state.last_interruption_time = now

    def natural_recovery(self) -> None:
        """Called periodically (e.g., tick loop) to restore budget over time."""
        now = time.time()
        time_since_last = now - self.state.last_interruption_time

        # If no interruptions for 1 hour, start recovering fatigue
        if time_since_last > 3600:
            self.state.approval_fatigue = max(0.0, self.state.approval_fatigue - 0.1)
            self.state.attention_budget = min(100.0, self.state.attention_budget + 5.0)

    def can_interrupt(self, action_criticality: float) -> bool:
        """Determine if we should interrupt the user or defer/cancel the task."""
        if self.state.focus_mode_active and action_criticality < 0.9:
            logger.info("HumanAttentionModel: Cannot interrupt. User is in focus mode.")
            return False

        if self.state.approval_fatigue > 0.8 and action_criticality < 0.7:
            logger.warning("HumanAttentionModel: Cannot interrupt. User is highly fatigued.")
            return False

        if self.state.attention_budget < 20.0 and action_criticality < 0.5:
            logger.info("HumanAttentionModel: Cannot interrupt. Attention budget exhausted.")
            return False

        return True
