"""Human Cognitive Load Model.

Tracks the human's attention budget and interruption fatigue to ensure
the system stays quiet when the user is exhausted or focused.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from hbllm.brain.snn import LIFConfig, SpikingAccumulator

logger = logging.getLogger(__name__)


@dataclass
class HumanAttentionState:
    """The estimated cognitive state of the human user."""

    attention_budget: float = 100.0  # Max 100
    approval_fatigue: float = 0.0  # 0.0 to 1.0 (1.0 = completely fatigued)
    focus_mode_active: bool = False  # e.g., in a meeting or deep work
    last_interruption_time: float = 0.0


class HumanAttentionModel:
    """Calculates and manages the user's interruption tolerance using spiking dynamics."""

    def __init__(self, config: LIFConfig | None = None) -> None:
        self.state = HumanAttentionState()

        # Default config for interruption fatigue:
        # Firing threshold of 0.8 fatigue, decay half-life of 300 seconds (5 mins),
        # reset to 0.0, and 600 seconds (10 mins) refractory period to block prompts.
        self.config = config or LIFConfig(
            threshold=0.8,
            decay_half_life=300.0,
            reset_potential=0.0,
            refractory_period=600.0,
        )
        self.fatigue_accumulator = SpikingAccumulator(self.config)

    def record_interruption(self, severity: float = 0.1) -> None:
        """Called whenever the system prompts the user (e.g., Explanation-First approval)."""
        now = time.time()

        # Interruption is modeled as an input current spike.
        # Quick successive prompts accumulate charge; sparse prompts leak away.
        charge = 0.15 + severity
        spike_event = self.fatigue_accumulator.stimulate(charge, timestamp=now)

        self.state.approval_fatigue = self.fatigue_accumulator.get_potential(now)
        self.state.attention_budget = max(0.0, 100.0 * (1.0 - self.state.approval_fatigue))
        self.state.last_interruption_time = now

        if spike_event.fired:
            logger.warning(
                "HumanAttentionModel: Interruption threshold breached (strength %.2f)! "
                "Activating focus mode protection.",
                spike_event.strength
            )
            self.state.focus_mode_active = True

    def natural_recovery(self) -> None:
        """Called periodically (e.g., tick loop) to restore budget over time."""
        now = time.time()

        # Simulating step with 0.0 charge automatically applies real-time exponential leak decay
        current_fatigue = self.fatigue_accumulator.get_potential(now)

        self.state.approval_fatigue = current_fatigue
        self.state.attention_budget = min(100.0, max(0.0, 100.0 * (1.0 - current_fatigue)))

        # Deactivate focus mode if refractory period has ended
        if self.fatigue_accumulator.neuron.refractory_time_remaining <= 0.0:
            self.state.focus_mode_active = False

    def can_interrupt(self, action_criticality: float) -> bool:
        """Determine if we should interrupt the user or defer/cancel the task."""
        now = time.time()
        current_fatigue = self.fatigue_accumulator.get_potential(now)

        # Update current state representation
        self.state.approval_fatigue = current_fatigue
        self.state.attention_budget = min(100.0, max(0.0, 100.0 * (1.0 - current_fatigue)))

        # Focus mode is active if explicitly set or if within SNN refractory period
        in_refractory = self.fatigue_accumulator.neuron.refractory_time_remaining > 0.0
        is_focused = self.state.focus_mode_active or in_refractory

        if is_focused and action_criticality < 0.9:
            logger.info("HumanAttentionModel: Cannot interrupt. User is in focus/refractory mode.")
            return False

        if current_fatigue > 0.8 and action_criticality < 0.7:
            logger.warning("HumanAttentionModel: Cannot interrupt. User is highly fatigued.")
            return False

        if self.state.attention_budget < 20.0 and action_criticality < 0.5:
            logger.info("HumanAttentionModel: Cannot interrupt. Attention budget exhausted.")
            return False

        return True

