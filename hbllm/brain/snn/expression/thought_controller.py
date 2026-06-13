"""
ThoughtController — SNN-gated thought sequencer.

Uses a pair of LIF neurons (readiness + coherence) to decide when each
thought goal is "ready" to fire the LLM for generation.  This prevents
the LLM from being called when context is insufficient or when the
previous fragment hasn't settled.

The controller gates **when** to generate, not **what** to generate.
The ThoughtPlanner decides *what* (the goals), the ThoughtController
decides *when* (the gating), and the LLM decides *how* (the text).

Design:
    readiness neuron
        Accumulates salience + memory density signals.
        When it spikes, the goal has enough context to generate.

    coherence neuron
        Accumulates inter-goal transition signals.
        When it spikes, the transition from the previous fragment
        is smooth enough to proceed.

Both must spike (or be bypassed) before the controller emits a
``fire`` signal for the goal.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hbllm.brain.snn.expression.models import ThoughtGoal
from hbllm.brain.snn.lif import LIFConfig, LIFNeuron

if TYPE_CHECKING:
    from hbllm.brain.snn.plasticity import PlasticWeightMatrix

logger = logging.getLogger(__name__)


@dataclass
class GateSignal:
    """Result of a ThoughtController gate check.

    Attributes:
        fire: Whether the LLM should generate for this goal.
        readiness: Current readiness neuron potential.
        coherence: Current coherence neuron potential.
        bypass: Whether gating was bypassed (e.g. first goal).
    """

    fire: bool = False
    readiness: float = 0.0
    coherence: float = 0.0
    bypass: bool = False


class ThoughtController:
    """SNN-gated thought sequencer.

    Uses two LIF neurons to gate LLM generation per thought goal:
    - **readiness**: fires when the goal has enough context
    - **coherence**: fires when the transition from previous is smooth

    Args:
        readiness_threshold: Threshold for the readiness neuron.
        coherence_threshold: Threshold for the coherence neuron.
        max_wait_steps: Maximum signal steps before bypass (prevents deadlock).
    """

    def __init__(
        self,
        readiness_threshold: float = 0.6,
        coherence_threshold: float = 0.5,
        max_wait_steps: int = 5,
        plastic_weights: PlasticWeightMatrix | None = None,
    ) -> None:
        self._readiness = LIFNeuron(
            config=LIFConfig(
                threshold=readiness_threshold,
                decay_half_life=0.5,
                reset_potential=0.0,
                refractory_period=0.01,
            ),
            neuron_id="expression.readiness",
        )
        self._coherence = LIFNeuron(
            config=LIFConfig(
                threshold=coherence_threshold,
                decay_half_life=0.3,
                reset_potential=0.0,
                refractory_period=0.01,
            ),
            neuron_id="expression.coherence",
        )
        self._max_wait_steps = max_wait_steps
        self._prev_fragment_text: str | None = None
        self._step_count = 0
        self.plastic_weights = plastic_weights

        # Static weight defaults (used when no plasticity)
        self._static_weights: dict[str, dict[str, float]] = {
            "readiness": {
                "salience": 0.5,
                "memory_density": 0.3,
                "budget": 0.2,
            },
            "coherence": {
                "base": 0.4,
                "overlap": 1.0,
                "constraint_penalty": 1.0,
            },
        }

    def gate(
        self,
        goal: ThoughtGoal,
        prev_fragment_text: str | None = None,
    ) -> GateSignal:
        """Check whether the LLM should fire for this goal.

        Feeds goal metadata as signals to readiness + coherence neurons.
        Returns a GateSignal indicating whether to proceed.

        Args:
            goal: The thought goal to check.
            prev_fragment_text: Text of the previously generated fragment
                (None for the first goal).

        Returns:
            GateSignal with fire decision and neuron states.
        """
        t = time.time()
        self._step_count += 1

        # First goal always bypasses gating
        if prev_fragment_text is None:
            self.reset()
            return GateSignal(fire=True, bypass=True)

        # Compute readiness signal from goal metadata
        readiness_current = self._compute_readiness(goal)

        # Compute coherence signal from goal + previous fragment
        coherence_current = self._compute_coherence(goal, prev_fragment_text)

        # Step both neurons
        readiness_spike = self._readiness.step(readiness_current, t)
        coherence_spike = self._coherence.step(coherence_current, t)

        # Record STDP timing if plasticity is enabled
        if self.plastic_weights is not None:
            signals = {
                "salience": min(1.0, goal.salience),
                "memory_density": min(1.0, len(goal.memory_hints) * 0.3),
                "budget": min(1.0, goal.max_tokens / 512),
            }
            self.plastic_weights.record_signals(signals, t)

            fired_channels: list[str] = []
            if readiness_spike.fired:
                fired_channels.append("readiness")
            if coherence_spike.fired:
                fired_channels.append("coherence")
            if fired_channels:
                self.plastic_weights.record_spikes(fired_channels, t)

        # Check if both neurons agree to fire
        fire = readiness_spike.fired and coherence_spike.fired

        # Bypass if we've waited too long (deadlock prevention)
        bypass = False
        if not fire and self._step_count >= self._max_wait_steps:
            fire = True
            bypass = True
            logger.debug(
                "ThoughtController bypass: waited %d steps for goal '%s'",
                self._step_count,
                goal.id,
            )

        if fire:
            self._step_count = 0
            self._prev_fragment_text = None  # Will be set after generation

        return GateSignal(
            fire=fire,
            readiness=self._readiness.v,
            coherence=self._coherence.v,
            bypass=bypass,
        )

    def _compute_readiness(self, goal: ThoughtGoal) -> float:
        """Compute readiness current from goal metadata.

        Higher salience, more memories, and larger token budget all
        increase readiness (the goal has enough context to generate).
        """
        salience_signal = min(1.0, goal.salience)
        memory_density = min(1.0, len(goal.memory_hints) * 0.3)
        budget_signal = min(1.0, goal.max_tokens / 512)

        # Use learned or static weights
        if self.plastic_weights is not None:
            w = self.plastic_weights.get_weights("readiness")
        else:
            w = self._static_weights["readiness"]

        current = (
            salience_signal * w.get("salience", 0.5)
            + memory_density * w.get("memory_density", 0.3)
            + budget_signal * w.get("budget", 0.2)
        )
        return current

    def _compute_coherence(self, goal: ThoughtGoal, prev_text: str) -> float:
        """Compute coherence current from goal + previous fragment.

        Checks lexical overlap and domain continuity to estimate
        how smoothly this goal follows the previous fragment.
        """
        # Domain continuity: same domain = high coherence
        prev_words = set(prev_text.lower().split()[-20:])
        goal_words = set(goal.text.lower().split())

        overlap = len(prev_words & goal_words)
        overlap_signal = min(1.0, overlap * 0.2)

        # Constraint goals need more coherence
        constraint_penalty = 0.0
        if goal.constraints:
            constraint_penalty = 0.1

        # Base coherence from salience continuity
        base_coherence = 0.4

        current = base_coherence + overlap_signal - constraint_penalty
        return max(0.0, min(1.0, current))

    def record_generation(self, fragment_text: str) -> None:
        """Record the generated fragment for coherence tracking.

        Call this after the LLM generates text for a goal.
        """
        self._prev_fragment_text = fragment_text

    def reset(self) -> None:
        """Reset controller state for a new expression session."""
        self._readiness.v = 0.0
        self._readiness.last_update_time = None
        self._readiness.refractory_time_remaining = 0.0
        self._coherence.v = 0.0
        self._coherence.last_update_time = None
        self._coherence.refractory_time_remaining = 0.0
        self._prev_fragment_text = None
        self._step_count = 0
