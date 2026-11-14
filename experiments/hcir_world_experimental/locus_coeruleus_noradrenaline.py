"""Locus Coeruleus (LC) Tonic vs. Phasic Noradrenergic Exploration-Exploitation Switch.

Biologically grounded in Aston-Jones & Cohen (2005) Adaptive Gain Theory:
1. Phasic Noradrenergic Mode (Focused Exploitation):
   - Driven by high task utility and steady progress toward goal states.
   - Brief phasic LC bursts elevate gain in active cortical decision circuits,
     sharpening decision boundaries (low softmax exploration temperature).
2. Tonic Noradrenergic Mode (Diffuse Exploration):
   - Triggered by declining task utility, stagnation, or prolonged deadlocks.
   - Elevated baseline tonic discharge increases global neural noise, desaturating
     dominant action choices and driving behavioral variability.
3. Disengaged Reset Regime:
   - When stagnation persists beyond an acute tolerance threshold, tonic LC activity
     peaks, inducing strategic task disengagement and triggering executive goal resets.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class ExplorationRegime(str, enum.Enum):
    """Behavioral mode dictated by noradrenergic gain dynamics."""

    EXPLOITATION_FOCUSED = "exploitation_focused"
    EXPLORATION_DIFFUSE = "exploration_diffuse"
    DISENGAGED_RESET = "disengaged_reset"


@dataclass
class NoradrenergicState:
    """Active LC neuromodulatory state snapshot."""

    tonic_level: float  # [0.0, 1.0] baseline tonic discharge rate
    phasic_burst: float  # [0.0, 1.0] transient task-related burst amplitude
    task_utility: float  # Current moving average utility [-1.0, 1.0]
    stagnation_counter: int  # Consecutive steps without tangible progress
    active_regime: ExplorationRegime
    exploration_temperature: float  # Dynamic softmax temperature [0.1, 3.0]

    def to_causal_predicate(self) -> CausalPredicate:
        """Export noradrenergic state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="noradrenergic_state",
            operator="==",
            value=f"regime:{self.active_regime.value}:temp:{self.exploration_temperature:.2f}",
            relation="lc_noradrenaline_adaptive_gain",
            target_entity="locus_coeruleus",
        )


class LocusCoeruleusNoradrenalineEngine:
    """Locus coeruleus adaptive gain engine modulating exploration vs. exploitation regimes."""

    def __init__(
        self,
        base_tonic: float = 0.20,
        stagnation_threshold: int = 8,
        disengagement_threshold: int = 20,
        utility_decay: float = 0.15,
    ) -> None:
        """Initialize locus coeruleus noradrenaline engine.

        Args:
            base_tonic: Baseline resting tonic LC discharge rate.
            stagnation_threshold: Stagnation steps before transitioning to diffuse exploration.
            disengagement_threshold: Stagnation steps before triggering complete disengaged reset.
            utility_decay: Exponential moving average rate for perceived task utility.
        """
        self.base_tonic = float(base_tonic)
        self.stagnation_threshold = max(2, int(stagnation_threshold))
        self.disengagement_threshold = max(
            self.stagnation_threshold + 2, int(disengagement_threshold)
        )
        self.utility_decay = max(0.01, min(1.0, float(utility_decay)))

        self.current_tonic: float = self.base_tonic
        self.current_phasic: float = 0.80
        self.current_utility: float = 0.50
        self.stagnation_steps: int = 0
        self.total_updates: int = 0

    def update_gain_state(
        self,
        immediate_utility: float,
        progress_made: bool,
    ) -> NoradrenergicState:
        """Update LC noradrenergic firing modes based on utility and progress feedback.

        Args:
            immediate_utility: Current step utility/reward signal in [-1.0, 1.0].
            progress_made: True if state changed closer to goal, False if stagnant.

        Returns:
            NoradrenergicState containing active regime and dynamic exploration temperature.
        """
        self.total_updates += 1
        clamped_util = max(-1.0, min(1.0, float(immediate_utility)))

        # Update running utility estimate
        self.current_utility = (
            1.0 - self.utility_decay
        ) * self.current_utility + self.utility_decay * clamped_util

        if progress_made:
            # Steady progress resets stagnation, boosts phasic responsiveness, dampens tonic
            self.stagnation_steps = 0
            self.current_tonic = max(self.base_tonic, self.current_tonic * 0.85)
            self.current_phasic = min(1.0, 0.40 + 0.60 * max(0.0, self.current_utility))
        else:
            # Stagnation accumulates, tonic discharge climbs, phasic burst drops
            self.stagnation_steps += 1
            tonic_increment = 0.05 * (1.0 + max(0.0, -self.current_utility))
            self.current_tonic = min(1.0, self.current_tonic + tonic_increment)
            self.current_phasic = max(0.05, self.current_phasic * 0.80)

        # Determine Exploration Regime and Temperature
        if self.stagnation_steps >= self.disengagement_threshold:
            regime = ExplorationRegime.DISENGAGED_RESET
            # Maximum noise to shatter deadlocks
            temp = 2.50
        elif self.stagnation_steps >= self.stagnation_threshold or self.current_tonic > 0.60:
            regime = ExplorationRegime.EXPLORATION_DIFFUSE
            # Elevated temperature for stochastic exploratory actions
            temp = max(1.0, 1.0 + 1.5 * (self.current_tonic - self.base_tonic))
        else:
            regime = ExplorationRegime.EXPLOITATION_FOCUSED
            # Sharpened choice distribution favoring highest-value actions
            temp = max(0.15, 0.50 * (1.0 - self.current_phasic * 0.6))

        return NoradrenergicState(
            tonic_level=float(self.current_tonic),
            phasic_burst=float(self.current_phasic),
            task_utility=float(self.current_utility),
            stagnation_counter=self.stagnation_steps,
            active_regime=regime,
            exploration_temperature=float(temp),
        )

    def trigger_phasic_burst(self, magnitude: float = 1.0) -> float:
        """Trigger an event-related phasic noradrenergic pulse upon major discovery."""
        burst = max(0.0, min(1.0, float(magnitude)))
        self.current_phasic = min(1.0, self.current_phasic + burst)
        # Phasic burst temporarily suppresses tonic noise
        self.current_tonic = max(self.base_tonic, self.current_tonic * 0.70)
        return self.current_phasic

    def reset(self) -> None:
        """Reset noradrenergic state to resting equilibrium."""
        self.current_tonic = self.base_tonic
        self.current_phasic = 0.80
        self.current_utility = 0.50
        self.stagnation_steps = 0
        self.total_updates = 0
