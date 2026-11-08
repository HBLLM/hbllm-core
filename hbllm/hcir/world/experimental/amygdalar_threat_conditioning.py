"""Amygdalar Fear Conditioning & Fast Threat Potentiation.

Biologically grounded in lateral and central amygdala fear conditioning
(LeDoux, 2000; Phelps & LeDoux, 2005; Fanselow & Poulos, 2005):
1. Single-Trial Pavlovian Threat Association:
   - Rapidly binds neutral sensory features or entity signatures (Conditioned Stimulus, CS)
     to aversive outcomes or lethal hazards (Unconditioned Stimulus, US).
   - Threat potency P in [0.0, 1.0] scales with perceived damage / penalty.
2. Defensive Vigilance Cascades:
   - Evaluates active sensory scenes for conditioned threat cues and shifts vigilance state:
     "baseline_alert" -> "hyper_vigilant" -> "defensive_freezing" -> "evasive_flight".
3. Pre-Attentive Action Inhibition & Extinction:
   - Inhibits exploratory approach actions toward conditioned danger sources.
   - Extinguishes conditioned fear responses exponentially when the CS is repeatedly
     experienced without aversive consequences.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class DefensiveVigilanceMode(str, enum.Enum):
    """Amygdalar defense cascade regime."""

    BASELINE_ALERT = "baseline_alert"
    HYPER_VIGILANT = "hyper_vigilant"
    DEFENSIVE_FREEZING = "defensive_freezing"
    EVASIVE_FLIGHT = "evasive_flight"


@dataclass
class ConditionedThreatStimulus:
    """A sensory stimulus or entity signature conditioned to an aversive shock."""

    stimulus_id: str
    threat_potency: float  # [0.0, 1.0] conditioned fear intensity
    reinforcement_count: int = 1
    unreinforced_encounters: int = 0
    extinction_resistance: float = 0.80


@dataclass
class DefensiveVigilanceState:
    """Active amygdalar threat evaluation across the sensory field."""

    composite_threat_level: float  # [0.0, 1.0]
    vigilance_mode: DefensiveVigilanceMode
    active_threat_cues: list[str]
    inhibited_approaches: list[str] = field(default_factory=list)
    evasion_recommended: bool = False

    def to_causal_predicate(self) -> CausalPredicate:
        """Export defensive vigilance state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="defensive_vigilance_state",
            operator="==",
            value=f"mode:{self.vigilance_mode.value}:threat:{self.composite_threat_level:.2f}:cues:{len(self.active_threat_cues)}",
            relation="amygdalar_threat_conditioning",
            target_entity="amygdala",
        )


class AmygdalarThreatConditioningEngine:
    """Amygdala fear conditioning engine regulating defense cascades and fast threat potentiation."""

    def __init__(
        self,
        learning_rate: float = 0.50,
        extinction_rate: float = 0.15,
        max_stimuli: int = 500,
    ) -> None:
        """Initialize amygdalar threat conditioning engine.

        Args:
            learning_rate: Acquisition speed for conditioned threat potentiation.
            extinction_rate: Decay rate of threat potency upon unreinforced exposures.
            max_stimuli: Maximum number of conditioned stimuli to retain.
        """
        self.learning_rate = max(0.01, min(1.0, float(learning_rate)))
        self.extinction_rate = max(0.01, min(1.0, float(extinction_rate)))
        self.max_stimuli = max(10, int(max_stimuli))

        self.conditioned_stimuli: dict[str, ConditionedThreatStimulus] = {}
        self.last_state: DefensiveVigilanceState | None = None
        self.total_evaluations: int = 0

    def associate_threat(
        self,
        stimulus_key: str,
        aversive_magnitude: float,
    ) -> ConditionedThreatStimulus:
        """Rapidly potentiate or update a conditioned threat association (CS -> US).

        Args:
            stimulus_key: Perceptual signature or entity identifier.
            aversive_magnitude: Severity of aversive outcome in [0.0, 1.0].

        Returns:
            ConditionedThreatStimulus with updated potency.
        """
        clamped_mag = max(0.0, min(1.0, float(aversive_magnitude)))

        if stimulus_key in self.conditioned_stimuli:
            cs = self.conditioned_stimuli[stimulus_key]
            cs.reinforcement_count += 1
            cs.unreinforced_encounters = 0
            cs.threat_potency = min(
                1.0, cs.threat_potency + self.learning_rate * (clamped_mag - cs.threat_potency)
            )
            return cs

        if len(self.conditioned_stimuli) >= self.max_stimuli:
            least_key = min(
                self.conditioned_stimuli.keys(),
                key=lambda k: self.conditioned_stimuli[k].threat_potency,
            )
            self.conditioned_stimuli.pop(least_key, None)

        cs = ConditionedThreatStimulus(
            stimulus_id=stimulus_key,
            threat_potency=clamped_mag,
            reinforcement_count=1,
            unreinforced_encounters=0,
            extinction_resistance=0.85,
        )
        self.conditioned_stimuli[stimulus_key] = cs
        return cs

    def extinguish_threat(self, stimulus_key: str) -> None:
        """Gradually extinguish conditioned threat upon safe, unreinforced exposure."""
        if stimulus_key in self.conditioned_stimuli:
            cs = self.conditioned_stimuli[stimulus_key]
            cs.unreinforced_encounters += 1
            cs.threat_potency = max(0.0, cs.threat_potency * (1.0 - self.extinction_rate))
            if cs.threat_potency < 0.05:
                self.conditioned_stimuli.pop(stimulus_key, None)

    def evaluate_sensory_vigilance(
        self,
        present_stimuli: list[str],
    ) -> DefensiveVigilanceState:
        """Evaluate active sensory cues against conditioned threat stimuli.

        Args:
            present_stimuli: List of perceptual features or entity IDs present in sensory field.

        Returns:
            DefensiveVigilanceState indicating threat level, mode, and action constraints.
        """
        self.total_evaluations += 1
        active_cues: list[str] = []
        max_threat: float = 0.0
        threat_sum: float = 0.0

        for stim in present_stimuli:
            if stim in self.conditioned_stimuli:
                cs = self.conditioned_stimuli[stim]
                if cs.threat_potency > 0.10:
                    active_cues.append(stim)
                    threat_sum += cs.threat_potency
                    if cs.threat_potency > max_threat:
                        max_threat = cs.threat_potency

        # Composite threat level
        composite_threat = min(1.0, max_threat + 0.10 * len(active_cues))

        # Determine Defensive Regime
        if composite_threat >= 0.85:
            mode = DefensiveVigilanceMode.EVASIVE_FLIGHT
            evasion = True
        elif composite_threat >= 0.60:
            mode = DefensiveVigilanceMode.DEFENSIVE_FREEZING
            evasion = False
        elif composite_threat >= 0.25:
            mode = DefensiveVigilanceMode.HYPER_VIGILANT
            evasion = False
        else:
            mode = DefensiveVigilanceMode.BASELINE_ALERT
            evasion = False

        state = DefensiveVigilanceState(
            composite_threat_level=float(composite_threat),
            vigilance_mode=mode,
            active_threat_cues=active_cues,
            inhibited_approaches=active_cues.copy(),
            evasion_recommended=evasion,
        )
        self.last_state = state
        return state

    def reset(self) -> None:
        """Clear conditioned threat stimuli and evaluations."""
        self.conditioned_stimuli.clear()
        self.last_state = None
        self.total_evaluations = 0
