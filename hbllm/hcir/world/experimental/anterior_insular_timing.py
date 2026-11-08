"""Anterior Insular Cortex (AIC) Subjective Time Dilation & Interoceptive Pacing.

Biological Foundations:
- Craig (2009): "How do you feel -- now? The anterior insula and human awareness"
  (Nat Rev Neurosci 10: 59-70).
- Wittmann (2013): "The inner sense of time: how the brain creates a representation
  of time" (Nat Rev Neurosci 14: 217-223).
- Van Wassenhove (2009): "Minding time: an amodal representational space for mental time".

Key Cognitive Capabilities:
1. Interoceptive Salience & Subjective Time Dilation:
   - Dilates subjective time (dt_subj = dt * dilation_factor) under acute threat, high visceral
     conflict, or novel salience bursts, granting additional simulation depth and deliberation.
   - Contracts subjective time during routine, familiar, or low-arousal execution.
2. Interoceptive Pulse Accumulator:
   - Tracks homeostatic arousal pulses as an internal pacemaker, providing an embodied
     subjective clock complementary to hippocampal CA1 metric interval timers.
3. Temporal Compression Ratio:
   - Quantifies the ratio between subjective perceptual duration and objective clock ticks,
     enabling metacognitive awareness of stress-induced cognitive distortion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from hbllm.hcir.world.causal_discovery import CausalPredicate


class PacingMode(str, Enum):
    """Operational mode of the anterior insular temporal pacemaker."""

    ROUTINE_COMPRESSED = "routine_compressed"
    CALIBRATED_NOMINAL = "calibrated_nominal"
    DILATED_SALIENT = "dilated_salient"
    ACUTE_HYPER_DILATED = "acute_hyper_dilated"


@dataclass(frozen=True)
class InteroceptivePacingState:
    """Instantaneous pacing and time dilation state emitted by the anterior insular cortex."""

    timestamp: float
    objective_dt: float
    subjective_dt: float
    dilation_factor: float
    arousal_level: float
    salience_density: float
    pacing_mode: PacingMode
    pulse_accumulator: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Serialize insular pacing state into an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="anterior_insula_subjective_timing",
            operator="==",
            value=f"mode:{self.pacing_mode.value}:dilation:{self.dilation_factor:.2f}:subj_dt:{self.subjective_dt:.3f}:arousal:{self.arousal_level:.2f}",
            relation="aic_subjective_time_dilation",
            target_entity="anterior_insular_cortex",
        )


class AnteriorInsularTimingEngine:
    """Orchestrator for anterior insular interoceptive timing and subjective dilation."""

    def __init__(
        self,
        base_pacemaker_hz: float = 10.0,
        threat_gain: float = 1.50,
        salience_gain: float = 1.00,
        max_dilation: float = 4.0,
        min_dilation: float = 0.5,
    ) -> None:
        """Initialize anterior insular timing engine.

        Args:
            base_pacemaker_hz: Baseline interoceptive pulse frequency in Hz.
            threat_gain: Sensitivity coefficient for threat-induced time dilation.
            salience_gain: Sensitivity coefficient for novel salience time dilation.
            max_dilation: Maximum subjective time dilation factor (slow-motion ceiling).
            min_dilation: Minimum dilation factor during routine/automatic tasks.
        """
        self.base_pacemaker_hz: float = max(1.0, float(base_pacemaker_hz))
        self.threat_gain: float = max(0.0, float(threat_gain))
        self.salience_gain: float = max(0.0, float(salience_gain))
        self.max_dilation: float = max(1.0, float(max_dilation))
        self.min_dilation: float = max(0.1, min(1.0, float(min_dilation)))

        self._pulse_accumulator: float = 0.0
        self._current_dilation: float = 1.0

    def compute_subjective_dilation(
        self,
        objective_dt: float,
        threat_level: float = 0.0,
        salience_score: float = 0.0,
        visceral_conflict: float = 0.0,
    ) -> InteroceptivePacingState:
        """Compute subjective time dilation and advance the interoceptive pacemaker.

        Args:
            objective_dt: Objective elapsed real or simulated time in seconds.
            threat_level: Amygdalar or somatic threat score in [0.0, 1.0].
            salience_score: Visual / environmental novelty salience in [0.0, 1.0].
            visceral_conflict: ACC conflict level or somatic alarm in [0.0, 1.0].

        Returns:
            InteroceptivePacingState detailing dilation factor, subjective dt, and pacing mode.
        """
        dt = max(0.0, float(objective_dt))
        threat = max(0.0, min(1.0, float(threat_level)))
        salience = max(0.0, min(1.0, float(salience_score)))
        conflict = max(0.0, min(1.0, float(visceral_conflict)))

        # Effective interoceptive arousal composite
        composite_arousal = min(1.0, 0.50 * threat + 0.30 * conflict + 0.20 * salience)

        # Biological dilation formula: dilation expands under high arousal/salience
        dilation = 1.0 + self.threat_gain * threat + self.salience_gain * salience + 0.50 * conflict

        # If perfectly peaceful/routine (zero threat, salience, and conflict), compress slightly
        if composite_arousal < 0.10 and salience < 0.10:
            dilation = max(self.min_dilation, dilation * 0.80)

        dilation = float(max(self.min_dilation, min(self.max_dilation, dilation)))
        self._current_dilation = dilation

        subjective_dt = dt * dilation

        # Pacemaker pulse integration: pulses accumulate faster during dilation
        effective_hz = self.base_pacemaker_hz * dilation
        pulse_increment = effective_hz * dt
        self._pulse_accumulator += pulse_increment

        # Determine pacing mode
        if dilation >= 2.50:
            mode = PacingMode.ACUTE_HYPER_DILATED
        elif dilation >= 1.30:
            mode = PacingMode.DILATED_SALIENT
        elif dilation <= 0.85:
            mode = PacingMode.ROUTINE_COMPRESSED
        else:
            mode = PacingMode.CALIBRATED_NOMINAL

        return InteroceptivePacingState(
            timestamp=0.0,
            objective_dt=dt,
            subjective_dt=subjective_dt,
            dilation_factor=dilation,
            arousal_level=composite_arousal,
            salience_density=salience,
            pacing_mode=mode,
            pulse_accumulator=self._pulse_accumulator,
        )

    def calibrate_deliberation_budget(
        self,
        base_rollouts: int,
        dilation_factor: float | None = None,
    ) -> int:
        """Calibrate mental simulation / search budget based on subjective time dilation.

        High subjective dilation grants expanded deliberation rollouts before action commitment.
        """
        factor = self._current_dilation if dilation_factor is None else float(dilation_factor)
        base = max(1, int(base_rollouts))
        scaled = int(math.ceil(base * factor))
        return max(1, scaled)

    def reset_accumulator(self) -> None:
        """Reset internal pulse accumulator."""
        self._pulse_accumulator = 0.0
