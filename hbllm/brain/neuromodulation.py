"""
Neuromodulation — global brain-state signals that modulate ALL subsystems.

Elevated from ``snn/neuromodulation.py`` to global ``brain/`` scope because
neuromodulation influences memory, planning, reasoning, and sleep — not
just the SNN.

Six neurotransmitter analogs::

    Dopamine        → Reward prediction error → learning rate, exploration
    Serotonin       → Stress / calm axis     → planning depth, patience
    Norepinephrine  → Novelty / alertness     → attention switch, interrupt
    Acetylcholine   → Focused attention       → encoding, selectivity
    GABA            → Global inhibition       → suppression, noise gating
    Glutamate       → Primary excitatory drive → signal amplification

Modulation domains::

    SNN:       excitatory_gain = weight × glutamate × acetylcholine
    Memory:    encoding_priority, retrieval_bias
    Planning:  planning_depth, exploration_rate
    Reasoning: reasoning_persistence
    Sleep:     consolidation_aggressiveness

Integration points::

    UserModelEngine  → stress_level   → serotonin
    RewardEvaluator  → reward score   → dopamine
    CuriosityNode    → novelty detect → norepinephrine
    AttentionManager → focus strength → acetylcholine
    SleepNode        → sleep phase    → GABA, ACh
    WorkspaceNode    → workload       → glutamate
    STDPRule         → reads dopamine  (learning rate)
    LIFNeuron        → reads serotonin (threshold)
    SaliencyEval     → reads NE + ACh  (attention)

Bus Topics::

    brain.neuromodulation.updated → Published when any modulator changes

Usage::

    from hbllm.brain.neuromodulation import (
        NeuromodulationEngine, NeuromodulatorState,
    )

    engine = NeuromodulationEngine()
    engine.signal_reward(0.9)
    engine.signal_attention_focus(0.8)

    # SNN gain computation
    gain = engine.get_excitatory_gain()  # w × glutamate × ACh
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Modulator State ──────────────────────────────────────────────────────────


@dataclass
class NeuromodulatorState:
    """Current levels of all six neuromodulators.

    Each value is in [0.0, 1.0] with 0.5 as the neutral baseline.

    Extended from the original 3-modulator design (dopamine, serotonin,
    norepinephrine) with acetylcholine, GABA, and glutamate for full
    cognitive modulation coverage.
    """

    dopamine: float = 0.5  # Reward → learning rate, exploration
    serotonin: float = 0.5  # Stress/calm → planning depth, patience
    norepinephrine: float = 0.5  # Novelty → attention switch, interrupt
    acetylcholine: float = 0.5  # Attention → encoding, selectivity
    gaba: float = 0.5  # Global inhibition
    glutamate: float = 0.5  # Primary excitatory drive

    # Decay tracking
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, float]:
        """Serialize all modulator levels."""
        return {
            "dopamine": round(self.dopamine, 4),
            "serotonin": round(self.serotonin, 4),
            "norepinephrine": round(self.norepinephrine, 4),
            "acetylcholine": round(self.acetylcholine, 4),
            "gaba": round(self.gaba, 4),
            "glutamate": round(self.glutamate, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuromodulatorState:
        """Deserialize from dict. Unknown keys are ignored."""
        return cls(
            dopamine=d.get("dopamine", 0.5),
            serotonin=d.get("serotonin", 0.5),
            norepinephrine=d.get("norepinephrine", 0.5),
            acetylcholine=d.get("acetylcholine", 0.5),
            gaba=d.get("gaba", 0.5),
            glutamate=d.get("glutamate", 0.5),
        )


# ── Engine ───────────────────────────────────────────────────────────────────


class NeuromodulationEngine:
    """Global neuromodulation state manager.

    Maintains six neuromodulator levels that decay toward baseline (0.5)
    over time, and provides scaling factors for all cognitive subsystems.

    Args:
        decay_half_life: Seconds for modulators to decay halfway to baseline.
            Default 30.0 (modulators return to neutral in ~2 minutes).
        dopamine_sensitivity: How strongly reward signals affect dopamine.
        serotonin_sensitivity: How strongly stress signals affect serotonin.
        norepinephrine_sensitivity: How strongly novelty signals affect NE.
        acetylcholine_sensitivity: How strongly focus signals affect ACh.
        gaba_sensitivity: How strongly inhibition signals affect GABA.
        glutamate_sensitivity: How strongly excitation signals affect glutamate.
    """

    def __init__(
        self,
        decay_half_life: float = 30.0,
        dopamine_sensitivity: float = 0.3,
        serotonin_sensitivity: float = 0.25,
        norepinephrine_sensitivity: float = 0.2,
        acetylcholine_sensitivity: float = 0.25,
        gaba_sensitivity: float = 0.2,
        glutamate_sensitivity: float = 0.2,
    ) -> None:
        self.state = NeuromodulatorState()
        self._decay_half_life = max(1.0, decay_half_life)
        self._dopamine_sens = dopamine_sensitivity
        self._serotonin_sens = serotonin_sensitivity
        self._ne_sens = norepinephrine_sensitivity
        self._ach_sens = acetylcholine_sensitivity
        self._gaba_sens = gaba_sensitivity
        self._glutamate_sens = glutamate_sensitivity

        # Telemetry
        self._signal_count = 0
        self._last_significant_change = 0.0

        logger.info(
            "NeuromodulationEngine initialized (decay_half_life=%.1fs, 6 modulators)",
            decay_half_life,
        )

    # ── Signal Inputs ────────────────────────────────────────────────

    def signal_reward(self, reward: float) -> None:
        """Signal a reward event to modulate dopamine.

        Args:
            reward: Reward magnitude in [-1.0, 1.0].
                Positive → dopamine rises (more learning).
                Negative → dopamine drops (less learning).
        """
        self._apply_decay()
        reward = max(-1.0, min(1.0, reward))
        delta = reward * self._dopamine_sens
        self.state.dopamine = max(0.0, min(1.0, self.state.dopamine + delta))
        self._signal_count += 1
        logger.debug("Neuromod: reward=%.2f → dopamine=%.3f", reward, self.state.dopamine)

    def signal_stress(self, stress_level: float) -> None:
        """Signal stress level to modulate serotonin.

        Args:
            stress_level: Stress intensity in [0.0, 1.0].
                High stress → serotonin drops (higher thresholds, more cautious).
                Low stress → serotonin rises (lower thresholds, more responsive).
        """
        self._apply_decay()
        # Invert: high stress → low serotonin
        target = 1.0 - stress_level
        delta = (target - self.state.serotonin) * self._serotonin_sens
        self.state.serotonin = max(0.0, min(1.0, self.state.serotonin + delta))
        self._signal_count += 1
        logger.debug("Neuromod: stress=%.2f → serotonin=%.3f", stress_level, self.state.serotonin)

    def signal_novelty(self, novelty: float) -> None:
        """Signal novelty detection to modulate norepinephrine.

        Args:
            novelty: Novelty intensity in [0.0, 1.0].
                High novelty → NE rises (sharper attention, faster reactions).
        """
        self._apply_decay()
        delta = (novelty - 0.5) * self._ne_sens * 2
        self.state.norepinephrine = max(0.0, min(1.0, self.state.norepinephrine + delta))
        self._signal_count += 1
        logger.debug(
            "Neuromod: novelty=%.2f → norepinephrine=%.3f",
            novelty,
            self.state.norepinephrine,
        )

    def signal_attention_focus(self, focus: float) -> None:
        """Signal attention focus level to modulate acetylcholine.

        Args:
            focus: Focus intensity in [0.0, 1.0].
                High focus → ACh rises (better encoding, sharper selectivity).
                Low focus → ACh drops (broader but shallower processing).
        """
        self._apply_decay()
        delta = (focus - 0.5) * self._ach_sens * 2
        self.state.acetylcholine = max(0.0, min(1.0, self.state.acetylcholine + delta))
        self._signal_count += 1
        logger.debug(
            "Neuromod: focus=%.2f → acetylcholine=%.3f",
            focus,
            self.state.acetylcholine,
        )

    def signal_inhibition(self, inhibition: float) -> None:
        """Signal global inhibition level to modulate GABA.

        Args:
            inhibition: Inhibition intensity in [0.0, 1.0].
                High inhibition → GABA rises (more suppression, noise gating).
        """
        self._apply_decay()
        delta = (inhibition - 0.5) * self._gaba_sens * 2
        self.state.gaba = max(0.0, min(1.0, self.state.gaba + delta))
        self._signal_count += 1
        logger.debug("Neuromod: inhibition=%.2f → gaba=%.3f", inhibition, self.state.gaba)

    def signal_excitation(self, excitation: float) -> None:
        """Signal excitatory drive to modulate glutamate.

        Args:
            excitation: Excitation intensity in [0.0, 1.0].
                High excitation → glutamate rises (stronger signals).
        """
        self._apply_decay()
        delta = (excitation - 0.5) * self._glutamate_sens * 2
        self.state.glutamate = max(0.0, min(1.0, self.state.glutamate + delta))
        self._signal_count += 1
        logger.debug(
            "Neuromod: excitation=%.2f → glutamate=%.3f",
            excitation,
            self.state.glutamate,
        )

    def signal_from_user_model(self, user_model: Any, tenant_id: str) -> None:
        """Bulk-update neuromodulators from UserModel state.

        Reads stress_level and engagement_level from the user model
        and converts them to neuromodulator signals.

        Args:
            user_model: UserModelEngine instance.
            tenant_id: Tenant to read model for.
        """
        try:
            model = user_model.get_model(tenant_id)
            self.signal_stress(model.stress_level)
            # High engagement → mild dopamine boost (engaged user = reward)
            if model.engagement_level > 0.6:
                self.signal_reward(0.3)
        except Exception as e:
            logger.debug("Failed to read UserModel for neuromodulation: %s", e)

    # ── Modulation Factors (read by ALL cognitive subsystems) ─────────

    # -- SNN factors --

    def get_learning_rate_factor(self) -> float:
        """Dopamine-modulated learning rate multiplier.

        Returns a factor in [0.5, 2.0]:
            dopamine=0.0 → 0.5× (very conservative learning)
            dopamine=0.5 → 1.0× (baseline)
            dopamine=1.0 → 2.0× (aggressive learning after reward)
        """
        self._apply_decay()
        return 0.5 + 1.5 * self.state.dopamine

    def get_threshold_factor(self) -> float:
        """Serotonin-modulated threshold multiplier.

        Returns a factor in [0.8, 1.2]:
            serotonin=0.0 → 1.2× (stressed → more cautious)
            serotonin=0.5 → 1.0× (baseline)
            serotonin=1.0 → 0.8× (calm → more responsive)
        """
        self._apply_decay()
        return 1.2 - 0.4 * self.state.serotonin

    def get_attention_factor(self) -> float:
        """Norepinephrine-modulated attention sensitivity multiplier.

        Returns a factor in [0.7, 1.5]:
            NE=0.0 → 0.7× (drowsy)
            NE=0.5 → 1.0× (baseline)
            NE=1.0 → 1.5× (hyper-alert)
        """
        self._apply_decay()
        return 0.7 + 0.8 * self.state.norepinephrine

    def get_excitatory_gain(self) -> float:
        """Combined excitatory gain: glutamate × acetylcholine.

        This is the core SNN gain computation::

            effective_weight = permanent_weight × excitatory_gain

        Returns a factor in [0.0, 2.25]:
            Both at 0.0 → 0.0 (complete suppression)
            Both at 0.5 → ~0.56 (baseline, slightly sub-1.0 by design)
            Both at 1.0 → 2.25 (maximum amplification)
        """
        self._apply_decay()
        # Scale each from [0,1] → [0, 1.5] so baseline (0.5) → 0.75
        glu_factor = self.state.glutamate * 1.5
        ach_factor = self.state.acetylcholine * 1.5
        return glu_factor * ach_factor

    def get_inhibition_factor(self) -> float:
        """GABA-modulated inhibition multiplier.

        Returns a factor in [0.5, 1.5]:
            GABA=0.0 → 0.5× (weak inhibition → noisy)
            GABA=0.5 → 1.0× (baseline)
            GABA=1.0 → 1.5× (strong inhibition → clean but slow)
        """
        self._apply_decay()
        return 0.5 + 1.0 * self.state.gaba

    # -- Memory factors --

    def get_encoding_priority(self) -> float:
        """ACh-modulated memory encoding priority.

        High ACh → prioritize encoding new memories.
        Low ACh → focus on retrieval instead.

        Returns a factor in [0.3, 1.0].
        """
        self._apply_decay()
        return 0.3 + 0.7 * self.state.acetylcholine

    def get_retrieval_bias(self) -> float:
        """Serotonin-modulated retrieval broadness.

        Low serotonin (stressed) → narrow, focused retrieval.
        High serotonin (calm) → broader, more associative retrieval.

        Returns a factor in [0.5, 1.5].
        """
        self._apply_decay()
        return 0.5 + 1.0 * self.state.serotonin

    # -- Planning factors --

    def get_planning_depth(self) -> float:
        """Serotonin-modulated planning depth.

        High serotonin (calm, patient) → deeper planning.
        Low serotonin (stressed) → shallow, reactive planning.

        Returns a factor in [0.5, 1.5].
        """
        self._apply_decay()
        return 0.5 + 1.0 * self.state.serotonin

    def get_exploration_rate(self) -> float:
        """Dopamine-modulated exploration rate.

        High dopamine (rewarded) → more exploration.
        Low dopamine → more exploitation of known strategies.

        Returns a factor in [0.3, 1.0].
        """
        self._apply_decay()
        return 0.3 + 0.7 * self.state.dopamine

    # -- Reasoning factors --

    def get_reasoning_persistence(self) -> float:
        """NE + serotonin modulated reasoning persistence.

        High NE + high serotonin → persistent, focused reasoning.
        Low values → quick, shallow reasoning.

        Returns a factor in [0.3, 1.5].
        """
        self._apply_decay()
        ne_contrib = self.state.norepinephrine * 0.6
        sero_contrib = self.state.serotonin * 0.6
        return 0.3 + ne_contrib + sero_contrib

    # -- Sleep factors --

    def get_consolidation_aggressiveness(self) -> float:
        """GABA-modulated sleep consolidation aggressiveness.

        High GABA during sleep → more aggressive memory consolidation.

        Returns a factor in [0.5, 2.0].
        """
        self._apply_decay()
        return 0.5 + 1.5 * self.state.gaba

    # ── Sleep-specific modulator presets ──────────────────────────────

    def enter_sleep_mode(self) -> None:
        """Shift modulators to sleep-appropriate levels.

        - ACh drops (stop encoding new memories)
        - GABA rises (strong inhibition for consolidation)
        - Serotonin rises (patience for consolidation)
        - NE drops (no attention interrupts)
        """
        self.state.acetylcholine = 0.1
        self.state.gaba = 0.9
        self.state.serotonin = 0.8
        self.state.norepinephrine = 0.1
        self.state.last_update = time.time()
        logger.info("Neuromod: entered sleep mode")

    def exit_sleep_mode(self) -> None:
        """Restore modulators to waking baseline."""
        self.state = NeuromodulatorState()
        logger.info("Neuromod: exited sleep mode → baseline")

    # ── Decay ────────────────────────────────────────────────────────

    def _apply_decay(self) -> None:
        """Decay all modulators toward baseline (0.5) based on elapsed time."""
        now = time.time()
        dt = now - self.state.last_update
        if dt < 0.1:
            return  # Skip sub-100ms updates

        self.state.last_update = now

        if dt <= 0:
            return

        # Exponential decay toward 0.5
        decay_factor = math.pow(2.0, -dt / self._decay_half_life)

        self.state.dopamine = 0.5 + (self.state.dopamine - 0.5) * decay_factor
        self.state.serotonin = 0.5 + (self.state.serotonin - 0.5) * decay_factor
        self.state.norepinephrine = 0.5 + (self.state.norepinephrine - 0.5) * decay_factor
        self.state.acetylcholine = 0.5 + (self.state.acetylcholine - 0.5) * decay_factor
        self.state.gaba = 0.5 + (self.state.gaba - 0.5) * decay_factor
        self.state.glutamate = 0.5 + (self.state.glutamate - 0.5) * decay_factor

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        self._apply_decay()
        return {
            "state": self.state.to_dict(),
            "factors": {
                "learning_rate": round(self.get_learning_rate_factor(), 3),
                "threshold": round(self.get_threshold_factor(), 3),
                "attention": round(self.get_attention_factor(), 3),
                "excitatory_gain": round(self.get_excitatory_gain(), 3),
                "inhibition": round(self.get_inhibition_factor(), 3),
                "encoding_priority": round(self.get_encoding_priority(), 3),
                "retrieval_bias": round(self.get_retrieval_bias(), 3),
                "planning_depth": round(self.get_planning_depth(), 3),
                "exploration_rate": round(self.get_exploration_rate(), 3),
                "reasoning_persistence": round(self.get_reasoning_persistence(), 3),
                "consolidation": round(self.get_consolidation_aggressiveness(), 3),
            },
            "signal_count": self._signal_count,
        }

    def reset(self) -> None:
        """Reset all modulators to baseline."""
        self.state = NeuromodulatorState()
        self._signal_count = 0
