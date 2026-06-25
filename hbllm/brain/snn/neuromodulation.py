"""
Neuromodulation — global brain-state signals that modulate SNN behavior.

Biological brains use neurotransmitters (dopamine, serotonin, norepinephrine)
to globally modulate learning, alertness, and mood.  This module provides
their computational analogs:

    Dopamine        → Reward prediction error → modulates STDP learning rate
    Serotonin       → Stress / calm axis → modulates LIF threshold globally
    Norepinephrine  → Novelty / alertness → modulates attention sensitivity

Each modulator is a float in [0.0, 1.0]:
    - 0.5 = baseline (neutral)
    - > 0.5 = elevated (e.g., high reward, high novelty, calm)
    - < 0.5 = depressed (e.g., punishment, stress, fatigue)

Integration points:
    UserModelEngine  → stress_level → serotonin
    RewardEvaluator  → reward score → dopamine
    CuriosityNode    → novelty detection → norepinephrine
    STDPRule         → reads dopamine to modulate learning rate
    LIFNeuron        → reads serotonin to modulate threshold
    AttentionGate    → reads norepinephrine to modulate sensitivity

Bus Topics:
    snn.neuromodulation.updated → Published when any modulator changes significantly

Usage::

    modulator = NeuromodulationEngine()
    modulator.signal_reward(0.9)   # High reward → dopamine spike
    modulator.signal_stress(0.8)   # High stress → serotonin drops
    modulator.signal_novelty(0.7)  # Novel input → norepinephrine rises

    # Read modulation factors for SNN components
    lr_factor = modulator.get_learning_rate_factor()   # 0.5–2.0
    thresh_factor = modulator.get_threshold_factor()    # 0.8–1.2
    attn_factor = modulator.get_attention_factor()      # 0.7–1.5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Modulator State ──────────────────────────────────────────────────────────


@dataclass
class NeuromodulatorState:
    """Current levels of all neuromodulators.

    Each value is in [0.0, 1.0] with 0.5 as the neutral baseline.
    """

    dopamine: float = 0.5  # Reward signal → learning rate
    serotonin: float = 0.5  # Stress/calm → threshold modulation
    norepinephrine: float = 0.5  # Novelty/alertness → attention sensitivity

    # Decay tracking
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, float]:
        return {
            "dopamine": round(self.dopamine, 4),
            "serotonin": round(self.serotonin, 4),
            "norepinephrine": round(self.norepinephrine, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuromodulatorState:
        return cls(
            dopamine=d.get("dopamine", 0.5),
            serotonin=d.get("serotonin", 0.5),
            norepinephrine=d.get("norepinephrine", 0.5),
        )


# ── Engine ───────────────────────────────────────────────────────────────────


class NeuromodulationEngine:
    """Global neuromodulation state manager.

    Maintains three neuromodulator levels that decay toward baseline (0.5)
    over time, and provides scaling factors for SNN components.

    Args:
        decay_half_life: Seconds for modulators to decay halfway to baseline.
            Default 30.0 (modulators return to neutral in ~2 minutes).
        dopamine_sensitivity: How strongly reward signals affect dopamine.
        serotonin_sensitivity: How strongly stress signals affect serotonin.
        norepinephrine_sensitivity: How strongly novelty signals affect NE.
    """

    def __init__(
        self,
        decay_half_life: float = 30.0,
        dopamine_sensitivity: float = 0.3,
        serotonin_sensitivity: float = 0.25,
        norepinephrine_sensitivity: float = 0.2,
    ) -> None:
        self.state = NeuromodulatorState()
        self._decay_half_life = max(1.0, decay_half_life)
        self._dopamine_sens = dopamine_sensitivity
        self._serotonin_sens = serotonin_sensitivity
        self._ne_sens = norepinephrine_sensitivity

        # Telemetry
        self._signal_count = 0
        self._last_significant_change = 0.0

        logger.info(
            "NeuromodulationEngine initialized (decay_half_life=%.1fs)",
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

    # ── Modulation Factors (read by SNN components) ──────────────────

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
            serotonin=0.0 → 1.2× (stressed → higher threshold → more cautious)
            serotonin=0.5 → 1.0× (baseline)
            serotonin=1.0 → 0.8× (calm → lower threshold → more responsive)
        """
        self._apply_decay()
        return 1.2 - 0.4 * self.state.serotonin

    def get_attention_factor(self) -> float:
        """Norepinephrine-modulated attention sensitivity multiplier.

        Returns a factor in [0.7, 1.5]:
            NE=0.0 → 0.7× (drowsy → less sensitive)
            NE=0.5 → 1.0× (baseline)
            NE=1.0 → 1.5× (alert → hypersensitive)
        """
        self._apply_decay()
        return 0.7 + 0.8 * self.state.norepinephrine

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
        decay_factor = 2.0 ** (-dt / self._decay_half_life)

        self.state.dopamine = 0.5 + (self.state.dopamine - 0.5) * decay_factor
        self.state.serotonin = 0.5 + (self.state.serotonin - 0.5) * decay_factor
        self.state.norepinephrine = 0.5 + (self.state.norepinephrine - 0.5) * decay_factor

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
            },
            "signal_count": self._signal_count,
        }

    def reset(self) -> None:
        """Reset all modulators to baseline."""
        self.state = NeuromodulatorState()
        self._signal_count = 0
