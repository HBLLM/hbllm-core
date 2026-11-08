"""Claustrum Dynamic Inter-Hemispheric Synchrony & Cross-Modal Gestalt Binding.

Biological Foundations:
- Crick & Koch (2005): "What is the function of the claustrum?" (Philos Trans R Soc
  Lond B Biol Sci 360: 1271-1279) — The Conductor Hypothesis.
- Smythies, Edelstein, Ramachandran (2012): "Hypotheses relating to the function of the
  claustrum" (Front Integr Neurosci 6: 53).
- Reser et al. (2017): "Claustrum and cortical synchrony: A review".

Key Cognitive Capabilities:
1. Conductor Gestalt Binding:
   - Integrates disparate multimodal sensory streams (spatial coordinates, tactile
     friction, somatic affect, temporal intervals, threat signals, rule status) into a
     temporally synchronous unified perceptual frame.
2. Inter-Hemispheric Coordination:
   - Harmonizes left-hemisphere symbolic, categorical, and rule-based processing with
     right-hemisphere holistic, spatial, and threat-vigilant representations.
3. Sensory Uncoupling & Attentional Gating:
   - Dynamically decouples peripheral or discordant sensory channels to protect the
     primary cognitive problem-solving loop from cross-modal interference.
4. Cross-Modal Coherence Metric (Phi):
   - Computes gestalt harmony versus sensory dissonance. When dissonance spikes (e.g.
     spatial path clear but somatic alarm active), Phi drops, triggering cortical verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass(frozen=True)
class CrossModalSynchronyPacket:
    """Synchronized cross-modal gestalt snapshot emitted by the claustrum."""

    timestamp: float
    gestalt_coherence: float
    bound_modalities: tuple[str, ...]
    coupled_channels: tuple[str, ...]
    interhemispheric_balance: float
    discordant_modalities: tuple[str, ...]

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert synchrony packet into an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="claustrum_gestalt_synchrony",
            operator="==",
            value=f"coherence:{self.gestalt_coherence:.2f}:bound:{len(self.bound_modalities)}:balance:{self.interhemispheric_balance:.2f}:discordant:{len(self.discordant_modalities)}",
            relation="claustrum_interhemispheric_binding",
            target_entity="claustrum",
        )


@dataclass
class InterHemisphericBindingState:
    """Ongoing dynamic state of claustrum inter-hemispheric binding."""

    coherence_history: list[float] = field(default_factory=list)
    active_channels: set[str] = field(
        default_factory=lambda: {"spatial", "somatic", "haptic", "temporal", "threat", "rule"}
    )
    decoupled_channels: set[str] = field(default_factory=set)
    gestalt_stable: bool = True


class ClaustrumInterhemisphericIntegrator:
    """Orchestrator for claustrum cross-modal binding and inter-hemispheric synchrony."""

    def __init__(self, coherence_threshold: float = 0.50, max_history: int = 50) -> None:
        """Initialize claustrum integrator.

        Args:
            coherence_threshold: Threshold below which gestalt is considered dissonant.
            max_history: Max number of history items preserved for coherence tracking.
        """
        self.coherence_threshold: float = max(0.1, min(0.9, float(coherence_threshold)))
        self.max_history: int = max(10, int(max_history))
        self._state: InterHemisphericBindingState = InterHemisphericBindingState()

    def couple_channel(self, channel_name: str) -> None:
        """Couple a sensory or cognitive channel into active gestalt binding."""
        cname = str(channel_name).strip().lower()
        self._state.active_channels.add(cname)
        self._state.decoupled_channels.discard(cname)

    def decouple_channel(self, channel_name: str) -> None:
        """Decouple a channel to filter distraction or severe noise."""
        cname = str(channel_name).strip().lower()
        self._state.decoupled_channels.add(cname)
        self._state.active_channels.discard(cname)

    def bind_cross_modal_streams(
        self,
        timestamp: float,
        spatial_state: dict[str, Any] | None = None,
        somatic_valence: float = 0.0,
        haptic_friction: float = 1.0,
        threat_vigilance: float = 0.0,
        temporal_progress: float = 0.0,
        rule_active: bool = False,
    ) -> CrossModalSynchronyPacket:
        """Bind multimodal streams into a unified cross-modal gestalt.

        Args:
            timestamp: Operational step / time index.
            spatial_state: High-level spatial perception dictionary (e.g. obstacles, target distance).
            somatic_valence: Gut-feeling affective valence in [-1.0, 1.0].
            haptic_friction: Tactile friction coefficient in [0.0, 1.0].
            threat_vigilance: Amygdalar threat level in [0.0, 1.0].
            temporal_progress: CA1 time cell interval progress in [0.0, 1.0].
            rule_active: Whether a dlPFC / vlPFC symbolic rule is actively constraining actions.

        Returns:
            CrossModalSynchronyPacket containing coherence score, channels, balance, and discordance.
        """
        bound: list[str] = []
        discordant: list[str] = []

        # 1. Evaluate Channel Inclusions
        if "spatial" in self._state.active_channels and spatial_state is not None:
            bound.append("spatial")
        if "somatic" in self._state.active_channels:
            bound.append("somatic")
        if "haptic" in self._state.active_channels:
            bound.append("haptic")
        if "threat" in self._state.active_channels:
            bound.append("threat")
        if "temporal" in self._state.active_channels:
            bound.append("temporal")
        if "rule" in self._state.active_channels:
            bound.append("rule")

        # 2. Check Cross-Modal Harmony vs Discordance
        # Discordance A: Spatial open / optimistic vs severe negative somatic alarm
        is_spatial_open = spatial_state.get("is_clear", True) if spatial_state else True
        if "spatial" in bound and "somatic" in bound:
            if is_spatial_open and somatic_valence <= -0.50:
                discordant.append("spatial_vs_somatic_alarm")

        # Discordance B: High speed / progression vs dangerously low friction (ice slip)
        if "haptic" in bound and haptic_friction < 0.20 and temporal_progress > 0.30:
            discordant.append("friction_slip_under_progress")

        # Discordance C: High threat vigilance with neutral / positive somatic valence
        if "threat" in bound and "somatic" in bound:
            if threat_vigilance > 0.70 and somatic_valence > 0.20:
                discordant.append("threat_vs_somatic_mismatch")

        # 3. Compute Gestalt Coherence Metric (Phi in [0.0, 1.0])
        penalty_per_discordance = 0.25
        base_coherence = 1.0 - (len(discordant) * penalty_per_discordance)
        # Down-modulate slightly if critical channels are decoupled
        if len(self._state.decoupled_channels) > 0:
            base_coherence -= 0.05 * len(self._state.decoupled_channels)
        coherence = float(max(0.05, min(1.0, base_coherence)))

        # 4. Compute Inter-Hemispheric Balance
        # Left hemisphere: Symbolic rules & precise temporal progress
        # Right hemisphere: Spatial holistic field & somatic / threat vigilance
        left_load = (1.0 if rule_active else 0.2) + temporal_progress * 0.8
        right_load = (
            (1.0 if not is_spatial_open else 0.3) + max(0.0, -somatic_valence) + threat_vigilance
        )
        total_load = left_load + right_load
        balance = float(
            max(-1.0, min(1.0, (right_load - left_load) / total_load)) if total_load > 1e-6 else 0.0
        )

        # 5. Update Internal State History
        self._state.coherence_history.append(coherence)
        if len(self._state.coherence_history) > self.max_history:
            self._state.coherence_history.pop(0)
        self._state.gestalt_stable = coherence >= self.coherence_threshold

        return CrossModalSynchronyPacket(
            timestamp=float(timestamp),
            gestalt_coherence=coherence,
            bound_modalities=tuple(bound),
            coupled_channels=tuple(sorted(list(self._state.active_channels))),
            interhemispheric_balance=balance,
            discordant_modalities=tuple(discordant),
        )

    def get_binding_state(self) -> InterHemisphericBindingState:
        """Retrieve current internal inter-hemispheric binding state."""
        return self._state
