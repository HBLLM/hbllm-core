"""Ventromedial Prefrontal Cortex (vmPFC) Somatic Marker Guidance.

Biologically grounded in Antonio Damasio's Somatic Marker Hypothesis
(Damasio, 1994, 1996; Bechara et al., 1999; Roy et al., 2012):
1. Visceral Value Integration:
   - vmPFC binds cognitive representations of state trajectories and subgoals to
     affective somatic markers (visceral "gut feelings"):
     Valence S in [-1.0, +1.0], Visceral Arousal A in [0.0, 1.0].
2. Rapid Pre-Deliberative Bias:
   - When candidate action trajectories lead into state patterns associated with
     prior catastrophic resets, lethal traps, or severe penalties, vmPFC activates
     a visceral alarm, depressing or inhibiting the action before forward MCTS
     expends cognitive deliberation effort.
3. Somatic Appetite Elevation:
   - Trajectories leading to high reward or novel breakthroughs receive positive
     affective bias, guiding active exploration toward promising corridors.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class SomaticMarker:
    """An affective somatic tag associated with a state signature."""

    marker_id: str
    state_hash: int
    valence: float  # [-1.0, +1.0] (negative = catastrophic, positive = appetitive)
    arousal: float = 0.5  # [0.0, 1.0] intensity of affective reaction
    is_visceral_alarm: bool = False
    visit_count: int = 1


@dataclass
class SomaticAffectiveBias:
    """The pre-deliberative bias landscape emitted by vmPFC for candidate actions."""

    action_biases: dict[Any, float]  # Additive somatic bias for each candidate action
    inhibited_actions: set[Any]  # Actions blocked by visceral alarms (S < -0.4)
    alarm_active: bool
    gut_feeling: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Export somatic affective bias as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="somatic_affective_bias",
            operator="==",
            value=f"alarm:{self.alarm_active}:feeling:{self.gut_feeling}",
            relation="vmpfc_somatic_marker_guidance",
            target_entity="prefrontal_cortex",
        )


class VentromedialSomaticMarkerEngine:
    """vmPFC executive controller evaluating somatic affective markers and biasing decision-making."""

    def __init__(
        self,
        alarm_threshold: float = -0.40,
        learning_rate: float = 0.20,
        max_markers: int = 2000,
    ) -> None:
        """Initialize ventromedial somatic marker engine.

        Args:
            alarm_threshold: Valence below which an action triggers visceral alarm inhibition.
            learning_rate: Incremental update rate for visceral marker valences.
            max_markers: Maximum number of distinct state signatures to retain.
        """
        self.alarm_threshold = float(alarm_threshold)
        self.learning_rate = max(0.01, float(learning_rate))
        self.max_markers = max(50, int(max_markers))

        self.markers: dict[int, SomaticMarker] = {}
        self.total_evaluations: int = 0

    def record_somatic_marker(
        self,
        state_hash: int,
        outcome_valence: float,
        arousal: float = 0.5,
    ) -> SomaticMarker:
        """Associate an environmental state signature with a somatic affective outcome."""
        clamped_val = max(-1.0, min(1.0, float(outcome_valence)))
        clamped_arousal = max(0.0, min(1.0, float(arousal)))
        is_alarm = clamped_val <= self.alarm_threshold

        if state_hash in self.markers:
            m = self.markers[state_hash]
            m.visit_count += 1
            # Exponential moving average update
            m.valence = (1.0 - self.learning_rate) * m.valence + self.learning_rate * clamped_val
            m.arousal = (
                1.0 - self.learning_rate
            ) * m.arousal + self.learning_rate * clamped_arousal
            m.is_visceral_alarm = m.valence <= self.alarm_threshold
            return m

        if len(self.markers) >= self.max_markers:
            # Evict marker with least visits
            least_key = min(self.markers.keys(), key=lambda k: self.markers[k].visit_count)
            self.markers.pop(least_key, None)

        marker = SomaticMarker(
            marker_id=f"sm_{uuid.uuid4().hex[:8]}",
            state_hash=state_hash,
            valence=clamped_val,
            arousal=clamped_arousal,
            is_visceral_alarm=is_alarm,
            visit_count=1,
        )
        self.markers[state_hash] = marker
        return marker

    def evaluate_somatic_bias(
        self,
        candidate_actions: list[Any],
        projected_state_hashes: dict[Any, int],
    ) -> SomaticAffectiveBias:
        """Evaluate somatic markers for projected action outcomes to bias deliberation.

        Args:
            candidate_actions: List of possible motor actions available.
            projected_state_hashes: Map of action -> predicted next state signature hash.

        Returns:
            SomaticAffectiveBias indicating action biases and visceral inhibitions.
        """
        self.total_evaluations += 1
        action_biases: dict[Any, float] = {}
        inhibited: set[Any] = set()
        any_alarm = False

        for act in candidate_actions:
            shash = projected_state_hashes.get(act)
            if shash is not None and shash in self.markers:
                marker = self.markers[shash]
                action_biases[act] = marker.valence * (0.5 + 0.5 * marker.arousal)
                if marker.is_visceral_alarm:
                    inhibited.add(act)
                    any_alarm = True
            else:
                action_biases[act] = 0.0  # Neutral somatic state

        # Determine qualitative gut feeling
        if any_alarm:
            gut_feeling = "visceral_alarm_threat_detected"
        elif any(v > 0.3 for v in action_biases.values()):
            gut_feeling = "appetitive_somatic_encouragement"
        else:
            gut_feeling = "neutral_equilibrium"

        return SomaticAffectiveBias(
            action_biases=action_biases,
            inhibited_actions=inhibited,
            alarm_active=any_alarm,
            gut_feeling=gut_feeling,
        )

    def reset(self) -> None:
        """Clear all stored somatic markers."""
        self.markers.clear()
        self.total_evaluations = 0
