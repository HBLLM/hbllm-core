"""Ventral Premotor (F5 / Mirror Neuron) Affordance Mimicry & Resonator.

Biologically grounded in primate ventral premotor cortex (area F5) mirror neuron
circuitry (Rizzolatti et al., 1996; Gallese et al., 1996; Fogassi et al., 2001):
1. Invariant Sensorimotor Resonance:
   - Area F5 discharges both when the agent performs a goal-directed action and
     when observing an external entity (patroller, dynamic block, mechanism)
     undergo an equivalent object interaction.
2. Vicarious Affordance Extraction:
   - When an external entity changes an environmental object's state (e.g. stepping
     on a pressure plate, opening a barrier, or deflecting off a boundary), mirror
     resonance maps the observed interaction vector into the agent's internal
     motor affordance inventory.
3. Rapid Mechanics Grounding:
   - Converts external dynamic observations into formal HCIR CausalPredicates
     without requiring the agent to spend execution steps or risk lethal failure
     in empirical trial-and-error.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class AffordanceResonancePattern:
    """An observed external interaction mapped onto internal motor affordances."""

    pattern_id: str
    observed_entity_id: str
    observed_entity_role: str
    target_entity_id: str
    target_role: str
    interaction_vector: tuple[int, int]
    inferred_action_type: str  # "interact", "push", "traverse", "collide"
    state_mutation: dict[str, Any]
    resonance_score: float  # [0, 1] confidence in functional equivalence

    def to_causal_predicate(self) -> CausalPredicate:
        """Export resonance pattern as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable=f"affordance_{self.target_role}",
            operator="==",
            value=f"action:{self.inferred_action_type}:res:{self.resonance_score:.2f}",
            relation="vicarious_mirror_resonance",
            target_entity=self.target_entity_id,
        )


@dataclass
class VicariousCausalInference:
    """Learned causal knowledge synthesized through mirror affordance resonance."""

    inference_id: str
    target_role: str
    inferred_affordance: str
    causal_predicate: CausalPredicate
    confidence: float
    support_count: int = 1


class VentralPremotorAffordanceResonator:
    """Mirror neuron affordance engine extracting vicarious causal mechanics from external interactions."""

    def __init__(
        self,
        resonance_threshold: float = 0.50,
        max_inferences: int = 100,
    ) -> None:
        """Initialize ventral premotor affordance resonator.

        Args:
            resonance_threshold: Minimum resonance score required to infer causal rule.
            max_inferences: Maximum number of vicarious inferences to store.
        """
        self.resonance_threshold = max(0.1, float(resonance_threshold))
        self.max_inferences = max(10, int(max_inferences))

        self.patterns: list[AffordanceResonancePattern] = []
        self.inferences: dict[str, VicariousCausalInference] = {}
        self.total_observations: int = 0

    def observe_interaction(
        self,
        entity_id: str,
        entity_role: str,
        initial_pos: tuple[int, int],
        final_pos: tuple[int, int],
        target_entity_id: str,
        target_role: str,
        state_mutation: dict[str, Any],
    ) -> AffordanceResonancePattern | None:
        """Process an observed interaction between an external entity and an object.

        Returns an AffordanceResonancePattern if the interaction exhibits functional resonance.
        """
        self.total_observations += 1
        dr = final_pos[0] - initial_pos[0]
        dc = final_pos[1] - initial_pos[1]
        dist = math.sqrt(dr * dr + dc * dc)

        # Classify inferred action type based on displacement and target role
        target_role_lower = str(target_role).lower()
        if not state_mutation:
            # Physical collision or boundary bounce
            inferred_action = "collide"
            resonance = 0.6 if dist <= 1.5 else 0.4
        elif any(k in target_role_lower for k in ("switch", "button", "lever", "door", "barrier")):
            inferred_action = "interact"
            resonance = 0.85 if dist <= 1.5 else 0.65
        elif any(k in target_role_lower for k in ("cargo", "box", "block")):
            inferred_action = "push"
            resonance = 0.80
        else:
            inferred_action = "traverse"
            resonance = 0.50

        if resonance < self.resonance_threshold:
            return None

        pattern = AffordanceResonancePattern(
            pattern_id=f"res_{uuid.uuid4().hex[:8]}",
            observed_entity_id=entity_id,
            observed_entity_role=entity_role,
            target_entity_id=target_entity_id,
            target_role=target_role,
            interaction_vector=(dr, dc),
            inferred_action_type=inferred_action,
            state_mutation=dict(state_mutation),
            resonance_score=float(resonance),
        )
        self.patterns.append(pattern)
        if len(self.patterns) > self.max_inferences:
            self.patterns.pop(0)

        # Synthesize causal inference
        self.synthesize_causal_rule(pattern)
        return pattern

    def synthesize_causal_rule(
        self,
        pattern: AffordanceResonancePattern,
    ) -> VicariousCausalInference:
        """Synthesize a persistent vicarious causal rule from an affordance pattern."""
        key = f"{pattern.target_role}_{pattern.inferred_action_type}"
        pred = pattern.to_causal_predicate()

        if key in self.inferences:
            inf = self.inferences[key]
            inf.support_count += 1
            inf.confidence = min(1.0, inf.confidence * 0.9 + pattern.resonance_score * 0.1)
            inf.causal_predicate = pred
            return inf

        inf = VicariousCausalInference(
            inference_id=f"inf_{uuid.uuid4().hex[:8]}",
            target_role=pattern.target_role,
            inferred_affordance=pattern.inferred_action_type,
            causal_predicate=pred,
            confidence=pattern.resonance_score,
            support_count=1,
        )
        self.inferences[key] = inf
        return inf

    def get_vicarious_predicates(self) -> list[CausalPredicate]:
        """Retrieve all currently synthesized vicarious causal predicates."""
        return [inf.causal_predicate for inf in self.inferences.values()]

    def reset(self) -> None:
        """Reset internal pattern and inference buffers."""
        self.patterns.clear()
        self.inferences.clear()
        self.total_observations = 0
