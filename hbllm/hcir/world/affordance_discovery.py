"""Functional Affordance Discovery Engine for HCIR World Kernel.

Enables an agent to discover functional action affordances across perceptual categories
through active sensorimotor experimentation, strict falsification, and schema induction.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    BeliefTransitionType,
)

logger = logging.getLogger(__name__)


@dataclass
class AffordanceHypothesis:
    """Hypothesis for object-action functional affordance."""

    hypothesis_id: str = field(default_factory=lambda: f"aff_{uuid.uuid4().hex[:6]}")
    action: Any = "ROLL"
    entity_shape: str = "ball"
    affordance_label: str = "ROLLABLE"
    confidence: float = 0.5
    interventions_tested: int = 0
    falsified: bool = False
    confirmed: bool = False
    supporting_episodes: list[str] = field(default_factory=list)
    counterexamples: list[str] = field(default_factory=list)

    def describe(self) -> str:
        act_val = self.action.value if hasattr(self.action, "value") else str(self.action)
        return f"AFFORDS({self.entity_shape}, {act_val}) => {self.affordance_label}"


class BaseAffordanceDiscoveryEngine:
    """Domain-agnostic functional affordance discovery engine."""

    def __init__(self) -> None:
        self.hypotheses: list[AffordanceHypothesis] = []
        self.confirmed_affordances: dict[
            str, list[str]
        ] = {}  # category -> list of affordance labels
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0

    def generate_hypotheses(
        self,
        categories: Sequence[str],
        action_affordance_pairs: Sequence[tuple[Any, str]],
    ) -> list[AffordanceHypothesis]:
        """Formulate candidate affordance hypotheses across observed categories and actions."""
        existing_pairs = {(h.entity_shape, h.action) for h in self.hypotheses}
        for cat in sorted(list(set(categories))):
            for action, label in action_affordance_pairs:
                if (cat, action) in existing_pairs:
                    continue
                hyp = AffordanceHypothesis(
                    action=action,
                    entity_shape=cat,
                    affordance_label=label,
                    confidence=0.5,
                )
                self.hypotheses.append(hyp)
                existing_pairs.add((cat, action))
                self._record_belief_event(
                    event_type=BeliefTransitionType.HYPOTHESIS_CREATED,
                    hyp=hyp,
                    prior_conf=0.0,
                    post_conf=0.5,
                )

        return self.hypotheses

    def update_affordance_from_evidence(
        self,
        hypothesis: AffordanceHypothesis,
        success: bool,
        target_id: str,
        step_index: int | None = None,
        target_store: dict[str, list[str]] | None = None,
    ) -> BeliefTransitionEvent:
        """Update belief posterior for an affordance hypothesis on interventional outcome."""
        hypothesis.interventions_tested += 1
        prior_conf = hypothesis.confidence
        step = self.interventions_count if step_index is None else step_index

        if success:
            hypothesis.confirmed = True
            hypothesis.confidence = 1.0
            hypothesis.supporting_episodes.append(target_id)
            # Register in self.confirmed_affordances (prevent duplicates)
            aff_list = self.confirmed_affordances.setdefault(hypothesis.entity_shape, [])
            if hypothesis.affordance_label not in aff_list:
                aff_list.append(hypothesis.affordance_label)
            # Register in external target_store if provided
            if target_store is not None:
                sub_list = target_store.setdefault(hypothesis.entity_shape, [])
                if hypothesis.affordance_label not in sub_list:
                    sub_list.append(hypothesis.affordance_label)
            event = self._record_belief_event(
                event_type=BeliefTransitionType.HYPOTHESIS_CONFIRMED,
                hyp=hypothesis,
                prior_conf=prior_conf,
                post_conf=1.0,
                step_index=step,
            )
            logger.info(
                "Affordance confirmed: AFFORDS(%s, %s) => %s",
                hypothesis.entity_shape,
                hypothesis.action,
                hypothesis.affordance_label,
            )
        else:
            hypothesis.falsified = True
            hypothesis.confidence = 0.0
            hypothesis.counterexamples.append(target_id)
            event = self._record_belief_event(
                event_type=BeliefTransitionType.HYPOTHESIS_FALSIFIED,
                hyp=hypothesis,
                prior_conf=prior_conf,
                post_conf=0.0,
                step_index=step,
            )
            logger.info(
                "Affordance falsified: AFFORDS(%s, %s) =/=> %s (counterexample: %s)",
                hypothesis.entity_shape,
                hypothesis.action,
                hypothesis.affordance_label,
                target_id,
            )

        return event

    @staticmethod
    def test_novel_entity_transfer(
        held_out_entities: Sequence[dict[str, Any]],
        confirmed_affordances: dict[str, list[str]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Test acquired affordance transfer on held-out unseen shapes/objects."""
        if not confirmed_affordances:
            return 0.0, []

        correct = 0
        total = 0
        eval_records: list[dict[str, Any]] = []

        for entity in held_out_entities:
            shape = entity.get("shape", "")
            base_shape = entity.get("base_shape", shape)
            expected_affordances = set(entity.get("ground_truth_affordances", []))

            # Retrieve predicted affordances based on acquired shape schema
            predicted_affordances = set(confirmed_affordances.get(base_shape, []))

            # Check precision & recall match
            is_match = predicted_affordances == expected_affordances
            if is_match:
                correct += 1
            total += 1

            eval_records.append(
                {
                    "entity_id": entity.get("id"),
                    "shape": shape,
                    "predicted_affordances": sorted(list(predicted_affordances)),
                    "expected_affordances": sorted(list(expected_affordances)),
                    "correct": is_match,
                }
            )

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, eval_records

    def evaluate_novel_entity_transfer(
        self,
        held_out_entities: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Evaluate novel entity transfer using engine's confirmed affordances."""
        return self.test_novel_entity_transfer(held_out_entities, self.confirmed_affordances)

    def _record_belief_event(
        self,
        event_type: BeliefTransitionType,
        hyp: AffordanceHypothesis,
        prior_conf: float,
        post_conf: float,
        step_index: int | None = None,
    ) -> BeliefTransitionEvent:
        step = self.interventions_count if step_index is None else step_index
        event = BeliefTransitionEvent(
            event_type=event_type,
            step_index=step,
            hypothesis_id=hyp.hypothesis_id,
            variable=hyp.entity_shape,
            condition=hyp.describe(),
            prior_confidence=prior_conf,
            posterior_confidence=post_conf,
            is_falsified=hyp.falsified,
        )
        self.belief_history.append(event)
        return event
