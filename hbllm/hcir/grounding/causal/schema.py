from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.causal.events import EventCandidate
from hbllm.hcir.grounding.contracts.enums import CausalStatus
from hbllm.hcir.grounding.contracts.references import GroundedRef


@dataclass
class CausalSchema:
    """A causal hypothesis or rule structured along Pearl's Three-Tier Hierarchy.

    Upholds Invariant 3 & SGE Section 10:
    Observational correlations P(E | A) are strictly separated from interventional
    invariants P(E | do(A)). Spurious confounders are prevented by requiring active
    intervention before counterfactual planning.
    """

    schema_id: str
    action_type: str
    target_role: str
    precondition: dict[str, Any] = field(default_factory=dict)
    expected_effect: dict[str, Any] = field(default_factory=dict)
    causal_status: CausalStatus = CausalStatus.OBSERVED_ASSOCIATION
    observational_co_occurrences: int = 1
    observational_trials: int = 1
    intervention_successes: int = 0
    intervention_trials: int = 0
    p_effect_given_action: float = 0.5
    p_effect_given_do_action: float = 0.0
    confounder_risk: float = 0.5
    counterfactual_accuracy: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def is_actionable_cause(self) -> bool:
        """True if hypothesis has achieved at least Rung 2 interventional support or identification."""
        return self.causal_status in (
            CausalStatus.CAUSAL_EFFECT_IDENTIFIED,
            CausalStatus.INTERVENTIONALLY_SUPPORTED,
            CausalStatus.COUNTERFACTUAL_MODEL_AVAILABLE,
            CausalStatus.CAUSALLY_IDENTIFIED,
        )


@dataclass(frozen=True)
class InterventionExperimentProposal:
    """A targeted experimental action proposed to establish P(E | do(A)) on Pearl's Ladder.

    Ensures that intervention experiments are themselves proposed as first-class actions
    that must pass through planning and the ActionValidationGateway before execution.
    """

    proposal_id: str
    schema_id: str
    target_entity_ref: GroundedRef
    action_intent: str
    action_parameters: dict[str, Any] = field(default_factory=dict)
    confounder_controls: dict[str, Any] = field(default_factory=dict)
    expected_outcome: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_symbolic_intent(self) -> Any:
        """Translates experimental intervention proposal into a SymbolicActionIntent for planner/gateway."""
        from hbllm.hcir.grounding.contracts.actions import SymbolicActionIntent

        return SymbolicActionIntent(
            intent_id=f"intent_exp_{self.proposal_id}",
            action_type=self.action_intent,
            target_ref=self.target_entity_ref,
            parameters=self.action_parameters,
            preconditions=tuple(f"{k} == {v}" for k, v in self.confounder_controls.items()),
            expected_outcomes=tuple(f"{k} == {v}" for k, v in self.expected_outcome.items()),
            context={"schema_id": self.schema_id, "is_intervention_experiment": True},
        )


class SchemaInducer:
    """Learns causal schemas through observation, hypothesis generation, and intervention.

    Governs transition across Pearl's causal hierarchy:
    1. Association: P(B | A)
    2. Predictive non-causal: P(B | A) forward-temporal
    3. Causal hypothesis: formulated for intervention testing
    4. Interventionally supported: P(B | do(A)) experimentally measured
    5. Causally identified: structural causal model counterfactually validated
    """

    def __init__(
        self,
        min_association_prob: float = 0.6,
        min_intervention_prob: float = 0.75,
        min_intervention_trials: int = 2,
    ) -> None:
        self._min_assoc_prob = min_association_prob
        self._min_interv_prob = min_intervention_prob
        self._min_interv_trials = min_intervention_trials
        self._schemas: dict[str, CausalSchema] = {}
        self._active_experiments: dict[str, InterventionExperimentProposal] = {}

    def record_observation(
        self,
        action_type: str,
        target_role: str,
        precondition: dict[str, Any],
        outcome_event: EventCandidate | None = None,
    ) -> CausalSchema:
        """Records an observational action-outcome pairing (Pearl Rung 1)."""
        schema_key = f"{action_type}_{target_role}"

        if schema_key not in self._schemas:
            schema = CausalSchema(
                schema_id=f"schema_{uuid.uuid4().hex[:8]}",
                action_type=action_type,
                target_role=target_role,
                precondition=dict(precondition),
                expected_effect=(
                    {
                        "event_type": outcome_event.event_type.value,
                        "post_state": outcome_event.post_state,
                    }
                    if outcome_event
                    else {}
                ),
                causal_status=CausalStatus.OBSERVED_ASSOCIATION,
                observational_co_occurrences=1 if outcome_event else 0,
                observational_trials=1,
            )
            self._schemas[schema_key] = schema
        else:
            schema = self._schemas[schema_key]
            schema.observational_trials += 1
            if outcome_event:
                schema.observational_co_occurrences += 1

        schema.p_effect_given_action = schema.observational_co_occurrences / max(
            1, schema.observational_trials
        )

        # Promote to PREDICTIVE_NON_CAUSAL if recurring reliably in observation
        if (
            schema.causal_status == CausalStatus.OBSERVED_ASSOCIATION
            and schema.observational_trials >= 3
            and schema.p_effect_given_action >= self._min_assoc_prob
        ):
            schema.causal_status = CausalStatus.PREDICTIVE_NON_CAUSAL

        return schema

    def propose_intervention(
        self,
        schema_id: str,
        target_entity_ref: GroundedRef,
        action_parameters: dict[str, Any] | None = None,
    ) -> InterventionExperimentProposal:
        """Formulates a targeted experiment to test P(E | do(A))."""
        schema = self._get_schema_by_id(schema_id)
        if not schema:
            raise KeyError(f"CausalSchema {schema_id} not found")

        # Advance status to CAUSAL_HYPOTHESIS awaiting experiment
        if schema.causal_status in (
            CausalStatus.OBSERVED_ASSOCIATION,
            CausalStatus.PREDICTIVE_NON_CAUSAL,
        ):
            schema.causal_status = CausalStatus.CAUSAL_HYPOTHESIS

        exp_id = f"exp_{uuid.uuid4().hex[:8]}"
        proposal = InterventionExperimentProposal(
            proposal_id=exp_id,
            schema_id=schema.schema_id,
            target_entity_ref=target_entity_ref,
            action_intent=schema.action_type,
            action_parameters=action_parameters or {},
            confounder_controls={"isolate_nearby_agents": True},
            expected_outcome=schema.expected_effect,
            rationale=f"Test intervention do({schema.action_type}) on {target_entity_ref.id}",
        )

        self._active_experiments[exp_id] = proposal
        return proposal

    def record_intervention_outcome(
        self,
        experiment_id: str,
        success: bool,
        observed_event: EventCandidate | None = None,
    ) -> CausalStatus:
        """Updates P(E | do(A)) following execution of an intervention experiment (Pearl Rung 2)."""
        proposal = self._active_experiments.pop(experiment_id, None)
        if not proposal:
            raise KeyError(f"InterventionExperimentProposal {experiment_id} not active")

        schema = self._get_schema_by_id(proposal.schema_id)
        if not schema:
            raise KeyError(f"Schema {proposal.schema_id} not found")

        schema.intervention_trials += 1
        if success:
            schema.intervention_successes += 1

        schema.p_effect_given_do_action = schema.intervention_successes / max(
            1, schema.intervention_trials
        )

        # Evaluate transition
        if schema.intervention_trials >= self._min_interv_trials:
            if schema.p_effect_given_do_action >= self._min_interv_prob:
                schema.causal_status = CausalStatus.INTERVENTIONALLY_SUPPORTED
                schema.confounder_risk = max(0.05, 1.0 - schema.p_effect_given_do_action)
            elif schema.p_effect_given_do_action < 0.3 and schema.p_effect_given_action > 0.7:
                # High observational correlation but failed intervention => Spurious Confounder!
                schema.causal_status = CausalStatus.PREDICTIVE_NON_CAUSAL
                schema.confounder_risk = 0.95

        return schema.causal_status

    def mark_counterfactually_identified(
        self,
        schema_id: str,
        validation_accuracy: float,
    ) -> None:
        """Promotes an interventionally supported schema to CAUSALLY_IDENTIFIED (Pearl Rung 3)."""
        schema = self._get_schema_by_id(schema_id)
        if not schema:
            raise KeyError(f"Schema {schema_id} not found")
        if schema.causal_status != CausalStatus.INTERVENTIONALLY_SUPPORTED:
            raise ValueError(
                f"Cannot identify counterfactual cause from status {schema.causal_status.value}"
            )
        schema.counterfactual_accuracy = validation_accuracy
        if validation_accuracy >= 0.85:
            schema.causal_status = CausalStatus.CAUSALLY_IDENTIFIED

    def get_schema(self, schema_id: str) -> CausalSchema | None:
        return self._get_schema_by_id(schema_id)

    def all_schemas(self) -> list[CausalSchema]:
        return list(self._schemas.values())

    def _get_schema_by_id(self, schema_id: str) -> CausalSchema | None:
        for s in self._schemas.values():
            if s.schema_id == schema_id:
                return s
        return None
