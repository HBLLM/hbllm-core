from __future__ import annotations

import logging
import uuid

from hbllm.hcir.grounding.causal.schema import SchemaInducer
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.dialogue import GoalProposal
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef
from hbllm.hcir.grounding.predictive.assimilator import GroundingSurpriseAlert

logger = logging.getLogger(__name__)


class ActiveInferenceFeedback:
    """Closes the active inference loop between prediction errors and internal world models.

    Upholds SGE Section 13 & Rule 1.8 (Internal Memory Governance):
    Surprise signals are not ignored or patched with ad-hoc flags. They route directly
    to WorkingConceptStore (degrading flawed prototypes), SchemaInducer (invalidating
    confounded rules), and emit Epistemic GoalProposals for curious exploration.
    """

    def __init__(
        self,
        concept_store: WorkingConceptStore | None = None,
        schema_inducer: SchemaInducer | None = None,
    ) -> None:
        self._concept_store = concept_store
        self._schema_inducer = schema_inducer

    def process_surprise_alert(
        self,
        alert: GroundingSurpriseAlert,
        associated_concept_ref: GroundedConceptRef | None = None,
    ) -> GoalProposal | None:
        """Processes a surprise alert, penalizes fragile hypotheses, and emits an info-gathering goal."""
        logger.info(
            "Active inference assimilating surprise %.2f for %s",
            alert.surprise_score,
            alert.entity_ref.id,
        )

        # 1. Feedback to WorkingConceptStore
        if self._concept_store and associated_concept_ref:
            concept = self._concept_store.get_concept(associated_concept_ref.id)
            if concept:
                # Trigger lifecycle evaluation with prediction error
                self._concept_store._lifecycle_manager.evaluate_transition(
                    concept, prediction_error=alert.surprise_score
                )

        # 2. Feedback to SchemaInducer
        if self._schema_inducer and alert.applied_schema_id:
            schema = self._schema_inducer.get_schema(alert.applied_schema_id)
            if schema:
                # Falsify or degrade schema confidence
                schema.confounder_risk = min(1.0, schema.confounder_risk + 0.2)
                schema.p_effect_given_do_action = max(0.0, schema.p_effect_given_do_action - 0.25)

        # 3. Formulate epistemic information-gathering goal proposal if surprise is severe
        if alert.surprise_score >= 0.5:
            return GoalProposal(
                proposal_id=f"goal_epistemic_explore_{uuid.uuid4().hex[:8]}",
                goal_intent="INSPECT_ENTITY",
                target_entity=alert.entity_ref,
                parameters={
                    "surprise_score": alert.surprise_score,
                    "position_error": alert.position_error,
                    "property_mismatches": {
                        k: {"pred": str(v[0]), "obs": str(v[1])}
                        for k, v in alert.property_mismatches.items()
                    },
                },
                priority=alert.surprise_score,
            )

        return None
