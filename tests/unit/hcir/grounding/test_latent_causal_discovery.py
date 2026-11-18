"""Unit tests for Latent Variable Causal Discovery and Counterfactual Reasoning (AGI Gap 7).

Verifies:
1. Consistent observations reinforce standard causal schemas without spawning latent variables.
2. Inconsistent observations under identical observable preconditions trigger autonomous
   latent boolean variable induction ("there must be something I can't see").
3. Schema bifurcation into latent-conditioned branches (L=true and L=false).
4. Pearl Level 3 counterfactual queries evaluated over hypothetical unobserved latent states.
5. Export to formal CausalPredicates and atomic TransactionOperations.
6. AutonomousEpistemicEngine integration for latent counterfactual queries.
"""

from __future__ import annotations

import uuid

from hbllm.hcir.grounding.causal.events import EventCandidate, EventType
from hbllm.hcir.grounding.causal.schema import LatentCausalVariable, SchemaInducer
from hbllm.hcir.grounding.contracts.enums import CausalStatus
from hbllm.hcir.grounding.contracts.references import GroundedRef
from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


def _make_dummy_event(event_type: EventType, post_state: dict) -> EventCandidate:
    return EventCandidate(
        event_id=f"ev_{uuid.uuid4().hex[:6]}",
        event_type=event_type,
        primary_entity_ref=GroundedRef(id="obj_target"),
        pre_state={"state": "closed"},
        post_state=post_state,
        confidence=1.0,
    )


def test_consistent_observations_do_not_trigger_latent_variable() -> None:
    """Validate that reliable deterministic observations do not spawn spurious latent variables."""
    inducer = SchemaInducer()

    event = _make_dummy_event(EventType.STATE_CHANGE, {"state": "open"})
    for _ in range(6):
        inducer.record_observation(
            action_type="OPEN",
            target_role="DOOR",
            precondition={"adjacent": True},
            outcome_event=event,
        )

    assert len(inducer.all_latent_variables()) == 0
    schema = inducer.get_schema("schema_open_door") or inducer.all_schemas()[0]
    assert schema.causal_status in (
        CausalStatus.PREDICTIVE_NON_CAUSAL,
        CausalStatus.OBSERVED_ASSOCIATION,
    )
    assert not schema.is_latent_conditioned


def test_inconsistent_observations_trigger_autonomous_latent_induction() -> None:
    """Validate that identical action/precondition with divergent outcomes induces a latent variable."""
    inducer = SchemaInducer()

    success_event = _make_dummy_event(EventType.STATE_CHANGE, {"state": "activated"})
    fail_event = _make_dummy_event(EventType.STATE_CHANGE, {"state": "inert"})

    # Alternate outcomes under identical observable precondition
    for _ in range(3):
        inducer.record_observation(
            action_type="PRESS",
            target_role="BUTTON",
            precondition={"adjacent": True, "charge": 10},
            outcome_event=success_event,
        )
        inducer.record_observation(
            action_type="PRESS",
            target_role="BUTTON",
            precondition={"adjacent": True, "charge": 10},
            outcome_event=fail_event,
        )

    latent_vars = inducer.all_latent_variables()
    assert len(latent_vars) == 1
    latent = latent_vars[0]

    assert latent.action_type == "PRESS"
    assert latent.target_role == "BUTTON"
    assert latent.precondition == {"adjacent": True, "charge": 10}
    assert len(latent.observed_effects) >= 2


def test_latent_bifurcation_into_conditioned_schemas() -> None:
    """Validate that the causal schema bifurcates into L=true and L=false conditioned branches."""
    inducer = SchemaInducer()

    eff_true = {"event_type": "STATE_TRANSFORMED", "post_state": {"door": "open"}}
    eff_false = {"event_type": "NULL_EFFECT", "post_state": {"door": "locked"}}

    latent_var = inducer.propose_latent_variable(
        action_type="PULL",
        target_role="LEVER",
        precondition={"at_lever": True},
        divergent_effects=[eff_true, eff_false],
    )

    assert isinstance(latent_var, LatentCausalVariable)
    schemas = inducer.all_schemas()
    branched_schemas = [s for s in schemas if s.is_latent_conditioned]
    assert len(branched_schemas) == 2

    true_schema = next(s for s in branched_schemas if s.latent_condition == "true")
    false_schema = next(s for s in branched_schemas if s.latent_condition == "false")

    assert true_schema.expected_effect == eff_true
    assert false_schema.expected_effect == eff_false
    assert true_schema.latent_variable_id == latent_var.variable_id
    assert false_schema.latent_variable_id == latent_var.variable_id


def test_pearl_level_3_counterfactual_queries() -> None:
    """Validate Pearl Level 3 counterfactual queries: 'What if latent state were X?'"""
    inducer = SchemaInducer()

    eff_active = {"event_type": "BEAM_EMITTED", "post_state": {"laser": "on"}}
    eff_inactive = {"event_type": "NO_CHANGE", "post_state": {"laser": "off"}}

    inducer.propose_latent_variable(
        action_type="TRIGGER",
        target_role="SENSOR",
        precondition={"beam_path": "clear"},
        divergent_effects=[eff_active, eff_inactive],
    )

    # Counterfactual 1: What if the unobserved latent variable were True?
    predicted_true = inducer.evaluate_latent_counterfactual(
        action_type="TRIGGER",
        target_role="SENSOR",
        hypothetical_latent_state=True,
    )
    assert predicted_true == eff_active

    # Counterfactual 2: What if the unobserved latent variable were False?
    predicted_false = inducer.evaluate_latent_counterfactual(
        action_type="TRIGGER",
        target_role="SENSOR",
        hypothetical_latent_state=False,
    )
    assert predicted_false == eff_inactive


def test_latent_variable_causal_predicate_and_transaction_export() -> None:
    """Validate formal export of latent causal variable to CausalPredicate and TransactionOperation."""
    latent = LatentCausalVariable(
        variable_id="latent_alpha_01",
        action_type="INTERACT",
        target_role="PORTAL",
        precondition={"powered": True},
        observed_effects=[{"state": "teleported"}, {"state": "blocked"}],
        probability_true=0.7,
        confidence=0.9,
    )

    pred = latent.to_causal_predicate()
    assert pred.variable == "latent_variable:latent_alpha_01"
    assert pred.relation == "latent_bifurcation"
    assert pred.target_entity == "INTERACT:PORTAL"

    ops = latent.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].op == TransactionOp.UPSERT_NODE
    assert ops[0].node_id == "latent_var_latent_alpha_01"
    assert ops[0].node_data["confidence"] == 0.9


def test_autonomous_epistemic_engine_latent_counterfactual_interface() -> None:
    """Validate AutonomousEpistemicEngine evaluates latent counterfactual queries."""
    engine = AutonomousEpistemicEngine()

    eff_open = {"event_type": "STATE_TRANSFORMED", "post_state": {"gate": "open"}}
    eff_closed = {"event_type": "NO_OP", "post_state": {"gate": "closed"}}

    engine.schema_inducer.propose_latent_variable(
        action_type="KEY_USE",
        target_role="GATE",
        precondition={"has_key": True},
        divergent_effects=[eff_open, eff_closed],
    )

    cf_open = engine.query_latent_counterfactual(
        action_type="KEY_USE",
        target_role="GATE",
        hypothetical_latent_state=True,
    )
    assert cf_open == eff_open

    cf_closed = engine.query_latent_counterfactual(
        action_type="KEY_USE",
        target_role="GATE",
        hypothetical_latent_state=False,
    )
    assert cf_closed == eff_closed
