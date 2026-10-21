"""Unit tests for POMDP Belief State & Latent Variable Induction under Persistent Surprise."""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import ActionModality, ActionNode, BeliefNode, NodeLifecycle, PredictionNode
from hbllm.hcir.types import Provenance, Scope, UncertaintyVector
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.prediction_error import PredictionErrorTypology
from hbllm.hcir.world.surprise_engine import SurpriseEngine
from hbllm.hcir.world.world_belief import WorldBeliefNode
from hbllm.hcir.world_kernel import WorldKernel


def test_surprise_engine_persistent_surprise_induction() -> None:
    engine = SurpriseEngine(surprise_threshold=0.15, persistent_surprise_threshold=2)

    # First prediction error
    eval_1 = engine.evaluate_surprise(
        prediction_id="pred_1",
        expected_state={"position_x": 10.0, "status": "open"},
        actual_state={"position_x": 0.0, "status": "stuck"},
        confidence=0.90,
        prediction_source="physics",
        context_signature="turn_valve",
    )
    assert eval_1.is_surprising is True
    assert eval_1.is_persistent is False
    assert eval_1.prediction_error_node is not None
    assert eval_1.prediction_error_node.typology == PredictionErrorTypology.MODEL_ERROR

    # Second prediction error with same context_signature triggers persistent surprise
    eval_2 = engine.evaluate_surprise(
        prediction_id="pred_2",
        expected_state={"position_x": 10.0, "status": "open"},
        actual_state={"position_x": 0.0, "status": "stuck"},
        confidence=0.90,
        prediction_source="physics",
        context_signature="turn_valve",
    )
    assert eval_2.is_surprising is True
    assert eval_2.is_persistent is True
    assert eval_2.prediction_error_node is not None
    assert eval_2.prediction_error_node.typology == PredictionErrorTypology.LATENT_CONFOUNDER


def test_world_belief_node_bayesian_update() -> None:
    belief = WorldBeliefNode(
        belief_id="belief_friction",
        subject="surface_friction",
        value="normal",
        confidence=0.5,
        is_latent=True,
        distribution={"normal": 0.5, "ice": 0.5},
    )

    # Observation provides strong evidence for "ice"
    likelihoods = {"normal": 0.1, "ice": 0.9}
    dist = belief.update_distribution(likelihoods)

    assert pytest.approx(dist["ice"], rel=1e-3) == 0.9
    assert pytest.approx(dist["normal"], rel=1e-3) == 0.1
    assert belief.value == "ice"
    assert pytest.approx(belief.confidence, rel=1e-3) == 0.9

    # Counter-evidence shifts distribution back towards "normal"
    counter_lh = {"normal": 9.0, "ice": 1.0}
    dist_2 = belief.update_distribution(counter_lh)

    # 0.1 * 9.0 = 0.9, 0.9 * 1.0 = 0.9 -> equal (0.5, 0.5)
    assert pytest.approx(dist_2["normal"], rel=1e-3) == 0.5
    assert pytest.approx(dist_2["ice"], rel=1e-3) == 0.5


def test_world_kernel_observe_and_update_induces_latent() -> None:
    ws = HCIRWorkspaceState()
    kernel = WorldKernel(workspace=ws)
    kernel.surprise_engine.persistent_surprise_threshold = 2

    action = ActionNode(
        id="act_pull_lever",
        modality=ActionModality.MANIPULATION,
        intent="pull_lever",
        properties={"predicted_state": {"gate_status": "open", "voltage": 12.0}},
    )
    prediction = PredictionNode(
        id="pred_lever_1",
        claim="Lever opens gate",
        predicted_outcome="gate_status=open",
        time_horizon_ms=1000,
        lifecycle=NodeLifecycle.ACTIVE,
        uncertainty=UncertaintyVector(confidence=0.92),
        provenance=Provenance(created_by="test"),
        scope=Scope(tenant_id="default"),
        properties={"predicted_state": {"gate_status": "open", "voltage": 12.0}},
    )
    ws.upsert_node(action)
    ws.upsert_node(prediction)

    actual_state_divergent = {"gate_status": "closed", "voltage": 0.0}

    # First attempt: surprising but not yet persistent
    eval_1, latent_1 = kernel.observe_and_update(
        action=action,
        actual_state=actual_state_divergent,
        prediction=prediction,
        prediction_source="physics",
    )
    assert eval_1.is_surprising is True
    assert eval_1.is_persistent is False
    assert latent_1 is None

    # Second attempt under same conditions: persistent surprise induces latent variable
    eval_2, latent_2 = kernel.observe_and_update(
        action=action,
        actual_state=actual_state_divergent,
        prediction=prediction,
        prediction_source="physics",
    )
    assert eval_2.is_surprising is True
    assert eval_2.is_persistent is True
    assert latent_2 is not None
    assert latent_2.is_latent is True
    assert "pull_lever" in latent_2.subject

    # Verify latent belief in belief_graph
    latents = kernel.belief_graph.get_latent_beliefs()
    assert len(latents) == 1
    assert latents[0].belief_id == latent_2.belief_id

    # Verify latent belief is synced into HCIR Workspace graph as BeliefNode
    graph_node = ws.graph.get_node(latent_2.belief_id)
    assert graph_node is not None
    assert isinstance(graph_node, BeliefNode)
    assert graph_node.properties.get("is_latent") is True
    assert "distribution" in graph_node.properties


def test_world_kernel_update_latent_belief_and_prediction_forwarding() -> None:
    ws = HCIRWorkspaceState()
    kernel = WorldKernel(workspace=ws)

    # Induce a latent variable directly
    latent = kernel.induce_latent_variable(
        action_intent="navigate",
        expected_state={"terrain": "flat"},
        actual_state={"terrain": "mud"},
    )
    assert latent.is_latent is True
    assert latent.subject.startswith("latent_navigate_")

    # Update latent belief distribution via Bayesian update
    updated_latent = kernel.update_latent_belief(
        latent_name_or_id=latent.subject,
        likelihoods={"mud": 0.85, "flat": 0.15},
    )
    assert updated_latent.value == "mud"
    assert pytest.approx(updated_latent.confidence, rel=1e-2) == 0.85

    # Check that WorldStateSummary reports the latent beliefs
    state_summary = kernel.get_current_world_state()
    assert latent.subject in state_summary.latent_beliefs
    assert state_summary.latent_beliefs[latent.subject]["value"] == "mud"

    # Forward prediction should now incorporate active latent beliefs
    action = ActionNode(
        id="act_nav_forward",
        modality=ActionModality.MANIPULATION,
        intent="navigate",
        properties={"step": 1},
    )
    pred_node = kernel.predict(action)
    assert pred_node is not None
    assert pred_node.claim == "Outcome of navigate"

    # Workspace synchronization test: sync_from_workspace restores latent belief
    kernel_2 = WorldKernel(workspace=ws)
    kernel_2.sync_from_workspace()
    kernel_2_latents = kernel_2.belief_graph.get_latent_beliefs()
    assert any(b.subject == latent.subject for b in kernel_2_latents)
