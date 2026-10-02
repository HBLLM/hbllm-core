"""Unit tests for Core HCIR Functional Affordance Discovery Engine."""

from __future__ import annotations

from hbllm.hcir.world.affordance_discovery import (
    AffordanceHypothesis,
    BaseAffordanceDiscoveryEngine,
)
from hbllm.hcir.world.causal_discovery import BeliefTransitionType


def test_affordance_hypotheses_generation() -> None:
    engine = BaseAffordanceDiscoveryEngine()

    categories = ["sphere", "cylinder", "cube"]
    actions = [("ROLL", "ROLLABLE"), ("PUSH", "SLIDABLE"), ("GRASP", "GRASPABLE")]

    hyps = engine.generate_hypotheses(categories, actions)
    assert len(hyps) == 9  # 3 categories x 3 actions
    assert len(engine.belief_history) == 9
    assert all(
        e.event_type == BeliefTransitionType.HYPOTHESIS_CREATED for e in engine.belief_history
    )

    # Subsequent generation with overlapping categories should not duplicate existing pairs
    hyps_again = engine.generate_hypotheses(["sphere", "torus"], [("ROLL", "ROLLABLE")])
    assert len(hyps_again) == 10  # Only torus added
    assert any(h.entity_shape == "torus" for h in engine.hypotheses)


def test_affordance_confirmation_and_falsification() -> None:
    engine = BaseAffordanceDiscoveryEngine()
    external_store: dict[str, list[str]] = {}

    roll_hyp = AffordanceHypothesis(
        action="ROLL",
        entity_shape="sphere",
        affordance_label="ROLLABLE",
        confidence=0.5,
    )
    push_hyp = AffordanceHypothesis(
        action="PUSH",
        entity_shape="fixed_wall",
        affordance_label="SLIDABLE",
        confidence=0.5,
    )

    # 1. Success confirmation
    event_conf = engine.update_affordance_from_evidence(
        hypothesis=roll_hyp,
        success=True,
        target_id="sphere_01",
        step_index=1,
        target_store=external_store,
    )
    assert roll_hyp.confirmed is True
    assert roll_hyp.confidence == 1.0
    assert "sphere_01" in roll_hyp.supporting_episodes
    assert event_conf.event_type == BeliefTransitionType.HYPOTHESIS_CONFIRMED
    assert "ROLLABLE" in engine.confirmed_affordances["sphere"]
    assert "ROLLABLE" in external_store["sphere"]

    # 2. Failure falsification
    event_fals = engine.update_affordance_from_evidence(
        hypothesis=push_hyp,
        success=False,
        target_id="wall_01",
        step_index=2,
        target_store=external_store,
    )
    assert push_hyp.falsified is True
    assert push_hyp.confidence == 0.0
    assert "wall_01" in push_hyp.counterexamples
    assert event_fals.event_type == BeliefTransitionType.HYPOTHESIS_FALSIFIED
    assert "fixed_wall" not in engine.confirmed_affordances


def test_affordance_novel_entity_transfer() -> None:
    confirmed = {
        "sphere": ["ROLLABLE", "SLIDABLE", "GRASPABLE"],
        "cube": ["SLIDABLE", "GRASPABLE"],
    }

    held_out = [
        {
            "id": "obj_novel_1",
            "shape": "ellipsoid",
            "base_shape": "sphere",
            "ground_truth_affordances": ["ROLLABLE", "SLIDABLE", "GRASPABLE"],
        },
        {
            "id": "obj_novel_2",
            "shape": "polyhedron",
            "base_shape": "cube",
            "ground_truth_affordances": ["SLIDABLE", "GRASPABLE"],
        },
    ]

    accuracy, records = BaseAffordanceDiscoveryEngine.test_novel_entity_transfer(
        held_out_entities=held_out,
        confirmed_affordances=confirmed,
    )

    assert accuracy == 1.0
    assert len(records) == 2
    assert all(r["correct"] for r in records)
