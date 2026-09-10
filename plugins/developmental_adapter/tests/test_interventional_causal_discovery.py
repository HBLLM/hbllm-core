"""Unit tests for Subwave A23.5 Active Interventional Causal Discovery."""

from __future__ import annotations

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.causal_discovery import InterventionalCausalDiscoveryEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.types import (
    BabyActionType,
    BeliefTransitionType,
)


def test_active_interventional_causal_discovery_resolves_confounding():
    env = BabyWorldEnvironment(seed=42)
    obs = env.reset("confounded_train_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    # 1. Observe and formulate initial hypotheses under observational correlation
    hypotheses = engine.observe_and_generate_hypotheses(obs, episodes_data=[])
    assert len(hypotheses) == 3
    var_names = [h.variable for h in hypotheses]
    assert "color" in var_names
    assert "mass_sensation" in var_names

    available_ids = list(env.objects.keys())

    # 2. Active Interventional Loop
    for _ in range(10):
        if any(h.confirmed and h.variable == "mass_sensation" for h in engine.hypotheses):
            break
        active_hyps = [h for h in engine.hypotheses if not h.falsified]
        target_id, target_hyp = engine.select_active_intervention(available_ids, active_hyps)
        engine.execute_interventional_probe(target_id, action=BabyActionType.PUSH)

    # 3. Assertions on Causal Inferences
    # Spurious color hypothesis MUST be falsified
    color_hyp = next(h for h in engine.hypotheses if h.variable == "color")
    assert color_hyp.falsified is True
    assert color_hyp.confidence == 0.0
    assert len(color_hyp.counterexamples) > 0

    # True mass causal hypothesis MUST be confirmed
    mass_hyp = next(h for h in engine.hypotheses if h.variable == "mass_sensation")
    assert mass_hyp.confirmed is True
    assert mass_hyp.confidence >= 0.95

    # Causal rule inducted into substrate
    assert len(substrate.causal_rules) >= 1
    inducted_rule = substrate.causal_rules[0]
    assert inducted_rule["action"] == "PUSH"
    assert inducted_rule["precondition"]["property"] == "mass_sensation"
    assert inducted_rule["consequence"] == "MOVES"


def test_three_level_generalization():
    env = BabyWorldEnvironment(seed=42)
    env.reset("confounded_train_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    obs = env.get_sensory_observation()
    engine.observe_and_generate_hypotheses(obs, episodes_data=[])

    # Run active discovery
    available_ids = list(env.objects.keys())
    for _ in range(8):
        if any(h.confirmed and h.variable == "mass_sensation" for h in engine.hypotheses):
            break
        active_hyps = [h for h in engine.hypotheses if not h.falsified]
        target_id, _ = engine.select_active_intervention(available_ids, active_hyps)
        engine.execute_interventional_probe(target_id, action=BabyActionType.PUSH)

    # Level 1: Training world objects
    train_objs = [
        {"id": o.id, "color": o.color, "shape": o.object_type.value, "mass": o.mass}
        for o in env.objects.values()
    ]
    l1_acc, _ = engine.evaluate_generalization(train_objs)
    assert l1_acc == 1.0

    # Level 2: Held-out unseen entities (green cylinder, yellow cone, purple torus)
    unseen_entities = [
        {"id": "ue1", "color": "green", "shape": "cylinder", "mass": 1.8},
        {"id": "ue2", "color": "yellow", "shape": "cone", "mass": 11.2},
        {"id": "ue3", "color": "purple", "shape": "torus", "mass": 0.9},
    ]
    l2_acc, _ = engine.evaluate_generalization(unseen_entities)
    assert l2_acc == 1.0

    # Level 3: Held-out unseen environment world
    unseen_world = [
        {"id": "uw1", "color": "orange", "shape": "block", "mass": 2.2},
        {"id": "uw2", "color": "brown", "shape": "box", "mass": 18.0},
    ]
    l3_acc, _ = engine.evaluate_generalization(unseen_world)
    assert l3_acc == 1.0


def test_event_sourced_belief_transitions_history():
    env = BabyWorldEnvironment(seed=42)
    env.reset("confounded_train_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    obs = env.get_sensory_observation()
    engine.observe_and_generate_hypotheses(obs, episodes_data=[])

    available_ids = list(env.objects.keys())
    for _ in range(6):
        if any(h.confirmed and h.variable == "mass_sensation" for h in engine.hypotheses):
            break
        active_hyps = [h for h in engine.hypotheses if not h.falsified]
        target_id, _ = engine.select_active_intervention(available_ids, active_hyps)
        engine.execute_interventional_probe(target_id, action=BabyActionType.PUSH)

    # Check event history
    event_types = [e.event_type for e in engine.belief_history]
    assert BeliefTransitionType.HYPOTHESIS_CREATED in event_types
    assert BeliefTransitionType.HYPOTHESIS_FALSIFIED in event_types
    assert BeliefTransitionType.CONFIDENCE_CHANGED in event_types
    assert BeliefTransitionType.HYPOTHESIS_CONFIRMED in event_types
    assert BeliefTransitionType.RULE_GENERALIZED in event_types
