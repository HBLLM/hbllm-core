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
    assert len(hypotheses) >= 3
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


def test_causal_variable_invariance_across_randomized_worlds():
    """A23.5-E2: Verify that Developmental HCIR discovers the mass invariant across varied surface confounders."""
    for mode in [0, 1, 2, 3]:
        env = BabyWorldEnvironment(seed=50 + mode)
        env.reset("randomized_confounded_world")
        env._setup_randomized_confounded_world(mode_override=mode)

        substrate = create_blank_brain_substrate()
        perception = DevelopmentalPerceptionAdapter()
        engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

        obs = env.get_sensory_observation()
        engine.observe_and_generate_hypotheses(obs, episodes_data=[])

        available_ids = list(env.objects.keys())
        for _ in range(10):
            if any(h.confirmed and h.variable == "mass_sensation" for h in engine.hypotheses):
                break
            active_hyps = [h for h in engine.hypotheses if not h.falsified]
            target_id, _ = engine.select_active_intervention(available_ids, active_hyps)
            engine.execute_interventional_probe(target_id, action=BabyActionType.PUSH)

        # In every randomized surface world, mass_sensation MUST be confirmed as the invariant cause!
        mass_hyp = next(h for h in engine.hypotheses if h.variable == "mass_sensation")
        assert mass_hyp.confirmed is True
        assert mass_hyp.confidence >= 0.95


def test_observational_demonstrations_and_hypothesis_induction_without_rule_leakage():
    """Verify that hypotheses are induced purely from observational demonstrations without ground-truth leakage."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("confounded_train_world")

    # 1. Generate observational demonstrations
    demos = env.generate_observational_demonstrations()
    assert len(demos) == 8  # 4 red light movers, 4 blue heavy stationary

    movers = [d for d in demos if d["moved"]]
    non_movers = [d for d in demos if not d["moved"]]
    assert len(movers) == 4
    assert len(non_movers) == 4
    assert all(d["features"]["color"] == "red" for d in movers)
    assert all(d["features"]["color"] == "blue" for d in non_movers)

    # 2. Induce hypotheses from demonstrations
    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    obs = env.get_sensory_observation()
    hyps = engine.observe_and_generate_hypotheses(obs, episodes_data=demos)

    # Color hypothesis correctly induced from mover color
    color_hyp = next(h for h in hyps if h.variable == "color")
    assert color_hyp.value == "red"

    # Mass threshold dynamically computed from boundary between movers and non-movers
    max_pos = max(d["features"]["mass_sensation"] for d in movers)
    min_neg = min(d["features"]["mass_sensation"] for d in non_movers)
    expected_boundary = round((max_pos + min_neg) / 2.0, 1)

    mass_hyp = next(h for h in hyps if h.variable == "mass_sensation")
    assert mass_hyp.value == expected_boundary
    assert mass_hyp.operator == "<"
