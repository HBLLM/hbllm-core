"""Adversarial & Held-Out Scenarios for Developmental Causal Discovery (Tier 1).

Constructed from first principles to test robustness against:
1. Dual Confounder Decoys (Simpson's Paradox: Color & Shape both correlated)
2. Latent Unobserved / Perceptually Indistinguishable Physical Resistance
3. Zero-Variance Negative Controls
"""

from __future__ import annotations

from typing import Any

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.causal_discovery import InterventionalCausalDiscoveryEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.types import (
    BabyActionType,
    BabyObjectState,
    BabyObjectType,
    Vector2D,
)


def test_adversarial_dual_confounder_decoy_simpson_paradox() -> None:
    """Scenario A: Both color and shape are deceptive confounders.

    Observational bias:
    - Red balls move (light)
    - Blue blocks stay stationary (heavy)

    Adversarial held-out objects:
    - Red heavy ball: Red and Ball, but DOES NOT move! (Falsifies both color and shape)
    - Blue light block: Blue and Block, but MOVES! (Falsifies negative correlation)

    The agent must reject both visual cues through active probing and discover the mass threshold.
    """
    env = BabyWorldEnvironment(seed=101)
    env.objects.clear()

    # Observational training demonstration data establishing dual correlation
    training_episodes: list[dict[str, Any]] = [
        {
            "features": {"color": "red", "shape": "ball", "mass_sensation": 1.5},
            "moved": True,
            "outcome": "MOVES",
        },
        {
            "features": {"color": "red", "shape": "ball", "mass_sensation": 2.0},
            "moved": True,
            "outcome": "MOVES",
        },
        {
            "features": {"color": "blue", "shape": "block", "mass_sensation": 12.0},
            "moved": False,
            "outcome": "STATIONARY",
        },
        {
            "features": {"color": "blue", "shape": "block", "mass_sensation": 15.0},
            "moved": False,
            "outcome": "STATIONARY",
        },
    ]

    # Populate physical environment with adversarial entities
    env.objects["obj_train_red_ball"] = BabyObjectState(
        id="obj_train_red_ball",
        object_type=BabyObjectType.BALL,
        color="red",
        mass=1.5,
        size=Vector2D(0.4, 0.4),
        position=Vector2D(1.0, 1.0),
    )
    env.objects["obj_train_blue_block"] = BabyObjectState(
        id="obj_train_blue_block",
        object_type=BabyObjectType.BLOCK,
        color="blue",
        mass=14.0,
        size=Vector2D(0.5, 0.5),
        position=Vector2D(1.0, -1.0),
    )
    # Adversarial Decoy 1: Red heavy ball (contradicts BOTH color and shape moving)
    env.objects["obj_adv_red_heavy_ball"] = BabyObjectState(
        id="obj_adv_red_heavy_ball",
        object_type=BabyObjectType.BALL,
        color="red",
        mass=16.0,
        size=Vector2D(0.4, 0.4),
        position=Vector2D(1.5, 0.5),
    )
    # Adversarial Decoy 2: Blue light block (contradicts blue and block staying stationary)
    env.objects["obj_adv_blue_light_block"] = BabyObjectState(
        id="obj_adv_blue_light_block",
        object_type=BabyObjectType.BLOCK,
        color="blue",
        mass=1.2,
        size=Vector2D(0.5, 0.5),
        position=Vector2D(0.5, -0.5),
    )

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    obs = env.get_sensory_observation()
    hypotheses = engine.observe_and_generate_hypotheses(obs, episodes_data=training_episodes)

    # Initial hypothesis formulation should capture color, shape, and mass
    variables = {h.variable for h in hypotheses}
    assert "color" in variables, "Should formulate color hypothesis from biased data"
    assert "shape" in variables, "Should formulate shape hypothesis from biased data"
    assert "mass_sensation" in variables, "Should formulate mass hypothesis"

    # Active interventional probing loop
    available_ids = list(env.objects.keys())
    for _ in range(12):
        if any(h.confirmed and h.variable == "mass_sensation" for h in engine.hypotheses):
            break
        active_hyps = [h for h in engine.hypotheses if not h.falsified]
        target_id, target_hyp = engine.select_active_intervention(available_ids, active_hyps)
        engine.execute_interventional_probe(target_id, action=BabyActionType.PUSH)

    # Verifications:
    # 1. Spurious color hypothesis must be strictly falsified
    color_hyp = next(h for h in engine.hypotheses if h.variable == "color")
    assert color_hyp.falsified is True, (
        "Color hypothesis failed to be falsified by adversarial probes!"
    )
    assert color_hyp.confidence == 0.0

    # 2. Spurious shape hypothesis must be strictly falsified
    shape_hyp = next(h for h in engine.hypotheses if h.variable == "shape")
    assert shape_hyp.falsified is True, (
        "Shape hypothesis failed to be falsified by adversarial probes!"
    )
    assert shape_hyp.confidence == 0.0

    # 3. Ground-truth mass hypothesis must be confirmed
    mass_hyp = next(h for h in engine.hypotheses if h.variable == "mass_sensation")
    assert mass_hyp.confirmed is True, "True physical mass invariant was not confirmed!"
    assert mass_hyp.confidence >= 0.95

    # 4. Out-of-distribution evaluation: test on novel objects
    unseen_eval_objects = [
        {"color": "green", "shape": "cylinder", "mass": 2.0, "surface_friction": 0.5},
        {"color": "yellow", "shape": "ramp", "mass": 18.0, "surface_friction": 0.5},
        {"color": "red", "shape": "ball", "mass": 20.0, "surface_friction": 0.5},
        {"color": "blue", "shape": "block", "mass": 0.5, "surface_friction": 0.5},
    ]
    accuracy, records = engine.evaluate_generalization(unseen_eval_objects)
    assert accuracy == 1.0, (
        f"Expected 100% causal generalization on held-out test suite, got {accuracy * 100}%"
    )


def test_adversarial_latent_unobserved_resistance_control() -> None:
    """Scenario B: Identical visual percepts with varying internal resistance.

    Two objects look 100% identical in color ("yellow"), shape ("ball"), and visual appearance.
    One is light (moves), one has lead core (heavy, does not move).
    The engine must recognize that vision cannot discriminate motion, falsify vision,
    and isolate mass sensation.
    """
    env = BabyWorldEnvironment(seed=202)
    env.objects.clear()

    env.objects["obj_yellow_light"] = BabyObjectState(
        id="obj_yellow_light",
        object_type=BabyObjectType.BALL,
        color="yellow",
        mass=2.0,
        size=Vector2D(0.4, 0.4),
        position=Vector2D(1.0, 0.5),
    )
    env.objects["obj_yellow_heavy"] = BabyObjectState(
        id="obj_yellow_heavy",
        object_type=BabyObjectType.BALL,
        color="yellow",
        mass=15.0,
        size=Vector2D(0.4, 0.4),
        position=Vector2D(1.0, -0.5),
    )

    training_episodes: list[dict[str, Any]] = [
        {
            "features": {"color": "yellow", "shape": "ball", "mass_sensation": 2.0},
            "moved": True,
            "outcome": "MOVES",
        },
        {
            "features": {"color": "yellow", "shape": "ball", "mass_sensation": 15.0},
            "moved": False,
            "outcome": "STATIONARY",
        },
    ]

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    obs = env.get_sensory_observation()
    _ = engine.observe_and_generate_hypotheses(obs, episodes_data=training_episodes)

    # Active probing
    available_ids = list(env.objects.keys())
    for _ in range(8):
        if any(h.confirmed and h.variable == "mass_sensation" for h in engine.hypotheses):
            break
        active_hyps = [h for h in engine.hypotheses if not h.falsified]
        target_id, target_hyp = engine.select_active_intervention(available_ids, active_hyps)
        engine.execute_interventional_probe(target_id, action=BabyActionType.PUSH)

    # Color and shape hypotheses must be falsified because both positive and negative are yellow balls
    color_hyp = next((h for h in engine.hypotheses if h.variable == "color"), None)
    if color_hyp:
        assert color_hyp.falsified is True

    mass_hyp = next(h for h in engine.hypotheses if h.variable == "mass_sensation")
    assert mass_hyp.confirmed is True


def test_adversarial_zero_variance_negative_control() -> None:
    """Scenario C: Negative control where all demonstrations are stationary.

    All objects are immovable heavy blocks. No object ever moved.
    Engine must not hallucinate false causal laws or divide by zero.
    """
    env = BabyWorldEnvironment(seed=303)
    env.objects.clear()

    env.objects["immovable_1"] = BabyObjectState(
        id="immovable_1",
        object_type=BabyObjectType.BLOCK,
        color="gray",
        mass=50.0,
        size=Vector2D(1.0, 1.0),
        position=Vector2D(1.0, 0.0),
    )

    all_stationary_episodes: list[dict[str, Any]] = [
        {
            "features": {"color": "gray", "shape": "block", "mass_sensation": 50.0},
            "moved": False,
            "outcome": "STATIONARY",
        },
        {
            "features": {"color": "gray", "shape": "block", "mass_sensation": 60.0},
            "moved": False,
            "outcome": "STATIONARY",
        },
    ]

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    obs = env.get_sensory_observation()
    hypotheses = engine.observe_and_generate_hypotheses(obs, episodes_data=all_stationary_episodes)

    # Fallback sampling occurs gracefully, but no hypothesis should be confirmed without evidence
    assert isinstance(hypotheses, list)
    assert not any(h.confirmed for h in hypotheses)
    assert len(engine.confirmed_causal_rules) == 0
