"""Unit tests for the Next-Generation Cognitive Architecture Faculties (Tracks 1 - 5).

Covers:
- Track 1: Global Neuronal Workspace (GNW) Ignition & Broadcast Engine
- Track 2: Four Neuromodulatory Systems (Dopamine, Noradrenaline, Acetylcholine, Serotonin)
- Track 3: Thalamocortical Gamma-Band Phase Synchronization & TRN Attentional Gating
- Track 4: Superposition Counterfactual Tree Search Planner
- Track 5: Native Rust Acceleration for Spatial MCTS & A* Grid Search
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.global_neuronal_workspace import (
    GlobalNeuronalWorkspace,
    IgnitionEvent,
)
from hbllm.hcir.world.neuromodulatory_system import (
    NeuromodulatorLevels,
    NeuromodulatorySystem,
)
from hbllm.hcir.world.superposition_planner import (
    SuperpositionEvaluationResult,
    SuperpositionPlanner,
)
from hbllm.hcir.world.thalamocortical_binding import (
    BoundFeatureBundle,
    ThalamocorticalBindingEngine,
)

try:
    import hbllm_compute  # type: ignore[import-not-found]  # noqa: F401

    HAS_HBLLM_COMPUTE = True
except ImportError:
    HAS_HBLLM_COMPUTE = False


# ─────────────────────────────────────────────────────────────────────────────
# Track 1: Global Neuronal Workspace (GNW)
# ─────────────────────────────────────────────────────────────────────────────


def test_gnw_ignition_and_broadcast() -> None:
    gnw = GlobalNeuronalWorkspace(base_threshold=0.6, gain=8.0, refractory_elevation=0.3)

    # Sub-threshold candidate should not ignite
    gnw.submit_candidate(
        source_cortex="perception",
        salience=0.3,
        content={"feature": "background_noise"},
    )
    ign = gnw.compete_and_ignite()
    assert ign is None
    assert gnw.last_ignited_content is None

    # Track listener notifications
    broadcast_log: list[IgnitionEvent] = []

    def on_broadcast(event: IgnitionEvent) -> None:
        broadcast_log.append(event)

    gnw.register_listener("cortex_deliberation", on_broadcast)

    # High salience candidate should ignite and broadcast
    gnw.submit_candidate(
        source_cortex="causal_induction",
        salience=0.85,
        content={"action": 1, "door_opened": True},
    )
    res_high = gnw.compete_and_ignite()
    assert res_high is not None
    assert res_high.winner.source_cortex == "causal_induction"
    assert res_high.winner.salience == 0.85
    assert len(broadcast_log) == 1
    assert broadcast_log[0].winner.content["door_opened"] is True
    assert gnw.last_ignited_content is not None
    assert gnw.current_threshold > 0.6  # Elevated by refractory


def test_gnw_decay_and_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)

    # Submit candidate to engine GNW
    ign = engine.submit_to_workspace(
        source_cortex="episodic_cortex",
        salience=0.9,
        content={"memory_key": "level_1_exit"},
    )
    assert ign is not None
    assert ign.winner.source_cortex == "episodic_cortex"
    assert engine.gnw.last_ignited_content is not None

    # Step decay
    engine.gnw.step_decay()
    assert engine.gnw.current_threshold <= 0.6 + 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Track 2: Neuromodulatory Systems (DA, NE, ACh, 5-HT)
# ─────────────────────────────────────────────────────────────────────────────


def test_neuromodulatory_system_dynamics() -> None:
    nms = NeuromodulatorySystem(
        baseline_da=1.0, baseline_ne=1.0, baseline_ach=1.0, baseline_5ht=1.0
    )
    base = nms.current
    assert math.isclose(base.dopamine, 1.0, abs_tol=1e-3)
    assert math.isclose(base.noradrenaline, 1.0, abs_tol=1e-3)
    assert math.isclose(base.acetylcholine, 1.0, abs_tol=1e-3)
    assert math.isclose(base.serotonin, 1.0, abs_tol=1e-3)

    # Step updates
    nms.update_reward_prediction_error(0.8)
    nms.update_unexpected_uncertainty(0.9, is_unpredicted=True)
    nms.update_expected_uncertainty(prediction_accuracy=0.8)
    nms.update_temporal_stress(risk_level=0.2)

    # Dopamine and noradrenaline must elevate above baseline
    assert nms.current.dopamine > 1.0
    assert nms.current.noradrenaline > 1.0
    assert nms.current.acetylcholine > 1.0

    # Check cognitive hyperparameter derivations
    params = nms.get_effective_hyperparameters()
    assert params["sensory_precision"] > 1.0
    assert params["discount_factor_gamma"] >= 0.85
    assert params["exploration_temperature"] > 0.5


def test_neuromodulatory_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)

    # Modulate cognition
    levels = engine.modulate_cognition(
        reward_prediction_error=0.5,
        unexpected_surprise=0.3,
        prediction_accuracy=0.8,
        risk_level=0.1,
    )
    assert isinstance(levels, NeuromodulatorLevels)
    params = engine.neuromodulatory_system.get_effective_hyperparameters()
    assert engine.hierarchical_active_inference.sensory_precision == params["sensory_precision"]


def test_thalamocortical_synchronization_and_gating() -> None:
    engine = ThalamocorticalBindingEngine(
        nominal_frequency_hz=40.0,
        coupling_strength_k=2.5,
        coherence_threshold=0.6,
    )

    # 3 features
    entities = [
        {"entity_id": "e1", "feature": 2, "pos": (1, 1)},
        {"entity_id": "e2", "feature": 2, "pos": (1, 2)},
        {"entity_id": "e3", "feature": 9, "pos": (9, 9)},
    ]

    bundles = engine.bind_features_oscillatory(entities)
    assert len(bundles) >= 1
    for b in bundles:
        assert isinstance(b, BoundFeatureBundle)
        assert 0.0 <= b.phase_coherence_r <= 1.0

    # TRN attentional gating
    gated = engine.apply_trn_attentional_gating(bundles, target_features={"feature": 2})
    assert len(gated) == len(bundles)
    matching = [b for b in gated if b.features.get("feature") == 2]
    assert len(matching) > 0


def test_thalamocortical_engine_integration() -> None:
    ep_engine = AutonomousEpistemicEngine(exploration_budget=50)
    entities = [
        {"entity_id": "e1", "feature": 3, "pos": (2, 2)},
        {"entity_id": "e2", "feature": 3, "pos": (2, 3)},
    ]
    bundles = ep_engine.synchronize_and_bind_entities(
        feature_entities=entities, target_features={"feature": 3}
    )
    assert isinstance(bundles, list)
    for b in bundles:
        assert isinstance(b, BoundFeatureBundle)


def test_superposition_counterfactual_planning() -> None:
    planner = SuperpositionPlanner(min_probability_threshold=0.05)

    # Two competing hypotheses about a switch:
    planner.add_hypothesis(
        hypothesis_id="h_win",
        probability=0.6,
        description="Switch opens door",
        latent_variables={"safe": True, "door_id": 1},
    )
    planner.add_hypothesis(
        hypothesis_id="h_trap",
        probability=0.4,
        description="Switch triggers trap",
        latent_variables={"safe": False, "door_id": 1},
    )

    def sim_step(action: int, latent_vars: dict[str, Any]) -> float:
        if latent_vars.get("safe", True):
            return 0.9 if action == 1 else 0.5
        else:
            return 0.2 if action == 1 else 0.6

    res = planner.evaluate_superposition_actions(
        candidate_actions=[1, 2],
        simulator_fn=sim_step,
    )

    assert isinstance(res, list)
    assert len(res) == 2
    # Action 2 has utilities 0.5 and 0.6 (both >= 0.5), so it is a universal invariant
    invariant_actions = [r.action_id for r in res if r.is_universal_invariant]
    assert 2 in invariant_actions

    # Test best action selection
    best = planner.select_best_action(
        candidate_actions=[1, 2],
        simulator_fn=sim_step,
    )
    assert isinstance(best, SuperpositionEvaluationResult)

    # Test Bayesian wavefunction collapse
    winner_id = planner.collapse_superposition(
        observation_evidence={"safe": True, "door_id": 1},
    )
    assert winner_id == "h_win"
    assert planner.hypotheses["h_win"].probability > 0.80


def test_superposition_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)
    engine.superposition_planner.add_hypothesis("h1", 0.5, latent_variables={"val": 1})
    engine.superposition_planner.add_hypothesis("h2", 0.5, latent_variables={"val": 2})

    def dummy_sim(action: int, latent_vars: dict[str, Any]) -> float:
        return 0.8

    res = engine.evaluate_superposition_actions(
        candidate_actions=[1, 2],
        simulator_fn=dummy_sim,
    )
    assert isinstance(res, list)
    assert len(res) == 2


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_fast_astar_search() -> None:
    # 5x5 grid with barrier wall with an opening
    barriers = [(2, 0), (2, 1), (2, 2), (2, 3)]  # (2, 4) is open
    start = (0, 0)
    goal = (4, 0)

    path = hbllm_compute.fast_astar_search(5, 5, start, goal, barriers)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

    # Verify no coordinates in path collide with barriers
    barrier_set = set(barriers)
    for coord in path:
        assert coord not in barrier_set

    # Verify all steps are adjacent
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_fast_astar_blocked() -> None:
    # Impassable barrier wall
    barriers = [(1, 0), (1, 1), (1, 2), (1, 3)]
    path = hbllm_compute.fast_astar_search(4, 4, (0, 0), (2, 0), barriers)
    assert path is None


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_bitboard_simulate_rollout() -> None:
    # Grid 4x4, start at (0, 0), goal at (2, 2)
    # Actions: Down (1, 0), Down (1, 0), Right (0, 1), Right (0, 1)
    actions = [(1, 0), (1, 0), (0, 1), (0, 1)]
    barriers: list[tuple[int, int]] = []

    reached, steps, final_pos = hbllm_compute.bitboard_simulate_rollout(
        4, 4, (0, 0), (2, 2), barriers, actions
    )
    assert reached is True
    assert steps == 4
    assert final_pos == (2, 2)


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_universal_engine_methods() -> None:
    engine = hbllm_compute.UniversalEngine()
    assert hasattr(engine, "fast_astar_search")
    assert hasattr(engine, "bitboard_simulate_rollout")

    path = engine.fast_astar_search(3, 3, (0, 0), (2, 2), [])
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
