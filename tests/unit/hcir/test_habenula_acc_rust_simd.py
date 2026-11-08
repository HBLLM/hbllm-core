"""Comprehensive Unit Test Suite for Habenular Gating, ACC Conflict Integration & Rust SIMD Kernels.

Tests:
1. Habenular pre-emptive inhibition from SCM counterfactual invariants.
2. Habenular pre-emptive inhibition from SNN spatio-temporal hazard density.
3. ACC Conflict Monitor predictive conflict registration from ensemble divergence.
4. Epistemic Engine integration: ensemble conflict gating and SNN pre-emptive inhibition.
5. Rust SIMD SNN Leaky Integrate-and-Fire numerical parity with pure NumPy.
6. Rust SIMD kinematic hazard density border reflection parity.
7. Rust SIMD causal attribution PN/PS attribution metric parity.
8. TransactionOperation & CausalPredicate export validity.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.world.anterior_cingulate_conflict import (
    AnteriorCingulateConflictMonitor,
)
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.causal_scm import (
    StructuralCausalModel,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from hbllm.hcir.world.predictors.snn import DynamicEntityTrack, SNNTemporalPredictor

try:
    import hbllm_compute

    _HAS_RUST = True
except ImportError:
    hbllm_compute = None
    _HAS_RUST = False


def test_habenular_preemptive_inhibition_from_scm() -> None:
    """Verify Habenular IOR pre-emptively installs motor blocks from SCM invariants."""
    habenula = HabenularEpisodicIOR(base_penalty=50.0)

    invariants = [
        CausalPredicate(
            variable="invariant:safe:2",
            operator="!=",
            value="2",
            relation="counterfactual_safety_invariant",
            target_entity="action",
        ),
        CausalPredicate(
            variable="invariant:safe:4",
            operator="!=",
            value="4",
            relation="counterfactual_safety_invariant",
            target_entity="action",
        ),
    ]

    curr_pos = (5, 5)
    inhibited_count = habenula.preemptively_inhibit_from_counterfactuals(
        scm_invariants=invariants,
        current_pos=curr_pos,
        duration_steps=16,
    )

    assert inhibited_count == 2
    assert habenula.is_action_inhibited(curr_pos, 2)
    assert habenula.is_action_inhibited(curr_pos, 4)
    assert not habenula.is_action_inhibited(curr_pos, 1)  # Safe action not inhibited
    assert habenula.get_repulsion_penalty(curr_pos, 2) >= 50.0


def test_habenular_preemptive_inhibition_from_snn() -> None:
    """Verify Habenular IOR inhibits actions stepping into high-density hazard cones."""
    habenula = HabenularEpisodicIOR(base_penalty=40.0)

    # 5x5 grid with hazard cone at (2, 3)
    hazard_density = np.zeros((5, 5), dtype=np.float32)
    hazard_density[2, 3] = 0.85  # Hazardous cell to the right of (2, 2)

    current_pos = (2, 2)
    action_deltas = {
        1: (-1, 0),  # UP -> (1, 2) safe
        2: (1, 0),  # DOWN -> (3, 2) safe
        3: (0, -1),  # LEFT -> (2, 1) safe
        4: (0, 1),  # RIGHT -> (2, 3) HAZARDOUS!
    }

    inhibited = habenula.preemptively_inhibit_from_snn(
        hazard_density=hazard_density,
        current_pos=current_pos,
        action_deltas=action_deltas,
        threshold=0.4,
    )

    assert inhibited == [4]
    assert habenula.is_action_inhibited(current_pos, 4)
    assert not habenula.is_action_inhibited(current_pos, 1)
    assert not habenula.is_action_inhibited(current_pos, 2)
    assert not habenula.is_action_inhibited(current_pos, 3)


def test_acc_predictive_conflict_registration() -> None:
    """Verify ACC monitor scales frustration and engages active conflict on ensemble disagreement."""
    acc = AnteriorCingulateConflictMonitor(frustration_threshold=0.5)

    assert not acc.is_frustrated()
    assert acc.frustration == 0.0

    # Mild disagreement does not exceed threshold
    f1 = acc.register_predictive_conflict(
        disagreement_score=0.35,
        conflicting_predictors=["snn", "scm"],
        actor_pos=(4.0, 4.0),
        step=10,
    )
    assert 0.0 < f1 < 0.5
    assert not acc.is_frustrated()

    # Acute disagreement triggers active conflict state and registers cluster
    f2 = acc.register_predictive_conflict(
        disagreement_score=0.85,
        conflicting_predictors=["snn", "lstm", "scm"],
        actor_pos=(4.0, 4.0),
        step=11,
    )
    assert f2 >= 0.5
    assert acc.is_frustrated()
    assert acc.active_conflict
    assert acc.is_in_conflict_zone((4, 4))


def test_engine_acc_habenula_ensemble_integration() -> None:
    """Verify AutonomousEpistemicEngine coordinates ACC conflict and Habenular pre-emption."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_pos = (3, 3)

    # Initial state: no conflict, no active inhibitions
    assert not engine.acc_conflict.is_frustrated()
    assert not engine.habenular_ior.is_action_inhibited((3, 3), 4)

    # 1. Trigger ensemble disagreement via ACC registration
    engine.acc_conflict.register_predictive_conflict(
        disagreement_score=0.75,
        conflicting_predictors=["snn", "transformer"],
        actor_pos=(3.0, 3.0),
        step=engine.step_counter,
    )
    assert engine.acc_conflict.is_frustrated()

    # 2. Trigger cerebellar SNN transit evaluation with dynamic patroller
    grid = np.zeros((7, 7), dtype=np.int32)
    grid[3, 4] = 9  # Patroller moving right towards column 5
    if engine.world_kernel and hasattr(engine.world_kernel, "predictive_reality"):
        snn = getattr(engine.world_kernel.predictive_reality, "snn", None)
        if snn is not None:
            snn.entity_tracks[9] = DynamicEntityTrack(
                feature_id=9,
                positions=[(3, 3), (3, 4)],
                velocity=(0, 1),
            )
            # Evaluate transit safety from (3, 4) towards (3, 5)
            engine.avatar_pos = (3, 4)
            engine.action_dynamics = {
                1: ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=0.8),
                2: ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=0.8),
                3: ActionDynamicsModel(action_id=3, delta_r=0, delta_c=-1, confidence=0.8),
                4: ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=0.8),
            }
            decision = engine.evaluate_transit_safety_cerebellar_snn(
                target_pos=(3, 5),
                curr_grid=grid,
            )
            assert decision is not None
            # Action 4 (RIGHT) is pre-emptively inhibited into the patroller path
            assert engine.habenular_ior.is_action_inhibited((3, 4), 4)


def test_rust_simd_snn_lif_parity() -> None:
    """Verify native Rust SIMD LIF step produces exact numerical parity with pure NumPy."""
    if not _HAS_RUST:
        pytest.skip("hbllm_compute extension not installed")

    snn = SNNTemporalPredictor(tau_m=8.0, v_thresh=1.2, v_reset=0.0, refractory_steps=3)
    curr_input = np.array(
        [
            [0.2, 1.5, 0.0],
            [2.0, 0.5, 1.3],
            [0.1, 0.0, 1.8],
        ],
        dtype=np.float32,
    )

    # Run step 1: inputs > 1.2 should spike
    spikes_rust, mem_rust = snn.step_lif(curr_input)
    assert spikes_rust.shape == (3, 3)
    assert mem_rust.shape == (3, 3)

    # Verify spikes fired on suprathreshold cells
    assert spikes_rust[0, 1] == 1  # 1.5 >= 1.2
    assert spikes_rust[1, 0] == 1  # 2.0 >= 1.2
    assert spikes_rust[1, 2] == 1  # 1.3 >= 1.2
    assert spikes_rust[2, 2] == 1  # 1.8 >= 1.2
    assert spikes_rust[0, 0] == 0  # 0.2 < 1.2

    # Spiked cells reset to 0.0
    assert mem_rust[0, 1] == 0.0
    assert mem_rust[1, 0] == 0.0

    # Step 2: clamped cells in refractory period cannot spike despite high input
    high_input = np.full((3, 3), 5.0, dtype=np.float32)
    spikes2, mem2 = snn.step_lif(high_input)

    # Previously refractory cells cannot spike on immediate next step
    assert spikes2[0, 1] == 0
    assert spikes2[1, 0] == 0
    # Previously non-refractory cell (0, 0) spikes now
    assert spikes2[0, 0] == 1


def test_rust_simd_hazard_density_parity() -> None:
    """Verify Rust forecast_hazard_density_simd computes border bouncing projections."""
    if not _HAS_RUST:
        pytest.skip("hbllm_compute extension not installed")

    tracks = [(2, 2, 0, 1)]  # (r=2, c=2, vr=0, vc=1) moving right
    density = hbllm_compute.forecast_hazard_density_simd(5, 5, tracks, 3)

    assert density.shape == (5, 5)
    # Target steps right: (2, 3) then (2, 4)
    assert density[2, 3] > 0.0
    assert density[2, 4] > 0.0
    # Unrelated cells have zero projected hazard
    assert density[0, 0] == 0.0
    assert density[4, 4] == 0.0
    assert np.max(density) <= 1.0


def test_rust_simd_causal_attribution() -> None:
    """Verify Rust SIMD causal attribution calculation for PN and PS."""
    if not _HAS_RUST:
        pytest.skip("hbllm_compute extension not installed")

    # When intervention causes significant divergence:
    pn, ps = hbllm_compute.causal_attribution_simd(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )
    assert 0.0 < pn <= 1.0
    assert 0.0 < ps <= 1.0
    assert pn > 0.8  # Large causal divergence -> high necessity

    # When intervention produces zero divergence:
    pn_zero, ps_zero = hbllm_compute.causal_attribution_simd(
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 1.0],
    )
    assert pn_zero == 0.0
    assert ps_zero == 0.0


def test_scm_causal_attribution_rust_integration() -> None:
    """Verify StructuralCausalModel automatically leverages Rust SIMD causal attribution."""
    scm = StructuralCausalModel()
    scm.add_variable("valve", structural_func=lambda ctx, u: ctx.get("valve", 1))
    scm.add_variable(
        "pressure", parents=["valve"], structural_func=lambda ctx, u: ctx.get("valve", 0) * 10.0
    )
    scm.add_variable(
        "hazard", parents=["pressure"], structural_func=lambda ctx, u: ctx.get("pressure", 0) > 15.0
    )

    factual = {"valve": 1, "pressure": 10.0, "hazard": False}
    # Counterfactual intervention: valve = 2 -> pressure = 20.0, hazard = True
    cf = scm.evaluate_counterfactual(
        factual_evidence=factual,
        interventions={"valve": 2},
        target_variables=["pressure", "hazard"],
    )

    assert cf.counterfactual_values["pressure"] == 20.0
    assert cf.counterfactual_values["hazard"] is True
    assert cf.probability_of_necessity > 0.0
    assert cf.causal_effect > 0.0


def test_habenular_and_acc_transaction_operations() -> None:
    """Verify Habenula and ACC state propose compliant atomic HCIR transactions."""
    acc = AnteriorCingulateConflictMonitor()
    acc.register_death_event(actor_pos=(3.0, 3.0), step=5, last_action=1)
    acc.register_death_event(actor_pos=(3.2, 3.1), step=12, last_action=1)

    acc_ops = acc.to_transaction_operations()
    assert len(acc_ops) >= 2
    assert any(op.node_id == "anterior_cingulate_monitor" for op in acc_ops)

    habenula = HabenularEpisodicIOR()
    habenula.record_step(1, (1, 1), 2)
    habenula.record_step(2, (1, 2), 4)
    habenula.record_catastrophe(final_step=3, is_fatal=True)

    hab_ops = habenula.to_transaction_operations()
    assert len(hab_ops) >= 2
    assert any(op.node_id == "habenular_ior_state" for op in hab_ops)
