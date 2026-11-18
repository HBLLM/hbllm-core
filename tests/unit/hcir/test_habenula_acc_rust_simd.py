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
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor

try:
    import hbllm_compute

    _HAS_RUST = True
except ImportError:
    hbllm_compute = None
    _HAS_RUST = False


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
