"""Unit tests for the Frontier Cognitive Synthesis Faculties (Tracks 1 - 5).

Covers:
- Track 1: Hierarchical Predictive Coding Cortical Columns (Rao & Ballard / Friston)
- Track 2: Entorhinal Grid Cells & Hippocampal Path Integration (Moser & O'Keefe)
- Track 3: Metacognitive Confidence Calibration & BA10 Agency Monitor (Fleming & Dolan)
- Track 4: Declarative Neuro-Symbolic Skill Synthesis & DSL Program Induction (DreamCoder)
- Track 5: Native Rust Acceleration for Kuramoto Oscillators & Grid Attractors (hbllm_compute)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hbllm.hcir.bytecode import Opcode
from hbllm.hcir.graph import (
    HCIRNodeType,
)
from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.workspace_tiers import (
    TieredWorkspace,
)
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.experimental.entorhinal_grid_cortex import (
    EntorhinalGridCortex,
    PlaceFieldRepresentation,
)
from hbllm.hcir.world.metacognitive_agency_cortex import (
    MetacognitiveAgencyCortex,
    MetacognitiveAssessment,
)
from hbllm.hcir.world.predictive_coding_cortex import (
    HierarchicalPredictiveCodingCortex,
    PredictiveCodingSnapshot,
)
from hbllm.hcir.world.skill_program_synthesizer import (
    ActionTraceStep,
    SkillProgramSynthesizer,
)

try:
    import hbllm_compute  # type: ignore[import-not-found]  # noqa: F401

    HAS_HBLLM_COMPUTE = True
except ImportError:
    HAS_HBLLM_COMPUTE = False


# ─────────────────────────────────────────────────────────────────────────────
# Track 1: Hierarchical Predictive Coding Cortical Columns
# ─────────────────────────────────────────────────────────────────────────────


def test_predictive_coding_bidirectional_cycle() -> None:
    cortex = HierarchicalPredictiveCodingCortex(learning_rate_kappa=0.3, anomaly_threshold=2.0)
    assert len(cortex.layers) == 3

    # Sensory input at Layer 0 (e.g. avatar coordinates and velocity)
    sensory = {"avatar_r": 2.0, "avatar_c": 5.0, "hazard_dist": 3.0}
    # Higher invariant prior at Layer 2
    priors = {"avatar_r": 2.0, "avatar_c": 4.0}

    snapshot = cortex.step_hierarchical_cycle(
        sensory_inputs=sensory,
        abstract_priors=priors,
    )

    assert isinstance(snapshot, PredictiveCodingSnapshot)
    assert len(snapshot.layer_errors) == 3
    assert snapshot.total_free_energy >= 0.0

    # Layer 0 error should reflect sensory discrepancy
    l0_err = cortex.layers[0].prediction_error
    assert "avatar_c" in l0_err

    # Check causal predicate export
    predicates = cortex.to_causal_predicates()
    assert len(predicates) == 3


def test_predictive_coding_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)
    snap = engine.step_predictive_coding(
        sensory_inputs={"target_r": 1.0, "target_c": 1.0},
        abstract_priors={"target_r": 1.0, "target_c": 1.0},
    )
    assert isinstance(snap, PredictiveCodingSnapshot)
    assert not snap.is_structural_anomaly


# ─────────────────────────────────────────────────────────────────────────────
# Track 2: Entorhinal Grid Cells & Path Integration
# ─────────────────────────────────────────────────────────────────────────────


def test_entorhinal_grid_cells_path_integration() -> None:
    grid_cortex = EntorhinalGridCortex(wavelengths=[4.0, 8.0, 16.0])
    grid_cortex.reset_anchor((0, 0))

    # Move right 2 steps
    est = grid_cortex.path_integrate_step(delta_r=0.0, delta_c=2.0)
    assert est == (0, 2)

    # Move down 3 steps
    est = grid_cortex.path_integrate_step(delta_r=3.0, delta_c=0.0)
    assert est == (3, 2)

    # Check theta phase precession forward trajectory compression
    trajectory = [(3, 3), (3, 4), (4, 4), (5, 4), (5, 5)]
    place = grid_cortex.generate_place_field_readout(planned_trajectory=trajectory)
    assert isinstance(place, PlaceFieldRepresentation)
    assert place.estimated_coord == (3, 2)
    assert len(place.theta_lookahead_waypoints) > 1
    assert len(place.population_vector) == 3 * 4  # 3 modules * (cos, sin for x & y)


def test_entorhinal_grid_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)
    engine.entorhinal_grid_cortex.reset_anchor((1, 1))
    new_pos = engine.path_integrate_odometry(delta_r=1.0, delta_c=2.0)
    assert new_pos == (2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Track 3: Metacognitive Confidence & Agency Monitor
# ─────────────────────────────────────────────────────────────────────────────


def test_metacognitive_agency_attribution_and_governor() -> None:
    cortex = MetacognitiveAgencyCortex(base_simulation_depth=5, max_simulation_depth=15)

    # Scenario A: Predicted avatar movement matches observed displacement perfectly -> High agency
    agency_high = cortex.compute_agency_attribution(
        predicted_avatar_pos=(2, 3),
        observed_avatar_pos=(2, 3),
        observed_other_changes_count=0,
    )
    assert agency_high > 0.90

    assess_reflex = cortex.calibrate_confidence_and_govern_compute(
        action_id=1,
        raw_confidence=0.92,
        agency_score=agency_high,
        empirical_success=True,
    )
    assert assess_reflex.recommended_exploration_mode == "reflex"
    assert assess_reflex.recommended_simulation_depth == 2

    # Scenario B: Predicted avatar movement failed (blocked) or unexpected drift -> Low agency
    agency_low = cortex.compute_agency_attribution(
        predicted_avatar_pos=(2, 3),
        observed_avatar_pos=(2, 2),
        observed_other_changes_count=5,  # Exogenous changes occurred
    )
    assert agency_low < 0.40

    assess_deep = cortex.calibrate_confidence_and_govern_compute(
        action_id=2,
        raw_confidence=0.35,
        agency_score=agency_low,
        empirical_success=False,
    )
    assert assess_deep.recommended_simulation_depth >= 8
    assert assess_deep.recommended_exploration_mode in ["epistemic_probe", "deep_simulation"]


def test_metacognitive_agency_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)
    assess = engine.assess_metacognitive_agency(
        action_id=1,
        raw_confidence=0.88,
        predicted_avatar_pos=(1, 2),
        observed_avatar_pos=(1, 2),
        observed_other_changes_count=0,
        empirical_success=True,
    )
    assert isinstance(assess, MetacognitiveAssessment)
    assert assess.agency_attribution_score > 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Track 4: Declarative Neuro-Symbolic Skill Synthesis
# ─────────────────────────────────────────────────────────────────────────────


def test_skill_program_synthesizer_induction() -> None:
    synthesizer = SkillProgramSynthesizer(min_support_traces=2)

    trace_a = [
        ActionTraceStep(intent=SpatialActionIntent.NAVIGATE_TO, target_query={"role": "key"}),
        ActionTraceStep(intent=SpatialActionIntent.INTERACT, target_query={"role": "door"}),
    ]
    trace_b = [
        ActionTraceStep(intent=SpatialActionIntent.NAVIGATE_TO, target_query={"role": "key"}),
        ActionTraceStep(intent=SpatialActionIntent.INTERACT, target_query={"role": "door"}),
        ActionTraceStep(intent=SpatialActionIntent.NAVIGATE_TO, target_query={"role": "exit"}),
    ]

    synthesizer.record_winning_trace(trace_a, required_actions={1, 2})
    synthesizer.record_winning_trace(trace_b, required_actions={1, 2, 3})

    # Find frequent subsequences
    frequent = synthesizer.find_frequent_subsequences(min_length=2)
    assert len(frequent) > 0
    top_pattern, count = frequent[0]
    assert count == 2
    assert top_pattern == (SpatialActionIntent.NAVIGATE_TO, SpatialActionIntent.INTERACT)

    # Induce DeclarativeNeuroSymbolicSkill
    skill = synthesizer.synthesize_skill_from_pattern(top_pattern, skill_name="KeyDoorUnlockSkill")
    assert isinstance(skill, DeclarativeNeuroSymbolicSkill)
    assert skill.skill_name == "KeyDoorUnlockSkill"
    assert skill.program is not None

    # Verify HCIR VM Bytecode compilation
    stream = skill.program.compile_to_stream(skill_name="KeyDoorUnlockSkill")
    assert len(stream.instructions) == 6  # 2 subgoals * 3 (QUERY, ASSERT, EXECUTE)
    opcodes = [ins.opcode for ins in stream.instructions]
    assert Opcode.QUERY in opcodes
    assert Opcode.ASSERT in opcodes
    assert Opcode.EXECUTE in opcodes


def test_skill_synthesizer_engine_integration() -> None:
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)
    trace = [
        ActionTraceStep(intent=SpatialActionIntent.NAVIGATE_TO),
        ActionTraceStep(intent=SpatialActionIntent.INTERACT),
    ]
    engine.skill_synthesizer.record_winning_trace(trace)
    engine.skill_synthesizer.record_winning_trace(trace)
    skills = engine.synthesize_procedural_skills_from_traces(min_support=2)
    assert len(skills) >= 1
    assert isinstance(skills[0], DeclarativeNeuroSymbolicSkill)


def test_active_inference_loop_and_consolidation_pipeline() -> None:
    """Verify active inference updates (odometry, predictive coding, agency) and persistent consolidation."""
    tiered_ws = TieredWorkspace()
    frame = tiered_ws.create_task_frame("goal_test")
    engine = AutonomousEpistemicEngine(exploration_budget=50)

    _restore_all(engine)
    engine.attach_tiered_workspace(tiered_ws, frame.frame_id)

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[2, 2] = 1  # avatar
    grid2 = np.zeros((10, 10), dtype=int)
    grid2[2, 3] = 1  # avatar displaced by (0, 1)

    # 1. First perception step
    engine.perceive(grid1)
    engine.prev_grid = grid1
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.avatar_pos = (2, 2)
    engine.last_action = 1
    engine.last_predicted_pos = (2, 3)

    # 2. Feedback assimilation triggers odometry, predictive coding, and agency attribution
    engine.assimilate_feedback(
        curr_grid=grid2,
        available_actions=[1, 2, 3, 4],
        action=1,
    )

    # Verify odometric displacement occurred
    assert engine.entorhinal_grid_cortex.current_estimated_coord != (0, 0)
    # Verify predictive coding layer errors recorded
    assert engine.predictive_coding_cortex.total_free_energy >= 0.0
    # Verify metacognitive agency history has assessment
    assert len(engine.metacognitive_agency_cortex.assessment_history) >= 1

    # 3. Simulate winning transitions in episodic buffer and test consolidation
    engine.episodic_cortex.record_transition(
        step=1,
        avatar_pos=(2, 2),
        action=1,
        reward=0.1,
    )
    engine.episodic_cortex.record_transition(
        step=2,
        avatar_pos=(2, 3),
        action=2,
        reward=1.0,
        is_success=True,
    )

    # Repeat once to achieve min_support=2 for trace pattern induction
    engine.skill_synthesizer.min_support_traces = 2
    engine.consolidate_episodic_memory(terminal_reward=1.0)
    engine.episodic_cortex.record_transition(
        step=1,
        avatar_pos=(2, 2),
        action=1,
        reward=0.1,
    )
    engine.episodic_cortex.record_transition(
        step=2,
        avatar_pos=(2, 3),
        action=2,
        reward=1.0,
        is_success=True,
    )
    res = engine.consolidate_episodic_memory(terminal_reward=1.0)

    assert "induced_skills" in res
    assert len(res["induced_skills"]) >= 1
    # Verify skill node promoted into persistent tier
    pers_nodes = tiered_ws.persistent.graph.nodes_by_type(HCIRNodeType.SKILL)
    assert len(pers_nodes) >= 1


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_kuramoto_phase_step_simd() -> None:
    phases = [0.1, 0.4]
    omegas = [2.0, 2.0]
    prox = [[1.0, 0.8], [0.8, 1.0]]
    coupling_k = 2.0
    dt = 0.05

    new_phases = hbllm_compute.kuramoto_phase_step_simd(phases, omegas, prox, coupling_k, dt)
    assert len(new_phases) == 2
    for p in new_phases:
        assert 0.0 <= p < 2.0 * math.pi


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_grid_cell_path_integrate_simd() -> None:
    px = [0.0, 0.0]
    py = [0.0, 0.0]
    wl = [4.0, 8.0]
    # Move delta_c = 2.0
    # For wl = 4.0: phase shift = (2*pi / 4.0) * 2.0 = pi = ~3.14159
    # For wl = 8.0: phase shift = (2*pi / 8.0) * 2.0 = pi / 2 = ~1.57079
    out_x, out_y = hbllm_compute.grid_cell_path_integrate_simd(px, py, wl, 0.0, 2.0)
    assert len(out_x) == 2
    assert abs(out_x[0] - math.pi) < 1e-4
    assert abs(out_x[1] - (math.pi / 2.0)) < 1e-4
    assert abs(out_y[0] - 0.0) < 1e-4


@pytest.mark.skipif(not HAS_HBLLM_COMPUTE, reason="hbllm_compute native module not compiled")
def test_rust_universal_engine_frontier_methods() -> None:
    engine = hbllm_compute.UniversalEngine()
    assert hasattr(engine, "kuramoto_phase_step_simd")
    assert hasattr(engine, "grid_cell_path_integrate_simd")

    res_k = engine.kuramoto_phase_step_simd([0.0], [1.0], [[1.0]], 1.0, 0.1)
    assert len(res_k) == 1

    out_x, out_y = engine.grid_cell_path_integrate_simd([0.0], [0.0], [6.0], 1.0, 0.0)
    assert len(out_x) == 1
    assert len(out_y) == 1
