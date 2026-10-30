"""Unit test suite for HCIR World Round 16 Cognitive Faculties & Native Substrates.

Covers:
- Track 1: Retrosplenial Heading Integration & Head Direction Cell Attractors (Taube-Burgess)
- Track 2: Thalamic Reticular Nucleus (TRN) Selective Inhibitory Gating (Crick-Halassa)
- Track 3: Dorsolateral Prefrontal Cortex (dlPFC) Rule Abstraction & Schema Transfer (Badre-D'Esposito)
- Track 4: Native Structural Graph Minimum Dominating Set & Maximum Independent Set (Greedy Heuristics)
- Engine Orchestration Integration: AutonomousEpistemicEngine Round 16 faculties
"""

from __future__ import annotations

import math

import pytest

from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.dorsolateral_rule_abstraction import (
    AbstractRuleSchema,
    DorsolateralRuleAbstractionEngine,
    RuleTransferResult,
)
from hbllm.hcir.world.retrosplenial_heading import (
    AllocentricHeadingVector,
    HeadDirectionAttractorState,
    RetrosplenialHeadingEngine,
)
from hbllm.hcir.world.thalamic_reticular_gating import (
    SensoryShieldingState,
    ThalamicReticularGatingEngine,
)

# ============================================================================
# Track 1: Retrosplenial Heading Integration & Attractor Tests
# ============================================================================


def test_retrosplenial_heading_integration_and_transforms() -> None:
    """Test angular velocity integration, coordinate rotations, and landmark recalibration."""
    engine = RetrosplenialHeadingEngine(initial_heading_rad=0.0)

    # 1. Rotate 90 degrees clockwise (pi/2 rad East)
    vec: AllocentricHeadingVector = engine.integrate_angular_velocity(
        omega_rad=math.pi / 2.0, dt=1.0
    )
    assert vec.heading_rad == pytest.approx(math.pi / 2.0, rel=1e-3)
    assert vec.heading_degrees == pytest.approx(90.0, rel=1e-3)
    assert vec.confidence > 0.0

    # 2. Transform egocentric forward vector (dist=5, ego_angle=0) while facing East
    dr, dc = engine.egocentric_to_allocentric(distance=5.0, ego_angle_rad=0.0)
    # Facing East: forward is along positive column axis
    assert dr == pytest.approx(0.0, abs=1e-3)
    assert dc == pytest.approx(5.0, abs=1e-3)

    # Inverse transform
    dist, ego_ang = engine.allocentric_to_egocentric(dr=0.0, dc=5.0)
    assert dist == pytest.approx(5.0, rel=1e-3)
    assert ego_ang == pytest.approx(0.0, abs=1e-3)

    # 3. Landmark recalibration with innovation
    # If landmark true bearing is 90 deg and observed bearing is 0 deg, expected matches
    recal = engine.recalibrate_from_landmark(
        observed_bearing_rad=0.0,
        landmark_true_bearing_rad=math.pi / 2.0,
        kalman_gain=0.50,
    )
    assert recal.heading_rad == pytest.approx(math.pi / 2.0, rel=1e-3)
    assert recal.confidence >= vec.confidence

    # 4. Ring attractor population state
    attractor: HeadDirectionAttractorState = engine.get_attractor_population()
    assert attractor.num_bins == 36
    assert len(attractor.tuning_curve) == 36
    assert attractor.bump_amplitude > 0.0

    pred = vec.to_causal_predicate()
    assert pred.variable == "allocentric_heading_vector"
    assert "deg:90.0" in pred.value


# ============================================================================
# Track 2: Thalamic Reticular Nucleus (TRN) Gating Tests
# ============================================================================


def test_thalamic_reticular_gating_and_shielding() -> None:
    """Test selective sensory disinhibition of attended modality and conflict suppression."""
    trn = ThalamicReticularGatingEngine()

    # 1. Attend to visual stream with low conflict
    state_low: SensoryShieldingState = trn.modulate_sensory_shielding(
        attended_modality="visual",
        conflict_level=0.10,
    )
    assert state_low.focused_modality == "visual"
    # Foveal visual channel is completely disinhibited
    ch_foveal = state_low.channels["visual_foveal"]
    assert ch_foveal.pass_ratio == 1.0
    assert ch_foveal.inhibitory_strength == 0.0
    assert ch_foveal.is_shielded is True

    # Signal transmission
    raw_sig = 100.0
    filtered_foveal = trn.filter_signal("visual_foveal", raw_sig)
    assert filtered_foveal == 100.0

    # 2. Elevated cognitive conflict suppresses distractor channels
    state_high: SensoryShieldingState = trn.modulate_sensory_shielding(
        attended_modality="visual",
        conflict_level=0.90,
    )
    ch_auditory = state_high.channels["auditory_ambient"]
    assert ch_auditory.pass_ratio < 0.25
    assert ch_auditory.inhibitory_strength > 0.70
    assert "auditory_ambient" in state_high.suppressed_channels

    filtered_auditory = trn.filter_signal("auditory_ambient", raw_sig)
    assert filtered_auditory < 25.0

    pred = state_high.to_causal_predicate()
    assert pred.variable == "sensory_shielding_state"
    assert "focused:visual" in pred.value


# ============================================================================
# Track 3: Dorsolateral Prefrontal Cortex (dlPFC) Rule Abstraction Tests
# ============================================================================


def test_dorsolateral_rule_abstraction_and_zero_shot_transfer() -> None:
    """Test relational Horn clause induction and zero-shot schema transfer."""
    engine = DorsolateralRuleAbstractionEngine(
        min_support_for_transfer=2, confidence_threshold=0.60
    )

    antecedents = {
        "adjacent_switch": True,
        "door_locked": True,
    }
    action = "toggle_switch"

    # 1. First observation induces rule
    schema1: AbstractRuleSchema = engine.induce_rule_schema(
        abstract_antecedents=antecedents,
        successful_action=action,
    )
    assert schema1.observed_support == 1

    # Before reaching min_support, transfer is not viable
    res_early: RuleTransferResult = engine.evaluate_rule_transfer(antecedents)
    assert res_early.is_transfer_viable is False

    # 2. Second observation reinforces rule
    schema2 = engine.induce_rule_schema(
        abstract_antecedents=antecedents,
        successful_action=action,
    )
    assert schema2.observed_support == 2
    assert schema2.confidence >= 0.70

    # 3. Zero-shot transfer to novel environment matching relational pattern
    res_transfer: RuleTransferResult = engine.evaluate_rule_transfer(antecedents)
    assert res_transfer.is_transfer_viable is True
    assert res_transfer.recommended_action == "toggle_switch"
    assert res_transfer.match_confidence >= 0.70

    # 4. State missing prerequisite does not trigger rule
    unmatched_state = {"adjacent_switch": False, "door_locked": True}
    res_unmatched = engine.evaluate_rule_transfer(unmatched_state)
    assert res_unmatched.is_transfer_viable is False
    assert res_unmatched.recommended_action is None

    # 5. Counter-example penalizes confidence
    conf_before = schema2.confidence
    engine.record_counter_example(abstract_antecedents=antecedents, failed_action=action)
    schema_penalized = engine.schemas[engine._generate_schema_key(antecedents, action)]
    assert schema_penalized.counter_examples == 1
    assert schema_penalized.confidence < conf_before

    pred = res_transfer.to_causal_predicate()
    assert pred.variable == "abstract_rule_transfer"
    assert "action:toggle_switch" in pred.value


# ============================================================================
# Track 4: Minimum Dominating Set & Maximum Independent Set Tests
# ============================================================================


def test_minimum_dominating_set_and_maximum_independent_set() -> None:
    """Test MDS and MIS on star and path topologies."""
    # 1. Star Graph: Center C with 3 leaves L1, L2, L3
    # Center C dominates all 4 nodes! Dominating set is {C}.
    # Maximum Independent Set consists of all 3 leaves {L1, L2, L3}.
    graph_star = CognitiveGraph(enable_native_substrate=False)
    for n in ["C", "L1", "L2", "L3"]:
        graph_star.add_node(FactNode(id=n, claim=n))

    graph_star.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["L1"])
    )
    graph_star.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["L2"])
    )
    graph_star.add_edge(
        HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["L3"])
    )

    dom_set = graph_star.minimum_dominating_set()
    assert dom_set == {"C"}

    ind_set = graph_star.maximum_independent_set()
    assert ind_set == {"L1", "L2", "L3"}

    # 2. Path Graph: A - B - C - D
    graph_path = CognitiveGraph(enable_native_substrate=False)
    for n in ["A", "B", "C", "D"]:
        graph_path.add_node(FactNode(id=n, claim=n))

    graph_path.add_edge(
        HCIREdge(id="e_ab", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    graph_path.add_edge(
        HCIREdge(id="e_bc", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
    )
    graph_path.add_edge(
        HCIREdge(id="e_cd", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["D"])
    )

    dom_path = graph_path.minimum_dominating_set()
    assert len(dom_path) == 2  # 2 nodes dominate all 4

    ind_path = graph_path.maximum_independent_set()
    assert len(ind_path) == 2  # Independent set of size 2


def test_dominating_and_independent_set_native_adapter_parity() -> None:
    """Test parity of MDS and MIS on NativeGraphAdapter."""
    adapter = NativeGraphAdapter()
    for n in ["C", "L1", "L2", "L3"]:
        adapter.add_node(node_id=n, node_type="concept")

    adapter.add_edge("e1", "causes", ["C"], ["L1"])
    adapter.add_edge("e2", "causes", ["C"], ["L2"])
    adapter.add_edge("e3", "causes", ["C"], ["L3"])

    dom = adapter.minimum_dominating_set()
    assert dom == {"C"}

    ind = adapter.maximum_independent_set()
    assert ind == {"L1", "L2", "L3"}


# ============================================================================
# Engine Orchestration Tests
# ============================================================================


def test_autonomous_epistemic_engine_round16_orchestration() -> None:
    """Test full integration of Round 16 faculties into AutonomousEpistemicEngine."""
    engine = AutonomousEpistemicEngine()

    # 1. Retrosplenial heading update
    heading_vec = engine.update_allocentric_heading(omega_rad=math.pi / 4.0, dt=1.0)
    assert heading_vec.heading_degrees == pytest.approx(45.0, rel=1e-3)

    # Recalibrate from landmark
    recal = engine.recalibrate_heading_from_landmark(
        observed_bearing_rad=0.0,
        landmark_true_bearing_rad=math.pi / 4.0,
    )
    assert recal.confidence > 0.0

    # 2. TRN sensory shielding
    shielding = engine.apply_trn_sensory_shielding(attended_modality="visual", conflict_level=0.75)
    assert shielding.focused_modality == "visual"
    assert "auditory_ambient" in shielding.suppressed_channels

    # 3. dlPFC rule abstraction and transfer
    antecedents = {"near_door": True, "has_key": True}
    transfer = engine.induce_and_transfer_abstract_rules(
        antecedents=antecedents, action="open_door"
    )
    assert transfer.is_transfer_viable is False  # Only 1 observation

    # Second observation reinforces
    transfer2 = engine.induce_and_transfer_abstract_rules(
        antecedents=antecedents, action="open_door"
    )
    assert transfer2.is_transfer_viable is True
    assert transfer2.recommended_action == "open_door"

    # 4. Graph MDS and MIS
    test_graph = CognitiveGraph(enable_native_substrate=False)
    for n in ["A", "B", "C"]:
        test_graph.add_node(FactNode(id=n, claim=n))
    test_graph.add_edge(
        HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
    )
    test_graph.add_edge(
        HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["C"])
    )

    dom_set = engine.compute_graph_dominating_set(test_graph)
    assert dom_set == {"A"}

    ind_set = engine.compute_graph_independent_set(test_graph)
    assert ind_set == {"B", "C"}

    # Verify causal predicates logged into symbolic theory
    assert engine.symbolic_theory is not None
    pred_variables = [p.variable for p in engine.symbolic_theory.causal_predicates]
    assert "allocentric_heading_vector" in pred_variables
    assert "sensory_shielding_state" in pred_variables
    assert "abstract_rule_transfer" in pred_variables
