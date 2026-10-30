"""Unit tests for HCIR World Round 19 faculties:
1. Hippocampal-Entorhinal Boundary-to-Grid Phase Alignment (Bush / Giocomo)
2. Medial Prefrontal Schema Induction & Invariant Abstraction (Euston / McKenzie)
3. Subthalamic Nucleus (STN) Decision Threshold Escalation & Conflict Braking (Frank)
4. Native Structural Graph Transitive Reduction & Essential Causal Skeleton
5. AutonomousEpistemicEngine Round 19 Integration
"""

from __future__ import annotations

from hbllm.hcir.graph import CognitiveGraph, FactNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.boundary_grid_phase_aligner import (
    BoundaryGridPhaseAligner,
    BoundaryPhaseAlignmentState,
)
from hbllm.hcir.world.medial_prefrontal_schema import (
    MedialPrefrontalSchemaEngine,
    SchemaGeneralizationReport,
)
from hbllm.hcir.world.subthalamic_threshold_escalator import (
    STNBrakingStatus,
    SubthalamicThresholdEscalator,
)


def test_boundary_grid_phase_aligner() -> None:
    """Test boundary-to-grid Kalman phase innovation and drift correction."""
    aligner = BoundaryGridPhaseAligner(base_kalman_gain=0.5, proximity_threshold=5.0)

    # Agent estimated at (2.0, 2.0) in a 16x16 arena.
    # Observed boundary distances: north wall is at distance 1.0 (actual r is 1.0, not 2.0)
    observed = {
        "north": 1.0,  # Expected 2.0 -> residual -1.0 -> should reduce r
        "west": 2.0,  # Expected 2.0 -> residual 0.0 -> no drift
    }

    state = aligner.align_grid_phase_with_boundaries(
        estimated_pos=(2.0, 2.0),
        grid_dimensions=(16, 16),
        observed_boundary_distances=observed,
    )

    assert isinstance(state, BoundaryPhaseAlignmentState)
    assert state.raw_position == (2.0, 2.0)
    assert state.corrected_position[0] < 2.0  # North correction shifted position toward 1.0
    assert state.total_drift_corrected > 0.0
    assert state.alignment_confidence > 0.40

    pred = state.to_causal_predicate()
    assert pred.variable == "boundary_grid_phase_alignment"
    assert "corr_r" in pred.value


def test_medial_prefrontal_schema_engine() -> None:
    """Test invariant relational schema induction, matching, and prior transfer."""
    engine = MedialPrefrontalSchemaEngine(match_threshold=0.30)

    # Observed scene with an actor, a key (mediator), and a door (barrier)
    entity_roles = {
        "avatar_1": "actor",
        "key_gold": "mediator",
        "door_iron": "barrier",
        "chest_target": "target",
    }
    relations = (
        ("avatar_1", "navigates_to", "key_gold"),
        ("key_gold", "disables", "door_iron"),
        ("avatar_1", "reaches", "chest_target"),
    )

    report = engine.match_schema_context(entity_roles, relations)
    assert isinstance(report, SchemaGeneralizationReport)
    assert report.dominant_schema is not None
    assert report.dominant_schema.archetype_name == "barrier_clearance"
    assert report.dominant_schema.match_confidence > 0.50
    assert "mediator" in report.dominant_schema.role_bindings
    assert report.dominant_schema.role_bindings["mediator"] == "key_gold"

    pred = report.dominant_schema.to_causal_predicate()
    assert pred.variable == "active_schema_binding"
    assert "barrier_clearance" in pred.value


def test_subthalamic_threshold_escalator() -> None:
    """Test dynamic DDM decision threshold escalation and conflict braking."""
    escalator = SubthalamicThresholdEscalator(
        baseline_threshold=1.0,
        conflict_escalation_gain=2.0,
    )

    # Scenario 1: Low conflict, clear dominant action (Q-values: 10.0 vs 2.0 -> margin 8.0)
    decision_clear = escalator.evaluate_decision_threshold([10.0, 2.0])
    assert decision_clear.status == STNBrakingStatus.DISPATCH_PERMITTED
    assert decision_clear.accumulated_evidence == 8.0
    assert decision_clear.recommended_deliberation_cycles == 0

    # Scenario 2: High conflict, close options (Q-values: 5.0 vs 4.9 -> margin 0.1)
    decision_conflict = escalator.evaluate_decision_threshold([5.0, 4.9])
    assert decision_conflict.status == STNBrakingStatus.BRAKE_ACTIVE
    assert decision_conflict.detected_conflict > 0.80
    assert decision_conflict.effective_threshold > decision_conflict.baseline_threshold
    assert decision_conflict.recommended_deliberation_cycles > 0

    # Scenario 3: High urgency overrides conflict after prolonged deliberation
    decision_urgency = escalator.evaluate_decision_threshold(
        [5.0, 4.8], elapsed_deliberation_steps=30
    )
    assert decision_urgency.status in (
        STNBrakingStatus.EMERGENCY_OVERRIDE,
        STNBrakingStatus.DISPATCH_PERMITTED,
    )

    pred = decision_conflict.to_causal_predicate()
    assert pred.variable == "stn_conflict_threshold"
    assert "brake_active" in pred.value


def test_transitive_reduction_and_causal_skeleton() -> None:
    """Test DAG transitive reduction removing redundant shortcut edges."""
    adapter = NativeGraphAdapter()

    # Create a DAG: A -> B -> C and direct shortcut A -> C
    adapter.add_node("A", "FACT")
    adapter.add_node("B", "FACT")
    adapter.add_node("C", "FACT")

    adapter.add_edge("e_ab", "CAUSAL_LINKS", ["A"], ["B"])
    adapter.add_edge("e_bc", "CAUSAL_LINKS", ["B"], ["C"])
    adapter.add_edge("e_ac", "CAUSAL_LINKS", ["A"], ["C"])

    skeleton = adapter.essential_causal_skeleton()
    assert ("A", "B") in skeleton
    assert ("B", "C") in skeleton
    assert ("A", "C") not in skeleton  # Redundant direct edge eliminated!

    # CognitiveGraph equivalence
    graph = CognitiveGraph(enable_native_substrate=False)
    graph.add_node(FactNode(id="A", claim="A"))
    graph.add_node(FactNode(id="B", claim="B"))
    graph.add_node(FactNode(id="C", claim="C"))

    graph.add_edge(HCIREdge(sources=["A"], targets=["B"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(sources=["B"], targets=["C"], edge_type=HCIREdgeType.CAUSES))
    graph.add_edge(HCIREdge(sources=["A"], targets=["C"], edge_type=HCIREdgeType.CAUSES))

    graph_skeleton = graph.essential_causal_skeleton()
    assert ("A", "B") in graph_skeleton
    assert ("B", "C") in graph_skeleton
    assert ("A", "C") not in graph_skeleton


def test_autonomous_epistemic_engine_round19_integration() -> None:
    """Verify that AutonomousEpistemicEngine initializes and wires all Round 19 faculties."""
    engine = AutonomousEpistemicEngine(enable_logging=False)

    assert hasattr(engine, "boundary_grid_aligner")
    assert isinstance(engine.boundary_grid_aligner, BoundaryGridPhaseAligner)

    assert hasattr(engine, "mpfc_schema_engine")
    assert isinstance(engine.mpfc_schema_engine, MedialPrefrontalSchemaEngine)

    assert hasattr(engine, "stn_escalator")
    assert isinstance(engine.stn_escalator, SubthalamicThresholdEscalator)
