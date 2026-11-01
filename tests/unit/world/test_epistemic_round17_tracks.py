"""Unit tests for HCIR Round 17 Cognitive Substrate Enhancements.

Tracks Tested:
1. Track 1: Hippocampal CA1 Temporal Distance Cells & Relative Interval Timing (hippocampal_time_cell.py)
2. Track 2: Ventrolateral Prefrontal Cortex (vlPFC / rIFG) Motor Rule Retrieval & Response Inhibition (ventrolateral_motor_inhibition.py)
3. Track 3: Claustrum Dynamic Inter-Hemispheric Synchrony & Cross-Modal Gestalt Binding (claustrum_interhemispheric.py)
4. Track 4: Native Structural Graph Minimum Vertex Cover & Greedy Edge Coloring (native_adapter.py & graph.py)
5. Track 5: Autonomous Epistemic Engine Round 17 Integrated Orchestration
"""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import (
    CognitiveGraph,
    FactNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.claustrum_interhemispheric import (
    ClaustrumInterhemisphericIntegrator,
)
from hbllm.hcir.world.cortex_causal import (
    HCIRSymbolicWorldTheory,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.hippocampal_time_cell import (
    HippocampalTimeCellEngine,
)
from hbllm.hcir.world.ventrolateral_motor_inhibition import (
    InhibitionRegime,
    VentrolateralMotorInhibitionEngine,
)


class TestHippocampalTimeCells:
    """Track 1: CA1 temporal distance cells and relative interval timing."""

    def test_weber_scalar_timing_and_rescaling(self) -> None:
        engine = HippocampalTimeCellEngine(default_sigma_0=0.10, scalar_beta=0.20)
        cells = engine.register_interval("delay_phase", duration=10.0, num_cells=6, spacing="weber")

        assert len(cells) == 6
        # Peak times must be monotonically increasing
        peaks = [c.peak_time for c in cells]
        assert peaks == sorted(peaks)
        assert peaks[0] == pytest.approx(0.0, abs=1e-3)
        assert peaks[-1] == pytest.approx(10.0, abs=1e-3)

        # Weber-Fechner scalar widening: later cells must have larger tuning widths
        assert cells[-1].tuning_width > cells[0].tuning_width

        # Test peak firing rate at exact cell peak
        c2 = cells[2]
        rate_at_peak = c2.firing_rate(c2.peak_time)
        assert rate_at_peak == pytest.approx(1.0, abs=1e-3)
        rate_away = c2.firing_rate(c2.peak_time + 2.0)
        assert rate_away < rate_at_peak

        # Step time and check population state
        states = engine.step(dt=peaks[2], interval_id="delay_phase")
        state = states["delay_phase"]
        assert state.elapsed_time == pytest.approx(peaks[2], abs=1e-3)
        assert state.active_cell_id == c2.cell_id

        # Estimate remaining time
        remaining, conf = engine.estimate_time_remaining("delay_phase")
        assert remaining == pytest.approx(10.0 - peaks[2], abs=1e-3)
        assert 0.0 < conf <= 1.0

        # Rescale interval duration from 10.0 to 20.0
        rescaled_cells = engine.rescale_interval("delay_phase", new_duration=20.0)
        assert rescaled_cells[-1].peak_time == pytest.approx(20.0, abs=1e-3)
        assert rescaled_cells[2].peak_time == pytest.approx(peaks[2] * 2.0, abs=1e-3)

        # Causal credit bridging discount
        credit_short = engine.bridge_causal_credit(delta_t=1.0)
        credit_long = engine.bridge_causal_credit(delta_t=5.0)
        assert credit_short > credit_long > 0.0

        # Predicate conversion
        pred = state.to_causal_predicate()
        assert pred.variable == "temporal_interval_progress"
        assert "delay_phase" in str(pred.value)


class TestVentrolateralMotorInhibition:
    """Track 2: vlPFC motor rule retrieval and proactive/reactive response inhibition."""

    def test_proactive_and_reactive_inhibition_modes(self) -> None:
        engine = VentrolateralMotorInhibitionEngine(reactive_decay_rate=2.0)

        # 1. Unrestricted baseline
        cmd_base = engine.evaluate_action_gating(intended_action="move_forward")
        assert cmd_base.permitted is True
        assert cmd_base.regime == InhibitionRegime.UNRESTRICTED
        assert cmd_base.sanctioned_action == "move_forward"

        # 2. Proactive braking under high risk / caution
        engine.set_proactive_brake(0.50, reason="slippery_terrain")
        cmd_proactive = engine.evaluate_action_gating(intended_action="move_forward")
        assert cmd_proactive.permitted is True
        assert cmd_proactive.regime == InhibitionRegime.PROACTIVE_BRAKING
        assert cmd_proactive.brake_factor >= 0.50

        # 3. Reactive hyperdirect stopping
        engine.trigger_reactive_stop(source="sudden_hazard", severity=0.80)
        cmd_stop = engine.evaluate_action_gating(intended_action="move_forward")
        assert cmd_stop.permitted is False
        assert cmd_stop.regime == InhibitionRegime.REACTIVE_STOP
        assert "sudden_hazard" in cmd_stop.cancellation_reason

        # 4. Decay reactive stop
        engine.decay_inhibition(dt=0.50)
        cmd_decayed = engine.evaluate_action_gating(intended_action="move_forward")
        # After decaying below 0.5, permitted again (under proactive brake)
        assert cmd_decayed.permitted is True

        # 5. Emergency collision abort
        cmd_emergency = engine.evaluate_action_gating(
            intended_action="move_forward",
            risk_score=0.90,
            time_to_impact=0.0,
        )
        assert cmd_emergency.permitted is False
        assert cmd_emergency.regime == InhibitionRegime.EMERGENCY_ABORT

        # 6. Rule-based inhibitory motor gating
        engine.register_inhibitory_rule(
            condition_key="red_barrier_locked",
            forbidden_action="advance",
            safe_alternative="search_for_key",
        )
        cmd_rule = engine.evaluate_action_gating(
            intended_action="advance",
            active_conditions={"red_barrier_locked"},
        )
        assert cmd_rule.permitted is False
        assert cmd_rule.sanctioned_action == "search_for_key"
        assert "red_barrier_locked" in cmd_rule.cancellation_reason

        pred = cmd_rule.to_causal_predicate()
        assert pred.variable == "motor_response_inhibition"
        assert "permitted:False" in str(pred.value)


class TestClaustrumInterhemisphericGestalt:
    """Track 3: Claustrum cross-modal binding and inter-hemispheric synchrony."""

    def test_cross_modal_binding_and_dissonance(self) -> None:
        conductor = ClaustrumInterhemisphericIntegrator(coherence_threshold=0.60)

        # 1. Harmonic cross-modal packet
        p_harmonic = conductor.bind_cross_modal_streams(
            timestamp=1.0,
            spatial_state={"is_clear": True, "target_dist": 3.0},
            somatic_valence=0.40,
            haptic_friction=0.85,
            threat_vigilance=0.10,
            temporal_progress=0.50,
            rule_active=False,
        )
        assert p_harmonic.gestalt_coherence >= 0.90
        assert len(p_harmonic.discordant_modalities) == 0
        assert p_harmonic.to_causal_predicate().variable == "claustrum_gestalt_synchrony"

        # 2. Dissonance: Spatial reports open, but somatic reports severe visceral alarm
        p_dissonant = conductor.bind_cross_modal_streams(
            timestamp=2.0,
            spatial_state={"is_clear": True},
            somatic_valence=-0.75,
            haptic_friction=0.85,
            threat_vigilance=0.10,
            temporal_progress=0.50,
        )
        assert p_dissonant.gestalt_coherence < 0.85
        assert "spatial_vs_somatic_alarm" in p_dissonant.discordant_modalities

        # 3. Dynamic channel decoupling
        conductor.decouple_channel("haptic")
        assert "haptic" in conductor.get_binding_state().decoupled_channels
        p_decoupled = conductor.bind_cross_modal_streams(
            timestamp=3.0,
            spatial_state={"is_clear": True},
            somatic_valence=0.30,
            haptic_friction=0.05,  # slippery, but haptic is decoupled
            temporal_progress=0.80,
        )
        assert "haptic" not in p_decoupled.bound_modalities
        assert "friction_slip_under_progress" not in p_decoupled.discordant_modalities


class TestStructuralGraphVertexCoverAndColoring:
    """Track 4: Minimum Vertex Cover & Greedy Edge Coloring in Native Adapter & CognitiveGraph."""

    def test_minimum_vertex_cover(self) -> None:
        graph = CognitiveGraph(enable_native_substrate=False)
        for node in ["A", "B", "C", "D"]:
            graph.add_node(FactNode(id=node, claim=node))

        # Star graph with center A and leaves B, C, D
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["C"])
        )
        graph.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["D"])
        )

        mvc = graph.minimum_vertex_cover()
        # In a star graph, the single center vertex covers all 3 edges
        assert mvc == {"A"}

        # Triangle graph K3: A-B, B-C, C-A
        graph_tri = CognitiveGraph(enable_native_substrate=False)
        for node in ["A", "B", "C"]:
            graph_tri.add_node(FactNode(id=node, claim=node))
        graph_tri.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph_tri.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )
        graph_tri.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["A"])
        )

        mvc_tri = graph_tri.minimum_vertex_cover()
        assert len(mvc_tri) == 2
        # Verify valid cover: every edge has at least one endpoint in mvc_tri
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        for u, v in edges:
            assert u in mvc_tri or v in mvc_tri

        # Parity check on NativeGraphAdapter
        adapter = NativeGraphAdapter()
        for node in ["A", "B", "C"]:
            adapter.add_node(node, node_type="concept")
        adapter.add_edge("e1", "causes", ["A"], ["B"])
        adapter.add_edge("e2", "causes", ["B"], ["C"])
        adapter.add_edge("e3", "causes", ["C"], ["A"])
        mvc_adapter = adapter.minimum_vertex_cover()
        assert len(mvc_adapter) == 2
        for u, v in edges:
            assert u in mvc_adapter or v in mvc_adapter

    def test_greedy_edge_coloring(self) -> None:
        graph = CognitiveGraph(enable_native_substrate=False)
        for node in ["A", "B", "C", "D"]:
            graph.add_node(FactNode(id=node, claim=node))

        # Path graph: A-B, B-C, C-D (max degree = 2)
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["A"], targets=["B"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["B"], targets=["C"])
        )
        graph.add_edge(
            HCIREdge(id="e3", edge_type=HCIREdgeType.CAUSES, sources=["C"], targets=["D"])
        )

        coloring = graph.greedy_edge_coloring()
        assert len(coloring) == 3

        # Edges incident to B: (A, B) and (B, C) must have different colors
        c_ab = coloring[("A", "B")]
        c_bc = coloring[("B", "C")]
        c_cd = coloring[("C", "D")]
        assert c_ab != c_bc
        assert c_bc != c_cd

        # Parity check on NativeGraphAdapter
        adapter = NativeGraphAdapter()
        for node in ["A", "B", "C", "D"]:
            adapter.add_node(node, node_type="concept")
        adapter.add_edge("e1", "causes", ["A"], ["B"])
        adapter.add_edge("e2", "causes", ["B"], ["C"])
        adapter.add_edge("e3", "causes", ["C"], ["D"])

        coloring_adapter = adapter.greedy_edge_coloring()
        assert len(coloring_adapter) == 3
        assert coloring_adapter[("A", "B")] != coloring_adapter[("B", "C")]
        assert coloring_adapter[("B", "C")] != coloring_adapter[("C", "D")]


class TestAutonomousEpistemicEngineRound17Orchestration:
    """Track 5: Integrated AutonomousEpistemicEngine Round 17 faculties."""

    def test_engine_round17_orchestration(self) -> None:
        engine = AutonomousEpistemicEngine()

        _restore_all(engine)
        engine.symbolic_theory = HCIRSymbolicWorldTheory()

        # 1. Temporal Interval Tracking
        interval_state = engine.track_temporal_interval(
            interval_id="navigation_window",
            duration=15.0,
            dt=3.0,
        )
        assert interval_state.elapsed_time == pytest.approx(3.0, abs=1e-3)
        assert interval_state.progress == pytest.approx(3.0 / 15.0, abs=1e-3)
        assert any(
            p.variable == "temporal_interval_progress"
            for p in engine.symbolic_theory.causal_predicates
        )

        # 2. Motor Response Inhibition
        inhib_cmd = engine.evaluate_motor_response_inhibition(
            intended_action="forward_thrust",
            conflict_level=0.1,
            risk_score=0.2,
        )
        assert inhib_cmd.permitted is True
        assert any(
            p.variable == "motor_response_inhibition"
            for p in engine.symbolic_theory.causal_predicates
        )

        # 3. Claustrum Cross-Modal Gestalt Binding
        packet = engine.bind_cross_modal_gestalt(
            timestamp=1.0,
            spatial_state={"is_clear": True},
            somatic_valence=0.5,
            haptic_friction=0.9,
            threat_vigilance=0.1,
            temporal_progress=interval_state.progress,
        )
        assert packet.gestalt_coherence >= 0.85
        assert any(
            p.variable == "claustrum_gestalt_synchrony"
            for p in engine.symbolic_theory.causal_predicates
        )

        # 4. Graph Structural Operations
        graph = CognitiveGraph(enable_native_substrate=False)
        for n in ["X", "Y", "Z"]:
            graph.add_node(FactNode(id=n, claim=n))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.CAUSES, sources=["X"], targets=["Y"])
        )
        graph.add_edge(
            HCIREdge(id="e2", edge_type=HCIREdgeType.CAUSES, sources=["Y"], targets=["Z"])
        )

        mvc = engine.compute_graph_vertex_cover(graph)
        assert mvc == {"Y"}

        coloring = engine.compute_graph_edge_coloring(graph)
        assert len(coloring) == 2
        assert coloring[("X", "Y")] != coloring[("Y", "Z")]
