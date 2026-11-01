from __future__ import annotations

"""Unit Tests for Advanced Cortical Enhancements (Tracks 1 - 4).

1. Track 1: SNN Spatio-Temporal Hazard Forecasting & Cerebellar Phase Entrainment
2. Track 2: Transparent Native Substrate Backing in CognitiveGraph
3. Track 3: Global Neuronal Workspace Ignition & Closed-Loop Neuromodulation
4. Track 4: Frontopolar Recursive Subgoal Decomposition & Causal Precondition Chaining
"""

import numpy as np

from hbllm.hcir.graph import (
    CognitiveCategory,
    CognitiveGraph,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNode,
    HCIRNodeType,
)
from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine, EpistemicPhase
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor

# ══════════════════════════════════════════════════════════════════════════════
# TRACK 1: SNN Hazard Forecasting & Cerebellar Phase Entrainment
# ══════════════════════════════════════════════════════════════════════════════


class TestSNNHazardAndCerebellarPhase:
    """Validate SNN kinematic limit cycles, trajectory hazard exposure, and cerebellar gating."""

    def test_snn_kinematic_limit_cycle_detection(self) -> None:
        """Verify SNN detects oscillation period for a moving patroller."""
        snn = SNNTemporalPredictor()

        # Simulate a 1D oscillating patroller at feature 5 moving back and forth: (2, 2) -> (2, 3) -> (2, 2) -> (2, 3)
        grid_shape = (6, 6)
        traj = [(2, 2), (2, 3), (2, 4), (2, 3), (2, 2), (2, 3), (2, 4), (2, 3)]

        for pos in traj:
            grid = np.zeros(grid_shape, dtype=np.int32)
            grid[pos[0], pos[1]] = 5
            snn.update_tracks_from_grid(grid)

        track = snn.entity_tracks[5]
        assert track.period >= 2
        cycles = snn.extract_limit_cycles()
        assert len(cycles) >= 1
        assert cycles[0][0] == 5  # feature_id
        assert cycles[0][1] == track.period

    def test_snn_trajectory_hazard_evaluation(self) -> None:
        """Verify SNN computes collision risk for an agent trajectory intersecting future entity path."""
        snn = SNNTemporalPredictor()
        grid_shape = (6, 6)

        # Entity moving rightward: (2, 1) -> (2, 2) -> (2, 3) with velocity (0, 1)
        g1 = np.zeros(grid_shape, dtype=np.int32)
        g1[2, 1] = 4
        snn.update_tracks_from_grid(g1)

        g2 = np.zeros(grid_shape, dtype=np.int32)
        g2[2, 2] = 4
        snn.update_tracks_from_grid(g2)

        # Agent plans to step into (2, 3) on next step (step 1 ahead)
        hazard_intersecting = snn.evaluate_hazard_at_trajectory(
            planned_positions=[(2, 3)],
            grid_shape=grid_shape,
        )
        assert hazard_intersecting > 0.8  # Direct collision detected

        # Agent plans to step into safe cell (5, 5)
        hazard_safe = snn.evaluate_hazard_at_trajectory(
            planned_positions=[(5, 5)],
            grid_shape=grid_shape,
        )
        assert hazard_safe == 0.0

    def test_snn_and_cerebellar_phase_gating(self) -> None:
        """Verify SNN phase gating recommends waiting when immediate arrival collides."""
        snn = SNNTemporalPredictor()
        grid_shape = (6, 6)

        # Create track with known period
        for r, c in [(2, 2), (2, 3), (2, 2), (2, 3)]:
            g = np.zeros(grid_shape, dtype=np.int32)
            g[r, c] = 7
            snn.update_tracks_from_grid(g)

        gate = snn.evaluate_phase_gate_for_track(
            feature_id=7,
            current_step=4,
            target_pos=(2, 2),
            grid_shape=grid_shape,
            travel_steps=1,
        )
        assert "should_wait" in gate
        assert "wait_steps_recommended" in gate

    def test_acc_snn_hazard_conflict_monitoring(self) -> None:
        """Verify ACC flags high conflict for target positions with high forecasted hazard density."""
        acc = AnteriorCingulateConflictMonitor()
        hazard_density = np.zeros((8, 8), dtype=np.float32)
        hazard_density[3, 3] = 0.85  # Contested tile

        conflict_high = acc.evaluate_snn_hazard_conflict(
            target_pos=(3, 3),
            snn_hazard_density=hazard_density,
            threshold=0.35,
        )
        assert conflict_high > 0.5

        conflict_low = acc.evaluate_snn_hazard_conflict(
            target_pos=(0, 0),
            snn_hazard_density=hazard_density,
            threshold=0.35,
        )
        assert conflict_low == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TRACK 2: Transparent Native Substrate in CognitiveGraph
# ══════════════════════════════════════════════════════════════════════════════


class TestCognitiveGraphNativeSubstrate:
    """Validate CognitiveGraph delegation to NativeGraphAdapter and high-performance methods."""

    def test_native_substrate_sync(self) -> None:
        """Verify adding, upserting, and removing nodes/edges syncs to native adapter."""
        cg = CognitiveGraph(enable_native_substrate=True)
        assert cg.get_native_adapter() is not None

        n1 = GoalNode(id="n1", description="Goal 1")
        n2 = HCIRNode(id="n2", node_type=HCIRNodeType.ACTION, category=CognitiveCategory.EXECUTION)
        cg.add_node(n1)
        cg.add_node(n2)

        edge = HCIREdge(
            id="e1",
            edge_type=HCIREdgeType.CAUSES,
            sources=["n1"],
            targets=["n2"],
        )
        cg.add_edge(edge)

        adapter = cg.get_native_adapter()
        assert adapter.has_node("n1")
        assert adapter.has_node("n2")
        assert adapter.has_edge("e1")

        # Shortest path query
        path = cg.find_shortest_path("n1", "n2")
        assert path == ["n1", "n2"]

        # Structural hash
        h1 = cg.canonical_state_hash()
        assert len(h1) in (16, 64)

        # Subgraph extraction
        sub = cg.extract_subgraph_by_depth("n1", max_depth=1)
        assert sub.has_node("n1")
        assert sub.has_node("n2")
        assert sub.has_edge("e1")

        # Native snapshot
        snap = cg.native_snapshot()
        assert snap is not None
        assert snap.has_node("n1")

        # Deletion
        cg.remove_node("n1")
        assert not cg.has_node("n1")
        assert not adapter.has_node("n1")


# ══════════════════════════════════════════════════════════════════════════════
# TRACK 3: Global Neuronal Workspace Ignition & Neuromodulation
# ══════════════════════════════════════════════════════════════════════════════


class TestGNWAndNeuromodulationClosedLoop:
    """Validate closed-loop active inference GNW conscious ignition and neuromodulatory dynamics."""

    def test_assimilate_feedback_updates_snn_and_neuromodulators(self) -> None:
        """Verify feedback assimilation steps SNN, tunes neuromodulators, and submits to GNW."""
        engine = AutonomousEpistemicEngine(world_id="test_gnw_world")

        grid1 = np.zeros((6, 6), dtype=np.int32)
        grid1[1, 1] = 1  # avatar
        grid1[3, 3] = 4  # moving entity

        engine.assimilate_feedback(
            curr_grid=grid1,
            available_actions=[1, 2, 3, 4],
            is_success=False,
            is_failure=False,
        )

        assert engine.snn_predictor.step_counter >= 1
        assert 4 in engine.snn_predictor.entity_tracks

        # Simulate positive win: should surge Dopamine
        grid2 = grid1.copy()
        engine.assimilate_feedback(
            curr_grid=grid2,
            available_actions=[1, 2, 3, 4],
            is_success=True,
            is_failure=False,
        )
        assert engine.neuromodulatory_system.current.dopamine > 1.0

        # Simulate loss: should surge Noradrenaline and trigger GNW ignition
        grid3 = grid1.copy()
        engine.transition_to_phase(EpistemicPhase.MENTAL_SIMULATION)
        engine.assimilate_feedback(
            curr_grid=grid3,
            available_actions=[1, 2, 3, 4],
            is_success=False,
            is_failure=True,
        )
        assert engine.neuromodulatory_system.current.noradrenaline > 1.0
        # GNW broadcast should reset phase on loss
        assert engine.phase == EpistemicPhase.REPLANNING


# ══════════════════════════════════════════════════════════════════════════════
# TRACK 4: Frontopolar Recursive Subgoal Decomposition & Causal Chaining
# ══════════════════════════════════════════════════════════════════════════════


class TestFrontopolarCausalChaining:
    """Validate BA10 recursive causal prerequisite subgoal decomposition."""

    def test_recursive_causal_chain_decomposition(self) -> None:
        """Verify decomposing a target subgoal into ordered causal prerequisites (key -> door -> goal)."""
        stack = FrontopolarSubgoalStack(max_depth=8)

        target = FrontopolarSubgoal(
            subgoal_id="terminal_chest",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(8, 8),
            target_destination=(8, 8),
            priority=2.0,
            is_terminal=True,
        )

        # Causal prerequisites: collect key at (2, 2), then unlock door at (5, 5)
        prerequisites = [
            ("acquire_golden_key", (2, 2), SubgoalType.ACQUIRE_TOOL, 3),
            ("unlock_barrier_door", (5, 5), SubgoalType.UNLOCK_REMOTE_MECHANISM, None),
        ]

        chain = stack.decompose_causal_chain(target, prerequisites)
        assert len(chain) == 2

        # Active top of stack must be the first prerequisite (key)
        active = stack.current_subgoal
        assert active is not None
        assert active.subgoal_id == "acquire_golden_key"

        # Completing the key should activate the door
        stack.pop_subgoal()
        active2 = stack.current_subgoal
        assert active2 is not None
        assert active2.subgoal_id == "unlock_barrier_door"

        # Completing the door should reveal the terminal goal
        stack.pop_subgoal()
        active3 = stack.current_subgoal
        assert active3 is not None
        assert active3.subgoal_id == "terminal_chest"

    def test_engine_schedule_causal_subgoal_chain(self) -> None:
        """Verify AutonomousEpistemicEngine schedules and executes causal subgoal chains."""
        engine = AutonomousEpistemicEngine(world_id="causal_chain_test")

        chain = engine.schedule_causal_subgoal_chain(
            target_destination=(9, 9),
            target_type=SubgoalType.DELIVER_TO_GOAL,
            causal_prerequisites=[
                ("step_on_pressure_plate", (3, 3), SubgoalType.UNLOCK_REMOTE_MECHANISM, None),
            ],
            target_id="master_chest",
        )

        assert len(chain) == 1
        assert engine.subgoal_stack.current_subgoal is not None
        assert engine.subgoal_stack.current_subgoal.subgoal_id == "step_on_pressure_plate"
