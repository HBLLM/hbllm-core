from __future__ import annotations

from hbllm.hcir.graph import (
    CognitiveGraph,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.native_adapter import (
    NativeGraphAdapter,
)

"""Unit tests for Round 6 Epistemic & Cortical Neuro-Symbolic Enhancements.

Track 1: Dual Neuromodulation (VTA Dopamine RPE vs. LC Noradrenaline Adaptive Gain)
Track 2: Retrosplenial Egocentric-to-Allocentric Spatial Coordinate Frame Transform
Track 3: Pearl Level-3 Counterfactual Simulation & Regret Minimization
Track 4: Native Structural Graph Multi-Source Parallel Reachability Cones
"""

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_deliberation import (
    AdaptiveNeuromodulationGovernor,
)
from hbllm.hcir.world.counterfactual_simulation import (
    CounterfactualRegretEvaluator,
)
from hbllm.hcir.world.parietal_coordinate_transform import (
    EgocentricCoordinate,
    HeadingOrientation,
    RetrosplenialCoordinateTransformer,
)


class TestTrack1DualNeuromodulationAdaptiveGain:
    """Track 1: VTA Dopamine & LC Noradrenaline coupled dynamics."""

    def test_dopamine_rpe_and_focal_engagement(self):
        governor = AdaptiveNeuromodulationGovernor()

        # Step with positive surprise / reward -> Phasic DA & NA burst
        state = governor.step_dynamics(
            reward=1.0,
            predicted_value=0.2,
            actual_value=1.0,
            conflict_level=0.1,
            stuck_steps=0,
        )

        assert state.dopamine_phasic_rpe > 0.0
        assert state.dopamine_tonic > 1.0
        assert state.regime in ("FOCAL_ENGAGEMENT", "EXPLOITATION")
        assert not state.is_disengaged
        assert state.exploitation_gain >= 1.0

    def test_noradrenaline_tonic_accumulation_and_disengagement(self):
        governor = AdaptiveNeuromodulationGovernor()

        # Simulate persistent behavioral stuck loop (stuck_steps accumulating)
        state = governor.current_state
        for steps in range(1, 5):
            state = governor.step_dynamics(
                reward=0.0,
                predicted_value=0.5,
                actual_value=0.0,
                conflict_level=0.7,
                stuck_steps=steps,
            )

        assert state.noradrenaline_tonic > 0.4
        assert state.is_disengaged
        assert state.regime == "GLOBAL_EXPLORATION"
        assert state.exploration_gain > 1.5
        assert state.to_causal_predicate().variable == "neuromodulation:state"
        assert len(state.to_transaction_operations()) > 0

    def test_engine_neuromodulatory_integration(self):
        engine = AutonomousEpistemicEngine()
        engine.consecutive_stuck_steps = 4

        state = engine.step_neuromodulatory_dynamics(
            reward=0.0,
            predicted_value=0.8,
            actual_value=0.0,
            conflict_level=0.5,
        )

        assert state.is_disengaged
        assert engine.neuromodulation_governor.current_state.is_disengaged


class TestTrack2RetrosplenialSpatialCoordinateTransform:
    """Track 2: Retrosplenial Egocentric-to-Allocentric transformations."""

    def test_heading_inference_from_displacement(self):
        transformer = RetrosplenialCoordinateTransformer()

        # Movement North (delta_r = -1)
        h_north = transformer.infer_heading_from_displacement((5, 5), (4, 5))
        assert h_north == HeadingOrientation.NORTH

        # Movement East (delta_c = +1)
        h_east = transformer.infer_heading_from_displacement((5, 5), (5, 6))
        assert h_east == HeadingOrientation.EAST

        # Movement South (delta_r = +1)
        h_south = transformer.infer_heading_from_displacement((5, 5), (6, 5))
        assert h_south == HeadingOrientation.SOUTH

        # Movement West (delta_c = -1)
        h_west = transformer.infer_heading_from_displacement((5, 5), (5, 4))
        assert h_west == HeadingOrientation.WEST

    def test_bidirectional_ego_allo_transform(self):
        transformer = RetrosplenialCoordinateTransformer()
        agent_pos = (10, 10)

        # 1. Facing North: Forward 2, Right 1 -> (8, 11)
        allo_n = transformer.ego_to_allo(
            EgocentricCoordinate(forward=2, lateral=1),
            agent_pos=agent_pos,
            heading=HeadingOrientation.NORTH,
        )
        assert allo_n == (8, 11)
        ego_n = transformer.allo_to_ego(allo_n, agent_pos, HeadingOrientation.NORTH)
        assert ego_n.forward == 2
        assert ego_n.lateral == 1

        # 2. Facing East: Forward 2, Right 1 -> (11, 12)
        allo_e = transformer.ego_to_allo(
            EgocentricCoordinate(forward=2, lateral=1),
            agent_pos=agent_pos,
            heading=HeadingOrientation.EAST,
        )
        assert allo_e == (11, 12)
        ego_e = transformer.allo_to_ego(allo_e, agent_pos, HeadingOrientation.EAST)
        assert ego_e.forward == 2
        assert ego_e.lateral == 1

    def test_egocentric_view_rotation(self):
        transformer = RetrosplenialCoordinateTransformer()
        grid = np.zeros((7, 7), dtype=int)
        # Place salient landmark North of center
        grid[2, 3] = 9
        agent_pos = (3, 3)

        # Facing North: landmark remains at top center
        view_n = transformer.extract_egocentric_view(
            grid, agent_pos, HeadingOrientation.NORTH, view_radius=2
        )
        assert view_n[1, 2] == 9

        # Facing East: landmark should appear to the left of the agent
        view_e = transformer.extract_egocentric_view(
            grid, agent_pos, HeadingOrientation.EAST, view_radius=2
        )
        assert view_e.shape == (5, 5)

    def test_engine_retrosplenial_integration(self):
        engine = AutonomousEpistemicEngine()
        engine.prev_avatar_pos = (5, 5)
        engine.avatar_pos = (5, 6)  # Facing East

        # Target (6, 8): 2 cells ahead (East), 1 cell right (South)
        ego = engine.to_egocentric_coordinate((6, 8))
        assert ego is not None
        assert ego.forward == 2
        assert ego.lateral == 1

        # Reverse check
        allo = engine.to_allocentric_coordinate(ego)
        assert allo == (6, 8)

    def test_counterfactual_regret_evaluation(self):
        evaluator = CounterfactualRegretEvaluator(regret_threshold=0.10)
        grid = np.zeros((10, 10), dtype=int)
        goals = {(5, 8)}
        barriers = {(6, 5)}
        deltas = {
            1: (0, 1),  # East (towards goal)
            2: (0, -1),  # West (away from goal)
            3: (1, 0),  # South (blocked by barrier)
        }

        # Actual: Agent chose West (Action 2), moving to (5, 4) away from goal
        comparisons = evaluator.evaluate_counterfactuals(
            actual_action=2,
            actual_reward=0.0,
            current_pos=(5, 5),
            actual_next_pos=(5, 4),
            candidate_actions=[1, 2, 3],
            grid=grid,
            goal_positions=goals,
            static_barriers=barriers,
            action_deltas=deltas,
        )

        assert len(comparisons) == 2  # Alternatives: 1 and 3

        # Counterfactual Action 1 (East) was superior: moves closer to goal
        comp_1 = next(c for c in comparisons if c.counterfactual_action == 1)
        assert comp_1.is_counterfactually_superior
        assert comp_1.regret > 0.0

        # Counterfactual Action 3 (South) was blocked by barrier: inferior
        comp_3 = next(c for c in comparisons if c.counterfactual_action == 3)
        assert not comp_3.is_counterfactually_superior
        assert comp_3.counterfactual_outcome_value < 0.0

    def test_engine_counterfactual_regret_integration(self):
        engine = AutonomousEpistemicEngine()
        grid = np.zeros((10, 10), dtype=int)
        engine.avatar_pos = (4, 4)
        engine.learned_goal_positions.add((4, 7))
        engine.action_deltas = {
            1: (0, 1),  # East
            2: (0, -1),  # West
        }

        # Actual choice: West (Action 2) away from goal
        comps = engine.evaluate_counterfactual_regret(
            actual_action=2,
            actual_reward=0.0,
            actual_next_pos=(4, 3),
            candidate_actions=[1, 2],
            grid=grid,
        )

        assert len(comps) == 1
        assert comps[0].counterfactual_action == 1
        assert comps[0].is_counterfactually_superior
        assert comps[0].to_causal_predicate().variable == "counterfactual:regret"
        assert len(comps[0].to_transaction_operations()) > 0

    def test_native_adapter_parallel_reachability(self):
        adapter = NativeGraphAdapter()
        adapter.add_node("subgoal_a", "goal")
        adapter.add_node("subgoal_b", "goal")
        adapter.add_node("door_1", "barrier")
        adapter.add_node("door_2", "barrier")
        adapter.add_node("terminal", "goal")

        adapter.add_edge("e1", "depends_on", ["subgoal_a"], ["door_1"])
        adapter.add_edge("e2", "depends_on", ["door_1"], ["terminal"])
        adapter.add_edge("e3", "depends_on", ["subgoal_b"], ["door_2"])

        cones = adapter.parallel_reachability_cones(
            seed_nodes=["subgoal_a", "subgoal_b"], max_depth=5
        )

        assert "subgoal_a" in cones
        assert "subgoal_b" in cones
        assert cones["subgoal_a"] == {"subgoal_a", "door_1", "terminal"}
        assert cones["subgoal_b"] == {"subgoal_b", "door_2"}

    def test_cognitive_graph_parallel_reachability(self):
        graph = CognitiveGraph()
        n1 = GoalNode(id="g1", description="G1")
        n2 = GoalNode(id="g2", description="G2")
        n3 = GoalNode(id="g3", description="G3")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        graph.add_edge(
            HCIREdge(id="e12", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g1"], targets=["g2"])
        )
        graph.add_edge(
            HCIREdge(id="e23", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g2"], targets=["g3"])
        )

        cones = graph.parallel_reachability_cones(seed_nodes=["g1", "g2"])
        assert cones["g1"] == {"g1", "g2", "g3"}
        assert cones["g2"] == {"g2", "g3"}
