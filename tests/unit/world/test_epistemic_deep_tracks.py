"""Unit tests for the 4 Advanced Epistemic & Cortical tracks.

Track 1: Metacognitive Simulation Budget Governor & Entorhinal Path Integration Loop
Track 2: Domain-General Temporal Dependency & Delayed-Effect Pearl Calculus
Track 3: Relational Graph Isomorphism & Zero-Shot Structural Analogy Transfer
Track 4: Betti Homology & Topological Cul-de-Sac Cavity Pruning in Manifold Routing
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.experimental.entorhinal_grid_cortex import EntorhinalGridCortex
from hbllm.hcir.world.experimental.relational_graph_matcher import (
    RelationalGraphMatcher as VentralRelationalGraphMatcher,
)
from hbllm.hcir.world.metacognitive_agency_cortex import MetacognitiveAgencyCortex
from hbllm.hcir.world.prediction_budget import PredictionBudget
from hbllm.hcir.world.temporal_dependency_tracker import (
    DelayedEffectModeler,
    LongHorizonDependencyGraph,
)
from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter


class TestMetacognitiveBudgetAndEntorhinalLoop:
    """Track 1: Metacognitive compute budget governor and entorhinal path integration."""

    def test_entorhinal_path_integration_and_loop_closure(self) -> None:
        grid_cortex = EntorhinalGridCortex()
        grid_cortex.reset_anchor((5, 5))

        # Integrate displacements in a closed loop: (5,5) -> (6,5) -> (6,6) -> (5,6) -> (5,5)
        path = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        for dr, dc in path:
            grid_cortex.path_integrate_step(dr, dc)
            grid_cortex.generate_place_field_readout()

        assert grid_cortex.current_estimated_coord == (5, 5)
        # Should detect loop closure returning to the anchor position
        assert grid_cortex.detect_loop_closure(min_step_separation=3, similarity_threshold=0.90)

    def test_metacognitive_confidence_governor_budget_scaling(self) -> None:
        meta_cortex = MetacognitiveAgencyCortex(base_simulation_depth=5, max_simulation_depth=20)
        budget = PredictionBudget()

        # 1. High confidence and high agency -> reflex mode
        assess_reflex = meta_cortex.calibrate_confidence_and_govern_compute(
            action_id=1, raw_confidence=0.92, agency_score=0.95, empirical_success=True
        )
        assert assess_reflex.recommended_simulation_depth == 2
        assert assess_reflex.recommended_exploration_mode == "reflex"
        reflex_search_expansions = budget.compute_search_budget(
            metacognitive_depth=assess_reflex.recommended_simulation_depth,
            exploration_mode=assess_reflex.recommended_exploration_mode,
        )
        assert reflex_search_expansions == 300

        # 2. Low confidence under uncertainty -> deep simulation mode
        assess_deep = meta_cortex.calibrate_confidence_and_govern_compute(
            action_id=2, raw_confidence=0.35, agency_score=0.50, empirical_success=False
        )
        assert assess_deep.recommended_simulation_depth == 20
        assert assess_deep.recommended_exploration_mode == "deep_simulation"
        deep_search_expansions = budget.compute_search_budget(
            metacognitive_depth=assess_deep.recommended_simulation_depth,
            exploration_mode=assess_deep.recommended_exploration_mode,
        )
        assert deep_search_expansions >= 3000

    def test_engine_assimilate_feedback_updates_metacognition(self) -> None:
        engine = AutonomousEpistemicEngine()
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2, 2] = 1  # avatar
        engine.avatar_pos = (2, 2)
        engine.avatar_feature = 1

        engine.assimilate_feedback(
            curr_grid=grid,
            available_actions=[1, 2, 3, 4],
            is_failure=False,
            is_success=False,
            action=1,
            action_data={"dr": 0, "dc": 0},
        )

        assert engine.working_memory.last_metacognitive_assessment is not None
        assert engine.prediction_budget.counterfactual_budget.max_depth in (2, 5, 9, 20)


class TestTemporalDependencyAndDelayedEffects:
    """Track 2: Domain-general delayed effect tracking and Kahn's prerequisite DAG."""

    def test_delayed_effect_discovery(self) -> None:
        modeler = DelayedEffectModeler(max_lag=5)

        # Step 0: execute action 4 (e.g. press button)
        modeler.record_step(step=0, action_id=4, state_features={"pos": (3, 3)})

        # Step 1: intermediate movement
        modeler.record_step(step=1, action_id=1, state_features={"pos": (3, 4)})

        # Step 2: door opens (delayed consequence with lag = 2)
        records = modeler.record_step(
            step=2,
            action_id=1,
            state_features={"pos": (3, 5)},
            observed_mutations={"barrier_opened": True},
        )

        assert len(records) >= 1
        delayed = [r for r in records if r.trigger_action == 4]
        assert len(delayed) == 1
        assert delayed[0].observed_lag_steps == 2
        pred = delayed[0].to_causal_predicate()
        assert pred.variable == "barrier_opened"
        assert "delayed_by_2_steps" in str(pred.relation)

    def test_long_horizon_dependency_topological_sort(self) -> None:
        graph = LongHorizonDependencyGraph()
        # Prerequisites: Key -> Switch -> Door -> Terminal Chest
        graph.add_milestone("acquire_key", "Collect key", prerequisites=[])
        graph.add_milestone("trip_switch", "Actuate switch", prerequisites=["acquire_key"])
        graph.add_milestone("unlock_door", "Pass barrier", prerequisites=["trip_switch"])
        graph.add_milestone("reach_chest", "Terminal chest", prerequisites=["unlock_door"])

        assert graph.is_acyclic()
        order = graph.topological_sort()
        assert order == ["acquire_key", "trip_switch", "unlock_door", "reach_chest"]

        # Only root milestone is executable initially
        executable = graph.get_executable_milestones()
        assert len(executable) == 1
        assert executable[0].node_id == "acquire_key"

        graph.mark_satisfied("acquire_key")
        executable = graph.get_executable_milestones()
        assert len(executable) == 1
        assert executable[0].node_id == "trip_switch"


class TestRelationalGraphAndAnalogyMatching:
    """Track 3: Relational scene graph Hungarian matching and object mutation detection."""

    def test_scene_graph_construction_and_matching(self) -> None:
        grid_a = np.zeros((8, 8), dtype=np.int32)
        grid_a[1:3, 1:3] = 2  # object A (red block)
        grid_a[5:7, 5:7] = 3  # object B (green block)

        grid_b = np.zeros((8, 8), dtype=np.int32)
        grid_b[1:3, 1:3] = 4  # recolored object A (blue block)
        grid_b[5:7, 5:7] = 5  # recolored object B (yellow block)

        graph_a = VentralRelationalGraphMatcher.build_scene_graph(grid_a, bg_color=0)
        graph_b = VentralRelationalGraphMatcher.build_scene_graph(grid_b, bg_color=0)

        assert len(graph_a.nodes) == 2
        assert len(graph_b.nodes) == 2

        correspondences = VentralRelationalGraphMatcher.match_graphs(
            graph_a, graph_b, ignore_color=True, ignore_scale=True
        )
        assert len(correspondences) == 2
        for corr in correspondences:
            assert corr.match_score >= 0.70

    def test_detect_object_mutations(self) -> None:
        grid_prev = np.zeros((8, 8), dtype=np.int32)
        grid_prev[1:3, 1:3] = 2  # object 1 at (1,1)

        grid_curr = np.zeros((8, 8), dtype=np.int32)
        grid_curr[3:5, 1:3] = 2  # object 1 moved downwards by 2 rows

        graph_prev = VentralRelationalGraphMatcher.build_scene_graph(grid_prev, bg_color=0)
        graph_curr = VentralRelationalGraphMatcher.build_scene_graph(grid_curr, bg_color=0)

        mutations = VentralRelationalGraphMatcher.detect_object_mutations(graph_prev, graph_curr)
        assert len(mutations["moved"]) == 1
        assert abs(mutations["moved"][0]["delta"][0] - 2.0) < 0.1

    def test_engine_transfer_structural_analogies(self) -> None:
        engine = AutonomousEpistemicEngine()
        _restore_all(engine)
        curr_grid = np.zeros((8, 8), dtype=np.int32)
        curr_grid[2, 2] = 2
        curr_grid[6, 6] = 3
        engine.prev_grid = curr_grid

        proto_grid = np.zeros((8, 8), dtype=np.int32)
        proto_grid[2, 2] = 7  # analogous configuration under different feature ID
        proto_grid[6, 6] = 8

        corrs = engine.transfer_structural_analogies(proto_grid)
        assert len(corrs) == 2
        assert all(c.match_score >= 0.70 for c in corrs)


class TestBettiHomologyAndCavityPruning:
    """Track 4: Persistent Betti homology and cul-de-sac cavity geodesic pruning."""

    def test_betti_invariants_computation(self) -> None:
        # Construct an 8x8 corridor with a single cul-de-sac
        passable = np.zeros((8, 8), dtype=bool)
        # Main corridor: row 3, cols 1..6
        for c in range(1, 7):
            passable[3, c] = True
        # Cul-de-sac branch: col 2, rows 1..2
        passable[1, 2] = True
        passable[2, 2] = True

        betti = GeodesicManifoldRouter.compute_manifold_homology((8, 8), passable)
        assert betti.betti_0 == 1
        # Should detect the dead-end branch as a cul-de-sac cavity
        assert len(betti.cavities) >= 1
        assert (1, 2) in betti.cavities[0]

    def test_cul_de_sac_cavity_pruning_in_geodesic_routing(self) -> None:
        curr_grid = np.zeros((7, 7), dtype=np.int32)
        # Main path from (3,1) to (3,5)
        # Dead end cavity at (1,2) and (2,2) branching off (3,2)
        start = (3, 1)
        goal = (3, 5)

        # Static barriers enclosing the space
        walls = set()
        for r in range(7):
            for c in range(7):
                if not (r == 3 and 1 <= c <= 5) and not (c == 2 and 1 <= r <= 2):
                    walls.add((r, c))

        movable_actions = [
            (1, -1, 0),  # up
            (2, 1, 0),  # down
            (3, 0, -1),  # left
            (4, 0, 1),  # right
        ]

        route = GeodesicManifoldRouter.route_geodesic(
            curr_grid=curr_grid,
            start_pos=start,
            goals=[goal],
            static_barriers=walls,
            grid_walls=set(),
            unknown_entity_cells=set(),
            oriented_threats=[],
            movable_actions=movable_actions,
            bg=0,
            barrier_features=set(),
            avatar_features={0},
        )

        assert route is not None
        # Path should step purely right along row 3: (3,2), (3,3), (3,4), (3,5)
        path_positions = [pos for _, pos in route]
        assert path_positions == [(3, 2), (3, 3), (3, 4), (3, 5)]
        # Dead-end cul-de-sac (1,2), (2,2) was never entered
        assert (1, 2) not in path_positions
        assert (2, 2) not in path_positions
