"""Unit tests for Advanced Human Brain Faculties:
1. Orbitofrontal Cortex (OFC) Counterfactual Deadlock & Irreversibility Detector
2. dlPFC/IPS Visuospatial Working Memory & Feature-Location Binding
3. IPL Extended Body Schema & Tool Affordance Incorporation
4. LOC Ventral Stream Geometric Symmetry Discrepancy Completion
5. Premotor Spatiotemporal Collision Cones & Trajectory Extrapolation
6. Inferotemporal Cortex (IT / Ventral Stream) Affordance Centroid Segmentation
7. Parieto-Occipital Mental Imagery (V6/MST) Optical Ray Projection & Specular Reflection
8. Anterior Mid-Cingulate & Lateral Habenula Episodic IOR & Fatal Prefix Pruning
9. Cerebellar Phase Gating & Rhythm-Locked Motor Hesitation
10. Anterior Prefrontal Cortex (aPFC / BA10) Hierarchical Subgoal Stack & Holding Bays
11. Ventromedial Prefrontal Cortex (vmPFC) Remote Causal Attribution & Action-Effect Binding
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.causal_discovery import BeliefTransitionType
from hbllm.hcir.world.cerebellar_phase_clock import CerebellarPhaseClock
from hbllm.hcir.world.cortex_causal import CausalInductionCortex, HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortex_episodic import HippocampalEpisodicCortex
from hbllm.hcir.world.cortex_hypothesis import (
    HypothesisStatus,
    HypothesisType,
    InductiveHypothesisEngine,
)
from hbllm.hcir.world.cortex_morphology import (
    MorphologicalSaliencyEngine,
    MorphologyPrimitiveType,
)
from hbllm.hcir.world.cortex_motor import (
    EpistemicCuriosityExplorer,
    EpistemicProbeProposal,
    MotorCortexEffector,
)
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.inferotemporal_segmentation import InferotemporalSegmentationEngine
from hbllm.hcir.world.kinetic_stream import KineticEntity
from hbllm.hcir.world.optical_ray_projection import (
    MirrorOrientation,
    OpticalRayProjector,
)
from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAttributor
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones
from hbllm.hcir.world.spatiotemporal_tracker import DynamicCellPhase, SpatiotemporalHazardTracker
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer
from hbllm.hcir.world.visuospatial_working_memory import VisuospatialWorkingMemory


class TestCounterfactualDeadlockDetector:
    """Test OFC counterfactual deadlock simulation and irreversible state pruning."""

    def test_corner_deadlock_detected(self) -> None:
        goals = {(5, 5)}
        static_barriers = {(0, 1), (1, 0)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (0, 0), goals, static_barriers, {(0, 0)}, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert eval_res.deadlock_type == "corner"

        pred_dl = eval_res.to_causal_predicate()
        assert pred_dl.variable == "deadlock_evaluation"
        assert pred_dl.value is True

        tx_dl = eval_res.to_transaction_operations()
        assert len(tx_dl) == 1
        assert tx_dl[0].node_id == "deadlock_corner"

        res_t, txs_t = CounterfactualDeadlockDetector.evaluate_and_transact(
            (0, 0), goals, static_barriers, {(0, 0)}, grid_shape
        )
        assert res_t.is_deadlock is True
        assert len(txs_t) == 1

    def test_goal_reversal_prevents_deadlock(self) -> None:
        goals = {(0, 0)}
        static_barriers = {(0, 1), (1, 0)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (0, 0), goals, static_barriers, {(0, 0)}, grid_shape
        )
        assert eval_res.is_deadlock is False

    def test_2x2_square_deadlock_detected(self) -> None:
        goals = {(9, 9)}
        static_barriers = {(2, 2), (2, 3)}
        all_blocks = {(3, 2), (3, 3)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (3, 2), goals, static_barriers, all_blocks, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert eval_res.deadlock_type == "square_2x2"

    def test_line_freeze_deadlock_detected(self) -> None:
        goals = {(9, 9)}
        static_barriers = {(1, 2), (1, 3)}
        all_blocks = {(2, 2), (2, 3)}
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (2, 2), goals, static_barriers, all_blocks, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert (
            CounterfactualDeadlockDetector.is_line_freeze_deadlock(
                (2, 2), goals, static_barriers, all_blocks, grid_shape
            )
            is True
        )

    def test_interior_wall_deadlock_detected(self) -> None:
        goals = {(9, 9)}
        static_barriers = {
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 6),
        }
        grid_shape = (10, 10)
        eval_res = CounterfactualDeadlockDetector.evaluate_deadlock(
            (3, 3), goals, static_barriers, {(3, 3)}, grid_shape
        )
        assert eval_res.is_deadlock is True
        assert eval_res.deadlock_type == "interior_wall"


class TestVisuospatialWorkingMemory:
    """Test dlPFC visuospatial feature-location binding and pair matching."""

    def test_feature_location_binding_and_pair_matching(self) -> None:
        wm = VisuospatialWorkingMemory(capacity=16)

        wm.record_probe((2, 3), feature_id=7, step=1)
        assert wm.last_probed_coord == (2, 3)
        assert wm.last_probed_feature == 7

        wm.record_probe((8, 4), feature_id=3, step=2)
        assert wm.find_matching_pair((8, 4), 3) is None

        wm.record_probe((5, 6), feature_id=7, step=3)
        pair = wm.find_matching_pair((5, 6), 7)
        assert pair == (2, 3)

    def test_diff_stencil_recording(self) -> None:
        wm = VisuospatialWorkingMemory()
        center = (5, 5)
        changed = {(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)}
        wm.record_diff_stencil(center, changed)
        assert len(wm.discovered_stencils) == 1
        assert (0, 0) in wm.discovered_stencils[0]
        assert (-1, 0) in wm.discovered_stencils[0]
        assert (1, 0) in wm.discovered_stencils[0]

    def test_systematic_raster_scan(self) -> None:
        wm = VisuospatialWorkingMemory()
        wm.record_probe((1, 1), feature_id=2, step=1)
        candidates = [(3, 2), (1, 1), (1, 5), (2, 0)]
        next_cand = wm.get_systematic_unprobed_candidate(candidates)
        assert next_cand == (1, 5)


class TestExtendedBodySchema:
    """Test IPL extended body schema and tool-gated barrier affordances."""

    def test_tool_incorporation_and_resonance(self) -> None:
        schema = ExtendedBodySchema()
        assert schema.is_holding(3) is False

        schema.acquire_tool(feature_id=3, step=10, role="key")
        assert schema.is_holding(3) is True

        assert schema.is_barrier_permeable(8) is False

        schema.register_resonance(tool_feature=3, barrier_feature=8)
        assert schema.is_barrier_permeable(8) is True
        assert 8 in schema.get_permeable_barrier_features()

        assert schema.expend_tool(3) is True
        assert schema.is_holding(3) is False
        assert schema.is_barrier_permeable(8) is False


class TestVisualSymmetryAnalyzer:
    """Test LOC geometric symmetry discrepancy completion."""

    def test_extract_discrepancy_targets(self) -> None:
        # Create a 6x6 grid with dominant vertical symmetry: 4 matching pairs, 1 missing at (2, 4)
        grid = np.zeros((6, 6), dtype=int)
        grid[0, 1] = 4
        grid[0, 4] = 4
        grid[1, 1] = 4
        grid[1, 4] = 4
        grid[3, 1] = 4
        grid[3, 4] = 4
        grid[4, 1] = 4
        grid[4, 4] = 4
        grid[2, 1] = 4  # Left side has color 4, right side at (2, 4) is 0
        discrepancies = VisualSymmetryAnalyzer.extract_discrepancy_targets(
            grid, background_color=0, threshold=0.55
        )
        assert len(discrepancies) > 0
        coords = [(d[0], d[1]) for d in discrepancies]
        assert (2, 4) in coords
        target_feat = [d[2] for d in discrepancies if (d[0], d[1]) == (2, 4)][0]
        assert target_feat == 4

    def test_symmetry_structured_analysis_and_causal_predicate(self) -> None:
        """Test analyze_symmetry, analyze_tiling, and CausalPredicate generation."""
        # 4x4 vertically symmetric grid (top row only: left=right, top!=bottom)
        grid = np.zeros((4, 4), dtype=int)
        grid[0, 0] = 3
        grid[0, 3] = 3

        result = VisualSymmetryAnalyzer.analyze_symmetry(grid)
        assert result.dominant_axis == "vertical"
        assert result.is_symmetric is True
        pred = result.to_symbolic_predicate()
        assert pred.variable == "visual_symmetry"
        assert pred.value == "vertical"

        # Tiling
        tile_res = VisualSymmetryAnalyzer.analyze_tiling(grid)
        assert tile_res.confidence >= 0.0
        tile_pred = tile_res.to_symbolic_predicate()
        assert tile_pred.variable == "periodic_tiling"


class TestSpatiotemporalCollisionCones:
    """Test Premotor collision cone projection with static boundary rebound."""

    def test_trajectory_extrapolation_and_hazard_check(self) -> None:
        cones = SpatiotemporalCollisionCones(default_horizon=6)
        entity = KineticEntity(
            centroid=(2.0, 2.0),
            velocity=(0.0, 1.0),
            cells=[(2, 2)],
            bounding_box=(2, 2, 2, 2),
            area=1,
            features={5},
        )
        static_barriers = {(2, 4)}
        grid_shape = (10, 10)

        cones.update_trajectories([entity], static_barriers, grid_shape, horizon=5)

        assert cones.is_collision_hazard(2, 3, time_step=1) is True
        assert cones.is_collision_hazard(2, 2, time_step=2) is True
        assert cones.is_collision_hazard(5, 5, time_step=1) is False


class TestInferotemporalSegmentation:
    """Test Area IT ventral stream object token segmentation and affordance anchors."""

    def test_segment_objects_and_medial_anchors(self) -> None:
        grid = np.zeros((10, 10), dtype=int)
        # Create a 2x3 compact object of color 4
        grid[2:4, 3:6] = 4
        # Create a 1x1 point button of color 7
        grid[8, 8] = 7

        tokens = InferotemporalSegmentationEngine.segment_objects(grid, background_feature=0)
        assert len(tokens) == 2

        # Most salient should be the compact objects
        feat_ids = {t.feature_id for t in tokens}
        assert feat_ids == {4, 7}

        token_4 = next(t for t in tokens if t.feature_id == 4)
        assert token_4.area == 6
        assert token_4.is_compact is True
        # Anchor must be an interior cell
        assert token_4.anchor_coord in [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]

    def test_extract_affordance_anchors_with_ior(self) -> None:
        grid = np.zeros((10, 10), dtype=int)
        grid[5, 5] = 3
        grid[1, 1] = 6

        visits = {"click_5_5": 5}  # Heavily visited (habituation / IOR)
        anchors = InferotemporalSegmentationEngine.extract_affordance_anchors(
            grid,
            background_feature=0,
            visit_counts=visits,
        )
        assert len(anchors) == 2
        # Unvisited anchor at (1, 1) should have higher priority than habituated (5, 5)
        top_anchor = anchors[0]
        assert (top_anchor[0], top_anchor[1]) == (1, 1)

    def test_token_lifting_to_spatial_entity_and_manipulable_extraction(self) -> None:
        grid = np.zeros((10, 10), dtype=int)
        grid[2:4, 3:6] = 4
        grid[8, 8] = 7

        entities = InferotemporalSegmentationEngine.extract_manipulable_entities(
            grid, background_feature=0
        )
        assert len(entities) == 2
        for ent in entities:
            assert isinstance(ent, SpatialEntity)
            assert ent.role == EntityRole.MANIPULABLE
            assert ent.area > 0
            assert "salience_score" in ent.properties
            assert "is_compact" in ent.properties


class TestOpticalRayProjector:
    """Test Parieto-Occipital V6/MST optical ray tracing and specular reflection."""

    def test_specular_reflection_vectors(self) -> None:
        # Forward slash '/' reflects right (0, 1) to up (-1, 0)
        assert OpticalRayProjector.reflect_vector((0, 1), MirrorOrientation.FORWARD_SLASH) == (
            -1,
            0,
        )
        # Backslash '\' reflects right (0, 1) to down (1, 0)
        assert OpticalRayProjector.reflect_vector((0, 1), MirrorOrientation.BACKSLASH) == (1, 0)

    def test_ray_tracing_through_mirrors_to_receptor(self) -> None:
        # Emitter at (2, 0) firing right (0, 1)
        # Mirror at (2, 5) reflecting down (1, 0)
        # Receptor at (8, 5)
        grid_shape = (10, 10)
        barriers: set[tuple[int, int]] = set()
        mirrors = {(2, 5): MirrorOrientation.BACKSLASH}
        receptors = {(8, 5)}

        path = OpticalRayProjector.trace_ray(
            start_pos=(2, 0),
            initial_dir=(0, 1),
            grid_shape=grid_shape,
            barriers=barriers,
            mirrors=mirrors,
            receptors=receptors,
        )
        assert path.hit_receptor is True
        assert path.terminated_at == (8, 5)
        assert path.termination_reason == "receptor"

    def test_solve_mirror_placement_bidirectional_intersection(self) -> None:
        # Emitter at (3, 0) firing right (0, 1)
        # Receptor at (7, 4)
        # Intersection should be at (3, 4) with backslash mirror reflecting right -> down
        hypotheses = OpticalRayProjector.solve_mirror_placement(
            emitter_pos=(3, 0),
            emitter_dir=(0, 1),
            receptor_pos=(7, 4),
            grid_shape=(10, 10),
            barriers=set(),
        )
        assert len(hypotheses) > 0
        h0 = hypotheses[0]
        assert h0.mirror_pos == (3, 4)
        assert h0.required_orientation == MirrorOrientation.BACKSLASH
        assert h0.outgoing_dir == (1, 0)


class TestHabenularEpisodicIOR:
    """Test aMCC & Lateral Habenula catastrophic credit assignment and fatal prefix pruner."""

    def test_catastrophe_backpropagation_and_inhibition(self) -> None:
        ior = HabenularEpisodicIOR(trace_horizon=5, gamma=0.8, base_penalty=100.0)

        # Record a 5-step trajectory ending in death
        ior.record_step(1, avatar_pos=(0, 0), action=1)
        ior.record_step(2, avatar_pos=(0, 1), action=2)
        ior.record_step(3, avatar_pos=(0, 2), action=3)
        ior.record_step(4, avatar_pos=(0, 3), action=4)
        ior.record_step(5, avatar_pos=(0, 4), action=5)

        # Fatal event at step 5
        record = ior.record_catastrophe(final_step=5, is_lost=True)
        assert record is not None
        assert record.fatal_pos == (0, 4)
        assert record.fatal_action == 5

        # Terminal state-action should be actively inhibited
        assert ior.is_action_inhibited((0, 4), 5) is True
        # Penultimate action within fatal window should also be inhibited
        assert ior.is_action_inhibited((0, 3), 4) is True

        # Non-fatal action should not be inhibited
        assert ior.is_action_inhibited((0, 0), 9) is False

    def test_limit_cycle_oscillation_detection(self) -> None:
        ior = HabenularEpisodicIOR()
        # Simulate oscillating actions [1, 5, 1, 5, 1, 5]
        for act in [1, 5, 1, 5, 1, 5]:
            ior.record_step(1, avatar_pos=(2, 2), action=act)

        osc_act = ior.detect_action_oscillation()
        assert osc_act == 5

    def test_reset_and_reset_episode(self) -> None:
        ior = HabenularEpisodicIOR()
        ior.record_step(1, avatar_pos=(1, 1), action=1)
        ior.record_step(2, avatar_pos=(1, 2), action=2)
        record = ior.record_catastrophe(final_step=2, is_lost=True)
        assert record is not None
        assert len(ior.repulsion_table) > 0
        assert len(ior.fatal_prefixes) > 0
        assert ior.is_action_inhibited((1, 2), 2) is True

        # Transient episode reset retaining long term memory
        ior.reset_episode(retain_long_term=True)
        assert len(ior.episode_trace) == 0
        assert len(ior.recent_actions) == 0
        assert len(ior.active_inhibitions) == 0
        # Learned repulsion and fatal records should be retained
        assert len(ior.repulsion_table) > 0
        assert len(ior.fatal_prefixes) > 0

        # Full reset
        ior.reset()
        assert len(ior.repulsion_table) == 0
        assert len(ior.fatal_prefixes) == 0
        assert len(ior.active_inhibitions) == 0
        assert len(ior.episode_trace) == 0
        assert len(ior.recent_actions) == 0


class TestHippocampalEpisodicCortex:
    """Test Hippocampal episodic memory, SWR replay, and reset mechanics."""

    def test_reset_clears_all_memories_and_habenular_ior(self) -> None:
        cortex = HippocampalEpisodicCortex()
        cortex.record_transition(
            step=1,
            avatar_pos=(2, 3),
            action=4,
            reward=0.0,
            is_lost=False,
        )
        cortex.record_transition(
            step=2,
            avatar_pos=(2, 4),
            action=2,
            reward=-10.0,
            is_lost=True,
        )
        cortex.trigger_sharp_wave_ripple_replay(is_lost=True, is_win=False)

        assert len(cortex.catastrophic_replays) > 0
        assert len(cortex.grounded_lethal_transitions) > 0
        assert len(cortex.habenular_ior.repulsion_table) > 0

        # Full reset should clear everything including the underlying HabenularEpisodicIOR
        cortex.reset()
        assert len(cortex.current_episode) == 0
        assert len(cortex.past_episodes) == 0
        assert len(cortex.successful_trajectories) == 0
        assert len(cortex.catastrophic_replays) == 0
        assert len(cortex.grounded_lethal_transitions) == 0
        assert len(cortex.discovered_lethal_features) == 0
        assert len(cortex.habenular_ior.repulsion_table) == 0
        assert len(cortex.habenular_ior.fatal_prefixes) == 0


class TestCerebellarMotorPhaseGating:
    """Test Cerebellar interval timer and basal ganglia motor phase gating."""

    def test_evaluate_motion_hazard_gate_with_periodic_cell(self) -> None:
        clock = CerebellarPhaseClock()
        tracker = SpatiotemporalHazardTracker()

        # Set up a periodic hazard cell at (3, 3) with period T=4
        # Sequence: [0, 9, 0, 0] where 9 is lethal
        tracker.periodic_cells[(3, 3)] = DynamicCellPhase(
            r=3,
            c=3,
            period=4,
            cycle_values=[0, 9, 0, 0],
            hazardous_values={9},
        )

        avatar_pos = (3, 2)
        target_pos = (3, 3)

        # At current_step = 1, arrival at step 2 has phase (2-1)%4 = 1 -> value 9 (hazardous!)
        # Waiting 1 step makes arrival at step 3 with phase (3-1)%4 = 2 -> value 0 (safe!)
        decision = clock.evaluate_motion_hazard_gate(
            current_step=1,
            avatar_pos=avatar_pos,
            target_pos=target_pos,
            hazard_tracker=tracker,
        )
        assert decision.should_wait is True
        assert decision.wait_steps_recommended == 1


class TestFrontopolarSubgoalStack:
    """Test Brodmann Area 10 cognitive branching and holding bay allocation."""

    def test_subgoal_push_pop_and_satisfaction(self) -> None:
        stack = FrontopolarSubgoalStack(max_depth=4)

        sg1 = FrontopolarSubgoal(
            subgoal_id="g1",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(2, 2),
            target_destination=(5, 5),
        )
        sg2 = FrontopolarSubgoal(
            subgoal_id="g2",
            subgoal_type=SubgoalType.PARK_IN_HOLDING_BAY,
            target_entity_pos=(3, 3),
            target_destination=(1, 1),
        )

        stack.push_subgoal(sg1)
        stack.push_subgoal(sg2)

        assert stack.current_subgoal == sg2
        assert stack.is_subgoal_satisfied(sg2, current_entity_positions={(1, 1)}) is True

        popped = stack.pop_subgoal()
        assert popped == sg2
        assert stack.current_subgoal == sg1

    def test_find_safe_holding_bay_avoids_deadlocks(self) -> None:
        grid_shape = (10, 10)
        # Destination goal is at (9, 9)
        goals = {(9, 9)}
        static_barriers = {(0, 1), (1, 0)}  # Corner at (0, 0)
        dynamic_obstacles = {(3, 3)}

        # Search around (2, 2)
        holding_bay = FrontopolarSubgoalStack.find_safe_holding_bay(
            entity_pos=(2, 2),
            destination_goals=goals,
            static_barriers=static_barriers,
            dynamic_obstacles=dynamic_obstacles,
            grid_shape=grid_shape,
        )
        assert holding_bay is not None
        # Cannot be (0, 0) because (0, 0) is a corner deadlock!
        assert holding_bay != (0, 0)
        assert holding_bay not in static_barriers


class TestRemoteCausalAttributor:
    """Test vmPFC distal causal action-effect binding and remote barrier unlocking."""

    def test_distal_causal_learning_and_lookup(self) -> None:
        attributor = RemoteCausalAttributor(min_remote_distance=2, confidence_threshold=0.7)

        prev_grid = np.zeros((10, 10), dtype=int)
        # Pressure plate switch at (2, 2) of color 3
        prev_grid[2, 2] = 3
        # Distant door barrier at (8, 8) of color 8 (locked)
        prev_grid[8, 8] = 8

        curr_grid = prev_grid.copy()
        # Stepping on switch at (2, 2) causes door at (8, 8) to vanish (0)
        curr_grid[8, 8] = 0

        # Trial 1
        attributor.record_transition(
            prev_grid=prev_grid,
            curr_grid=curr_grid,
            action_pos=(2, 2),
            background_feature=0,
        )

        # Trial 2: repeated observation cements high confidence
        attributor.record_transition(
            prev_grid=prev_grid,
            curr_grid=curr_grid,
            action_pos=(2, 2),
            background_feature=0,
        )

        # Query what unlocks barrier at (8, 8)
        aff = attributor.get_trigger_for_barrier((8, 8), barrier_feature=8)
        assert aff is not None
        assert aff.trigger_pos == (2, 2)
        assert aff.trigger_feature == 3
        assert aff.confidence >= 0.7

        # Test alias method
        aff_alias = attributor.get_remote_trigger_for_barrier((8, 8), barrier_feature=8)
        assert aff_alias == aff

    def test_causal_induction_cortex_query_barrier_clearance(self) -> None:
        cortex = CausalInductionCortex()
        prev_grid = np.zeros((10, 10), dtype=int)
        prev_grid[2, 2] = 3
        prev_grid[8, 8] = 8

        curr_grid = prev_grid.copy()
        curr_grid[8, 8] = 0

        # Register distal transition
        for _ in range(2):
            cortex.causal_attributor.record_transition(
                prev_grid=prev_grid,
                curr_grid=curr_grid,
                action_pos=(2, 2),
                background_feature=0,
            )

        # Query barrier clearance via cortex
        trig_pos, req_tool = cortex.query_barrier_clearance(barrier_pos=(8, 8), barrier_feature=8)
        assert trig_pos == (2, 2)
        assert req_tool is None

        # Test reset_episode with retain_dynamics=False
        cortex.reset_episode(retain_dynamics=False)
        trig_pos_after, _ = cortex.query_barrier_clearance(barrier_pos=(8, 8), barrier_feature=8)
        assert trig_pos_after is None

        # Test attributor.reset()
        cortex.causal_attributor.co_occurrence_evidence[(((2, 2)), ((8, 8)))] = 5
        cortex.causal_attributor.reset()
        assert len(cortex.causal_attributor.co_occurrence_evidence) == 0

        # Test cortex.reset()
        cortex.reset()


class TestAnteriorCingulateConflictMonitor:
    """Test Anterior Cingulate Cortex (ACC) conflict monitoring, frustration scaling, and detour routing."""

    def test_frustration_scaling_on_recurring_cluster_deaths(self) -> None:
        from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor

        acc = AnteriorCingulateConflictMonitor(cluster_radius=3.0, frustration_threshold=0.5)

        # Single death: no active conflict
        f1 = acc.register_death_event((5.0, 5.0), step=10, last_action=2)
        assert f1 == 0.0
        assert acc.active_conflict is False
        assert acc.compute_heuristic_weight(1.0) == 1.0
        assert acc.get_chokepoint_penalty((5, 5)) == 0.0

        # Second death in immediate vicinity (5.0, 6.0): recurrence count = 2
        f2 = acc.register_death_event((5.0, 6.0), step=20, last_action=2)
        assert f2 >= 0.5
        assert acc.active_conflict is True
        assert acc.compute_heuristic_weight(1.0) <= 0.55

        # Chokepoint penalty should be positive near cluster center
        penalty = acc.get_chokepoint_penalty((5, 5))
        assert penalty > 15.0

        # Cells far away have zero penalty
        far_penalty = acc.get_chokepoint_penalty((15, 15))
        assert far_penalty == 0.0

    def test_synthesize_perimeter_detour(self) -> None:
        from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor

        acc = AnteriorCingulateConflictMonitor(cluster_radius=2.5)
        # Induce conflict at (5, 5)
        acc.register_death_event((5.0, 5.0), step=10, last_action=1)
        acc.register_death_event((5.0, 5.0), step=20, last_action=1)
        assert acc.active_conflict is True

        traversable = {
            (5, 5),
            (5, 6),
            (5, 4),  # In lethal cone
            (2, 5),
            (8, 5),
            (5, 1),
            (5, 9),  # Safe perimeter cells
        }
        detour = acc.synthesize_perimeter_detour(
            current_pos=(5, 3),
            goal_pos=(5, 8),
            grid_shape=(10, 10),
            traversable=traversable,
        )
        assert detour is not None
        # Detour must NOT be in the lethal cone
        assert detour not in {(5, 5), (5, 6), (5, 4)}

    def test_frontopolar_subgoal_causal_prerequisite_dag(self) -> None:
        stack = FrontopolarSubgoalStack(max_depth=4)
        sg_key = FrontopolarSubgoal(
            subgoal_id="collect_key",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(2, 2),
            target_destination=(2, 2),
        )
        sg_door = FrontopolarSubgoal(
            subgoal_id="unlock_door",
            subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
            target_entity_pos=(8, 8),
            target_destination=(8, 8),
            prerequisite_subgoal_ids=["collect_key"],
        )

        stack.push_subgoal(sg_door)
        stack.push_subgoal(sg_key)

        # sg_door prerequisites are not met yet
        assert sg_door.are_prerequisites_met(completed_ids=set()) is False
        assert sg_key.are_prerequisites_met(completed_ids=set()) is True

        # Stack executes top goal (sg_key) first
        assert stack.current_subgoal == sg_key
        stack.pop_subgoal()

        # Now collect_key is completed; sg_door prerequisites are satisfied!
        assert sg_door.are_prerequisites_met(completed_ids=set(stack.completed_subgoals)) is True
        assert stack.current_subgoal == sg_door


class TestMotorCortexEffector:
    """Test Motor Cortex Affordance Parsing, Motor Quantum Calibration, and Allocentric Targeting."""

    def test_parse_action_spec_variants(self) -> None:
        # 1. Bare int
        aff_bare = MotorCortexEffector.parse_action_spec(1)
        assert aff_bare.action_id == 1
        assert aff_bare.is_displacement is False
        assert aff_bare.requires_spatial_target is False

        # 2. Displacement dict
        aff_move = MotorCortexEffector.parse_action_spec({"action_id": 2, "name": "MOVE_UP"})
        assert aff_move.action_id == 2
        assert aff_move.is_displacement is True
        assert aff_move.requires_spatial_target is False

        # 3. Spatial effector click dict
        aff_click = MotorCortexEffector.parse_action_spec(
            {
                "action_id": 6,
                "name": "CLICK_CELL",
                "parameters": {"x": "int", "y": "int"},
            }
        )
        assert aff_click.action_id == 6
        assert aff_click.requires_spatial_target is True
        assert aff_click.target_param_keys == ("x", "y")

    def test_infer_motor_step_size(self) -> None:
        class DummyDyn:
            def __init__(self, dr: int, dc: int, is_disp: bool = True):
                self.dr = dr
                self.dc = dc
                self._is_disp = is_disp

            def is_displacement_action(self) -> bool:
                return self._is_disp

            def get_displacement(self) -> tuple[int, int]:
                return (self.dr, self.dc)

        dynamics = {
            1: DummyDyn(-1, 0),
            2: DummyDyn(3, 0),  # Leap of 3
            3: DummyDyn(0, 0, is_disp=False),
        }
        step = MotorCortexEffector.infer_motor_step_size([1, 2, 3], dynamics)
        assert step == 3

    def test_ground_effector_action_via_engine(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

        engine = AutonomousEpistemicEngine()
        engine.register_action_space(
            [
                {"action_id": 1, "name": "MOVE_UP"},
                {"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}},
            ]
        )
        assert engine.is_spatial_effector(6) is True
        assert engine.is_displacement_action(1) is True

        grid = np.zeros((10, 10), dtype=int)
        # Put an entity at (4, 7)
        grid[4, 7] = 3

        coords = engine.ground_effector_action(grid, action=6)
        assert coords is not None
        assert "x" in coords and "y" in coords
        assert 0 <= coords["x"] < 10
        assert 0 <= coords["y"] < 10

    def test_compute_epistemic_probe_proposal_and_atomic_apply(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

        engine = AutonomousEpistemicEngine()
        engine.avatar_pos = (2, 2)
        engine.avatar_feature = 1
        grid = np.zeros((10, 10), dtype=int)
        grid[2, 2] = 1

        # Calculate pure proposal without mutating engine state
        initial_probes = engine.total_epistemic_probes
        proposal = EpistemicCuriosityExplorer.compute_epistemic_probe(
            engine=engine,
            curr_grid=grid,
            available_actions=[1, 2, 3, 4],
        )

        assert isinstance(proposal, EpistemicProbeProposal)
        assert proposal.action in (1, 2, 3, 4)
        assert (
            engine.total_epistemic_probes == initial_probes
        )  # Reasoning invariant: zero direct mutation!

        # Now commit atomically via engine
        engine.apply_probe_proposal(proposal)
        assert engine.total_epistemic_probes == initial_probes + 1


class TestInductiveHypothesisEngine:
    """Test Prefrontal Cortex Inductive Logic, Relational Hypotheses & Counterexample Refutation."""

    def test_remote_mechanism_hypothesis_and_confirmation(self) -> None:
        hyp_engine = InductiveHypothesisEngine(min_support_to_confirm=2)

        # Transition 1: Avatar steps from (2, 2) onto switch at (2, 3) (feat 3).
        # A barrier of feat 5 at (8, 8) disappears (becomes 0).
        prev_grid = np.zeros((10, 10), dtype=int)
        prev_grid[2, 3] = 3  # switch
        prev_grid[8, 8] = 5  # barrier

        curr_grid = np.zeros((10, 10), dtype=int)
        curr_grid[2, 3] = 3  # switch remains

        rules = hyp_engine.observe_transition(
            prev_grid=prev_grid,
            action=4,
            curr_grid=curr_grid,
            prev_avatar_pos=(2, 2),
            curr_avatar_pos=(2, 3),
            background_feature=0,
        )

        assert len(rules) >= 1
        rule = hyp_engine.get_rule_for_barrier(5)
        assert rule is not None
        assert rule.rule_type == HypothesisType.REMOTE_MECHANISM
        assert rule.premise_feature == 3
        assert rule.target_feature == 5
        assert rule.status == HypothesisStatus.TENTATIVE

        # Transition 2: In another room, avatar steps onto switch 3 again, clearing another barrier 5
        prev_grid2 = np.zeros((10, 10), dtype=int)
        prev_grid2[4, 4] = 3
        prev_grid2[1, 1] = 5
        curr_grid2 = np.zeros((10, 10), dtype=int)
        curr_grid2[4, 4] = 3

        hyp_engine.observe_transition(
            prev_grid=prev_grid2,
            action=2,
            curr_grid=curr_grid2,
            prev_avatar_pos=(4, 3),
            curr_avatar_pos=(4, 4),
            background_feature=0,
        )

        # Now confirmed!
        confirmed = hyp_engine.get_confirmed_rules()
        assert len(confirmed) == 1
        assert confirmed[0].status == HypothesisStatus.CONFIRMED
        assert confirmed[0].confidence > 0.7

    def test_popperian_counterexample_refutation(self) -> None:
        hyp_engine = InductiveHypothesisEngine()

        # Step 1: initial tentative rule
        prev_grid = np.zeros((10, 10), dtype=int)
        prev_grid[1, 1] = 4
        prev_grid[9, 9] = 7
        curr_grid = np.zeros((10, 10), dtype=int)
        curr_grid[1, 1] = 4

        hyp_engine.observe_transition(
            prev_grid=prev_grid,
            action=1,
            curr_grid=curr_grid,
            prev_avatar_pos=(1, 0),
            curr_avatar_pos=(1, 1),
            background_feature=0,
        )
        assert hyp_engine.get_rule_for_barrier(7) is not None

        # Step 2: Avatar steps onto 4 again, but barrier 7 remains intact!
        prev_grid2 = np.zeros((10, 10), dtype=int)
        prev_grid2[3, 3] = 4
        prev_grid2[9, 9] = 7
        curr_grid2 = np.zeros((10, 10), dtype=int)
        curr_grid2[3, 3] = 4
        curr_grid2[9, 9] = 7  # Not cleared!

        hyp_engine.observe_transition(
            prev_grid=prev_grid2,
            action=1,
            curr_grid=curr_grid2,
            prev_avatar_pos=(3, 2),
            curr_avatar_pos=(3, 3),
            background_feature=0,
        )

        # Falsified by counterexample!
        assert hyp_engine.get_rule_for_barrier(7) is None
        rule = hyp_engine.hypotheses.get("remote_mech_4_clears_7")
        assert rule is not None
        assert rule.status == HypothesisStatus.REFUTED
        assert rule.counterexamples == 1

    def test_event_sourced_belief_transitions_and_predicate_export(self) -> None:
        hyp_engine = InductiveHypothesisEngine(min_support_to_confirm=2)

        prev_grid = np.zeros((8, 8), dtype=int)
        prev_grid[1, 1] = 2  # key switch
        prev_grid[5, 5] = 9  # door barrier

        curr_grid = np.zeros((8, 8), dtype=int)
        curr_grid[1, 1] = 2  # switch remains, barrier 9 clears

        hyp_engine.observe_transition(
            prev_grid=prev_grid,
            action=1,
            curr_grid=curr_grid,
            prev_avatar_pos=(1, 0),
            curr_avatar_pos=(1, 1),
            background_feature=0,
        )

        # Verify event log records hypothesis creation
        assert len(hyp_engine.event_log) >= 1
        create_event = hyp_engine.event_log[0]
        assert create_event.event_type == BeliefTransitionType.HYPOTHESIS_CREATED
        assert create_event.rule_id == "remote_mech_2_clears_9"

        # Transition 2 confirms
        prev_grid2 = np.zeros((8, 8), dtype=int)
        prev_grid2[3, 3] = 2
        prev_grid2[6, 6] = 9
        curr_grid2 = np.zeros((8, 8), dtype=int)
        curr_grid2[3, 3] = 2

        hyp_engine.observe_transition(
            prev_grid=prev_grid2,
            action=1,
            curr_grid=curr_grid2,
            prev_avatar_pos=(3, 2),
            curr_avatar_pos=(3, 3),
            background_feature=0,
        )

        confirmed_remote = [
            r
            for r in hyp_engine.get_confirmed_rules()
            if r.rule_type == HypothesisType.REMOTE_MECHANISM
        ]
        assert len(confirmed_remote) == 1
        conf_rule = confirmed_remote[0]

        # Verify CausalPredicate compilation
        pred = conf_rule.to_causal_predicate()
        assert pred.variable == "contact_feature"
        assert pred.value == 2
        assert pred.relation == "clears_barrier"
        assert pred.target_entity == "9"

        # Verify export to HCIRSymbolicWorldTheory
        theory = HCIRSymbolicWorldTheory()
        hyp_engine.export_to_world_theory(theory)
        assert 9 in theory.barrier_features
        assert len(theory.barrier_predicates) >= 1

    def test_frontopolar_spawn_causal_unlock(self) -> None:
        stack = FrontopolarSubgoalStack()
        subgoal = stack.spawn_causal_unlock_subgoal(
            barrier_pos=(8, 8),
            barrier_feat=5,
            trigger_pos=(2, 3),
            trigger_feat=3,
        )
        assert subgoal.subgoal_type == SubgoalType.UNLOCK_REMOTE_MECHANISM
        assert subgoal.target_destination == (2, 3)
        assert subgoal.required_feature == 3
        assert stack.current_subgoal == subgoal


class TestMorphologicalSaliencyEngine:
    """Test Lateral Occipital Complex Morphological Primitives, Enclosure, and Saliency."""

    def test_compute_saliency_map(self) -> None:
        grid = np.zeros((10, 10), dtype=int)
        # Add a rare focal point (feat 7)
        grid[5, 5] = 7

        saliency = MorphologicalSaliencyEngine.compute_saliency_map(grid, background_feature=0)
        assert saliency.shape == (10, 10)
        assert 0.0 <= np.min(saliency) and np.max(saliency) <= 1.0
        # The rare feature at (5, 5) must have higher salience than flat background
        assert saliency[5, 5] > saliency[0, 0]

    def test_extract_morphological_primitives(self) -> None:
        grid = np.zeros((12, 12), dtype=int)

        # 1. Hollow container (5x5 border of feat 4 with empty 3x3 interior)
        grid[1:6, 1] = 4
        grid[1:6, 5] = 4
        grid[1, 1:6] = 4
        grid[5, 1:6] = 4

        # 2. Linear structure (1x6 bar of feat 2)
        grid[8, 1:7] = 2

        # 3. Singleton (feat 9 at (10, 10))
        grid[10, 10] = 9

        # 4. Solid blob (2x2 square of feat 3)
        grid[8:10, 8:10] = 3

        primitives = MorphologicalSaliencyEngine.extract_morphological_primitives(
            grid, background_feature=0
        )
        types_by_feat = {p.feature_id: p.primitive_type for p in primitives}

        assert types_by_feat.get(4) == MorphologyPrimitiveType.HOLLOW_CONTAINER
        assert types_by_feat.get(2) == MorphologyPrimitiveType.LINEAR_STRUCTURE
        assert types_by_feat.get(9) == MorphologyPrimitiveType.SINGLETON
        assert types_by_feat.get(3) == MorphologyPrimitiveType.SOLID_BLOB

    def test_detect_enclosure(self) -> None:
        grid = np.zeros((10, 10), dtype=int)
        # Ring of barrier feature 8 enclosing (5, 5)
        for r in range(3, 8):
            grid[r, 3] = 8
            grid[r, 7] = 8
        for c in range(3, 8):
            grid[3, c] = 8
            grid[7, c] = 8

        # (5, 5) is completely enclosed inside the ring
        assert (
            MorphologicalSaliencyEngine.detect_enclosure(grid, (5, 5), barrier_features={8}) is True
        )

        # (1, 1) is outside the ring
        assert (
            MorphologicalSaliencyEngine.detect_enclosure(grid, (1, 1), barrier_features={8})
            is False
        )

    def test_filter_salt_pepper_noise(self) -> None:
        grid = np.zeros((8, 8), dtype=int)
        # Add single isolated noise pixels
        grid[1, 1] = 2
        grid[6, 2] = 2

        # Add a 2x2 solid block of feat 3 (area 4)
        grid[3:5, 3:5] = 3

        cleaned = MorphologicalSaliencyEngine.filter_salt_pepper_noise(
            grid, background_feature=0, min_component_area=2
        )
        # Noise speckles should be erased
        assert cleaned[1, 1] == 0
        assert cleaned[6, 2] == 0
        # Multi-cell entity must be preserved
        assert np.all(cleaned[3:5, 3:5] == 3)


class TestPrefrontalDeliberationEngine:
    """Test Prefrontal Cortex Deliberation Engine & Cognitive Arbitration."""

    def test_deliberation_engine_delegation(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_deliberation import PrefrontalDeliberationEngine

        engine = AutonomousEpistemicEngine()
        assert hasattr(engine, "deliberation_engine")
        assert isinstance(engine.deliberation_engine, PrefrontalDeliberationEngine)

        # Basic decision step with blank grid
        grid = np.zeros((10, 10), dtype=int)
        grid[2, 2] = 1  # avatar
        actions = [1, 2, 3, 4]
        action, data = engine.decide(grid, actions)
        assert action in actions
        assert engine.step_counter == 1

    def test_validate_plan_safety_hazard_detection(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_deliberation import PrefrontalDeliberationEngine
        from hbllm.hcir.world.cortex_planner import MentalSimulationStep

        delib = PrefrontalDeliberationEngine()
        engine = AutonomousEpistemicEngine()
        engine.avatar_pos = (5, 5)

        # Mark feature 9 as lethal
        engine.hazard_tracker.known_lethal_features.add(9)

        grid = np.zeros((10, 10), dtype=int)
        grid[5, 6] = 9  # lethal hazard

        step = MentalSimulationStep(
            action=2,
            predicted_avatar_pos=(5, 6),
        )

        safe, should_wait, wait_steps = delib.validate_plan_safety(engine, grid, [1, 2, 3, 4], step)
        assert safe is False
        assert should_wait is False

    def test_oscillatory_temporal_pacing_when_waiting(self) -> None:
        """Verify oscillatory pacing step executes cleanly using engine.action_deltas when should_wait is True."""
        from hbllm.hcir.world.autonomous_epistemic_engine import (
            AutonomousEpistemicEngine,
            EpistemicPhase,
        )
        from hbllm.hcir.world.cortex_deliberation import PrefrontalDeliberationEngine
        from hbllm.hcir.world.cortex_planner import MentalSimulationStep

        delib = PrefrontalDeliberationEngine()
        engine = AutonomousEpistemicEngine()
        engine.avatar_pos = (5, 5)
        engine.phase = EpistemicPhase.EXPLOITATION
        engine.working_memory.update_salience_baseline(engine)

        # Plan has a step that is temporarily blocked / needs wait
        next_step = MentalSimulationStep(
            action=1,
            predicted_avatar_pos=(4, 5),
            expected_mutation="step_up",
        )
        engine.mental_plan.append(next_step)

        # Mock validate_plan_safety to simulate a looming transient threat requiring a wait
        delib.validate_plan_safety = lambda *args, **kwargs: (True, True, 1)  # type: ignore[assignment]

        grid = np.zeros((10, 10), dtype=int)
        available_actions = [1, 2, 3, 4]  # Purely directional actions

        chosen_act, _ = delib.decide(engine, grid, available_actions)
        # Should pick a valid pacing action (not 1 since 1 is the waiting step)
        assert chosen_act in [2, 3, 4]
        # And return step was enqueued in mental_plan
        assert len(engine.mental_plan) > 0


class TestParietalCoordinateTransformer:
    """Test Posterior Parietal Cortex coordinate conversion and spatial symmetry projector."""

    def test_allo_to_ego_and_ego_to_allo(self) -> None:
        from hbllm.hcir.world.parietal_coordinate_transform import (
            ParietalCoordinateTransformer,
            SpatialEgoVector,
        )

        avatar_pos = (5, 5)
        target_pos = (3, 7)  # 2 north, 2 east -> NE, dist 4

        ego_vec = ParietalCoordinateTransformer.allo_to_ego(target_pos, avatar_pos)
        assert isinstance(ego_vec, SpatialEgoVector)
        assert ego_vec.delta_r == -2
        assert ego_vec.delta_c == 2
        assert ego_vec.manhattan_dist == 4
        assert ego_vec.bearing == "NE"

        # Invert back to allocentric
        recovered_allo = ParietalCoordinateTransformer.ego_to_allo(ego_vec, avatar_pos)
        assert recovered_allo == target_pos

    def test_dihedral_projections(self) -> None:
        from hbllm.hcir.world.parietal_coordinate_transform import ParietalCoordinateTransformer

        shape = (10, 10)
        pos = (2, 3)

        refl_h = ParietalCoordinateTransformer.project_dihedral_transform(
            pos, shape, "reflect_horizontal"
        )
        assert refl_h == (2, 6)

        refl_v = ParietalCoordinateTransformer.project_dihedral_transform(
            pos, shape, "reflect_vertical"
        )
        assert refl_v == (7, 3)

        rot_180 = ParietalCoordinateTransformer.project_dihedral_transform(pos, shape, "rotate_180")
        assert rot_180 == (7, 6)

    def test_symmetry_inference_and_subgoal_transfer(self) -> None:
        from hbllm.hcir.world.parietal_coordinate_transform import ParietalCoordinateTransformer

        # Symmetrical grid with horizontal reflection symmetry
        grid = np.zeros((8, 8), dtype=int)
        grid[2:6, 1] = 4
        grid[2:6, 6] = 4  # symmetric counterpart

        planes = ParietalCoordinateTransformer.infer_symmetry_planes(grid, background_feature=0)
        assert planes["horizontal"] == 1.0

        transferred = ParietalCoordinateTransformer.transfer_subgoal_across_symmetry(
            subgoal_pos=(3, 1),
            grid_shape=(8, 8),
            grid=grid,
            background_feature=0,
        )
        assert transferred == (3, 6)


class TestPhase25CognitiveBridgeCausalAndTransactions:
    """Validate declarative causal predicates and atomic HCIR transactions for Phase 25 components."""

    def test_affordance_discovery_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.affordance_discovery import (
            AffordanceHypothesis,
            BaseAffordanceDiscoveryEngine,
        )

        hyp = AffordanceHypothesis(
            hypothesis_id="aff_ball_roll",
            action="ROLL",
            entity_shape="ball",
            affordance_label="ROLLABLE",
            confidence=0.85,
        )
        pred = hyp.to_causal_predicate()
        assert pred.variable == "affordance:ball"
        assert pred.operator == "=="
        assert pred.value == "ROLLABLE"

        ops = hyp.to_transaction_operations()
        assert len(ops) == 1
        assert ops[0].node_id == f"aff_hyp_{hyp.hypothesis_id}"

        engine = BaseAffordanceDiscoveryEngine()
        engine.hypotheses.append(hyp)
        engine_preds = engine.export_causal_predicates()
        assert len(engine_preds) == 1
        engine_ops = engine.to_transaction_operations()
        assert len(engine_ops) == 1

    def test_anterior_cingulate_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.anterior_cingulate_conflict import (
            AnteriorCingulateConflictMonitor,
            FailureCluster,
        )

        cluster = FailureCluster(center_r=5.0, center_c=5.0, count=3)
        c_pred = cluster.to_causal_predicate()
        assert "failure_cluster" in c_pred.variable
        c_ops = cluster.to_transaction_operations()
        assert len(c_ops) == 1

        acc = AnteriorCingulateConflictMonitor()
        acc.clusters.append(cluster)
        acc_preds = acc.export_causal_predicates()
        assert len(acc_preds) == 2
        acc_ops = acc.to_transaction_operations()
        assert len(acc_ops) == 2

    def test_bilateral_coordination_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.bilateral_coordination import (
            BilateralCoordinateIntegrator,
            BilateralState,
            SymmetryAxis,
        )

        state = BilateralState(
            pos1=(2, 3),
            pos2=(2, 7),
            symmetry_axis=SymmetryAxis.HORIZONTAL,
            features={1, 2},
            confidence=0.9,
        )
        state_pred = state.to_causal_predicate()
        assert state_pred.variable == "bilateral_symmetry"
        state_ops = state.to_transaction_operations()
        assert len(state_ops) == 1

        integrator = BilateralCoordinateIntegrator()
        integrator.active_bilateral_state = state
        preds = integrator.to_causal_predicates()
        assert len(preds) == 1
        ops = integrator.to_transaction_operations()
        assert len(ops) == 1

    def test_capability_evidence_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.capability_evidence import (
            BenchmarkStatus,
            CapabilityEvidence,
            CapabilityLedger,
            GeneralizationStatus,
            ImplementationStatus,
            IntegrationStatus,
            UnitTestStatus,
        )

        ev = CapabilityEvidence(
            capability_id="cortex_planner",
            name="Cortex Planner",
            domain="planning",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            integration_status=IntegrationStatus.ACTIVE_RUNTIME,
            unit_test_status=UnitTestStatus.PASSING,
            benchmark_status=BenchmarkStatus.BENCHMARKED,
            generalization_status=GeneralizationStatus.HELD_OUT_EVIDENCED,
        )
        ev_pred = ev.to_causal_predicate()
        assert ev_pred.variable == "capability:cortex_planner"
        ev_ops = ev.to_transaction_operations()
        assert len(ev_ops) == 1

        ledger = CapabilityLedger()
        ledger.register(ev)
        ledger_preds = ledger.export_causal_predicates()
        assert len(ledger_preds) == 1
        ledger_ops = ledger.to_transaction_operations()
        assert len(ledger_ops) == 2

    def test_cerebellar_clock_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cerebellar_phase_clock import (
            CerebellarPhaseClock,
            PhaseGateDecision,
        )

        dec = PhaseGateDecision(
            should_wait=True,
            wait_steps_recommended=2,
            current_phase=1,
            environmental_period=4,
        )
        dec_pred = dec.to_causal_predicate()
        assert dec_pred.variable == "motor_phase_gate"
        assert dec_pred.value == "WAIT"
        dec_ops = dec.to_transaction_operations()
        assert len(dec_ops) == 1

        clock = CerebellarPhaseClock()
        clock.detected_macro_period = 16
        clock_preds = clock.export_causal_predicates()
        assert len(clock_preds) == 1
        clock_ops = clock.to_transaction_operations()
        assert len(clock_ops) == 1

    def test_habenular_ior_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.habenular_episodic_inhibition import (
            FatalPrefixRecord,
            HabenularEpisodicIOR,
        )

        rec = FatalPrefixRecord(
            terminal_step=12,
            fatal_pos=(4, 4),
            fatal_action=3,
            branch_point_step=9,
        )
        rec_pred = rec.to_causal_predicate()
        assert "fatal_prefix:12" in rec_pred.variable
        rec_ops = rec.to_transaction_operations()
        assert len(rec_ops) == 1

        ior = HabenularEpisodicIOR()
        ior.fatal_prefixes.append(rec)
        ior_preds = ior.export_causal_predicates()
        assert len(ior_preds) == 2
        ior_ops = ior.to_transaction_operations()
        assert len(ior_ops) == 2


class TestPhase26CognitiveBridgeCausalAndTransactions:
    """Validate declarative causal predicates and atomic HCIR transactions for Phase 26 components."""

    def test_symbolic_constraints_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.symbolic_constraints import (
            LocalCardinalityConstraint,
            SymbolicConstraintSolver,
        )

        c = LocalCardinalityConstraint(
            center=(3, 3),
            radius=1,
            required_count=2,
            scope={(2, 3), (3, 2)},
        )
        c_pred = c.to_causal_predicate()
        assert c_pred.variable == "constraint_cardinality:(3, 3)"
        assert c_pred.value == 2
        c_ops = c.to_transaction_operations()
        assert len(c_ops) == 1
        assert c_ops[0].node_id == "cardinality_constraint_3_3"

        solver = SymbolicConstraintSolver()
        solver.constraints.append(c)
        solver_pred = solver.to_causal_predicate()
        assert solver_pred.variable == "symbolic_constraint_solver"
        solver_preds = solver.export_causal_predicates()
        assert len(solver_preds) == 2
        solver_ops = solver.to_transaction_operations()
        assert len(solver_ops) == 2

    def test_morphology_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.morphology import (
            MorphologicalConcept,
            ShapeArchetype,
        )

        arch = ShapeArchetype.from_coords([(0, 0), (0, 1), (1, 0)])
        arch_pred = arch.to_causal_predicate()
        assert "shape_archetype:" in arch_pred.variable
        arch_ops = arch.to_transaction_operations()
        assert len(arch_ops) == 1

        concept = MorphologicalConcept(
            canonical_id=arch.canonical_id,
            archetype=arch,
            canonical_name="L_tromino",
        )
        c_pred = concept.to_causal_predicate()
        assert c_pred.variable == "morphology:L_tromino"
        c_ops = concept.to_transaction_operations()
        assert len(c_ops) == 1
        assert c_ops[0].node_id == "morph_concept_L_tromino"

    def test_whole_grid_predictor_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.predictors.whole_grid import (
            DimensionMode,
            DimensionRule,
            WholeGridPredictor,
        )

        rule = DimensionRule(mode=DimensionMode.FIXED, fixed_shape=(10, 10))
        rule_pred = rule.to_causal_predicate()
        assert rule_pred.variable == "dimension_rule"
        assert rule_pred.value == "FIXED"
        rule_ops = rule.to_transaction_operations()
        assert len(rule_ops) == 1

        predictor = WholeGridPredictor()
        p_pred = predictor.to_causal_predicate()
        assert p_pred.variable == "predictor:whole_grid"
        p_ops = predictor.to_transaction_operations()
        assert len(p_ops) == 1

    def test_ensemble_predictors_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.predictors.llm import LLMReasoningPredictor
        from hbllm.hcir.world.predictors.neural import NeuralWorldModel
        from hbllm.hcir.world.predictors.physics import PhysicsPredictor
        from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor
        from hbllm.hcir.world.predictors.statistical import StatisticalPredictor

        phys = PhysicsPredictor()
        assert phys.to_causal_predicate().variable == "predictor:physics"
        assert len(phys.to_transaction_operations()) == 1

        llm = LLMReasoningPredictor()
        assert llm.to_causal_predicate().variable == "predictor:llm"
        assert len(llm.to_transaction_operations()) == 1

        neural = NeuralWorldModel()
        assert neural.to_causal_predicate().variable == "predictor:neural"
        assert len(neural.to_transaction_operations()) == 1

        snn = SNNTemporalPredictor()
        assert snn.to_causal_predicate().variable == "predictor:snn"
        assert len(snn.to_transaction_operations()) == 1

        stat = StatisticalPredictor()
        assert stat.to_causal_predicate().variable == "predictor:statistical"
        assert len(stat.to_transaction_operations()) == 1


class TestPhase27CognitiveBridgeCausalAndTransactions:
    """Validate declarative causal predicates and atomic HCIR transactions for Phase 27 components."""

    def test_rule_induction_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.rule_induction import (
            RuleFamily,
            RuleInductionEngine,
            RuleInductionResult,
            RulePosteriorResult,
            TransformationRule,
        )

        rule = TransformationRule(
            rule_id="rule_rot90",
            family=RuleFamily.AFFINE,
            params={"angle": 90},
            complexity=1.5,
        )
        r_pred = rule.to_causal_predicate()
        assert r_pred.variable == "transformation_rule:rule_rot90"
        assert r_pred.value == "AFFINE"
        r_ops = rule.to_transaction_operations()
        assert len(r_ops) == 1
        assert r_ops[0].node_id == "rule_rule_rot90"

        posterior = RulePosteriorResult(
            best_rule=rule,
            probabilities={"rule_rot90": 0.95},
            entropy_bits=0.1,
            confidence=0.95,
            is_ambiguous=False,
        )
        res = RuleInductionResult(
            selected_rule=rule,
            surviving_rules=[rule],
            posterior=posterior,
            refutations=[],
            metadata={},
        )
        res_pred = res.to_causal_predicate()
        assert res_pred.variable == "rule_induction_selected"
        res_preds = res.export_causal_predicates()
        assert len(res_preds) >= 2
        res_ops = res.to_transaction_operations()
        assert len(res_ops) >= 2

        engine = RuleInductionEngine()
        engine_pred = engine.to_causal_predicate()
        assert engine_pred.variable == "rule_induction_engine"
        engine_ops = engine.to_transaction_operations()
        assert len(engine_ops) == 1

    def test_global_invariant_induction_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.global_invariant_induction import (
            BranchRule,
            GlobalScalarConditionedOperator,
        )

        branch = BranchRule(
            rule_type="recolor",
            params={"src_color": 1, "dst_color": 2},
        )
        b_pred = branch.to_causal_predicate()
        assert b_pred.variable == "branch_rule:recolor"
        b_ops = branch.to_transaction_operations()
        assert len(b_ops) == 1
        assert b_ops[0].node_id == "branch_rule_recolor"

        op = GlobalScalarConditionedOperator(
            name="test_scalar_op",
            property_name="color_count",
            modulus=2,
            branches={0: branch},
            complexity=2.5,
        )
        op_pred = op.to_causal_predicate()
        assert op_pred.variable == "global_invariant:color_count"
        op_preds = op.export_causal_predicates()
        assert len(op_preds) == 2
        op_ops = op.to_transaction_operations()
        assert len(op_ops) == 2

    def test_topological_graph_induction_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.topological_graph_induction import (
            GeodesicManifoldRouter,
            GeodesicPathOperator,
        )

        router_pred = GeodesicManifoldRouter.to_causal_predicate()
        assert router_pred.variable == "geodesic_manifold_router"
        router_ops = GeodesicManifoldRouter.to_transaction_operations()
        assert len(router_ops) == 1

        path_op = GeodesicPathOperator(
            name="geodesic_test_op",
            start_color=1,
            goal_color=2,
            path_color=3,
        )
        pop_pred = path_op.to_causal_predicate()
        assert pop_pred.variable == "geodesic_path:1->2"
        pop_ops = path_op.to_transaction_operations()
        assert len(pop_ops) == 1
        assert pop_ops[0].node_id == "op_geodesic_test_op"

    def test_relational_graph_matcher_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.inferotemporal_segmentation import VentralObjectToken
        from hbllm.hcir.world.relational_graph_matcher import (
            ObjectCorrespondence,
            ObjectGraph,
            RelationalEdge,
            SpatialRelation,
        )

        edge = RelationalEdge(
            source_id=1,
            target_id=2,
            relation=SpatialRelation.LEFT_OF,
            distance=2.0,
            dx=0.0,
            dy=-2.0,
        )
        e_pred = edge.to_causal_predicate()
        assert e_pred.variable == "relational_edge:1->2"
        e_ops = edge.to_transaction_operations()
        assert len(e_ops) == 1

        token = VentralObjectToken(
            object_id=1,
            feature_id=3,
            area=4,
            centroid=(1, 1),
            anchor_coord=(1, 1),
            bounding_box=(0, 0, 2, 2),
            aspect_ratio=1.0,
            is_compact=True,
        )
        graph = ObjectGraph()
        graph.add_node(token)
        graph.add_edge(edge)
        g_pred = graph.to_causal_predicate()
        assert g_pred.variable == "object_graph"
        g_preds = graph.export_causal_predicates()
        assert len(g_preds) == 2
        g_ops = graph.to_transaction_operations()
        assert len(g_ops) == 2

        corr = ObjectCorrespondence(
            src_object_id=1,
            dst_object_id=10,
            match_score=0.92,
            matched_relations=2,
            invariants_preserved=["color"],
        )
        c_pred = corr.to_causal_predicate()
        assert c_pred.variable == "object_match:1->10"
        c_ops = corr.to_transaction_operations()
        assert len(c_ops) == 1

    def test_representation_expansion_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.representation_expansion import (
            RepresentationExpansionEngine,
            RepresentationRevisionTrace,
            ResidualEvent,
            SynthesizedCellularFluxOperator,
        )

        event = ResidualEvent(
            r=2,
            c=3,
            x_val=0,
            y_val=5,
        )
        ev_pred = event.to_causal_predicate()
        assert ev_pred.variable == "residual_event:2_3"
        ev_ops = event.to_transaction_operations()
        assert len(ev_ops) == 1

        op = SynthesizedCellularFluxOperator(
            name="flux_op_fill",
            flux_mode="flood_infill",
            direction=(0, 1),
            params={"target_color": 5},
            complexity=2.0,
        )
        op_pred = op.to_causal_predicate()
        assert op_pred.variable == "flux_operator:flux_op_fill"
        op_ops = op.to_transaction_operations()
        assert len(op_ops) == 1

        trace = RepresentationRevisionTrace(
            task_id="task_rev_01",
            family="flood",
            phase1_solved=False,
            phase1_inadequacy_detected=True,
            phase1_epistemic_uncertainty=0.8,
            unexplained_residual_pixels=3,
            synthesized_operator_name="flux_op_fill",
            out_of_construction_verified=True,
            phase2_solved=True,
            phase2_exact_match=True,
            phase2_epistemic_uncertainty=0.1,
            reusable_on_novel_task=True,
            control_ablation_passed=True,
        )
        tr_pred = trace.to_causal_predicate()
        assert tr_pred.variable == "representation_revision:task_rev_01"
        tr_ops = trace.to_transaction_operations()
        assert len(tr_ops) == 1

        eng_pred = RepresentationExpansionEngine.to_causal_predicate()
        assert eng_pred.variable == "representation_expansion_engine"
        eng_ops = RepresentationExpansionEngine.to_transaction_operations()
        assert len(eng_ops) == 1


class TestPhase28CognitiveBridgeCausalAndTransactions:
    """Validate declarative causal predicates and atomic HCIR transactions for Phase 28 perceptual & tracking faculties."""

    def test_cortex_perception_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_perception import (
            EpistemicObservationDiff,
            FrameDiffType,
            OrientedThreat,
            PerceptionEngine,
        )

        diff = EpistemicObservationDiff(
            changed_pixel_count=5,
            diff_type=FrameDiffType.TRANSLATION,
            translation_delta=(1, 0),
            moved_object_feature=2,
            moved_object_size=5,
        )
        d_pred = diff.to_causal_predicate()
        assert d_pred.variable == "observation_diff:TRANSLATION"
        assert d_pred.value == 5
        d_ops = diff.to_transaction_operations()
        assert len(d_ops) == 1
        assert d_ops[0].node_id == "diff_TRANSLATION_5"

        threat = OrientedThreat(
            pos=(3, 3),
            facing=(0, 1),
            gaze_pos=(3, 4),
            feature_id=7,
        )
        t_pred = threat.to_causal_predicate()
        assert t_pred.variable == "oriented_threat:7"
        t_ops = threat.to_transaction_operations()
        assert len(t_ops) == 1
        assert t_ops[0].node_id == "threat_7_3_3"

        eng_pred = PerceptionEngine.to_causal_predicate()
        assert eng_pred.variable == "cortex_perception_engine"
        eng_ops = PerceptionEngine.to_transaction_operations()
        assert len(eng_ops) == 1

    def test_inferotemporal_segmentation_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.inferotemporal_segmentation import (
            InferotemporalSegmentationEngine,
            VentralObjectToken,
        )

        token = VentralObjectToken(
            object_id=2,
            feature_id=4,
            area=6,
            centroid=(3, 3),
            anchor_coord=(3, 3),
            bounding_box=(2, 2, 4, 4),
            aspect_ratio=1.0,
            is_compact=True,
            salience_score=0.9,
        )
        tok_pred = token.to_causal_predicate()
        assert tok_pred.variable == "ventral_token:4_2"
        assert tok_pred.value == 6
        tok_ops = token.to_transaction_operations()
        assert len(tok_ops) == 1
        assert tok_ops[0].node_id == "ventral_token_4_2"

        eng_pred = InferotemporalSegmentationEngine.to_causal_predicate()
        assert eng_pred.variable == "inferotemporal_segmentation_engine"
        eng_ops = InferotemporalSegmentationEngine.to_transaction_operations()
        assert len(eng_ops) == 1

    def test_cortex_morphology_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_morphology import (
            MorphologicalEntity,
            MorphologicalSaliencyEngine,
            MorphologyPrimitiveType,
            ScaleInferenceResult,
            StampInstance,
            WallpaperSymmetryGroup,
            WallpaperTilingModel,
        )

        entity = MorphologicalEntity(
            entity_id="entity_m1",
            primitive_type=MorphologyPrimitiveType.SOLID_BLOB,
            feature_id=3,
            area=9,
            bounding_box=(1, 1, 3, 3),
            centroid=(2, 2),
            aspect_ratio=1.0,
        )
        e_pred = entity.to_causal_predicate()
        assert e_pred.variable == "morph_entity:SOLID_BLOB_3"
        assert e_pred.value == 9
        e_ops = entity.to_transaction_operations()
        assert len(e_ops) == 1
        assert e_ops[0].node_id == "morph_entity_entity_m1"

        scale_res = ScaleInferenceResult(
            scale_r=2.0,
            scale_c=2.0,
            is_isotropic=True,
            residual_error=0.0,
        )
        s_pred = scale_res.to_causal_predicate()
        assert s_pred.variable == "scale_inference_result"
        s_ops = scale_res.to_transaction_operations()
        assert len(s_ops) == 1

        stamp = StampInstance(row=1, col=2, rotation_k=1, fidelity=0.95)
        st_pred = stamp.to_causal_predicate()
        assert st_pred.variable == "stamp_instance:1_2"
        st_ops = stamp.to_transaction_operations()
        assert len(st_ops) == 1

        wall = WallpaperTilingModel(
            group=WallpaperSymmetryGroup.P1,
            period_r=3,
            period_c=3,
            unit_cell=np.zeros((3, 3), dtype=int),
            concordance_score=0.92,
            coverage_ratio=1.0,
        )
        w_pred = wall.to_causal_predicate()
        assert w_pred.variable == "wallpaper_tiling:p1"
        w_ops = wall.to_transaction_operations()
        assert len(w_ops) == 1

        eng_pred = MorphologicalSaliencyEngine.to_causal_predicate()
        assert eng_pred.variable == "morphological_saliency_engine"
        eng_ops = MorphologicalSaliencyEngine.to_transaction_operations()
        assert len(eng_ops) == 1

    def test_visual_symmetry_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.visual_symmetry import (
            CountingRelationResult,
            CountingRelationType,
            DiscrepancyTarget,
            PeriodicTilingResult,
            SymmetryAnalysisResult,
            SymmetryAxis,
            VisualSymmetryAnalyzer,
        )

        target = DiscrepancyTarget(r=1, c=2, expected_feature=3, confidence=0.9)
        t_pred = target.to_causal_predicate()
        assert t_pred.variable == "symmetry_discrepancy:1_2"
        t_ops = target.to_transaction_operations()
        assert len(t_ops) == 1

        sym = SymmetryAnalysisResult(
            dominant_axis=SymmetryAxis.HORIZONTAL,
            dominant_score=0.88,
            scores={"horizontal": 0.88},
            is_symmetric=True,
        )
        sym_pred = sym.to_causal_predicate()
        assert sym_pred.variable == "symmetry_axis:horizontal"
        sym_ops = sym.to_transaction_operations()
        assert len(sym_ops) == 1

        tiling = PeriodicTilingResult(
            tile_shape=(3, 3),
            repetition_counts=(2, 2),
            confidence=0.95,
        )
        ti_pred = tiling.to_causal_predicate()
        assert ti_pred.variable == "periodic_tiling_unit"
        ti_ops = tiling.to_transaction_operations()
        assert len(ti_ops) == 1

        rel = CountingRelationResult(
            is_relation=True,
            relation_type=CountingRelationType.CONSERVED,
            confidence=0.85,
        )
        r_pred = rel.to_causal_predicate()
        assert r_pred.variable == "counting_relation:conserved"
        r_ops = rel.to_transaction_operations()
        assert len(r_ops) == 1

        eng_pred = VisualSymmetryAnalyzer.to_causal_predicate()
        assert eng_pred.variable == "visual_symmetry_analyzer"
        eng_ops = VisualSymmetryAnalyzer.to_transaction_operations()
        assert len(eng_ops) == 1

    def test_remote_causal_attribution_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.remote_causal_attribution import (
            RemoteCausalAffordance,
            RemoteCausalAttributor,
        )

        aff = RemoteCausalAffordance(
            trigger_pos=(1, 1),
            trigger_feature=2,
            remote_pos=(8, 8),
            original_remote_feature=5,
            mutated_remote_feature=0,
            confidence=0.85,
        )
        a_pred = aff.to_causal_predicate()
        assert a_pred.variable == "remote_affordance:2->5"
        a_ops = aff.to_transaction_operations()
        assert len(a_ops) == 1

        attributor = RemoteCausalAttributor()
        attributor.remote_actuators[(8, 8)].append(aff)
        attributor.co_occurrence_evidence[(((1, 1)), ((8, 8)))] = 3
        preds = attributor.export_causal_predicates()
        assert len(preds) == 2
        ops = attributor.to_transaction_operations()
        assert len(ops) == 2

    def test_spatiotemporal_tracker_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.spatiotemporal_tracker import (
            DeformationRecord,
            DynamicCellPhase,
            LifecycleTrackingResult,
            MorphologicalDeformationTracker,
            MorphologicalDeformationType,
            PhasePrecessionEstimator,
            SpatiotemporalHazardTracker,
        )

        phase = DynamicCellPhase(
            r=4,
            c=4,
            period=3,
            cycle_values=[0, 1, 2],
            hazardous_values={2},
        )
        p_pred = phase.to_causal_predicate()
        assert p_pred.variable == "cell_phase:4_4"
        p_ops = phase.to_transaction_operations()
        assert len(p_ops) == 1

        est_pred = PhasePrecessionEstimator.to_causal_predicate()
        assert est_pred.variable == "phase_precession_estimator"
        est_ops = PhasePrecessionEstimator.to_transaction_operations()
        assert len(est_ops) == 1

        tracker = SpatiotemporalHazardTracker()
        tracker.periodic_cells[(4, 4)] = phase
        tracker_preds = tracker.export_causal_predicates()
        assert len(tracker_preds) == 2
        tracker_ops = tracker.to_transaction_operations()
        assert len(tracker_ops) == 2

        rec = DeformationRecord(
            entity_id="entity_e1",
            prior_cells_count=4,
            current_cells_count=5,
            iou=0.8,
            euler_characteristic=1,
            deformation_type=MorphologicalDeformationType.TOPOLOGICAL_GROWTH,
        )
        rec_pred = rec.to_causal_predicate()
        assert rec_pred.variable == "deformation_record:entity_e1"
        rec_ops = rec.to_transaction_operations()
        assert len(rec_ops) == 1

        lifecycle_res = LifecycleTrackingResult(
            one_to_one_mappings={"entity_e1": "entity_e1_next"},
            deformation_records=[rec],
            fission_records=[],
            fusion_records=[],
            unmatched_prior_ids=[],
            unmatched_current_ids=[],
        )
        lc_preds = lifecycle_res.export_causal_predicates()
        assert len(lc_preds) == 2
        lc_ops = lifecycle_res.to_transaction_operations()
        assert len(lc_ops) == 2

        eng_pred = MorphologicalDeformationTracker.to_causal_predicate()
        assert eng_pred.variable == "morphological_deformation_tracker"
        eng_ops = MorphologicalDeformationTracker.to_transaction_operations()
        assert len(eng_ops) == 1

    def test_spatial_containment_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.spatial_containment import (
            BaseSpatialContainmentEngine,
            RoomDoor,
            RoomTopologyExtractor,
            RoomTopologyGraph,
            SpatialRelationFact,
            SpatialRelationType,
        )

        fact = SpatialRelationFact(
            relation=SpatialRelationType.INSIDE,
            subject_id="item_ball",
            object_id="container_box",
            confidence=0.95,
        )
        f_pred = fact.to_causal_predicate()
        assert f_pred.variable == "spatial_relation:INSIDE"
        f_ops = fact.to_transaction_operations()
        assert len(f_ops) == 1

        engine = BaseSpatialContainmentEngine()
        eng_preds = engine.export_causal_predicates()
        assert len(eng_preds) == 1
        eng_ops = engine.to_transaction_operations()
        assert len(eng_ops) == 1

        door = RoomDoor(door_coord=(3, 5), connects_rooms=(0, 1))
        d_pred = door.to_causal_predicate()
        assert d_pred.variable == "room_door:(3,5)"
        d_ops = door.to_transaction_operations()
        assert len(d_ops) == 1

        topology = RoomTopologyGraph(
            rooms={0: [(1, 1)], 1: [(4, 4)]},
            doors=[door],
            adjacency={0: [1], 1: [0]},
        )
        top_preds = topology.export_causal_predicates()
        assert len(top_preds) == 2
        top_ops = topology.to_transaction_operations()
        assert len(top_ops) == 2

        ext_pred = RoomTopologyExtractor.to_causal_predicate()
        assert ext_pred.variable == "room_topology_extractor"
        ext_ops = RoomTopologyExtractor.to_transaction_operations()
        assert len(ext_ops) == 1

    def test_surprise_engine_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.surprise_engine import (
            SurpriseEngine,
            SurpriseEvaluation,
        )

        eval_res = SurpriseEvaluation(
            surprise_score=0.45,
            is_surprising=True,
            salience_boost=0.35,
            is_persistent=True,
        )
        ev_pred = eval_res.to_causal_predicate()
        assert ev_pred.variable == "surprise_evaluation"
        assert ev_pred.value == 0.45
        ev_ops = eval_res.to_transaction_operations()
        assert len(ev_ops) == 1

        engine = SurpriseEngine()
        engine._surprise_ledger["physics_world"] = [0.45, 0.50]
        eng_preds = engine.export_causal_predicates()
        assert len(eng_preds) == 2
        eng_ops = engine.to_transaction_operations()
        assert len(eng_ops) == 1


class TestPhase29CognitiveBridgeCausalAndTransactions:
    """Validate declarative causal predicates and atomic HCIR transactions for Phase 29 components."""

    def test_frontopolar_subgoal_stack_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.frontopolar_subgoal_stack import (
            FrontopolarSubgoal,
            FrontopolarSubgoalStack,
            SubgoalType,
        )

        subgoal = FrontopolarSubgoal(
            subgoal_id="sg_1",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(2, 3),
            target_destination=(5, 5),
            priority=2.0,
        )
        sg_pred = subgoal.to_causal_predicate()
        assert "subgoal:" in sg_pred.variable
        sg_ops = subgoal.to_transaction_operations()
        assert len(sg_ops) == 1

        stack = FrontopolarSubgoalStack()
        stack.push(subgoal)
        stack_pred = stack.to_causal_predicate()
        assert stack_pred.variable == "frontopolar_subgoal_stack"
        stack_preds = stack.export_causal_predicates()
        assert len(stack_preds) == 2
        stack_ops = stack.to_transaction_operations()
        assert len(stack_ops) == 2

    def test_cortex_planner_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_planner import (
            DynamicReplanner,
            EpistemicGoalProposal,
            MentalSimulationPlanner,
            MentalSimulationStep,
        )

        step = MentalSimulationStep(
            action=1,
            predicted_avatar_pos=(3, 4),
            is_deterministic_geodesic=True,
        )
        step_pred = step.to_causal_predicate()
        assert "simulation_step:1" in step_pred.variable
        step_ops = step.to_transaction_operations()
        assert len(step_ops) == 1

        prop = EpistemicGoalProposal(
            goals=[(6, 6)],
            confirmed_goals=[(6, 6)],
            new_learned_goal_positions={(6, 6)},
        )
        prop_pred = prop.to_causal_predicate()
        assert "epistemic_goal_proposal" in prop_pred.variable
        prop_ops = prop.to_transaction_operations()
        assert len(prop_ops) == 1

        planner_pred = MentalSimulationPlanner.to_causal_predicate()
        assert planner_pred.variable == "mental_simulation_planner"
        planner_ops = MentalSimulationPlanner.to_transaction_operations()
        assert len(planner_ops) == 1

        replanner_pred = DynamicReplanner.to_causal_predicate()
        assert replanner_pred.variable == "dynamic_replanner"
        replanner_ops = DynamicReplanner.to_transaction_operations()
        assert len(replanner_ops) == 1

    def test_cortex_deliberation_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_deliberation import (
            PlanSafetyEvaluator,
            PrefrontalDeliberationEngine,
            SafetyGateResult,
        )

        gate = SafetyGateResult(
            is_safe=True,
            should_wait=False,
            wait_steps=0,
            reason="clear",
        )
        gate_pred = gate.to_causal_predicate()
        assert gate_pred.variable == "safety_gate_result"
        assert gate_pred.value == "SAFE"
        gate_ops = gate.to_transaction_operations()
        assert len(gate_ops) == 1

        eval_pred = PlanSafetyEvaluator.to_causal_predicate()
        assert eval_pred.variable == "plan_safety_evaluator"
        eval_ops = PlanSafetyEvaluator.to_transaction_operations()
        assert len(eval_ops) == 1

        engine_pred = PrefrontalDeliberationEngine.to_causal_predicate()
        assert engine_pred.variable == "prefrontal_deliberation_engine"
        engine_ops = PrefrontalDeliberationEngine.to_transaction_operations()
        assert len(engine_ops) == 1

    def test_cortex_motor_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_motor import (
            ACCConflictMonitor,
            ACCConflictRecord,
            EpistemicProbeProposal,
            MotorCortexEffector,
        )

        rec = ACCConflictRecord(
            step=10,
            cycle_period=2,
            cyclic_actions=[1, 2],
            suppressed_actions=[2],
        )
        rec_pred = rec.to_causal_predicate()
        assert "acc_conflict:10" in rec_pred.variable
        rec_ops = rec.to_transaction_operations()
        assert len(rec_ops) == 1

        monitor = ACCConflictMonitor()
        monitor.conflict_records.append(rec)
        monitor_pred = monitor.to_causal_predicate()
        assert monitor_pred.variable == "acc_conflict_monitor"
        monitor_preds = monitor.export_causal_predicates()
        assert len(monitor_preds) == 2
        monitor_ops = monitor.to_transaction_operations()
        assert len(monitor_ops) == 2

        probe = EpistemicProbeProposal(
            action=1,
            new_probe_target=(3, 3),
            exploration_mode="curiosity",
        )
        probe_pred = probe.to_causal_predicate()
        assert "probe_proposal:1" in probe_pred.variable
        probe_ops = probe.to_transaction_operations()
        assert len(probe_ops) == 1

        effector_pred = MotorCortexEffector.to_causal_predicate()
        assert effector_pred.variable == "motor_cortex_effector"
        effector_ops = MotorCortexEffector.to_transaction_operations()
        assert len(effector_ops) == 1

    def test_cortex_assimilator_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_assimilator import EpistemicFeedbackAssimilator

        assimilator_pred = EpistemicFeedbackAssimilator.to_causal_predicate()
        assert assimilator_pred.variable == "epistemic_feedback_assimilator"
        assimilator_ops = EpistemicFeedbackAssimilator.to_transaction_operations()
        assert len(assimilator_ops) == 1

    def test_cortex_episodic_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.cortex_episodic import (
            ConsolidatedKnowledgeItem,
            DualStoreConsolidationEngine,
            EpisodicTransition,
            HippocampalEpisodicCortex,
            SchemaMorphism,
        )

        trans = EpisodicTransition(
            step=5,
            avatar_pos=(1, 2),
            action_id=0,
            reward=1.0,
            is_win=True,
        )
        t_pred = trans.to_causal_predicate()
        assert "episodic_transition:step_5" in t_pred.variable
        t_ops = trans.to_transaction_operations()
        assert len(t_ops) == 1

        cortex = HippocampalEpisodicCortex()
        cortex.current_episode.append(trans)
        c_pred = cortex.to_causal_predicate()
        assert c_pred.variable == "hippocampal_episodic_cortex"
        c_preds = cortex.export_causal_predicates()
        assert len(c_preds) == 2
        c_ops = cortex.to_transaction_operations()
        assert len(c_ops) == 2

        morphism = SchemaMorphism(
            source_domain="grid_a",
            target_domain="grid_b",
            predicate_mapping={"Key": "Switch"},
        )
        m_pred = morphism.to_causal_predicate()
        assert "schema_morphism:grid_a->grid_b" in m_pred.variable
        m_ops = morphism.to_transaction_operations()
        assert len(m_ops) == 1

        item = ConsolidatedKnowledgeItem(
            concept_id="invariant_wall",
            representation={"solid": True},
        )
        item_pred = item.to_causal_predicate()
        assert item_pred.variable == "consolidated_item:invariant_wall"
        item_ops = item.to_transaction_operations()
        assert len(item_ops) == 1

        engine = DualStoreConsolidationEngine()
        engine.consolidated_store["invariant_wall"] = item
        eng_pred = engine.to_causal_predicate()
        assert eng_pred.variable == "dual_store_consolidation_engine"
        eng_preds = engine.export_causal_predicates()
        assert len(eng_preds) == 2
        eng_ops = engine.to_transaction_operations()
        assert len(eng_ops) == 2

    def test_grid_operator_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.grid_operator import (
            AffineOperator,
            DecisionAction,
            OperationalEpistemicDecision,
            OperatorBinding,
            TransformationProgramSearch,
        )

        binding = OperatorBinding(
            operator_name="affine",
            params={"rotate": 90},
            complexity=1.5,
        )
        b_pred = binding.to_causal_predicate()
        assert b_pred.variable == "operator_binding:affine"
        b_ops = binding.to_transaction_operations()
        assert len(b_ops) == 1

        decision = OperationalEpistemicDecision(
            action=DecisionAction.PREDICT,
            selected_prediction=None,
            probing_coordinate=None,
            expected_information_gain=0.0,
            abstention_reason=None,
            brier_score=0.1,
        )
        d_pred = decision.to_causal_predicate()
        assert d_pred.variable == "operational_epistemic_decision"
        d_ops = decision.to_transaction_operations()
        assert len(d_ops) == 1

        op = AffineOperator()
        op_pred = op.to_causal_predicate()
        assert op_pred.variable == "grid_operator:affine"
        op_ops = op.to_transaction_operations()
        assert len(op_ops) == 1

        search = TransformationProgramSearch(max_depth=1)
        s_pred = search.to_causal_predicate()
        assert s_pred.variable == "transformation_program_search"
        s_preds = search.export_causal_predicates()
        assert len(s_preds) >= 2
        s_ops = search.to_transaction_operations()
        assert len(s_ops) == 1

    def test_autonomous_epistemic_engine_causal_and_transactions(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import (
            AutonomousEpistemicEngine,
            EpistemicObservation,
            EpistemicTruthValue,
        )

        obs = EpistemicObservation(
            variable="door_status",
            value="OPEN",
            uncertainty=0.05,
            truth_value=EpistemicTruthValue.TRUE,
        )
        obs_pred = obs.to_causal_predicate()
        assert obs_pred.variable == "door_status"
        obs_ops = obs.to_transaction_operations()
        assert len(obs_ops) == 1

        engine = AutonomousEpistemicEngine(enable_logging=False)
        engine.learned_goal_positions.add((7, 7))
        engine.learned_barriers.add((1, 1))
        engine_pred = engine.to_causal_predicate()
        assert engine_pred.variable == "autonomous_epistemic_engine"
        engine_preds = engine.export_causal_predicates()
        assert len(engine_preds) >= 1
        engine_ops = engine.to_transaction_operations()
        assert len(engine_ops) >= 3


class TestMemoryAndKnowledgeBaseSingleSourceOfTruth:
    """Validate that formal internal memory stores (PrefrontalWorkingMemory, VisuospatialWorkingMemory,
    HabenularEpisodicIOR, and HCIRSymbolicWorldTheory) serve as the unified Single Source of Truth (SSOT).
    """

    def test_symbolic_theory_single_source_of_truth(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

        engine = AutonomousEpistemicEngine(enable_logging=False)

        # Spelke core knowledge: background 0 is seeded as walkable space by default
        assert engine.symbolic_theory.is_walkable(0)
        assert 0 in engine.verified_safe_features

        # Verified safe features strictly maps to symbolic_theory.walkable_features
        engine.verified_safe_features.add(4)
        assert engine.symbolic_theory.is_walkable(4)
        assert 4 in engine.symbolic_theory.walkable_features

        engine.symbolic_theory.induce_walkable(7)
        assert 7 in engine.verified_safe_features

        # Goal features unification
        engine.learned_goal_features.add(2)
        assert engine.symbolic_theory.is_goal(2)
        assert 2 in engine.symbolic_theory.goal_features

        # Cargo features unification
        engine.learned_cargo_features.add(5)
        assert engine.symbolic_theory.is_cargo(5)
        assert 5 in engine.symbolic_theory.cargo_features

        # Barrier features unification
        engine.learned_barrier_features.add(8)
        assert engine.symbolic_theory.is_barrier(8)
        assert 8 in engine.symbolic_theory.barrier_features

    def test_prefrontal_working_memory_executive_ssot(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_planner import MentalSimulationStep

        engine = AutonomousEpistemicEngine(enable_logging=False)

        # Active plan queue resides in working_memory
        step = MentalSimulationStep(
            action=1,
            action_data={},
            predicted_avatar_pos=(2, 2),
        )
        engine.mental_plan.append(step)
        assert len(engine.working_memory.mental_plan) == 1
        assert engine.working_memory.mental_plan[0].action == 1

        engine.working_memory.mental_plan.clear()
        assert len(engine.mental_plan) == 0

        # Metacognitive cooldowns & failure counters
        engine.consecutive_simulation_failures = 3
        engine.simulation_cooldown = 5
        engine.consecutive_plan_failures = 2
        engine.exploration_cooldown = 4
        engine.consecutive_stuck_steps = 1

        assert engine.working_memory.consecutive_simulation_failures == 3
        assert engine.working_memory.simulation_cooldown == 5
        assert engine.working_memory.consecutive_plan_failures == 2
        assert engine.working_memory.exploration_cooldown == 4
        assert engine.working_memory.consecutive_stuck_steps == 1

        # Resetting working memory clears cooldowns
        engine.working_memory.reset_episode()
        assert engine.consecutive_simulation_failures == 0
        assert engine.simulation_cooldown == 0
        assert engine.consecutive_plan_failures == 0
        assert engine.exploration_cooldown == 0
        assert engine.consecutive_stuck_steps == 0

    def test_visuospatial_and_habenular_spatial_ssot(self) -> None:
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

        engine = AutonomousEpistemicEngine(enable_logging=False)

        # Level 0 barriers & failed transitions
        engine.learned_barriers.add((3, 3))
        assert (3, 3) in engine.working_memory.visuospatial.learned_barriers
        assert (3, 3) in engine.working_memory.visuospatial.level_learned_barriers[0]

        engine.failed_transitions.add(((1, 1), 2))
        assert ((1, 1), 2) in engine.habenular_ior.failed_transitions
        assert ((1, 1), 2) in engine.habenular_ior.level_failed_transitions[0]

        # Switch to Level 1: working memory buffers automatically switch active partition
        engine.current_level = 1
        assert len(engine.learned_barriers) == 0
        assert len(engine.failed_transitions) == 0

        engine.learned_barriers.add((5, 5))
        engine.failed_transitions.add(((4, 4), 3))

        assert (5, 5) in engine.working_memory.visuospatial.level_learned_barriers[1]
        assert ((4, 4), 3) in engine.habenular_ior.level_failed_transitions[1]

        # Switch back to Level 0: Level 0 memory is intact
        engine.current_level = 0
        assert (3, 3) in engine.learned_barriers
        assert ((1, 1), 2) in engine.failed_transitions

    def test_transaction_consolidation_across_all_memory_stores(self) -> None:
        from hbllm.hcir.transactions import TransactionOp
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

        engine = AutonomousEpistemicEngine(enable_logging=False)

        engine.symbolic_theory.induce_goal(3)
        engine.symbolic_theory.induce_barrier(9)
        engine.working_memory.visuospatial.record_probe((4, 4), feature_id=2, step=1)
        engine.working_memory.unlocked_barriers.add((6, 6))

        ops = engine.to_transaction_operations()
        node_ids = {op.node_id for op in ops if op.node_id is not None}

        assert any("ontology_goal_3" in nid for nid in node_ids)
        assert any("ontology_barrier_9" in nid for nid in node_ids)
        assert any("wm_slot_4_4" in nid for nid in node_ids)
        assert any("unlocked_barrier_6_6" in nid for nid in node_ids)
        assert all(op.op == TransactionOp.UPSERT_NODE for op in ops)

    def test_spelke_physical_axioms_and_evaluators(self) -> None:
        from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory, SpelkePhysicalAxiom

        theory = HCIRSymbolicWorldTheory()
        theory.seed_core_knowledge(bg_feature=0)

        # 1. All 5 core Spelke physical principles are grounded
        assert SpelkePhysicalAxiom.FREE_SPACE in theory.spelke_axioms
        assert SpelkePhysicalAxiom.SOLIDITY_CONTINUITY in theory.spelke_axioms
        assert SpelkePhysicalAxiom.CONTACT_CAUSALITY in theory.spelke_axioms
        assert SpelkePhysicalAxiom.COHESION in theory.spelke_axioms
        assert SpelkePhysicalAxiom.DIRECTIONAL_INERTIA in theory.spelke_axioms
        assert 0 in theory.walkable_features

        # 2. Spelke spatiotemporal continuity evaluation
        # Unobstructed path
        assert (
            theory.evaluates_spelke_continuity(
                origin=(1, 1),
                destination=(1, 4),
                barrier_set={(2, 1), (2, 2)},
                grid_bounds=(10, 10),
            )
            is True
        )
        # Wall of barriers completely partitioning the space
        wall = {(r, 3) for r in range(10)}
        assert (
            theory.evaluates_spelke_continuity(
                origin=(1, 1),
                destination=(1, 5),
                barrier_set=wall,
                grid_bounds=(10, 10),
            )
            is False
        )

        # 3. Spelke mechanical contact causality evaluation
        # 4-way adjacent
        assert theory.evaluates_spelke_contact((2, 2), (2, 3), adjacency="4-way") is True
        assert theory.evaluates_spelke_contact((2, 2), (1, 2), adjacency="4-way") is True
        # Diagonal: false for 4-way, true for 8-way
        assert theory.evaluates_spelke_contact((2, 2), (3, 3), adjacency="4-way") is False
        assert theory.evaluates_spelke_contact((2, 2), (3, 3), adjacency="8-way") is True
        # Distant
        assert theory.evaluates_spelke_contact((2, 2), (5, 5), adjacency="8-way") is False

        # 4. Axioms serialize into canonical HCIR transactions
        ops = theory.to_transaction_operations()
        node_ids = {op.node_id for op in ops if op.node_id is not None}
        assert "spelke_axiom_free_space" in node_ids
        assert "spelke_axiom_solidity_continuity" in node_ids
        assert "spelke_axiom_contact_causality" in node_ids

    def test_procedural_skill_synthesis_and_retrieval(self) -> None:
        from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill
        from hbllm.hcir.world.prefrontal_working_memory import (
            PrefrontalWorkingMemory,
            SubgoalSchema,
            SubgoalStage,
        )

        wm = PrefrontalWorkingMemory()
        schema = SubgoalSchema(
            schema_name="KEY_DOOR_GOAL_SCHEMA",
            stages=[
                SubgoalStage(stage_id="ACQUIRE_KEY", target_role="KEY", target_positions={(2, 2)}),
                SubgoalStage(
                    stage_id="UNLOCK_BARRIER", target_role="BARRIER", target_positions={(4, 4)}
                ),
                SubgoalStage(stage_id="REACH_GOAL", target_role="GOAL", target_positions={(8, 8)}),
            ],
        )

        # Synthesize skill
        skill = wm.synthesize_skill_from_schema(schema, required_actions={1, 2, 3, 4})
        assert isinstance(skill, DeclarativeNeuroSymbolicSkill)
        assert skill.skill_name == "skill_key_door_goal_schema"
        assert skill.program is not None
        assert len(skill.program) == 3

        # Query procedural library
        retrieved = wm.query_procedural_skill("KEY_DOOR_GOAL_SCHEMA")
        assert retrieved is skill

        # Evaluate declarative signature
        dummy_grid = np.zeros((10, 10), dtype=int)
        assert skill.can_handle(dummy_grid, available_actions=[1, 2, 3, 4]) is True
        assert skill.can_handle(dummy_grid, available_actions=[1, 2]) is False

        # Verify transaction serialization
        ops = wm.to_transaction_operations()
        node_ids = {op.node_id for op in ops if op.node_id is not None}
        assert "procedural_skill_KEY_DOOR_GOAL_SCHEMA" in node_ids

        # Register custom skill without explicit description attribute
        class CustomSkillWithoutDesc(DeclarativeNeuroSymbolicSkill):
            skill_name = "custom_door_opener"

        custom_skill = CustomSkillWithoutDesc()
        assert custom_skill.description == ""
        wm.register_procedural_skill("CUSTOM_SKILL", custom_skill)

        # Ensure to_transaction_operations succeeds without AttributeError
        ops2 = wm.to_transaction_operations()
        node_map = {op.node_id: op.node_data for op in ops2 if op.node_id is not None}
        assert "procedural_skill_CUSTOM_SKILL" in node_map
        assert (
            node_map["procedural_skill_CUSTOM_SKILL"]["description"]
            == "Procedural skill for CUSTOM_SKILL"
        )

    def test_tiered_workspace_transaction_mirroring_and_promotion(self) -> None:
        from hbllm.hcir.workspace_tiers import TieredWorkspace
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.prefrontal_working_memory import SubgoalSchema, SubgoalStage

        engine = AutonomousEpistemicEngine(enable_logging=False)
        tiered = TieredWorkspace()

        # Attach workspace and create active task frame
        frame = engine.attach_tiered_workspace(tiered, goal_id="reach_goal_test")
        assert engine.transaction_manager is not None
        assert engine.active_task_frame is frame

        # Induce knowledge in symbolic theory and working memory
        engine.symbolic_theory.seed_core_knowledge(bg_feature=0)
        engine.symbolic_theory.induce_goal(7)
        engine.record_learned_goal((3, 3))
        engine.record_learned_barrier((5, 5))

        # Commit transactions to the active task frame
        tx = engine.commit_working_memory_transactions()
        assert tx is not None
        assert tx.is_committed

        # Verify state exists in WORKING task frame
        assert frame.workspace.get_node("ontology_goal_7") is not None
        assert frame.workspace.get_node("learned_goal_3_3") is not None
        assert frame.workspace.get_node("learned_barrier_5_5") is not None
        assert frame.workspace.get_node("spelke_axiom_solidity_continuity") is not None

        # Prepare active schema in working memory
        schema = SubgoalSchema(
            schema_name="WINNING_SUBGOAL_SCHEMA",
            stages=[
                SubgoalStage(stage_id="STEP_ONE", target_role="GOAL", target_positions={(3, 3)}),
            ],
        )
        engine.working_memory.active_schema = schema

        # Simulate level win -> triggers automatic promotion from WORKING to PERSISTENT
        curr_grid = np.zeros((10, 10), dtype=int)
        curr_grid[3, 3] = 7
        engine.avatar_pos = (3, 2)
        engine.prev_grid = curr_grid.copy()
        engine.assimilate_feedback(
            curr_grid=curr_grid,
            available_actions=[1, 2, 3, 4],
            is_win=True,
            action=1,
        )

        # Assert nodes have been promoted to WorkspaceTier.PERSISTENT
        persistent_goal = tiered.persistent.get_node("ontology_goal_7")
        assert persistent_goal is not None

        persistent_goal_pos = tiered.persistent.get_node("learned_goal_3_3")
        assert persistent_goal_pos is not None

        persistent_skill = tiered.persistent.get_node("procedural_skill_WINNING_SUBGOAL_SCHEMA")
        assert persistent_skill is not None

    def test_habitual_procedural_skill_execution_in_planner(self) -> None:
        """Test Option 1: Habitual procedural skill matching, schema instantiation, and execution in planner."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_planner import EpistemicGoalScheduler
        from hbllm.hcir.world.prefrontal_working_memory import SubgoalSchema, SubgoalStage

        engine = AutonomousEpistemicEngine(enable_logging=False)
        curr_grid = np.zeros((10, 10), dtype=int)
        curr_grid[5, 5] = 2  # Tool
        curr_grid[8, 8] = 3  # Goal
        engine.avatar_pos = (1, 1)

        # Synthesize a procedural skill in working memory
        schema = SubgoalSchema(
            schema_name="ACQUIRE_THEN_REACH",
            stages=[
                SubgoalStage(
                    stage_id="ACQUIRE_TOOL", target_role="tool", target_positions={(5, 5)}
                ),
                SubgoalStage(stage_id="REACH_GOAL", target_role="goal", target_positions={(8, 8)}),
            ],
        )
        engine.working_memory.synthesize_skill_from_schema(schema, required_actions=[1, 2, 3, 4])
        engine.working_memory.active_schema = (
            None  # Reset active schema to test automatic habitual recall
        )

        # Verify skill is stored
        assert "ACQUIRE_THEN_REACH" in engine.working_memory.procedural_library

        # Run goal scheduling with available actions
        proposal = EpistemicGoalScheduler.schedule_goals(
            engine=engine,
            curr_grid=curr_grid,
            entities=[],
            bg=0,
            av_feats=set(),
            target_role=EntityRole.GOAL,
            available_actions=[1, 2, 3, 4],
        )

        # Verify active schema was habitually instantiated and first stage target (5, 5) prioritized
        assert engine.working_memory.active_schema is not None
        assert engine.working_memory.active_schema.schema_name == "ACQUIRE_THEN_REACH"
        assert (5, 5) in proposal.goals
        assert proposal.goals[0] == (5, 5)

        # Advance avatar to (5, 5) -> current stage satisfied
        engine.avatar_pos = (5, 5)
        proposal2 = EpistemicGoalScheduler.schedule_goals(
            engine=engine,
            curr_grid=curr_grid,
            entities=[],
            bg=0,
            av_feats=set(),
            target_role=EntityRole.GOAL,
            available_actions=[1, 2, 3, 4],
        )

        # Verify schema advanced to stage 2 and prioritized (8, 8)
        assert engine.working_memory.active_schema.current_stage_idx == 1
        assert (8, 8) in proposal2.goals
        assert proposal2.goals[0] == (8, 8)

    def test_persistent_workspace_lifelong_memory_hydration(self) -> None:
        """Test Option 2: Lifelong memory hydration from TieredWorkspace persistent tier."""
        from hbllm.hcir.graph import ConceptNode, ConstraintNode, GoalNode, SkillNode
        from hbllm.hcir.workspace_tiers import TieredWorkspace
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_causal import SpelkePhysicalAxiom

        tiered = TieredWorkspace()

        # Seed persistent tier directly with promoted nodes
        tiered.persistent.upsert_node(
            SkillNode(
                id="procedural_skill_PERSISTENT_UNLOCK",
                name="PERSISTENT_UNLOCK",
                description="Long-term key-door solver",
                properties={"schema_name": "PERSISTENT_UNLOCK"},
            ),
            author="test",
        )
        tiered.persistent.upsert_node(
            ConceptNode(
                id="ontology_goal_8",
                label="goal_feature_8",
                definition="Known winning entity feature",
                properties={"feature_id": 8},
            ),
            author="test",
        )
        tiered.persistent.upsert_node(
            GoalNode(
                id="learned_goal_4_4",
                description="Persistent goal coordinate",
                properties={"position": [4, 4]},
            ),
            author="test",
        )
        tiered.persistent.upsert_node(
            ConstraintNode(
                id="learned_barrier_6_6",
                name="barrier_6_6",
                description="Persistent static barrier",
                properties={"position": [6, 6]},
            ),
            author="test",
        )
        tiered.persistent.upsert_node(
            ConceptNode(
                id="spelke_axiom_solidity_continuity",
                label="solidity_continuity",
                definition="Spelke solidity axiom",
                properties={"axiom_name": "solidity_continuity"},
            ),
            author="test",
        )

        # Initialize a new epistemic engine (tabula rasa)
        engine = AutonomousEpistemicEngine(enable_logging=False)
        assert 8 not in engine.symbolic_theory.goal_features
        assert (4, 4) not in engine.learned_goal_positions
        assert (6, 6) not in engine.learned_barriers
        assert "PERSISTENT_UNLOCK" not in engine.working_memory.procedural_library

        # Attach workspace with automatic persistent hydration
        engine.attach_tiered_workspace(
            tiered, goal_id="lifelong_learning_task", hydrate_persistent=True
        )

        # Assert all persistent nodes were hydrated into engine stores
        assert 8 in engine.symbolic_theory.goal_features
        assert (4, 4) in engine.learned_goal_positions
        assert (6, 6) in engine.learned_barriers
        assert "PERSISTENT_UNLOCK" in engine.working_memory.procedural_library
        assert SpelkePhysicalAxiom.SOLIDITY_CONTINUITY in engine.symbolic_theory.spelke_axioms

    def test_spelke_topological_pruning_in_counterfactual_rollout(self) -> None:
        """Test Option 3: Spelke continuity and contact causality in counterfactual rollout expansion."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

        engine = AutonomousEpistemicEngine(enable_logging=False)
        engine.symbolic_theory.seed_core_knowledge(bg_feature=0)
        engine.avatar_pos = (2, 2)
        engine.avatar_features = {1}
        engine.action_dynamics = {
            1: ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, probes_tested=5),
            2: ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, probes_tested=5),
            3: ActionDynamicsModel(action_id=3, delta_r=0, delta_c=-1, probes_tested=5),
            4: ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, probes_tested=5),
        }

        # Grid with barrier separating avatar at (2, 2) from goal at (2, 5)
        grid = np.zeros((6, 6), dtype=int)
        grid[2, 2] = 1  # Avatar
        grid[2, 5] = 7  # Goal
        grid[:, 3] = 9  # Solid wall barrier

        engine.symbolic_theory.induce_barrier(9)
        engine.symbolic_theory.induce_goal(7)
        engine.record_learned_goal((2, 5))

        # Evaluate Spelke continuity: trajectory crossing col 3 is invalid
        wall_barrier_set = {(r, 3) for r in range(6)}
        assert not engine.symbolic_theory.evaluates_spelke_continuity(
            (2, 2), (2, 4), barrier_set=wall_barrier_set, grid_bounds=(6, 6)
        )
        assert engine.symbolic_theory.evaluates_spelke_continuity(
            (2, 2), (2, 1), barrier_set=wall_barrier_set, grid_bounds=(6, 6)
        )

        # Evaluate Spelke contact: avatar at (2, 2) cannot push cargo at (2, 4)
        assert not engine.symbolic_theory.evaluates_spelke_contact((2, 2), (2, 4), adjacency=True)
        assert engine.symbolic_theory.evaluates_spelke_contact((2, 2), (2, 3), adjacency=True)

    def test_metacognitive_conflict_schema_invalidation(self) -> None:
        """Test Option 4: Metacognitive conflict-driven schema revision and phase transition."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
        from hbllm.hcir.world.prefrontal_working_memory import SubgoalSchema, SubgoalStage

        engine = AutonomousEpistemicEngine(enable_logging=False)
        engine._phase = EpistemicPhase.EXPLOITATION
        schema = SubgoalSchema(
            schema_name="FRAGILE_SCHEMA",
            stages=[
                SubgoalStage(stage_id="RISKY_STEP", target_role="goal", target_positions={(4, 4)})
            ],
        )
        engine.working_memory.active_schema = schema

        curr_grid = np.zeros((8, 8), dtype=int)
        engine.avatar_pos = (4, 3)
        engine.prev_grid = curr_grid.copy()

        # First failure: confidence drops from 1.0 to 0.5, failure_count = 1
        engine.assimilate_feedback(
            curr_grid=curr_grid,
            available_actions=[1, 2, 3, 4],
            is_lost=True,
            action=1,
        )
        assert engine.working_memory.active_schema is not None
        assert engine.working_memory.active_schema.failure_count == 1
        assert engine.working_memory.active_schema.confidence == 0.5

        # Second failure in same localized area: triggers recurring conflict and invalidation
        engine.assimilate_feedback(
            curr_grid=curr_grid,
            available_actions=[1, 2, 3, 4],
            is_lost=True,
            action=1,
        )
        # Active schema should now be completely invalidated
        assert engine.working_memory.active_schema is None
        # Engine shifts from EXPLOITING to EPISTEMIC_PROBE for new discovery
        assert engine._phase == EpistemicPhase.EPISTEMIC_PROBE

    def test_spelke_cohesion_and_inertia_evaluators(self) -> None:
        """Test Spelke cohesion (rigid body multi-cell motion) and directional inertia."""
        from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

        theory = HCIRSymbolicWorldTheory()

        # Multi-cell rigid body (2x2 square)
        block_cells = {(2, 2), (2, 3), (3, 2), (3, 3)}
        barriers = {(4, 2)}

        # Displace by (1, 0) -> touches barrier at (4, 2) -> False
        assert not theory.evaluates_spelke_cohesion(
            entity_cells=block_cells,
            displacement=(1, 0),
            barrier_set=barriers,
            grid_bounds=(10, 10),
        )

        # Displace by (0, 1) -> moves safely to {(2, 3), (2, 4), (3, 3), (3, 4)} -> True
        assert theory.evaluates_spelke_cohesion(
            entity_cells=block_cells,
            displacement=(0, 1),
            barrier_set=barriers,
            grid_bounds=(10, 10),
        )

        # Out-of-bounds displacement: (8, 0) exceeds H=10
        assert not theory.evaluates_spelke_cohesion(
            entity_cells=block_cells,
            displacement=(8, 0),
            barrier_set=set(),
            grid_bounds=(10, 10),
        )

        # Directional inertia: position (1, 1), velocity (0, 1), 3 steps
        traj = theory.evaluates_spelke_inertia(
            position=(1, 1),
            velocity=(0, 1),
            steps=3,
            obstacles=set(),
            grid_bounds=(10, 10),
        )
        assert traj == [(1, 2), (1, 3), (1, 4)]

        # Inertia with inelastic obstacle collision
        traj_blocked = theory.evaluates_spelke_inertia(
            position=(1, 1),
            velocity=(0, 1),
            steps=5,
            obstacles={(1, 3)},
            grid_bounds=(10, 10),
        )
        assert traj_blocked == [(1, 2)]

        # Inertia with elastic boundary bounce
        traj_bounce = theory.evaluates_spelke_inertia(
            position=(1, 8),
            velocity=(0, 1),
            steps=2,
            obstacles=set(),
            bounce=True,
            grid_bounds=(10, 10),
        )
        assert len(traj_bounce) == 2
        assert traj_bounce[0] == (1, 9)
        assert traj_bounce[1] == (1, 8)

    def test_composite_declarative_macro_skill_composition(self) -> None:
        """Test hierarchical macro-skill chaining and composition in procedural memory."""
        from hbllm.hcir.world.prefrontal_working_memory import (
            PrefrontalWorkingMemory,
            SubgoalSchema,
            SubgoalStage,
        )

        wm = PrefrontalWorkingMemory()

        # Skill 1: Acquire Tool
        schema1 = SubgoalSchema(
            schema_name="ACQUIRE_KEY",
            stages=[
                SubgoalStage(stage_id="GRAB_KEY", target_role="tool", target_positions={(2, 2)})
            ],
        )
        wm.synthesize_skill_from_schema(schema1, required_actions=[1, 2, 3, 4])

        # Skill 2: Unlock Door
        schema2 = SubgoalSchema(
            schema_name="UNLOCK_DOOR",
            stages=[
                SubgoalStage(
                    stage_id="OPEN_BARRIER", target_role="barrier", target_positions={(4, 4)}
                )
            ],
        )
        wm.synthesize_skill_from_schema(schema2, required_actions=[1, 2, 3, 4])

        # Compose macro-skill
        macro = wm.compose_macro_skill(
            skill_names=["ACQUIRE_KEY", "UNLOCK_DOOR"],
            macro_name="KEY_THEN_DOOR_MACRO",
            description="Sequenced tool-use and door opening macro",
        )
        assert macro is not None
        assert "KEY_THEN_DOOR_MACRO" in wm.procedural_library
        assert macro.program is not None
        assert len(macro.program.subgoals) == 2

        # Instantiate as active schema
        schema = wm.instantiate_skill_as_schema("KEY_THEN_DOOR_MACRO")
        assert schema is not None
        assert schema.schema_name == "KEY_THEN_DOOR_MACRO"
        assert len(schema.stages) == 2
        assert schema.stages[0].stage_id == "GRAB_KEY"
        assert schema.stages[1].stage_id == "OPEN_BARRIER"

    def test_pearl_epistemic_intervention_information_gain(self) -> None:
        """Test Pearl do(a) causal intervention planning for ambiguous entity disambiguation."""
        from hbllm.hcir.spatial_planner import SpatialEntity
        from hbllm.hcir.world.cortex_causal import CausalInductionCortex

        cortex = CausalInductionCortex()
        avatar_pos = (1, 1)

        # Ambiguous entity 1: near avatar at (2, 2)
        e1 = SpatialEntity(
            id="unknown_switch",
            feature_id=5,
            grid_pos=(2, 2),
            area=1,
            centroid=(2.0, 2.0),
            bounding_box=(2, 2, 2, 2),
        )
        # Ambiguous entity 2: far from avatar at (8, 8)
        e2 = SpatialEntity(
            id="unknown_box",
            feature_id=6,
            grid_pos=(8, 8),
            area=1,
            centroid=(8.0, 8.0),
            bounding_box=(8, 8, 8, 8),
        )

        proposal = cortex.plan_epistemic_intervention(
            avatar_pos=avatar_pos,
            candidate_entities=[e1, e2],
            available_actions=[1, 2, 3, 4],
            bg_feature=0,
        )

        assert proposal is not None
        assert proposal.target_coord == (2, 2)
        assert proposal.target_feature == 5
        assert proposal.expected_information_gain > 0.0

        # Verify transaction proposal serialization
        ops = proposal.to_transaction_operations()
        assert len(ops) == 1
        assert ops[0].node_id == "active_epistemic_intervention"
        assert ops[0].node_data["target_feature"] == 5

    def test_parietal_egocentric_polar_and_entity_lifting(self) -> None:
        """Test Posterior Parietal Cortex viewpoint-invariant polar transformations and entity frame lifting."""
        from hbllm.hcir.spatial_planner import SpatialEntity
        from hbllm.hcir.world.parietal_coordinate_transform import (
            HeadingOrientation,
            ParietalCoordinateTransformer,
        )

        avatar_pos = (5, 5)

        # 1. Target directly north (ahead when facing NORTH)
        polar_north = ParietalCoordinateTransformer.allo_to_ego_polar(
            allo_coord=(3, 5),
            avatar_pos=avatar_pos,
            heading=HeadingOrientation.NORTH,
        )
        assert polar_north.forward_depth == 2.0
        assert polar_north.lateral_offset == 0.0
        assert polar_north.relative_bearing == "FORWARD"

        # Invert back to allocentric
        allo_north = ParietalCoordinateTransformer.ego_polar_to_allo(
            polar=polar_north,
            avatar_pos=avatar_pos,
            heading=HeadingOrientation.NORTH,
        )
        assert allo_north == (3, 5)

        # 2. Target east (to the right when facing NORTH)
        polar_east = ParietalCoordinateTransformer.allo_to_ego_polar(
            allo_coord=(5, 8),
            avatar_pos=avatar_pos,
            heading=HeadingOrientation.NORTH,
        )
        assert polar_east.relative_bearing == "RIGHT"
        assert polar_east.lateral_offset == 3.0
        assert polar_east.forward_depth == 0.0

        # Rotate heading: when facing EAST, target at (5, 8) is now FORWARD
        polar_facing_east = ParietalCoordinateTransformer.allo_to_ego_polar(
            allo_coord=(5, 8),
            avatar_pos=avatar_pos,
            heading=HeadingOrientation.EAST,
        )
        assert polar_facing_east.relative_bearing == "FORWARD"
        assert polar_facing_east.forward_depth == 3.0

        # 3. Entity Frame Lifting
        e1 = SpatialEntity(
            id="key_1",
            feature_id=3,
            grid_pos=(3, 5),
            area=1,
            centroid=(3.0, 5.0),
            bounding_box=(3, 5, 3, 5),
        )
        lifted = ParietalCoordinateTransformer.lift_entities_to_ego_frame(
            entities=[e1],
            avatar_pos=avatar_pos,
            heading=HeadingOrientation.NORTH,
        )
        assert len(lifted) == 1
        assert lifted[0].entity_id == "key_1"
        assert lifted[0].polar_coord.relative_bearing == "FORWARD"
        pred = lifted[0].to_causal_predicate()
        assert pred.variable == "ego_entity:key_1"

    def test_hippocampal_offline_td_experience_replay(self) -> None:
        """Test Hippocampal offline Temporal Difference (TD) experience replay and Q-value consolidation."""
        from hbllm.hcir.world.cortex_episodic import HippocampalEpisodicCortex

        cortex = HippocampalEpisodicCortex()

        # Record a 4-step trajectory leading to success
        cortex.record_transition(step=0, avatar_pos=(1, 1), action=1, reward=0.0)
        cortex.record_transition(step=1, avatar_pos=(1, 2), action=2, reward=0.0)
        cortex.record_transition(step=2, avatar_pos=(1, 3), action=3, reward=0.0)
        cortex.record_transition(
            step=3, avatar_pos=(1, 4), action=4, reward=1.0, is_win=True, is_terminal=True
        )

        summary = cortex.consolidate_episodic_replay_td(
            learning_rate=0.5,
            discount=0.9,
            prioritized=True,
            max_batch_size=10,
        )

        assert summary.replayed_steps == 4
        assert summary.max_td_error > 0.0
        assert len(summary.updated_q_values) > 0

        # Verify transaction serialization
        ops = summary.to_transaction_operations()
        assert len(ops) == len(summary.updated_q_values)
        assert ops[0].node_id.startswith("td_q_")

    def test_frontopolar_subgoal_mcts_hierarchical_planning(self) -> None:
        """Test BA10 Frontopolar Subgoal MCTS with Spelke continuity and Habenular lethal pruning."""
        from hbllm.hcir.world.cortex_planner import (
            FrontopolarSubgoalMCTSPlanner,
        )
        from hbllm.hcir.world.frontopolar_subgoal_stack import (
            FrontopolarSubgoal,
            SubgoalType,
        )

        avatar_pos = (1, 1)

        # Candidate subgoals
        sg_key = FrontopolarSubgoal(
            subgoal_id="grab_key",
            subgoal_type=SubgoalType.ACQUIRE_TOOL,
            target_entity_pos=(2, 2),
            target_destination=(2, 2),
            required_feature=3,
        )
        sg_door = FrontopolarSubgoal(
            subgoal_id="open_door",
            subgoal_type=SubgoalType.CLEAR_OBSTACLE,
            target_entity_pos=(4, 4),
            target_destination=(4, 4),
        )
        # Lethal trap subgoal: should be pruned by Habenular check
        sg_trap = FrontopolarSubgoal(
            subgoal_id="walk_into_lava",
            subgoal_type=SubgoalType.NAVIGATE_TO,
            target_entity_pos=(7, 7),
            target_destination=(7, 7),
        )

        result = FrontopolarSubgoalMCTSPlanner.plan_subgoal_mcts(
            avatar_pos=avatar_pos,
            candidate_subgoals=[sg_key, sg_door, sg_trap],
            goal_positions={(8, 8)},
            barriers={(4, 3), (4, 5)},
            grid_bounds=(10, 10),
            lethal_positions={(7, 7)},
            max_iterations=32,
        )

        assert result.total_iterations == 32
        assert result.pruned_dead_ends >= 1
        # Lethal trap should never be selected in optimal plan
        for sg in result.best_subgoals:
            assert sg.subgoal_id != "walk_into_lava"

        # Verify causal predicate & transaction generation
        pred = result.to_causal_predicate()
        assert pred.variable == "frontopolar_subgoal_mcts"
        ops = result.to_transaction_operations()
        assert len(ops) == len(result.best_subgoals)

    def test_parietal_motor_egocentric_dispatch(self) -> None:
        """Test Parietal-Motor egocentric intention dispatch and heading coordination."""
        from hbllm.hcir.world.cortex_motor import (
            EgocentricMotorIntent,
            ParietalMotorDispatcher,
        )
        from hbllm.hcir.world.parietal_coordinate_transform import HeadingOrientation

        # Standard 4-way cardinal actions: 1=UP(-1, 0), 2=DOWN(1, 0), 3=LEFT(0, -1), 4=RIGHT(0, 1)
        action_deltas = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
        }

        # Facing NORTH
        res_fwd = ParietalMotorDispatcher.dispatch_egocentric_intent(
            intent=EgocentricMotorIntent.MOVE_FORWARD,
            current_heading=HeadingOrientation.NORTH,
            action_deltas=action_deltas,
        )
        assert res_fwd.chosen_action_id == 1
        assert res_fwd.target_displacement == (-1, 0)
        assert res_fwd.resulting_heading == HeadingOrientation.NORTH

        res_right = ParietalMotorDispatcher.dispatch_egocentric_intent(
            intent=EgocentricMotorIntent.MOVE_RIGHT,
            current_heading=HeadingOrientation.NORTH,
            action_deltas=action_deltas,
        )
        assert res_right.chosen_action_id == 4
        assert res_right.target_displacement == (0, 1)

        # Rotations
        res_turn_r = ParietalMotorDispatcher.dispatch_egocentric_intent(
            intent=EgocentricMotorIntent.TURN_RIGHT,
            current_heading=HeadingOrientation.NORTH,
            action_deltas=action_deltas,
        )
        assert res_turn_r.chosen_action_id is None
        assert res_turn_r.resulting_heading == HeadingOrientation.EAST

        # Facing EAST: MOVE_FORWARD should now map to action 4 (0, 1)
        res_fwd_east = ParietalMotorDispatcher.dispatch_egocentric_intent(
            intent=EgocentricMotorIntent.MOVE_FORWARD,
            current_heading=res_turn_r.resulting_heading,
            action_deltas=action_deltas,
        )
        assert res_fwd_east.chosen_action_id == 4
        assert res_fwd_east.target_displacement == (0, 1)

        # Predicate & Transaction operations
        pred = res_fwd.to_causal_predicate()
        assert pred.variable == "ego_dispatch:MOVE_FORWARD"
        ops = res_fwd.to_transaction_operations()
        assert len(ops) == 1
        assert ops[0].node_id.startswith("ego_dispatch_MOVE_FORWARD_")

    def test_hippocampal_td_replay_consolidation_on_reset(self) -> None:
        """Test automatic Hippocampal TD experience replay into SymbolicWorldTheory upon reset."""
        from unittest.mock import MagicMock

        from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
        from hbllm.hcir.world.cortex_episodic import HippocampalEpisodicCortex

        cortex = HippocampalEpisodicCortex()
        theory = HCIRSymbolicWorldTheory()

        mock_engine = MagicMock()
        mock_engine.symbolic_theory = theory
        mock_engine.working_memory = MagicMock()

        # Record a 3-step episode
        cortex.record_transition(step=0, avatar_pos=(1, 1), action=1, reward=0.0)
        cortex.record_transition(step=1, avatar_pos=(1, 2), action=4, reward=0.0)
        cortex.record_transition(
            step=2, avatar_pos=(1, 3), action=4, reward=1.0, is_win=True, is_terminal=True
        )

        assert len(theory.domain_predicates) == 0

        # Consolidate and reset on level win
        cortex.consolidate_schema_and_reset(
            engine=mock_engine,
            retain_dynamics=True,
            is_new_level=True,
            level=2,
        )

        # Verify Q-values consolidated and predicates transferred to symbolic theory
        assert len(cortex.q_table) > 0
        assert len(theory.domain_predicates) > 0
        assert any("q_value:" in p.variable for p in theory.domain_predicates)

    def test_prefrontal_working_memory_ssot_delegation(self) -> None:
        """Test that AutonomousEpistemicEngine delegates exploration state directly to PrefrontalWorkingMemory."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

        engine = AutonomousEpistemicEngine(exploration_budget=50, enable_logging=False)

        # Mutate through engine properties
        engine.active_probe_target = (4, 7)
        engine.active_probe_id = "entity_4_7"
        engine.probe_target_steps = 3
        engine.entity_visit_counts = {"e_door": 2}
        engine.inhibited_actions = {1: 5}

        # Verify underlying PrefrontalWorkingMemory SSOT updated
        assert engine.working_memory.active_probe_target == (4, 7)
        assert engine.working_memory.active_probe_id == "entity_4_7"
        assert engine.working_memory.probe_target_steps == 3
        assert engine.working_memory.entity_visit_counts == {"e_door": 2}
        assert engine.working_memory.inhibited_actions == {1: 5}

        # Verify reading via engine properties
        assert engine.active_probe_target == (4, 7)
        assert engine.entity_visit_counts["e_door"] == 2
        assert engine.inhibited_actions[1] == 5

        # Reset episode and verify transient fields cleared
        engine.working_memory.reset_episode(retain_long_term=False)
        assert engine.active_probe_target is None
        assert engine.probe_target_steps == 0
        assert len(engine.entity_visit_counts) == 0
        assert len(engine.inhibited_actions) == 0

    def test_anterior_cingulate_conflict_predicate_safety_gate(self) -> None:
        """Test ACC conflict gating: rejecting trajectories into recurrent death clusters."""
        from unittest.mock import MagicMock

        import numpy as np

        from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor
        from hbllm.hcir.world.cortex_deliberation import (
            AnteriorCingulateConflictPredicate,
        )
        from hbllm.hcir.world.cortex_planner import MentalSimulationStep

        acc = AnteriorCingulateConflictMonitor(cluster_radius=2.0)
        # Register two death events at (5, 5) to form a recurrent failure cluster
        acc.register_death_event(actor_pos=(5.0, 5.0), step=1, last_action=1)
        acc.register_death_event(actor_pos=(5.0, 5.0), step=2, last_action=1)
        assert acc.is_in_conflict_zone((5, 5))

        mock_engine = MagicMock()
        mock_engine.working_memory.acc_conflict = acc
        mock_engine.avatar_pos = (4, 5)
        mock_engine.collision_cones = None

        cand_step = MentalSimulationStep(
            action=1,
            predicted_avatar_pos=(5, 5),
            expected_mutation="move_into_danger",
        )

        pred = AnteriorCingulateConflictPredicate()
        curr_grid = np.zeros((10, 10), dtype=int)
        res = pred.evaluate(
            engine=mock_engine,
            curr_grid=curr_grid,
            next_step=cand_step,
            future_steps=[cand_step],
            available_actions=[1, 2, 3, 4],
        )
        assert not res.is_safe
        assert "ACC recurrent failure cluster" in res.reason

        # Safe position away from death cluster
        safe_step = MentalSimulationStep(
            action=2,
            predicted_avatar_pos=(1, 1),
            expected_mutation="safe_move",
        )
        res_safe = pred.evaluate(
            engine=mock_engine,
            curr_grid=curr_grid,
            next_step=safe_step,
            future_steps=[safe_step],
            available_actions=[1, 2, 3, 4],
        )
        assert res_safe.is_safe

    def test_frontopolar_subgoal_mcts_with_body_schema(self) -> None:
        """Test Frontopolar Subgoal MCTS with ExtendedBodySchema and tool-barrier resonance."""
        import numpy as np

        from hbllm.hcir.world.cortex_planner import (
            FrontopolarSubgoal,
            FrontopolarSubgoalMCTSPlanner,
            SubgoalType,
        )
        from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema

        body_schema = ExtendedBodySchema()
        # Key feature 4 unlocks barrier feature 5
        body_schema.register_resonance(tool_feature=4, barrier_feature=5)

        # 5x5 grid with barrier of feature 5 separating key and goal
        # (2, 2) is a barrier cell of feature 5
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 2] = 5

        # Subgoal 1: Acquire key at (1, 1) (feature 4)
        sg_key = FrontopolarSubgoal(
            subgoal_id="acquire_key",
            subgoal_type=SubgoalType.ACQUIRE_TOOL,
            target_entity_pos=(1, 1),
            target_destination=(1, 1),
            required_feature=4,
            priority=80.0,
        )
        # Subgoal 2: Clear/navigate barrier at (2, 2)
        sg_barrier = FrontopolarSubgoal(
            subgoal_id="unlock_door",
            subgoal_type=SubgoalType.CLEAR_OBSTACLE,
            target_entity_pos=(2, 2),
            target_destination=(2, 2),
            required_feature=4,
            priority=60.0,
        )

        mcts_res = FrontopolarSubgoalMCTSPlanner.plan_subgoal_mcts(
            avatar_pos=(0, 0),
            candidate_subgoals=[sg_key, sg_barrier],
            goal_positions={(4, 4)},
            barriers={(2, 2)},
            grid_bounds=(5, 5),
            held_items=set(),
            body_schema=body_schema,
            tool_barrier_affinities=body_schema.tool_barrier_affinities,
            curr_grid=grid,
            max_iterations=40,
        )

        assert len(mcts_res.best_subgoals) > 0
        # Optimal sequence should prioritize acquiring the key first
        assert mcts_res.best_subgoals[0].subgoal_id == "acquire_key"

    def test_parietal_motor_dispatcher_tool_effector_targets(self) -> None:
        """Test ParietalMotorDispatcher.resolve_tool_effector_targets with body schema extension."""
        from hbllm.hcir.world.cortex_motor import ParietalMotorDispatcher
        from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
        from hbllm.hcir.world.parietal_coordinate_transform import HeadingOrientation

        schema = ExtendedBodySchema()
        schema.acquire_tool(feature_id=3, step=1, relative_offset=(0, 2))

        # Heading EAST: fwd is (0, 1)
        targets = ParietalMotorDispatcher.resolve_tool_effector_targets(
            avatar_pos=(3, 3),
            current_heading=HeadingOrientation.EAST,
            body_schema=schema,
        )

        # Must include avatar_pos (3, 3)
        assert (3, 3) in targets
        # Must include immediate forward cell (3, 4)
        assert (3, 4) in targets
        # Must include relative offset (3, 3 + 2) = (3, 5)
        assert (3, 5) in targets
