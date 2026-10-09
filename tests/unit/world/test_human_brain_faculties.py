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

from hbllm.hcir.world.cerebellar_phase_clock import CerebellarPhaseClock
from hbllm.hcir.world.cortex_causal import CausalInductionCortex
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
from hbllm.hcir.world.cortex_motor import MotorCortexEffector
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
        assert "x" in coords and "y" in coords
        assert 0 <= coords["x"] < 10
        assert 0 <= coords["y"] < 10


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
