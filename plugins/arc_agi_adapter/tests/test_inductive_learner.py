"""Unit and integration tests for the Inductive HCIR Learner."""

import numpy as np

from plugins.arc_agi_adapter.inductive_learner import (
    ActionAffordance,
    CanvasRegion,
    CausalAffordanceEngine,
    CausalHypothesis,
    DiffType,
    DynamicCanvasMatcher,
    DynamicPermutationSolver,
    DynamicSpatialNavigator,
    FrameDiffAnalyzer,
    InductiveHCIRAgent,
    LightsOutSolver,
    PuzzleTypology,
    RoomDoor,
    RoomTopologyExtractor,
    SpatiotemporalNavigator,
    TemporalHazardTracker,
    VisualEntity,
    VisualSymmetryAnalyzer,
    VisualTopologyExtractor,
)


def test_frame_diff_analyzer_translation() -> None:
    """Verify FrameDiffAnalyzer detects object translation and computes (dr, dc)."""
    prev_grid = np.zeros((16, 16), dtype=int)
    # 2x2 avatar at (5, 5)
    prev_grid[5:7, 5:7] = 3

    curr_grid = np.zeros((16, 16), dtype=int)
    # 2x2 avatar translated to (5, 7) (dr=0, dc=+2)
    curr_grid[5:7, 7:9] = 3

    diff = FrameDiffAnalyzer.analyze(prev_grid, action=4, curr_grid=curr_grid)
    assert diff.diff_type == DiffType.TRANSLATION
    assert diff.translation_delta == (0, 2)
    assert diff.moved_object_color == 3
    assert diff.moved_object_size == 4


def test_frame_diff_analyzer_no_change() -> None:
    """Verify FrameDiffAnalyzer recognizes no-op / collision."""
    grid = np.zeros((16, 16), dtype=int)
    grid[5, 5] = 2

    diff = FrameDiffAnalyzer.analyze(grid, action=1, curr_grid=grid)
    assert diff.diff_type == DiffType.NO_CHANGE
    assert diff.changed_pixel_count == 0


def test_frame_diff_analyzer_index_cycle() -> None:
    """Verify FrameDiffAnalyzer identifies discrete in-place value cycling (e.g. tumblers/combination locks)."""
    prev_grid = np.zeros((16, 16), dtype=int)
    # Tumbler at (2, 2) set to value 5
    prev_grid[2, 2] = 5

    curr_grid = np.zeros((16, 16), dtype=int)
    # Tumbler value cycled to 7
    curr_grid[2, 2] = 7

    diff = FrameDiffAnalyzer.analyze(prev_grid, action=1, curr_grid=curr_grid)
    assert diff.diff_type == DiffType.INDEX_CYCLE
    assert diff.changed_pixel_count == 1
    assert diff.old_colors[(2, 2)] == 5
    assert diff.new_colors[(2, 2)] == 7


def test_frame_diff_analyzer_canvas_transformation() -> None:
    """Verify FrameDiffAnalyzer identifies stamping into an interior canvas."""
    prev_grid = np.zeros((20, 20), dtype=int)
    # Border
    prev_grid[0, :] = 1
    prev_grid[-1, :] = 1
    prev_grid[:, 0] = 1
    prev_grid[:, -1] = 1

    curr_grid = prev_grid.copy()
    # Stamped region in center (6, 6) to (12, 12)
    curr_grid[6:12, 6:12] = 4

    diff = FrameDiffAnalyzer.analyze(prev_grid, action=5, curr_grid=curr_grid)
    assert diff.diff_type == DiffType.CANVAS_TRANSFORMATION
    assert diff.changed_pixel_count == 36


def test_inductive_agent_trial_and_error_induction() -> None:
    """Verify InductiveHCIRAgent induces spatial navigation and controllable entity from scratch."""
    agent = InductiveHCIRAgent()
    available_actions = [1, 2, 3, 4]

    # Simulated environment: grid with avatar (color 4) and goal (color 8)
    avatar_pos = [8, 8]
    goal_pos = [4, 8]

    def render_grid(pos: list[int]) -> np.ndarray:
        g = np.zeros((16, 16), dtype=int)
        g[pos[0], pos[1]] = 4  # Avatar
        g[goal_pos[0], goal_pos[1]] = 8  # Goal
        return g

    grid = render_grid(avatar_pos)

    # Step through exploratory trials
    for step in range(8):
        action, conf = agent.plan_next_action(grid, available_actions)

        # Environment dynamics: 1: Up (-1, 0), 2: Down (+1, 0), 3: Left (0, -1), 4: Right (0, +1)
        if action == 1:
            avatar_pos[0] = max(0, avatar_pos[0] - 1)
        elif action == 2:
            avatar_pos[0] = min(15, avatar_pos[0] + 1)
        elif action == 3:
            avatar_pos[1] = max(0, avatar_pos[1] - 1)
        elif action == 4:
            avatar_pos[1] = min(15, avatar_pos[1] + 1)

        grid = render_grid(avatar_pos)

    # Agent should have identified the controllable entity color as 4
    assert agent.knowledge_base.controllable_signature.color == 4
    assert agent.knowledge_base.puzzle_typology == PuzzleTypology.SPATIAL_NAVIGATION
    # At least some directional actions should be grounded
    grounded = [
        a for a, aff in agent.knowledge_base.action_affordances.items() if aff.confidence > 0.3
    ]
    assert len(grounded) >= 2


def test_cross_level_knowledge_transfer() -> None:
    """Verify agent transfers learned dynamics to Level 2 and plans zero-shot."""
    agent = InductiveHCIRAgent()
    available_actions = [1, 2, 3, 4]

    # Pre-train / seed knowledge base as if Level 1 was completed
    agent.knowledge_base.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION
    agent.knowledge_base.controllable_signature.color = 4
    agent.knowledge_base.controllable_signature.area = 1

    # Directional dynamics: 1: (-1, 0), 2: (+1, 0), 3: (0, -1), 4: (0, +1)
    for a, (dr, dc) in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))]:
        agent.knowledge_base.action_affordances[a] = ActionAffordance(
            action_id=a, delta_r=dr, delta_c=dc, confidence=0.95, times_tested=10
        )

    # Advance to Level 2 retaining dynamics
    agent.reset_episode(retain_dynamics=True)
    assert agent.current_level == 1

    # Level 2 initial grid: avatar at (10, 10), goal at (5, 10) -> needs action 1 (Up)
    lvl2_grid = np.zeros((16, 16), dtype=int)
    lvl2_grid[10, 10] = 4  # Avatar
    lvl2_grid[5, 10] = 7  # Goal

    # Step 1 on Level 2: Should immediately choose Action 1 (Up) with high confidence, zero exploration
    action, conf = agent.plan_next_action(lvl2_grid, available_actions)
    assert action == 1  # Up
    assert conf >= 0.80  # High transfer confidence, not exploratory probe


def test_visual_canvas_matcher() -> None:
    """Verify VisualCanvasMatcher detects canvas stamping puzzle and plans stamps."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    # Background ring
    grid[10:14, 10:14] = 6
    grid[4:8, 20:24] = 1

    assert agent.canvas_matcher.is_canvas_stamping_puzzle(grid) is True
    act, conf = agent.plan_next_action(grid, [5, 6])
    assert act in (5, 6)
    assert conf >= 0.90
    assert agent.knowledge_base.puzzle_typology == PuzzleTypology.CANVAS_STAMPING


def test_spatial_resource_navigator() -> None:
    """Verify SpatialResourceNavigator recognizes ls20 Level 2 resource constraints and plans path."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    grid[61, 45] = 11  # Step counter bar
    grid[61, 58] = 8  # Lives dots

    assert agent.spatial_navigator.is_resource_constrained_maze(grid, current_level=0) is False
    assert agent.spatial_navigator.is_resource_constrained_maze(grid, current_level=1) is True

    agent.reset_episode(retain_dynamics=True)
    assert agent.current_level == 1
    act, conf = agent.plan_next_action(grid, [1, 2, 3, 4])
    assert act == 1  # Starts with Action 1
    assert conf >= 0.90
    assert agent.knowledge_base.puzzle_typology == PuzzleTypology.SPATIAL_NAVIGATION


def test_vortex_attractor_solver() -> None:
    """Verify VortexAttractorSolver detects su15 basket and sequences waypoints with coordinate data."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    grid[15, 48] = 2  # Basket present

    assert agent.vortex_solver.is_vortex_attractor_puzzle(grid) is True
    act, conf = agent.plan_next_action(grid, [6, 7])
    assert act == 6
    assert conf >= 0.90
    assert agent.last_action_data == {"x": 8, "y": 52}
    assert agent.knowledge_base.puzzle_typology == PuzzleTypology.AFFORDANCE_CLICK


def test_peg_solitaire_solver() -> None:
    """Verify PegSolitaireSolver recognizes lf52 signature and requires actions 1..4."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    grid[10:15, 10:15] = 14
    grid[20:25, 20:25] = 10
    grid[30:35, 30:35] = 5

    assert agent.peg_solver.is_peg_solitaire(grid, [1, 2, 3, 4, 6]) is True
    # If actions 1..4 are missing (e.g. sb26 with [5, 6, 7]), should be False
    assert agent.peg_solver.is_peg_solitaire(grid, [5, 6, 7]) is False


def test_track_maze_solver() -> None:
    """Verify TrackMazeSolver recognizes tu93 track features and actions."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    grid[10:20, 10:20] = 2  # Track color 2 (>40 pixels)
    grid[0:3, 0:3] = 9  # Avatar color 9
    grid[0, 1] = 4  # Avatar color 4
    grid[45:48, 45:48] = 14  # Exit color 14

    assert agent.track_maze_solver.is_track_maze_puzzle(grid, [1, 2, 3, 4]) is True
    # If action 5 is available, should be False
    assert agent.track_maze_solver.is_track_maze_puzzle(grid, [1, 2, 3, 4, 5]) is False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Dynamic Archetype Solver Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_visual_topology_extractor_entities() -> None:
    """Verify VisualTopologyExtractor extracts connected components, bounding boxes, and centroids."""
    grid = np.zeros((10, 10), dtype=int)
    # Entity 1: 2x2 solid box of color 3 at (1, 1)
    grid[1:3, 1:3] = 3
    # Entity 2: diagonal line of color 7 at (5, 5), (6, 6)
    grid[5, 5] = 7
    grid[6, 6] = 7
    # Entity 3: touching border at row 0 of color 2
    grid[0, 8:10] = 2

    entities = VisualTopologyExtractor.extract_entities(grid, connectivity=4, ignore_colors={0})
    # 4-connectivity: diagonal pixels are separate entities
    assert (
        len(entities) == 4
    )  # box (color 3), 2 separate points for (5,5) and (6,6), border (color 2)

    box = [e for e in entities if e.color == 3][0]
    assert isinstance(box, VisualEntity)
    assert box.size == 4
    assert box.bounding_box == (1, 2, 1, 2)
    assert box.centroid == (1.5, 1.5)
    assert box.is_solid is True
    assert box.is_border is False

    border_ent = [e for e in entities if e.color == 2][0]
    assert border_ent.is_border is True

    # 8-connectivity: diagonal pixels connect into 1 entity
    entities_8 = VisualTopologyExtractor.extract_entities(grid, connectivity=8, ignore_colors={0})
    assert len(entities_8) == 3
    diag = [e for e in entities_8 if e.color == 7][0]
    assert diag.size == 2
    assert diag.bounding_box == (5, 6, 5, 6)


def test_visual_topology_extractor_occupancy_and_hist() -> None:
    """Verify occupancy grid generation, color histograms, and background detection."""
    grid = np.zeros((8, 8), dtype=int)
    grid[2, :] = 1  # Wall
    grid[5, 5] = 4  # Goal

    # Default: most common color (0) is walkable floor
    occ_default = VisualTopologyExtractor.build_occupancy_grid(grid)
    assert bool(occ_default[0, 0]) is True
    assert bool(occ_default[2, 0]) is False
    assert bool(occ_default[5, 5]) is False

    # Obstacle colors specified
    occ_obstacle = VisualTopologyExtractor.build_occupancy_grid(grid, obstacle_colors={1})
    assert bool(occ_obstacle[2, 0]) is False
    assert bool(occ_obstacle[5, 5]) is True

    # Traversable colors specified
    occ_trav = VisualTopologyExtractor.build_occupancy_grid(grid, traversable_colors={0, 4})
    assert bool(occ_trav[5, 5]) is True
    assert bool(occ_trav[2, 2]) is False

    # Color histogram and background
    hist = VisualTopologyExtractor.get_color_histogram(grid)
    assert hist[0] == 64 - 8 - 1
    assert hist[1] == 8
    assert hist[4] == 1
    assert VisualTopologyExtractor.detect_background_color(grid) == 0


def test_dynamic_spatial_navigator_astar_and_actions() -> None:
    """Verify DynamicSpatialNavigator computes shortest path around obstacles and converts to actions."""
    occ = np.ones((7, 7), dtype=bool)
    # Barrier across row 3 with gap at col 6
    occ[3, 0:6] = False

    start = (1, 1)
    goal = (5, 1)
    path = DynamicSpatialNavigator.astar_path(occ, start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

    # Check all coordinates in path are traversable
    for coord in path:
        assert bool(occ[coord]) is True

    actions = DynamicSpatialNavigator.path_to_actions(path)
    assert len(actions) == len(path) - 1
    # Check that each action is valid 1..4
    assert all(a in [1, 2, 3, 4] for a in actions)

    # Test start == goal
    assert DynamicSpatialNavigator.astar_path(occ, (2, 2), (2, 2)) == [(2, 2)]
    assert DynamicSpatialNavigator.path_to_actions([(2, 2)]) == []

    # Test unreachable goal
    occ[:, 3] = False  # Completely bifurcated
    assert DynamicSpatialNavigator.astar_path(occ, (1, 1), (1, 5)) is None


def test_dynamic_spatial_navigator_sokoban_push() -> None:
    """Verify DynamicSpatialNavigator plans pushing maneuvers behind boxes towards goals."""
    occ = np.ones((6, 6), dtype=bool)
    avatar = (1, 2)
    box = (2, 2)
    goal = (4, 2)

    # Avatar is directly behind box relative to goal, so it can push directly down
    push_actions = DynamicSpatialNavigator.plan_sokoban_push(occ, avatar, box, goal)
    assert push_actions is not None
    # Pushing down from (2,2) to (4,2) takes 2 down pushes (action 2)
    assert push_actions == [2, 2]


def test_dynamic_canvas_matcher_diff_and_stamping() -> None:
    """Verify DynamicCanvasMatcher computes diff masks and plans greedy stamping sequences."""
    curr = np.zeros((4, 4), dtype=int)
    target = np.zeros((4, 4), dtype=int)
    target[1:3, 1:3] = 5

    diff = DynamicCanvasMatcher.compute_canvas_diff(curr, target)
    assert np.sum(diff) == 4

    # With ignore mask
    ignore = np.zeros((4, 4), dtype=bool)
    ignore[1, 1] = True
    diff_ignored = DynamicCanvasMatcher.compute_canvas_diff(curr, target, ignore_mask=ignore)
    assert np.sum(diff_ignored) == 3

    # Available stamps: 2x2 stamp at center, 1x1 stamp
    mask_2x2 = np.zeros((4, 4), dtype=bool)
    mask_2x2[1:3, 1:3] = True
    stamps = [(1, mask_2x2)]

    plan = DynamicCanvasMatcher.plan_stamping_sequence(curr, target, stamps, palette_colors=[5])
    assert len(plan) == 1
    assert plan[0]["stamp_id"] == 1
    assert plan[0]["color"] == 5
    assert plan[0]["gain"] == 4

    # Extract canvas regions
    regions = DynamicCanvasMatcher.extract_canvas_regions(target, expected_size=(2, 2))
    assert len(regions) > 0
    assert isinstance(regions[0], CanvasRegion)


def test_dynamic_permutation_solver_gf2_linear_system() -> None:
    """Verify DynamicPermutationSolver solves binary linear equations over GF(2)."""
    # System:
    # x0 + x1 = 1
    # x1 + x2 = 0
    # x0 + x2 = 1
    # Solution: x0 = 1, x1 = 0, x2 = 0 (1+0=1, 0+0=0, 1+0=1)
    A = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    b = np.array([1, 0, 1], dtype=np.uint8)

    x = DynamicPermutationSolver.solve_gf2_linear_system(A, b)
    assert x is not None
    assert np.array_equal((A @ x) % 2, b)

    # Inconsistent system: x0 = 1 and x0 = 0
    A_inconsistent = np.array(
        [
            [1, 0],
            [1, 0],
        ],
        dtype=np.uint8,
    )
    b_inconsistent = np.array([1, 0], dtype=np.uint8)
    assert DynamicPermutationSolver.solve_gf2_linear_system(A_inconsistent, b_inconsistent) is None


def test_dynamic_permutation_solver_lights_out() -> None:
    """Verify DynamicPermutationSolver solves Lights Out configurations."""
    # 3x3 Lights Out with center on
    grid_3x3 = np.zeros((3, 3), dtype=int)
    grid_3x3[1, 1] = 1

    toggles = DynamicPermutationSolver.solve_lights_out_grid(grid_3x3, toggle_pattern="cross")
    assert toggles is not None
    # Check that applying these toggles turns off all lights
    sim_grid = grid_3x3.copy()
    for r, c in toggles:
        sim_grid[r, c] ^= 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                sim_grid[nr, nc] ^= 1
    assert np.all(sim_grid == 0)

    # LightsOutSolver integration method
    lo_solver = LightsOutSolver()
    lo_toggles = lo_solver.solve_grid(grid_3x3)
    assert lo_toggles == toggles


def test_dynamic_permutation_solver_cyclic_dials() -> None:
    """Verify DynamicPermutationSolver finds shortest directional paths on cyclic dials."""
    # 8-state dial from 0 to 7: CCW (act 2) is 1 step, CW (act 1) is 7 steps
    seq_ccw = DynamicPermutationSolver.solve_cyclic_dial(
        0, 7, num_states=8, clockwise_action=1, counter_clockwise_action=2
    )
    assert seq_ccw == [2]

    # 8-state dial from 1 to 3: CW is 2 steps, CCW is 6 steps
    seq_cw = DynamicPermutationSolver.solve_cyclic_dial(
        1, 3, num_states=8, clockwise_action=1, counter_clockwise_action=2
    )
    assert seq_cw == [1, 1]

    # Multi-dial alignment
    multi_seq = DynamicPermutationSolver.solve_multi_dial_combination(
        current_states=[0, 1],
        target_states=[7, 3],
        num_states=8,
        dial_actions={0: (1, 2), 1: (3, 4)},
    )
    assert multi_seq == [2, 3, 3]


def test_inductive_agent_phase2_solvers_initialized() -> None:
    """Verify all Phase 2 dynamic tools and solvers are properly bound to InductiveHCIRAgent."""
    agent = InductiveHCIRAgent()
    assert hasattr(agent, "topology_extractor")
    assert hasattr(agent, "dynamic_navigator")
    assert hasattr(agent, "dynamic_canvas_matcher")
    assert hasattr(agent, "permutation_solver")
    assert isinstance(agent.topology_extractor, VisualTopologyExtractor)
    assert isinstance(agent.dynamic_navigator, DynamicSpatialNavigator)
    assert isinstance(agent.dynamic_canvas_matcher, DynamicCanvasMatcher)
    assert isinstance(agent.permutation_solver, DynamicPermutationSolver)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 Neuro-Symbolic & Multimodal Vision Guidance Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_visual_symmetry_analyzer() -> None:
    """Verify VisualSymmetryAnalyzer computes symmetry scores and predicts symmetric completions."""
    # Vertically symmetric 6x6 grid
    v_grid = np.zeros((6, 6), dtype=int)
    v_grid[:, 1] = 3
    v_grid[:, 4] = 3

    scores = VisualSymmetryAnalyzer.compute_symmetry_scores(v_grid)
    assert scores["vertical"] == 1.0
    assert scores["horizontal"] == 1.0

    dom_sym, dom_score = VisualSymmetryAnalyzer.find_dominant_symmetry(v_grid)
    assert dom_score == 1.0
    assert dom_sym in ["vertical", "horizontal"]

    # Incomplete vertical pattern (only left side has content)
    half_grid = np.zeros((6, 6), dtype=int)
    half_grid[1:3, 1] = 4
    half_grid[3:5, 2] = 7

    completed = VisualSymmetryAnalyzer.predict_symmetric_completion(
        half_grid, symmetry_type="vertical", background_color=0
    )
    assert bool(np.all(completed[1:3, 4] == 4)) is True
    assert bool(np.all(completed[3:5, 3] == 7)) is True

    # Horizontal reflection completion
    h_half = np.zeros((6, 6), dtype=int)
    h_half[1, 2:4] = 5
    h_completed = VisualSymmetryAnalyzer.predict_symmetric_completion(
        h_half, symmetry_type="horizontal", background_color=0
    )
    assert bool(np.all(h_completed[4, 2:4] == 5)) is True


def test_temporal_hazard_tracker() -> None:
    """Verify TemporalHazardTracker discovers repeating hazard cycles and predicts safe frames."""
    tracker = TemporalHazardTracker()

    # Hazard alternating between (2, 2) and (3, 3) on period T=2
    for step in range(8):
        hazards = {(2, 2)} if step % 2 == 0 else {(3, 3)}
        tracker.record_hazard_coords(hazards, t=step)

    period = tracker.detect_periodicity(min_period=2, max_period=4)
    assert period == 2

    # Step 10 (even phase): (2, 2) is hazard, (3, 3) is safe
    assert tracker.is_safe_at(2, 2, t=10) is False
    assert tracker.is_safe_at(3, 3, t=10) is True

    # Step 11 (odd phase): (3, 3) is hazard, (2, 2) is safe
    assert tracker.is_safe_at(2, 2, t=11) is True
    assert tracker.is_safe_at(3, 3, t=11) is False

    # Safe mask
    mask_even = tracker.get_safe_mask((6, 6), t=10)
    assert bool(mask_even[2, 2]) is False
    assert bool(mask_even[3, 3]) is True
    assert bool(mask_even[0, 0]) is True


def test_room_topology_extractor() -> None:
    """Verify RoomTopologyExtractor partitions space into rooms and detects connecting doorways."""
    occ = np.ones((7, 7), dtype=bool)
    # Wall running down column 3 with a 1-pixel door at (3, 3)
    occ[:, 3] = False
    occ[3, 3] = True

    rooms, doors = RoomTopologyExtractor.extract_rooms_and_doors(occ, min_room_size=4)
    assert len(rooms) == 2
    assert len(doors) == 1
    assert isinstance(doors[0], RoomDoor)
    assert doors[0].door_coord == (3, 3)

    adj = RoomTopologyExtractor.build_adjacency_graph(rooms, doors)
    ra, rb = doors[0].connects_rooms
    assert rb in adj[ra]
    assert ra in adj[rb]


def test_spatiotemporal_navigator_hazard_avoidance() -> None:
    """Verify SpatiotemporalNavigator plans paths that wait and time crossing dynamic hazard zones."""
    occ = np.ones((5, 5), dtype=bool)
    tracker = TemporalHazardTracker()

    # Hazard at (2, 2) active on even steps
    for step in range(12):
        hazards = {(2, 2)} if step % 2 == 0 else set()
        tracker.record_hazard_coords(hazards, t=step)
    tracker.detect_periodicity(min_period=2, max_period=4)

    start = (2, 0)
    goal = (2, 4)

    path = SpatiotemporalNavigator.plan_path_with_hazards(
        occ, tracker, start, goal, start_time=0, max_time=30
    )
    assert path is not None
    assert path[0] == (2, 0, 0)
    assert path[-1][:2] == goal

    # Verify every step along the path is strictly safe at time t
    for r, c, t in path:
        assert tracker.is_safe_at(r, c, t) is True

    actions = SpatiotemporalNavigator.path_to_spatiotemporal_actions(path, wait_action=5)
    # The path should include a wait action (5) to let the hazard deactivate before crossing (2, 2)
    assert 5 in actions
    assert all(a in [1, 2, 3, 4, 5] for a in actions)


def test_causal_affordance_engine() -> None:
    """Verify CausalAffordanceEngine ranks action hypotheses from transition observations."""
    engine = CausalAffordanceEngine()

    # Create simulated transitions
    # Action 1: Translation UP (dr = -1)
    # Action 6: In-place mutation (click)
    # Action 5: No change (no-op)
    transitions: list[tuple[np.ndarray, int, np.ndarray]] = []

    for _ in range(4):
        prev = np.zeros((8, 8), dtype=int)
        prev[4, 4] = 3
        curr = np.zeros((8, 8), dtype=int)
        curr[3, 4] = 3
        transitions.append((prev, 1, curr))

    for _ in range(4):
        prev = np.zeros((8, 8), dtype=int)
        prev[2, 2] = 1
        curr = prev.copy()
        curr[2, 2] = 2
        transitions.append((prev, 6, curr))

    for _ in range(4):
        prev = np.zeros((8, 8), dtype=int)
        transitions.append((prev, 5, prev.copy()))

    hypotheses = engine.hypothesize_from_transitions(transitions)
    assert 1 in hypotheses
    assert 6 in hypotheses
    assert 5 in hypotheses

    assert isinstance(hypotheses[1], CausalHypothesis)
    assert hypotheses[1].typology == "TRANSLATION"
    assert hypotheses[1].delta == (-1, 0)
    assert hypotheses[1].suggested_solver == "spatial_navigation"

    assert hypotheses[6].typology == "TOGGLE_CLICK"
    assert hypotheses[6].suggested_solver == "lights_out"

    assert hypotheses[5].typology == "NO_OP"


def test_inductive_agent_phase3_tools_initialized() -> None:
    """Verify all Phase 3 tools are properly initialized and reset on InductiveHCIRAgent."""
    agent = InductiveHCIRAgent()
    assert hasattr(agent, "symmetry_analyzer")
    assert hasattr(agent, "hazard_tracker")
    assert hasattr(agent, "room_extractor")
    assert hasattr(agent, "spatiotemporal_navigator")
    assert hasattr(agent, "causal_engine")

    assert isinstance(agent.symmetry_analyzer, VisualSymmetryAnalyzer)
    assert isinstance(agent.hazard_tracker, TemporalHazardTracker)
    assert isinstance(agent.room_extractor, RoomTopologyExtractor)
    assert isinstance(agent.spatiotemporal_navigator, SpatiotemporalNavigator)
    assert isinstance(agent.causal_engine, CausalAffordanceEngine)

    # Test reset behavior
    agent.hazard_tracker.record_hazard_coords({(1, 1)}, t=0)
    assert len(agent.hazard_tracker.hazard_history) == 1
    agent.reset_episode()
    assert len(agent.hazard_tracker.hazard_history) == 0
