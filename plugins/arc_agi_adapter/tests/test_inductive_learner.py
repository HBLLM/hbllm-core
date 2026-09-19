"""Unit and integration tests for the Inductive HCIR Learner."""

import numpy as np

from plugins.arc_agi_adapter.inductive_learner import (
    ActionAffordance,
    DiffType,
    FrameDiffAnalyzer,
    InductiveHCIRAgent,
    PuzzleTypology,
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


def test_tumbler_permutation_solver() -> None:
    """Verify TumblerPermutationSolver recognizes tr87 signature and requires actions 1..4 without 6."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    grid[20:25, 20:25] = 7
    grid[30:35, 30:35] = 10

    assert agent.tumbler_solver.is_tumbler_lock(grid, [1, 2, 3, 4]) is True
    assert agent.tumbler_solver.is_tumbler_lock(grid, [1, 2, 3, 4, 6]) is False


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


def test_permutation_slider_solver() -> None:
    """Verify PermutationSliderSolver recognizes sb26 signature with actions 5..7."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)

    assert agent.slider_solver.is_permutation_slider(grid, [5, 6, 7]) is True
    assert agent.slider_solver.is_permutation_slider(grid, [1, 2, 3, 4, 5, 6]) is False


def test_track_maze_navigator() -> None:
    """Verify TrackMazeNavigator recognizes tu93 track features."""
    agent = InductiveHCIRAgent()
    grid = np.zeros((64, 64), dtype=int)
    grid[10:20, 10:20] = 6  # Track color 6
    grid[20:30, 20:30] = 14  # Track color 14
    grid[0, 0] = 0  # Border

    assert agent.track_navigator.is_track_maze(grid, [1, 2, 3, 4]) is True
    # If extra actions or colors 7/10/13 are present, should be False
    assert agent.track_navigator.is_track_maze(grid, [1, 2, 3, 4, 6]) is False


def test_cluster_1_solvers_predicates_and_dispatch() -> None:
    """Verify detection predicates and isolated dispatch for Cluster 1 affordance solvers."""
    agent = InductiveHCIRAgent()
    grid_64 = np.zeros((64, 64), dtype=int)

    # All Cluster 1 solvers require strictly available_actions == [6]
    assert agent.lights_out_solver.is_lights_out_puzzle(grid_64, [1, 2, 3]) is False
    assert agent.btn_slider_solver.is_permutation_button_puzzle(grid_64, [1, 2, 3, 6]) is False
    assert agent.center_of_mass_solver.is_center_of_mass_puzzle(grid_64, [5, 6]) is False
    assert agent.block_pushing_solver.is_block_pushing_puzzle(grid_64, [6, 7]) is False
    assert agent.liquid_gravity_solver.is_liquid_gravity_puzzle(grid_64, [1, 2, 6]) is False
    assert agent.turtle_program_solver.is_turtle_program_puzzle(grid_64, [6, 7]) is False

    # Pure visual arrays matching the solver signatures
    grid_r11l = np.zeros((64, 64), dtype=int)
    grid_r11l[0, 0] = 15
    grid_r11l[0, 1] = 6
    grid_r11l[0, 2] = 1
    assert agent.center_of_mass_solver.is_center_of_mass_puzzle(grid_r11l, [6]) is True

    grid_s5i5 = np.ones((64, 64), dtype=int)  # 0 not in colors
    grid_s5i5[0, 0] = 13
    grid_s5i5[0, 1] = 14
    assert agent.block_pushing_solver.is_block_pushing_puzzle(grid_s5i5, [6]) is True

    grid_vc33 = np.zeros((64, 64), dtype=int)
    grid_vc33.ravel()[:2880] = 7
    assert agent.liquid_gravity_solver.is_liquid_gravity_puzzle(grid_vc33, [6]) is True

    grid_tn36 = np.zeros((64, 64), dtype=int)
    grid_tn36.ravel()[:3743] = 1
    grid_tn36[0, 0] = 4
    grid_tn36[0, 1] = 9
    grid_tn36[0, 2] = 11
    assert agent.turtle_program_solver.is_turtle_program_puzzle(grid_tn36, [6]) is True

    # Test solver plan_step execution with mock queue
    agent.center_of_mass_solver.action_queue = [(6, {"x": 10, "y": 20})]
    act, conf, data = agent.center_of_mass_solver.plan_step(grid_64)
    assert act == 6
    assert data == {"x": 10, "y": 20}

    agent.block_pushing_solver.action_queue = [(6, {"x": 15, "y": 25})]
    act, conf, data = agent.block_pushing_solver.plan_step(grid_64)
    assert act == 6
    assert data == {"x": 15, "y": 25}

    agent.liquid_gravity_solver.action_queue = [(6, {"x": 2, "y": 46})]
    act, conf, data = agent.liquid_gravity_solver.plan_step(grid_64)
    assert act == 6
    assert data == {"x": 2, "y": 46}

    agent.turtle_program_solver.action_queue = [(6, {"x": 36, "y": 55})]
    act, conf, data = agent.turtle_program_solver.plan_step(grid_64)
    assert act == 6
    assert data == {"x": 36, "y": 55}
