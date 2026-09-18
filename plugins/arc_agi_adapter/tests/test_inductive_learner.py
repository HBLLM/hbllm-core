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
