"""Domain-agnostic embodied agency, cut-set topology, and replay invariants."""

from __future__ import annotations

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.world.assimilation.assimilator import EpistemicFeedbackAssimilator
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.causal.transition_kb import TransitionKnowledgeBase, TransitionOutcome
from hbllm.hcir.world.deliberation.engine import PrefrontalDeliberationEngine
from hbllm.hcir.world.motor.curiosity import EpistemicCuriosityExplorer
from hbllm.hcir.world.motor.effector import MotorCortexEffector
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from hbllm.hcir.world.percept import (
    EpisodeBoundary,
    EpisodeBoundaryKind,
)
from hbllm.hcir.world.perception.engine import PerceptionEngine
from hbllm.hcir.world.planning.goal_scheduler import EpistemicGoalScheduler


def test_point_actuator_schema_marks_spatial_effector_and_grounds_pure_click_world() -> None:
    """ActuatorSchema(kind='point') binds spatial effector affordances and never emits None coords."""
    spec = MotorCortexEffector.parse_action_spec({"action_id": 6, "kind": "point"})
    assert spec.requires_spatial_target is True
    assert spec.target_param_keys == ("x", "y")

    engine = AutonomousEpistemicEngine()
    engine.register_action_space([{"action_id": 6, "kind": "point"}])
    assert engine.is_spatial_effector(6)

    # Simulate pure click-only board where an exogenous corner pixel ticked on previous step
    # and all grid targets were transiently marked quiescent: ensure_effector_grounded still returns x/y.
    grid = np.zeros((16, 16), dtype=int)
    grid[0, 15] = 4  # exogenous corner tick
    grid[8, 8] = 2
    engine.prev_grid = grid.copy()
    engine.avatar_pos = (0, 15)  # simulate prior misattribution or focus without displacement caps
    engine.quiescent_actuation_targets.add((8, 8))
    coords = MotorCortexEffector.ground_effector_action(engine, grid, 6)
    assert coords is not None and "x" in coords and "y" in coords

    act, data = PrefrontalDeliberationEngine.ensure_effector_grounded(engine, grid, [6], 6, None)
    assert act == 6
    assert isinstance(data, dict) and "x" in data and "y" in data


def test_impassable_background_cutset_excludes_enclosed_indicator_goals() -> None:
    """When dominant background is an impassable barrier, sealed HUD indicators are excluded."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 9
    engine.avatar_features = {9}
    engine.bg_feature = 4
    engine.symbolic_theory.induce_walkable(3, evidence="floor")
    engine.symbolic_theory.induce_barrier(4, evidence="wall_collision")
    engine.symbolic_theory.induce_barrier(5, evidence="border_collision")
    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0, probes_tested=5
    )

    grid = np.full((16, 16), 4, dtype=int)
    grid[2:10, 2:10] = 3  # walkable arena
    grid[4, 4] = 9  # avatar in arena
    grid[6, 8] = 8  # reachable entity inside arena
    grid[12, :] = 5  # horizontal divider wall
    grid[14, 3] = 9  # enclosed HUD indicator behind wall=4/5

    filtered, _ = EpistemicGoalScheduler._filter_reachable_and_unlocked(
        engine,
        grid,
        16,
        16,
        4,
        {9},
        (4, 4),
        [(14, 3)],
    )
    assert (14, 3) not in filtered


def test_motor_babbling_rotates_across_uninhibited_actions_on_multicell_avatar() -> None:
    """Motor babbling evaluates cohesive multi-pixel bodies and avoids inhibited/recent directions."""
    engine = AutonomousEpistemicEngine()
    grid = np.zeros((16, 16), dtype=int)
    grid[7:10, 7:10] = 6  # 3x3 (9-pixel) cohesive avatar body
    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=0.5, probes_tested=0
    )
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=0.5, probes_tested=0
    )
    engine.action_dynamics[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=0.5, probes_tested=0
    )
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=0.5, probes_tested=0
    )
    engine.inhibited_actions[4] = 3
    engine.recent_actions.extend([4, 4, 4])

    prop = EpistemicCuriosityExplorer._probe_motor_babble(engine, [1, 2, 3, 4], grid)
    assert prop is not None
    assert prop.action != 4


def test_non_displacement_noop_inhibits_action_despite_exogenous_timer_tick() -> None:
    """Exogenous corner/timer ticks must not mask no-op discrete actions or reset stuck steps."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 9
    engine.avatar_features = {9}
    engine.avatar_pos = (4, 4)
    engine.bg_feature = 0
    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0, probes_tested=5
    )

    prev = np.zeros((32, 32), dtype=int)
    prev[4, 4] = 9
    prev[28, 3] = 7
    curr = prev.copy()
    curr[28, 3] = 0
    curr[28, 4] = 7  # inner colinear exogenous timer bar tick (diff.changed_pixel_count == 2)
    for i in range(4):
        engine.working_memory.exogenous_tracker.observe_transition(
            [(28, i), (28, i + 1)], action=5, avatar_locus=(4, 4)
        )

    engine.prev_grid = prev
    engine.last_action = 5  # non-displacement discrete action
    diff = PerceptionEngine.compute_frame_diff(prev, curr)
    EpistemicFeedbackAssimilator._assimilate_proprioception_and_barriers(
        engine,
        curr,
        diff,
        action=5,
        action_data=None,
        is_effector_action=False,
        is_known_displacement=False,
        av_moved=[],
        manip_moved=False,
        available_actions=[1, 5],
        is_success=False,
        is_failure=False,
        prev_avatar_pos=(4, 4),
        bg=0,
        H=32,
        W=32,
    )
    assert engine.working_memory.consecutive_stuck_steps >= 1
    assert engine.inhibited_actions.get(5, 0) > 0
    assert ((4, 4), 5) in engine.failed_transitions


def test_same_level_death_restores_initial_state_locks_and_rearms_transformers() -> None:
    """After DEATH retry on a multi-state level, S_0 gate locks restore and transformers re-arm."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 2
    engine.avatar_features = {2}
    engine.bg_feature = 0
    engine.symbolic_theory.induce_walkable(0, evidence="bg")
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=1.0, probes_tested=5
    )

    bs = engine.working_memory.body_schema
    # State S_0: gate (3, 3)+action 2 blocked
    bs.record_conditioned_block((3, 3), 2, target_pos=(4, 3))
    # Transition onto transformer pad (6, 6) moves latent state to S_1
    bs.advance_latent_state(12345, trigger_pos=(6, 6))
    engine.exhausted_candidate_goals.add((6, 6))
    engine.symbolic_theory.mark_locus_explored((6, 6), reason="used_pad")

    # Simulate same-level DEATH retry
    engine.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.DEATH, same_level=True))

    assert bs.latent_state_signature == 0
    assert ((3, 3), 2) in engine.failed_transitions
    assert (4, 3) in engine.exhausted_candidate_goals
    assert (6, 6) not in engine.exhausted_candidate_goals
    assert not engine.symbolic_theory.is_locus_explored((6, 6))
    # Transformer record is cleared only on new_context level change
    engine.on_boundary(EpisodeBoundary(kind=EpisodeBoundaryKind.LEVEL_COMPLETE, same_level=False))
    assert bs.transformer_state_transitions == {}


def test_locus_blocked_ambient_bg_cell_excluded_from_interaction_stances() -> None:
    """Locus-blocked ambient bg cells must not be treated as walkable stances."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 9
    engine.avatar_features = {9}
    engine.avatar_pos = (4, 1)
    engine.bg_feature = 0
    engine.symbolic_theory.induce_walkable(0, evidence="floor")
    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=1.0, probes_tested=5
    )

    grid = np.zeros((9, 9), dtype=int)
    grid[4, 1] = 9
    grid[4, 4] = 7

    engine.symbolic_theory.transition_kb.record_outcome(
        (3, 4), TransitionOutcome.BLOCKED, scope="locus", evidence="invisible_wall"
    )
    engine.learned_barriers.add((5, 4))

    assert not engine.symbolic_theory.is_cell_feasible(
        0,
        locus=(3, 4),
        bg_feature=0,
        learned_barriers=engine.learned_barriers,
        hazard_loci=engine.current_context_hazards,
    )
    assert not engine.symbolic_theory.is_cell_feasible(
        0,
        locus=(5, 4),
        bg_feature=0,
        learned_barriers=engine.learned_barriers,
        hazard_loci=engine.current_context_hazards,
    )

    target_goals: list[tuple[int, int]] = []
    satisfied_when: dict[tuple[int, int], dict[str, object]] = {}
    EpistemicGoalScheduler._enqueue_adjacent_or_contact(
        engine,
        grid,
        9,
        9,
        0,
        (4, 1),
        (4, 4),
        7,
        target_goals,
        satisfied_when,
    )
    assert (3, 4) not in target_goals
    assert (5, 4) not in target_goals
    assert (4, 3) in target_goals
    assert (4, 5) in target_goals


def test_distal_point_modality_targets_enclosed_entity_without_walkable_neighbors() -> None:
    """Pure point-actuator worlds must target enclosed entities without walkable stances."""
    engine = AutonomousEpistemicEngine()
    engine.register_action_space([{"action_id": 6, "kind": "point"}])
    engine.bg_feature = 0
    engine.symbolic_theory.induce_barrier(5, evidence="solid_wall")

    grid = np.zeros((11, 11), dtype=int)
    grid[3:8, 3:8] = 5  # sealed enclosure around (5, 5)
    grid[5, 5] = 7  # enclosed interactive target
    engine.learned_barriers.update(
        [(r, c) for r in range(3, 8) for c in range(3, 8) if (r, c) != (5, 5)]
    )

    target_goals: list[tuple[int, int]] = []
    satisfied_when: dict[tuple[int, int], dict[str, object]] = {}
    EpistemicGoalScheduler._enqueue_adjacent_or_contact(
        engine,
        grid,
        11,
        11,
        0,
        None,
        (5, 5),
        7,
        target_goals,
        satisfied_when,
        set_contact_meta=True,
        available_actions=[6],
    )
    assert (5, 5) in target_goals

    entities = [SpatialEntity(grid_pos=(5, 5), area=1, feature_id=7, role=EntityRole.UNKNOWN)]
    proposal = EpistemicGoalScheduler.schedule_goals(
        engine,
        grid,
        entities,
        bg=0,
        av_feats=set(),
        available_actions=[6],
    )
    assert (5, 5) in proposal.goals


def test_non_cardinal_diagonal_and_stride2_kinematics_reachability_and_coupling() -> None:
    """Diagonal and stride-2 calibrated dynamics support operator-closure reachability and stances."""
    # Case 1: Diagonal mover escapes a 4-connected sealed box across corner gap
    grid = np.zeros((7, 7), dtype=int)
    barriers = {(2, 3), (3, 2), (3, 4), (4, 3)}
    for r, c in barriers:
        grid[r, c] = 5
    comp_diag = TopologicalCutSetAnalyzer.get_reachable_component(
        (3, 3),
        barriers,
        (7, 7),
        transition_deltas=[(-1, -1), (-1, 1), (1, -1), (1, 1)],
    )
    assert (1, 1) in comp_diag
    comp_cardinal = TopologicalCutSetAnalyzer.get_reachable_component((3, 3), barriers, (7, 7))
    assert (1, 1) not in comp_cardinal

    # Case 2: Stride-2 jumper queues {target - Δ_a} stances even across 1-cell moat
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 9
    engine.avatar_features = {9}
    engine.avatar_pos = (4, 0)
    engine.bg_feature = 0
    engine.symbolic_theory.induce_walkable(0, evidence="floor")
    engine.symbolic_theory.induce_barrier(5, evidence="moat")
    engine.action_dynamics[1] = ActionDynamicsModel(
        action_id=1, delta_r=-2, delta_c=0, confidence=1.0, probes_tested=5
    )
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=2, delta_c=0, confidence=1.0, probes_tested=5
    )
    engine.action_dynamics[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-2, confidence=1.0, probes_tested=5
    )
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=2, confidence=1.0, probes_tested=5
    )

    grid2 = np.zeros((9, 9), dtype=int)
    grid2[4, 0] = 9
    # 1-cell barrier ring around target at (4, 4) — cardinal 1-step neighbors are barriers!
    for r, c in ((3, 4), (5, 4), (4, 3), (4, 5)):
        grid2[r, c] = 5
    grid2[4, 4] = 7

    target_goals: list[tuple[int, int]] = []
    satisfied_when: dict[tuple[int, int], dict[str, object]] = {}
    EpistemicGoalScheduler._enqueue_adjacent_or_contact(
        engine,
        grid2,
        9,
        9,
        0,
        (4, 0),
        (4, 4),
        7,
        target_goals,
        satisfied_when,
        available_actions=[1, 2, 3, 4],
    )
    assert any(st in target_goals for st in ((2, 4), (6, 4), (4, 2), (4, 6)))


def test_modality_conditioned_transition_kb_separates_locomote_from_distal() -> None:
    """Locomotor BLOCKED leaves distal feasibility open, while TERMINAL_FAIL blocks both."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(5, TransitionOutcome.BLOCKED, scope="feature", evidence="wall")
    assert not kb.evaluate_transition(5, modality="locomote").traversable
    assert kb.evaluate_transition(5, modality="distal").traversable

    kb.record_outcome(8, TransitionOutcome.TERMINAL_FAIL, scope="feature", evidence="hazard")
    assert not kb.evaluate_transition(8, modality="locomote").traversable
    assert not kb.evaluate_transition(8, modality="distal").traversable


def test_avatar_traversal_clears_false_exogenous_corridor_and_never_marks_stuck() -> None:
    """Self-displacement across a previously exogenous-flagged corridor clears spurious flags."""
    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.avatar_pos = (7, 4)
    engine.bg_feature = 0
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=1.0, probes_tested=5
    )

    prev = np.zeros((10, 10), dtype=int)
    prev[7, 4] = 1
    curr = np.zeros((10, 10), dtype=int)
    curr[7, 5] = 1

    engine.working_memory.exogenous_tracker.exogenous_loci.update({(7, 4), (7, 5)})
    engine.prev_grid = prev
    engine.last_action = 4
    engine.last_predicted_pos = (7, 5)
    engine.assimilate_feedback(curr, [1, 2, 3, 4], action=4)

    assert engine.working_memory.consecutive_stuck_steps == 0
    assert ((7, 4), 4) not in engine.failed_transitions
    assert engine.inhibited_actions.get(4, 0) == 0
    assert (7, 5) not in engine.working_memory.exogenous_tracker.exogenous_loci
