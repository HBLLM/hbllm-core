from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_deliberation import (
    OFCCounterfactualMentalSimulationPredicate,
    PlanSafetyEvaluator,
)
from hbllm.hcir.world.cortex_episodic import (
    EpisodicTransition,
    ProceduralMacro,
    compress_action_trajectory,
)
from hbllm.hcir.world.cortex_planner import MentalSimulationStep
from hbllm.hcir.world.intuitive_physics import (
    GravityDirection,
    IntuitivePhysicsEngine,
)
from hbllm.hcir.world.macro_actions import ProceduralReplayAction
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones


def test_ofc_counterfactual_mental_simulation_gating() -> None:
    """Validate OFC Counterfactual Mental Simulation Gating:
    Evaluates 3-ply forward rollouts in imagination, rejecting trajectories that lead
    into irreversible corner deadlocks or spatiotemporal collision cones before motor dispatch.
    """
    engine = AutonomousEpistemicEngine()
    engine.learned_goal_positions = {(5, 5)}
    engine.learned_barriers = {(0, 1), (1, 0)}  # Forms a corner around (0, 0)
    grid = np.zeros((10, 10), dtype=int)

    evaluator = PlanSafetyEvaluator(
        predicates=[OFCCounterfactualMentalSimulationPredicate(simulation_depth=3)]
    )

    # 1. Step into corner deadlock (0, 0) bounded by grid bounds and barriers
    deadlock_step = MentalSimulationStep(
        action=1,
        predicted_avatar_pos=(0, 0),
    )
    result = evaluator.evaluate(
        engine=engine,
        curr_grid=grid,
        next_step=deadlock_step,
        available_actions=[1, 2, 3, 4],
    )
    assert not result.is_safe
    assert "corner deadlock" in result.reason

    # 2. Step into open safe space (3, 3)
    safe_step = MentalSimulationStep(
        action=2,
        predicted_avatar_pos=(3, 3),
    )
    safe_result = evaluator.evaluate(
        engine=engine,
        curr_grid=grid,
        next_step=safe_step,
        available_actions=[1, 2, 3, 4],
    )
    assert safe_result.is_safe

    # 3. Step intersecting a future looming collision cone
    engine.collision_cones = SpatiotemporalCollisionCones()
    engine.collision_cones.step_occupied_cells[1] = {(4, 4)}
    hazard_step = MentalSimulationStep(
        action=3,
        predicted_avatar_pos=(4, 4),
    )
    hazard_result = evaluator.evaluate(
        engine=engine,
        curr_grid=grid,
        next_step=hazard_step,
        available_actions=[1, 2, 3, 4],
    )
    assert not hazard_result.is_safe
    assert "collision cone" in hazard_result.reason


def test_intuitive_physics_multi_directional_field_induction() -> None:
    """Validate Dynamic Physics & Multi-Directional Environmental Field Induction:
    Detects uncommanded left/right/up/down drift, induces environmental force vector g,
    and projects passive ballistics along g.
    """
    physics = IntuitivePhysicsEngine()

    # Frame 1: Action commanded was stationary (0, 0), but entity moved left: dr=0, dc=-1
    physics.record_transition(
        prev_pos=(5, 5),
        curr_pos=(5, 4),
        is_displacement_action=False,
        commanded_delta=(0, 0),
    )
    assert not physics.has_gravity

    # Frame 2: Uncommanded motion left repeated: confirms environmental force field
    physics.record_transition(
        prev_pos=(5, 4),
        curr_pos=(5, 3),
        is_displacement_action=False,
        commanded_delta=(0, 0),
    )
    assert physics.has_gravity
    assert physics.gravity_vector == (0, -1)
    assert physics.infer_gravity_direction() == GravityDirection.LEFT

    # Test projection of fall along leftward gravity vector
    grid = np.zeros((10, 10), dtype=int)
    grid[:, 1] = 9  # Vertical barrier wall at column 1 (feature 9)
    landing_pos, path, is_lethal = physics.project_fall(
        start_r=5,
        start_c=8,
        grid=grid,
        barrier_features={9},
    )
    # Falling left from c=8 stops when supported by barrier at c=1 (landing at c=2)
    assert landing_pos == (5, 2)
    assert (5, 2) in path
    assert not is_lethal

    # Multi-body third-party passive entity motion induction
    physics_up = IntuitivePhysicsEngine()
    physics_up.record_passive_entity_motion(prev_pos=(6, 5), curr_pos=(5, 5))  # dr=-1, dc=0
    physics_up.record_passive_entity_motion(prev_pos=(5, 5), curr_pos=(4, 5))
    assert physics_up.has_gravity
    assert physics_up.gravity_vector == (-1, 0)
    assert physics_up.infer_gravity_direction() == GravityDirection.UP


def test_dynamic_macro_skill_compression_loop_elimination() -> None:
    """Validate Dynamic Macro Skill Compression & Hierarchical Chunking:
    Eliminates zero-mutation cyclic wandering from winning episodes and replays optimal subgoals.
    """
    # Create raw trajectory:
    # (1, 1) -> (1, 2) -> (1, 1) -> (2, 1) -> (3, 1)[WIN]
    # Actions: 4 (RIGHT), 3 (LEFT), 2 (DOWN), 2 (DOWN)
    transitions = [
        EpisodicTransition(
            step=0,
            avatar_pos=(1, 1),
            action_id=4,
            reward=0.0,
            is_win=False,
        ),
        EpisodicTransition(
            step=1,
            avatar_pos=(1, 2),
            action_id=3,
            reward=0.0,
            is_win=False,
        ),
        EpisodicTransition(
            step=2,
            avatar_pos=(1, 1),
            action_id=2,
            reward=0.0,
            is_win=False,
        ),
        EpisodicTransition(
            step=3,
            avatar_pos=(2, 1),
            action_id=2,
            reward=1.0,
            is_win=True,
        ),
    ]

    comp_actions, comp_data = compress_action_trajectory(transitions)
    # The redundant (1, 1) -> (1, 2) -> (1, 1) loop is excised!
    assert comp_actions == [2, 2]
    assert len(comp_actions) == 2

    # Verify ProceduralMacro integration and replay execution
    macro = ProceduralMacro(
        macro_id="compressed_macro_test",
        action_sequence=[4, 3, 2, 2],
        action_data_sequence=[None, None, None, None],
        compressed_action_sequence=comp_actions,
        compressed_action_data_sequence=comp_data,
        available_actions={2, 3, 4},
        level=0,
        context_id="test_game",
    )

    exec_acts, exec_data = macro.get_execution_sequence()
    assert exec_acts == [2, 2]

    # Verify ProceduralReplayAction executes compressed sequence
    replay = ProceduralReplayAction(macro=macro)
    assert replay._action_seq == [2, 2]
    assert len(replay._action_seq) == 2
