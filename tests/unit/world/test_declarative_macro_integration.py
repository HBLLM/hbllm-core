from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.skills.declarative import (
    CompositeDeclarativeSkill,
    DeclarativeNeuroSymbolicSkill,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.macro_actions import (
    DepositCargo,
    InteractWith,
    MacroActionPlan,
    MacroPlanSynthesizer,
    NavigateTo,
    ObjectSpec,
    PickUpCargo,
    attribute_macro_action_outcome,
)
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel


@pytest.fixture
def calibrated_engine() -> AutonomousEpistemicEngine:
    engine = AutonomousEpistemicEngine()
    engine.action_dynamics = {
        1: ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=1.0),
        2: ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=1.0),
        3: ActionDynamicsModel(action_id=3, delta_r=0, delta_c=-1, confidence=1.0),
        4: ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=1.0),
    }
    engine.avatar_pos = (5, 5)
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.bg_feature = 0
    engine.symbolic_theory.walkable_features.add(0)
    return engine


class KeyDoorExitSkill(DeclarativeNeuroSymbolicSkill):
    """Declarative skill for key pickup, door unlocking, and exit navigation."""

    program = SubgoalSequence(
        SymbolicSubgoal(
            intent=SpatialActionIntent.PICKUP,
            target_query={"role": "cargo"},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.DROP,
            target_query={"role": "receptacle"},
        ),
        SymbolicSubgoal(
            intent=SpatialActionIntent.NAVIGATE,
            target_query={"role": "goal"},
        ),
    )


def test_declarative_skill_compile_to_macro_plan(calibrated_engine: AutonomousEpistemicEngine):
    """Verify DeclarativeNeuroSymbolicSkill compiles its SubgoalSequence directly into MacroActionPlan."""
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1  # Avatar
    grid[3, 3] = 6  # Key cargo
    grid[7, 7] = 8  # Door receptacle
    grid[9, 9] = 3  # Goal exit

    objects = [
        ObjectSpec(id="brass_key", pos=(3, 3), feature_id=6, role="cargo"),
        ObjectSpec(id="cell_door", pos=(7, 7), feature_id=8, role="receptacle"),
        ObjectSpec(id="portal_exit", pos=(9, 9), feature_id=3, role="goal"),
    ]
    MacroPlanSynthesizer.ingest_knowledge(calibrated_engine, objects=objects)

    skill = KeyDoorExitSkill()
    plan = skill.compile_to_macro_plan(calibrated_engine, grid, [1, 2, 3, 4])

    assert plan is not None
    assert len(plan.actions) == 3

    # Op 1: PickUpCargo at (3, 3)
    op1 = plan.actions[0]
    assert isinstance(op1, PickUpCargo)
    assert op1.cargo_pos == (3, 3)
    assert op1.cargo_feature == 6

    # Op 2: DepositCargo at (7, 7)
    op2 = plan.actions[1]
    assert isinstance(op2, DepositCargo)
    assert op2.receptacle_pos == (7, 7)

    # Op 3: NavigateTo at (9, 9)
    op3 = plan.actions[2]
    assert isinstance(op3, NavigateTo)
    assert op3.destination == (9, 9)


def test_pickup_cargo_body_schema_docking(calibrated_engine: AutonomousEpistemicEngine):
    """Verify that PickUpCargo completion docks the tool directly into ExtendedBodySchema."""
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1
    grid[5, 6] = 6  # Key at (5, 6)

    plan = MacroActionPlan(plan_id="pickup_test")
    pickup_op = PickUpCargo(cargo_pos=(5, 6), cargo_feature=6)
    plan.append(pickup_op)

    # Avatar steps right (Action 4) onto/adjacent to key
    calibrated_engine.avatar_pos = (5, 6)
    grid[5, 6] = 1

    assert pickup_op.is_satisfied(calibrated_engine, grid)
    # Plan advancement should trigger attribute_macro_action_outcome
    plan.advance_if_satisfied(calibrated_engine, grid)

    assert plan.is_empty()
    # Verified: Body schema acquired tool 6!
    assert calibrated_engine.working_memory.body_schema.is_holding(6)
    assert 6 in calibrated_engine.symbolic_theory.cargo_features


def test_deposit_cargo_pearl_do_causal_discovery(calibrated_engine: AutonomousEpistemicEngine):
    """Verify that DepositCargo clearing a door registers Pearl's do(a) causal resonance and affinity."""
    # Pre-condition: Avatar holds key 6
    calibrated_engine.working_memory.body_schema.acquire_tool(6, step=1, role="key")
    assert calibrated_engine.working_memory.body_schema.is_holding(6)

    prev_grid = np.zeros((10, 10), dtype=int)
    prev_grid[7, 7] = 8  # Door feature 8
    calibrated_engine.symbolic_theory.barrier_features.add(8)

    # Action clears the door cell
    curr_grid = np.zeros((10, 10), dtype=int)
    curr_grid[7, 7] = 0  # Cleared to background

    deposit_op = DepositCargo(receptacle_pos=(7, 7), cargo_feature=6, barrier_feature=8)
    deposit_op.consecutive_steps = 1
    calibrated_engine.avatar_pos = (7, 6)  # Adjacent to door

    assert deposit_op.is_satisfied(calibrated_engine, curr_grid)

    attribute_macro_action_outcome(
        calibrated_engine,
        macro_action=deposit_op,
        prev_grid=prev_grid,
        curr_grid=curr_grid,
        is_success=True,
    )

    # Pearl do(a) assertions:
    # 1. Causal world theory learned tool 6 clears barrier 8
    assert 8 in calibrated_engine.symbolic_theory.tool_barrier_affinities.get(6, set())
    # 2. Body schema registered resonance
    assert 8 in calibrated_engine.body_schema.tool_barrier_affinities.get(6, set())
    assert 6 in calibrated_engine.body_schema.get_tools_for_barrier(8)
    # 3. Tool was expended
    assert not calibrated_engine.body_schema.is_holding(6)


def test_interact_with_remote_causal_attribution(calibrated_engine: AutonomousEpistemicEngine):
    """Verify InteractWith records distal state mutations via vmPFC RemoteCausalAttributor."""
    prev_grid = np.zeros((10, 10), dtype=int)
    prev_grid[2, 3] = 5  # Switch at (2, 3)
    prev_grid[8, 8] = 4  # Remote barrier at (8, 8)

    curr_grid = np.zeros((10, 10), dtype=int)
    curr_grid[2, 3] = 5
    curr_grid[8, 8] = 0  # Remote barrier opened by switch!

    interact_op = InteractWith(target_coord=(2, 3), effector_action=6)
    interact_op.has_executed = True

    attribute_macro_action_outcome(
        calibrated_engine,
        macro_action=interact_op,
        prev_grid=prev_grid,
        curr_grid=curr_grid,
        is_success=True,
    )

    # vmPFC causal discovery assertion
    affordance = calibrated_engine.remote_causal.get_trigger_for_barrier((8, 8), barrier_feature=4)
    assert affordance is not None
    assert affordance.trigger_pos == (2, 3)


def test_failed_macro_action_barrier_grounding(calibrated_engine: AutonomousEpistemicEngine):
    """Verify failed macro-action updates learned_barriers, barrier_features, and failed_transitions."""
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 7] = 4  # Solid obstacle at (5, 7)

    nav_op = NavigateTo(destination=(5, 7), radius=0)
    nav_op.consecutive_steps = 150  # Exceeded max steps

    attribute_macro_action_outcome(
        calibrated_engine,
        macro_action=nav_op,
        prev_grid=grid,
        curr_grid=grid,
        is_success=False,
        action_used=4,
    )

    assert (5, 7) in calibrated_engine.learned_barriers
    assert 4 in calibrated_engine.symbolic_theory.barrier_features
    assert ((5, 5), 4) in calibrated_engine.failed_transitions


def test_composite_declarative_skill_compile_to_macro_plan(
    calibrated_engine: AutonomousEpistemicEngine,
):
    """Verify CompositeDeclarativeSkill concatenates SubgoalSequences into a single MacroActionPlan."""

    class SkillA(DeclarativeNeuroSymbolicSkill):
        program = SubgoalSequence(
            SymbolicSubgoal(
                intent=SpatialActionIntent.INTERACT,
                parameters={"pos": (2, 2), "action": 6},
            )
        )

    class SkillB(DeclarativeNeuroSymbolicSkill):
        program = SubgoalSequence(
            SymbolicSubgoal(
                intent=SpatialActionIntent.NAVIGATE,
                parameters={"pos": (8, 8)},
            )
        )

    composite = CompositeDeclarativeSkill(skills=[SkillA(), SkillB()])
    grid = np.zeros((10, 10), dtype=int)

    plan = composite.compile_to_macro_plan(calibrated_engine, grid, [1, 2, 3, 4, 6])
    assert plan is not None
    assert len(plan.actions) == 2
    assert isinstance(plan.actions[0], InteractWith)
    assert plan.actions[0].target_coord == (2, 2)
    assert isinstance(plan.actions[1], NavigateTo)
    assert plan.actions[1].destination == (8, 8)
