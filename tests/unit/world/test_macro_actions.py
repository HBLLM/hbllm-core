from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.macro_actions import (
    CustomMacroAction,
    DepositCargo,
    InteractWith,
    MacroActionDriverManager,
    MacroActionPlan,
    MacroPlanSynthesizer,
    MicroActionSpec,
    NavigateTo,
    ObjectSpec,
    PickUpCargo,
    SwitchFocus,
)
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from plugins.arc_agi_adapter.arc_macro_actions import (
    ToggleSwitch,
    register_arc_macro_actions,
)


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


def test_navigate_to_operator(calibrated_engine: AutonomousEpistemicEngine):
    grid = np.zeros((10, 10), dtype=int)
    nav_op = NavigateTo(destination=(5, 8), radius=0)

    assert not nav_op.is_satisfied(calibrated_engine, grid)
    assert nav_op.is_feasible(calibrated_engine, grid, [1, 2, 3, 4])

    # First step toward (5, 8) from (5, 5) should be Action 4 (Right: delta_c=+1)
    act, data = nav_op.step(calibrated_engine, grid, [1, 2, 3, 4])
    assert act == 4
    assert data is None

    # When avatar reaches destination
    calibrated_engine.avatar_pos = (5, 8)
    assert nav_op.is_satisfied(calibrated_engine, grid)


def test_interact_with_operator(calibrated_engine: AutonomousEpistemicEngine):
    grid = np.zeros((10, 10), dtype=int)
    interact_op = InteractWith(target_coord=(2, 3), effector_action=6)

    assert not interact_op.is_satisfied(calibrated_engine, grid)
    assert interact_op.is_feasible(calibrated_engine, grid, [1, 2, 3, 4, 6])

    act, data = interact_op.step(calibrated_engine, grid, [1, 2, 3, 4, 6])
    assert act == 6
    assert data == {"x": 3, "y": 2}
    assert interact_op.is_satisfied(calibrated_engine, grid)


def test_switch_focus_operator(calibrated_engine: AutonomousEpistemicEngine):
    grid = np.zeros((10, 10), dtype=int)
    calibrated_engine.focus_switch_actions.add(5)
    calibrated_engine.failed_transitions.add(((5, 5), 1))

    switch_op = SwitchFocus(switch_action=5)
    assert switch_op.is_feasible(calibrated_engine, grid, [1, 2, 3, 4, 5])

    act, data = switch_op.step(calibrated_engine, grid, [1, 2, 3, 4, 5])
    assert act == 5
    assert data is None
    assert switch_op.is_satisfied(calibrated_engine, grid)
    # Verifies failed_transitions was purged upon focus switch
    assert len(calibrated_engine.failed_transitions) == 0


def test_macro_action_plan_advancement(calibrated_engine: AutonomousEpistemicEngine):
    grid = np.zeros((10, 10), dtype=int)
    plan = MacroActionPlan(plan_id="test_plan")

    op1 = NavigateTo(destination=(5, 6), radius=0)
    op2 = InteractWith(target_coord=(2, 2), effector_action=6)
    plan.append(op1)
    plan.append(op2)

    assert not plan.is_empty()
    assert plan.current() == op1

    # Step 1: Execute NavigateTo
    act1, data1 = plan.step(calibrated_engine, grid, [1, 2, 3, 4, 6])
    assert act1 == 4

    # Simulate avatar moving to (5, 6)
    calibrated_engine.avatar_pos = (5, 6)

    # Step 2: op1 is satisfied, plan advances automatically to op2
    act2, data2 = plan.step(calibrated_engine, grid, [1, 2, 3, 4, 6])
    assert act2 == 6
    assert data2 == {"x": 2, "y": 2}

    # Step 3: op2 is now satisfied, plan becomes empty
    plan.advance_if_satisfied(calibrated_engine, grid)
    assert plan.is_empty()


def test_macro_plan_synthesizer(calibrated_engine: AutonomousEpistemicEngine):
    grid = np.zeros((10, 10), dtype=int)
    grid[8, 8] = 3  # Place goal feature 3 at (8, 8)
    calibrated_engine.learned_goal_features.add(3)
    calibrated_engine.learned_goal_positions.add((8, 8))

    plan = MacroPlanSynthesizer.synthesize_plan(calibrated_engine, grid, [1, 2, 3, 4])
    assert plan is not None
    assert not plan.is_empty()
    cur = plan.current()
    assert isinstance(cur, NavigateTo)
    assert cur.destination == (8, 8)


def test_provide_objects_and_micro_actions_to_core():
    """Verify runner provides objects and micro-actions to core; core synthesizes and executes macro-plan."""
    engine = AutonomousEpistemicEngine()

    # Micro-action specs provided by runner
    micro_actions = [
        MicroActionSpec(action_id=1, name="UP", delta_r=-1, delta_c=0),
        MicroActionSpec(action_id=2, name="DOWN", delta_r=1, delta_c=0),
        MicroActionSpec(action_id=3, name="LEFT", delta_r=0, delta_c=-1),
        MicroActionSpec(action_id=4, name="RIGHT", delta_r=0, delta_c=1),
    ]

    # Object specs provided by runner
    objects = [
        ObjectSpec(id="exit_gate", pos=(5, 9), feature_id=7, role="goal"),
        ObjectSpec(
            id="stone_wall", pos=(5, 7), feature_id=4, role="barrier", properties={"is_solid": True}
        ),
    ]

    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1  # Avatar
    grid[5, 9] = 7  # Exit
    grid[5, 7] = 4  # Wall
    engine.avatar_pos = (5, 5)
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.bg_feature = 0
    engine.symbolic_theory.walkable_features.add(0)

    # Core receives objects and micro-actions directly via decide()
    act, data = engine.decide(
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        objects=objects,
        micro_actions=micro_actions,
    )

    # Ingested knowledge assertions
    assert (5, 9) in engine.learned_goal_positions
    assert (5, 7) in engine.learned_barriers
    assert 1 in engine.action_dynamics and engine.action_dynamics[1].delta_r == -1

    # Macro plan synthesized assertions
    assert not engine.macro_plan.is_empty()
    cur_op = engine.macro_plan.current()
    assert isinstance(cur_op, NavigateTo)
    assert cur_op.destination == (5, 9)

    # Action chosen should avoid wall at (5, 7)
    assert act in [1, 2]  # Detour UP or DOWN around wall


def test_key_door_multi_stage_macro_plan(calibrated_engine: AutonomousEpistemicEngine):
    """Verify core synthesizes a multi-stage MacroActionPlan: [PickUpCargo] -> [DepositCargo] -> [NavigateTo]."""
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1  # Avatar at (5, 5)
    grid[3, 3] = 6  # Key cargo at (3, 3)
    grid[7, 7] = 8  # Receptacle / Door at (7, 7)
    grid[9, 9] = 3  # Exit goal at (9, 9)

    objects = [
        ObjectSpec(id="gold_key", pos=(3, 3), feature_id=6, role="cargo"),
        ObjectSpec(id="iron_door", pos=(7, 7), feature_id=8, role="receptacle"),
        ObjectSpec(id="exit_portal", pos=(9, 9), feature_id=3, role="goal"),
    ]

    plan = MacroPlanSynthesizer.synthesize_plan(
        calibrated_engine,
        grid,
        available_actions=[1, 2, 3, 4],
        objects=objects,
    )

    assert plan is not None
    assert len(plan.actions) == 3
    op1, op2, op3 = list(plan.actions)
    assert isinstance(op1, PickUpCargo)
    assert op1.cargo_pos == (3, 3)
    assert isinstance(op2, DepositCargo)
    assert op2.receptacle_pos == (7, 7)
    assert isinstance(op3, NavigateTo)
    assert op3.destination == (9, 9)

    # Execute step on stage 1 (PickUpCargo)
    act = plan.step(calibrated_engine, grid, [1, 2, 3, 4])
    assert act is not None


def test_toggle_switch_macro_action(calibrated_engine: AutonomousEpistemicEngine):
    """Verify ToggleSwitch macro-action actuation via spatial click effector via driver manager."""
    register_arc_macro_actions()
    assert MacroActionDriverManager.get_registered_action("ToggleSwitch") is ToggleSwitch

    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1  # Avatar
    grid[2, 4] = 9  # Switch / Lever
    grid[8, 8] = 3  # Exit goal

    objects = [
        ObjectSpec(id="laser_switch", pos=(2, 4), feature_id=9, role="switch"),
        ObjectSpec(id="exit", pos=(8, 8), feature_id=3, role="goal"),
    ]
    micro_actions = [
        MicroActionSpec(action_id=1, name="UP", delta_r=-1, delta_c=0),
        MicroActionSpec(action_id=2, name="DOWN", delta_r=1, delta_c=0),
        MicroActionSpec(action_id=3, name="LEFT", delta_r=0, delta_c=-1),
        MicroActionSpec(action_id=4, name="RIGHT", delta_r=0, delta_c=1),
        MicroActionSpec(action_id=6, name="CLICK", is_spatial_effector=True),
    ]

    act, data = calibrated_engine.decide(
        curr_grid=grid,
        available_actions=[1, 2, 3, 4, 6],
        objects=objects,
        micro_actions=micro_actions,
    )

    # First macro action is ToggleSwitch, which triggers spatial click at (2, 4)
    assert act == 6
    assert data == {"x": 4, "y": 2}


def test_custom_macro_action_registration(calibrated_engine: AutonomousEpistemicEngine):
    """Verify custom user-defined macro action handlers can be registered and stepped."""
    executed_steps = []

    def custom_step(engine, curr_grid, available_actions):
        executed_steps.append("step_taken")
        return 4, None

    custom_op = CustomMacroAction(
        name="CustomDanceStep",
        satisfaction_fn=lambda eng, g: len(executed_steps) >= 2,
        feasibility_fn=lambda eng, g, acts: 4 in acts,
        step_fn=custom_step,
        description="Execute custom dance micro-actions",
    )

    grid = np.zeros((10, 10), dtype=int)
    plan = MacroActionPlan(plan_id="custom_plan")
    plan.append(custom_op)

    assert not custom_op.is_satisfied(calibrated_engine, grid)
    assert custom_op.is_feasible(calibrated_engine, grid, [1, 2, 3, 4])

    # Step 1
    act1, data1 = plan.step(calibrated_engine, grid, [1, 2, 3, 4])
    assert act1 == 4
    assert not custom_op.is_satisfied(calibrated_engine, grid)

    # Step 2
    act2, data2 = plan.step(calibrated_engine, grid, [1, 2, 3, 4])
    assert act2 == 4
    assert custom_op.is_satisfied(calibrated_engine, grid)

    # Plan automatically advances and clears satisfied custom operator
    plan.advance_if_satisfied(calibrated_engine, grid)
    assert plan.is_empty()


def test_bilateral_multi_agent_macro_handover(calibrated_engine: AutonomousEpistemicEngine):
    """Verify multi-agent MacroActionPlan incorporates SwitchFocus operator."""
    grid = np.zeros((10, 10), dtype=int)
    calibrated_engine.focus_switch_actions.add(5)
    calibrated_engine.working_memory.focus_switch_actions.add(5)

    objects = [
        ObjectSpec(id="agent1_goal", pos=(5, 7), feature_id=3, role="goal"),
    ]

    plan = MacroPlanSynthesizer.synthesize_plan(
        calibrated_engine,
        grid,
        available_actions=[1, 2, 3, 4, 5],
        objects=objects,
    )

    assert plan is not None
    assert len(plan.actions) == 2
    nav_op = plan.actions[0]
    switch_op = plan.actions[1]
    assert isinstance(nav_op, NavigateTo)
    assert nav_op.destination == (5, 7)
    assert isinstance(switch_op, SwitchFocus)
    assert switch_op.switch_action == 5
