from __future__ import annotations

"""Unit tests for ARC Adapter Object Lifting and MyAgent macro-action wiring.

Validates:
1. `extract_arc_objects` parses raw ARC 2D matrices into typed ObjectSpec instances.
2. Avatar position is excluded from extracted target objects.
3. Key/cargo, door/receptacle, switch/actuator, and goal/exit entities are properly typed.
4. `MyAgent.choose_action` passes extracted objects into `internal_engine.decide(..., objects=objects)`
   to synthesize and execute neuro-symbolic macro plans without hardcoded rules in the core.
"""

from typing import Any, cast

import numpy as np
import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.macro_actions import (
    NavigateTo,
)
from plugins.arc_agi_adapter.arc_macro_actions import register_arc_macro_actions
from plugins.arc_agi_adapter.arc_perception import extract_arc_objects
from plugins.arc_agi_adapter.my_agent import MyAgent


@pytest.fixture(autouse=True)
def setup_drivers() -> None:
    register_arc_macro_actions()


def test_extract_arc_objects_basic_lifting() -> None:
    """Test lifting a synthetic grid with avatar, key, door, and exit into typed ObjectSpecs."""
    grid = np.zeros((12, 12), dtype=int)

    # Walls (color 1)
    grid[0, :] = 1
    grid[11, :] = 1
    grid[:, 0] = 1
    grid[:, 11] = 1

    # Avatar (color 2)
    grid[2, 2] = 2

    # Key / cargo (color 6)
    grid[4, 4] = 6

    # Door / receptacle (color 8)
    grid[7, 7] = 8

    # Exit goal (color 3)
    grid[10, 10] = 3

    engine = AutonomousEpistemicEngine()
    engine.avatar_pos = (2, 2)
    engine.avatar_feature = 2
    engine.avatar_features.add(2)
    engine.symbolic_theory.induce_cargo(6, evidence="cargo")
    engine.symbolic_theory.induce_receptacle(8, evidence="receptacle")
    engine.symbolic_theory.induce_goal(3, evidence="terminal_reward")
    engine.symbolic_theory.induce_barrier(1, evidence="collision")

    objects = extract_arc_objects(grid, engine=engine)

    assert isinstance(objects, list)
    assert len(objects) > 0

    # Ensure avatar is excluded
    avatar_objs = [obj for obj in objects if obj.pos == (2, 2) or obj.feature_id == 2]
    assert len(avatar_objs) == 0

    # Verify cargo/key object
    cargo_objs = [obj for obj in objects if obj.pos == (4, 4)]
    assert len(cargo_objs) == 1
    assert cargo_objs[0].feature_id == 6
    assert cargo_objs[0].role in ("cargo", "manipulable", "resource")

    # Verify door/receptacle object
    rec_objs = [obj for obj in objects if obj.pos == (7, 7)]
    assert len(rec_objs) == 1
    assert rec_objs[0].feature_id == 8
    assert rec_objs[0].role in ("receptacle", "door", "portal")

    # Verify exit goal object
    goal_objs = [obj for obj in objects if obj.pos == (10, 10)]
    assert len(goal_objs) == 1
    assert goal_objs[0].feature_id == 3
    assert goal_objs[0].role == "goal"


def test_extract_arc_objects_switch_actuator() -> None:
    """Test lifting a switch/actuator widget."""
    grid = np.zeros((10, 10), dtype=int)
    grid[2, 3] = 9  # Switch
    grid[8, 8] = 3  # Goal

    engine = AutonomousEpistemicEngine()
    engine.avatar_pos = (2, 1)
    engine.avatar_feature = 2
    engine.symbolic_theory.induce_actuator(9, evidence="actuator")
    engine.symbolic_theory.induce_goal(3, evidence="terminal_reward")

    objects = extract_arc_objects(grid, engine=engine)
    switch_objs = [obj for obj in objects if obj.pos == (2, 3)]
    assert len(switch_objs) == 1
    assert switch_objs[0].role == "switch"
    assert switch_objs[0].feature_id == 9


def test_my_agent_choose_action_wires_objects_to_core() -> None:
    """Test MyAgent.choose_action routes minimal EmbodiedObservation through core to reach goal."""

    agent = MyAgent(game_id="test_game")
    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    grid[2, 4] = 9  # Switch
    grid[2, 6] = 3  # Goal

    # Ground engine motor dynamics
    agent.internal_engine.avatar_pos = (2, 2)
    agent.internal_engine.avatar_feature = 2
    agent.internal_engine.avatar_features.add(2)
    agent.internal_engine.symbolic_theory.induce_actuator(9, evidence="actuator")
    agent.internal_engine.symbolic_theory.induce_goal(3, evidence="terminal_reward")

    # Provide movement actions 1..4 (UP, DOWN, LEFT, RIGHT)
    for aid, dr, dc in [(1, -1, 0), (2, 1, 0), (3, 0, -1), (4, 0, 1)]:
        agent.internal_engine.action_deltas[aid] = (dr, dc)
        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

        agent.internal_engine.action_dynamics[aid] = ActionDynamicsModel(
            action_id=aid, delta_r=dr, delta_c=dc, confidence=1.0
        )

    # Mock FrameData
    class MockFrameData:
        def __init__(self) -> None:
            self.state = "NOT_FINISHED"
            self.frame = grid
            self.levels_completed = 0
            self.win_levels = 1
            self.available_actions = [1, 2, 3, 4]
            self.game_id = "test_game"

    frame = MockFrameData()
    action = agent.choose_action([], cast(Any, frame))

    # Chosen action should follow the optimal A* path around (2, 4) toward goal (2, 6)
    assert action.value in (1, 4)
    assert agent.internal_engine.mental_plan
    assert agent.internal_engine.mental_plan[-1].predicted_avatar_pos == (2, 6)


def test_arc_macro_actions_click_and_silhouette_planning() -> None:
    """Test synthesizing ArcClickCell and SwitchFocus plans via arc_macro_plan_synthesis_hook."""
    from hbllm.hcir.world.macro_actions import ObjectSpec, SwitchFocus
    from plugins.arc_agi_adapter.arc_macro_actions import (
        ArcClickCell,
        arc_macro_plan_synthesis_hook,
    )

    engine = AutonomousEpistemicEngine()
    grid = np.zeros((64, 64), dtype=int)

    # 1. Test click button synthesis
    btn_obj = ObjectSpec(
        id="btn_1",
        pos=(26, 62),
        feature_id=9,
        role="switch",
        properties={"click_coord": (26, 62)},
    )
    click_plan = arc_macro_plan_synthesis_hook(
        engine, grid, available_actions=[6], objects=[btn_obj]
    )
    assert click_plan is not None
    assert len(click_plan.actions) == 1
    assert isinstance(click_plan.actions[0], ArcClickCell)

    # 2. Test silhouette matching synthesis
    shape_obj = ObjectSpec(
        id="shape_1",
        pos=(20, 20),
        feature_id=11,
        role="manipulandum",
    )
    sil_obj = ObjectSpec(
        id="sil_1",
        pos=(15, 15),
        feature_id=11,
        role="receptacle",
    )
    shape_plan = arc_macro_plan_synthesis_hook(
        engine, grid, available_actions=[1, 2, 3, 4, 5], objects=[shape_obj, sil_obj]
    )
    assert shape_plan is not None
    assert any(isinstance(s, SwitchFocus) for s in shape_plan.actions)
    assert any(isinstance(s, NavigateTo) for s in shape_plan.actions)
