from __future__ import annotations

"""Unit tests for ARC Adapter Object Lifting and MyAgent macro-action wiring.

Validates:
1. `extract_arc_objects` parses raw ARC 2D matrices into typed ObjectSpec instances.
2. Avatar position is excluded from extracted target objects.
3. Key/cargo, door/receptacle, switch/actuator, and goal/exit entities are properly typed.
4. `MyAgent.choose_action` passes extracted objects into `internal_engine.decide(..., objects=objects)`
   to synthesize and execute neuro-symbolic macro plans without hardcoded rules in the core.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.macro_actions import (
    NavigateTo,
)
from kaggle_submission.my_agent import MyAgent
from plugins.arc_agi_adapter.arc_macro_actions import register_arc_macro_actions
from plugins.arc_agi_adapter.arc_perception import extract_arc_objects


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
    engine.symbolic_theory.cargo_features.add(6)
    engine.symbolic_theory.receptacle_features.add(8)
    engine.symbolic_theory.goal_features.add(3)
    engine.symbolic_theory.barrier_features.add(1)

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
    engine.symbolic_theory.actuator_features.add(9)
    engine.symbolic_theory.goal_features.add(3)

    objects = extract_arc_objects(grid, engine=engine)
    switch_objs = [obj for obj in objects if obj.pos == (2, 3)]
    assert len(switch_objs) == 1
    assert switch_objs[0].role == "switch"
    assert switch_objs[0].feature_id == 9


def test_extract_arc_objects_with_sprite_tags() -> None:
    """Test extracting cargo and receptacle objects from Arcade sprite tags."""
    grid = np.zeros((16, 16), dtype=int)

    # Mock Arcade Game Level with tagged sprites
    mock_game = MagicMock()
    mock_lvl = MagicMock()
    mock_game.current_level = mock_lvl

    # Cargo sprite (e.g. wa30 crate)
    cargo_sprite = MagicMock()
    cargo_sprite.x = 5
    cargo_sprite.y = 5
    cargo_sprite.width = 1
    cargo_sprite.height = 1
    cargo_sprite.pixels = [[4]]

    # Receptacle sprite (e.g. wa30 green goal area)
    rec_sprite = MagicMock()
    rec_sprite.x = 12
    rec_sprite.y = 12
    rec_sprite.width = 2
    rec_sprite.height = 2
    rec_sprite.pixels = [[7, 7], [7, 7]]

    def get_sprites_by_tag(tag: str) -> list[Any]:
        if tag == "geezpjgiyd":
            return [cargo_sprite]
        elif tag == "fsjjayjoeg":
            return [rec_sprite]
        return []

    mock_lvl.get_sprites_by_tag.side_effect = get_sprites_by_tag

    objects = extract_arc_objects(grid, game_obj=mock_game)

    cargos = [obj for obj in objects if obj.role == "cargo"]
    assert len(cargos) >= 1
    assert (5, 5) in [c.pos for c in cargos]

    recs = [obj for obj in objects if obj.role == "receptacle"]
    assert len(recs) >= 1
    assert any(r.pos[0] >= 12 and r.pos[1] >= 12 for r in recs)


def test_my_agent_choose_action_wires_objects_to_core() -> None:
    """Test MyAgent.choose_action feeds lifted ObjectSpecs into engine.decide and triggers macro execution."""
    agent = MyAgent(game_id="test_game")
    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Avatar
    grid[2, 4] = 9  # Switch
    grid[2, 6] = 3  # Goal

    # Ground engine motor dynamics
    agent.internal_engine.avatar_pos = (2, 2)
    agent.internal_engine.avatar_feature = 2
    agent.internal_engine.avatar_features.add(2)
    agent.internal_engine.symbolic_theory.actuator_features.add(9)
    agent.internal_engine.symbolic_theory.goal_features.add(3)

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
    action = agent.choose_action([], frame)

    # Chosen action should move RIGHT towards (2, 6)
    assert action.value == 4
    # Macro plan should be synthesized and active in working memory
    macro_plan = agent.internal_engine.working_memory.macro_plan
    assert macro_plan is not None
    assert not macro_plan.is_empty()
    from plugins.arc_agi_adapter.arc_macro_actions import ToggleSwitch

    assert isinstance(macro_plan.current_action(), (ToggleSwitch, NavigateTo))


def test_extract_arc_objects_sys_click_and_shapes() -> None:
    """Test lifting sys_click buttons, shapes, and silhouettes from game_obj."""
    grid = np.zeros((64, 64), dtype=int)
    grid[24:28, 60:64] = 9  # Button 1

    # Mock game_obj with level and camera
    game_obj = MagicMock()
    cur_lvl = MagicMock()
    cam = MagicMock()
    cam.width = 32
    cam.height = 32
    game_obj.camera = cam
    game_obj.current_level = cur_lvl

    # Mock button sprite
    btn_sprite = MagicMock()
    btn_sprite.name = "btn_1"
    btn_sprite.tags = ["sys_click"]
    btn_sprite.x = 30
    btn_sprite.y = 12
    btn_sprite.width = 2
    btn_sprite.height = 2

    # Mock shape sprite
    shape_sprite = MagicMock()
    shape_sprite.name = "shape_1"
    shape_sprite.tags = ["0031cppcuvqlbi"]
    shape_sprite.x = 10
    shape_sprite.y = 16
    shape_sprite.width = 20
    shape_sprite.height = 20
    shape_sprite.pixels = np.full((20, 20), 11)

    # Mock silhouette sprite
    sil_sprite = MagicMock()
    sil_sprite.name = "sil_1"
    sil_sprite.tags = ["0054xnsuqceejm"]
    sil_pixels = np.full((64, 64), -1)
    sil_pixels[10:20, 15:25] = 11
    sil_sprite.pixels = sil_pixels

    def get_sprites_by_tag(tag: str) -> list[Any]:
        if tag == "sys_click":
            return [btn_sprite]
        elif tag == "0031cppcuvqlbi":
            return [shape_sprite]
        elif tag == "0054xnsuqceejm":
            return [sil_sprite]
        return []

    cur_lvl.get_sprites_by_tag.side_effect = get_sprites_by_tag

    engine = AutonomousEpistemicEngine()
    objects = extract_arc_objects(grid, engine=engine, game_obj=game_obj)

    btn_objs = [o for o in objects if "btn" in o.id or "button" in o.role or o.role == "switch"]
    assert len(btn_objs) >= 1
    assert btn_objs[0].properties.get("click_data") == {"x": 62, "y": 26}

    shape_objs = [o for o in objects if o.role == "manipulandum"]
    assert len(shape_objs) >= 1
    assert shape_objs[0].feature_id == 11

    sil_objs = [o for o in objects if "silhouette" in o.id]
    assert len(sil_objs) >= 1
    assert sil_objs[0].feature_id == 11


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
