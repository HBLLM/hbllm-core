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
    grid[2, 6] = 3  # Goal

    # Ground engine motor dynamics
    agent.internal_engine.avatar_pos = (2, 2)
    agent.internal_engine.avatar_feature = 2
    agent.internal_engine.avatar_features.add(2)
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
    assert isinstance(macro_plan.current_action(), NavigateTo)
