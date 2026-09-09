"""
Upstream Byte-Verification & Integration Tests for Native Crafter.

Verifies that the Crafter perception and action adapters interface with 100% fidelity
against the real upstream published 'crafter' package, byte-verifying:
1. Material IDs, Object IDs, and Action definitions against upstream constants.
2. Native perception grid extraction, transpose alignment, and entity mapping.
3. Live vitals (health, food, drink, energy) tracking from engine state.
4. Milestone achievement unlocking on native crafter.Env.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_test_dir = Path(__file__).resolve().parent
_plugin_dir = _test_dir.parent
_plugins_root = _plugin_dir.parent
_core_root = _plugins_root.parent

for p in [str(_core_root), str(_plugins_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import crafter  # noqa: F401
    from crafter import constants as crafter_constants
    from crafter import objects as crafter_objects  # noqa: F401

    HAS_REAL_CRAFTER = True
except ImportError:
    HAS_REAL_CRAFTER = False

pytestmark = pytest.mark.skipif(
    not HAS_REAL_CRAFTER,
    reason="Requires upstream 'crafter' package installed.",
)

from crafter_adapter import (
    ACTION_NAMES,
    CrafterAchievement,
    CrafterAction,
    CrafterGoal,
    CrafterObject,
    CrafterPerceptionAdapter,
    NativeCrafterWrapper,
    PureHCIRCrafterAgent,
    make_crafter_env,
)


def test_crafter_constants_byte_verification() -> None:
    """Byte-verify action names and achievement names against upstream constants."""
    # 1. Action verification
    assert list(crafter_constants.actions) == ACTION_NAMES
    for i, act_name in enumerate(crafter_constants.actions):
        assert CrafterAction(i).name.lower() == act_name

    # 2. Achievement verification
    for ach in crafter_constants.achievements:
        assert ach in [a.value for a in CrafterAchievement]

    # 3. Material IDs in upstream World
    # Upstream World sets materials as [None] + constants.materials
    expected_materials = [
        "water",
        "grass",
        "stone",
        "path",
        "sand",
        "tree",
        "lava",
        "coal",
        "iron",
        "diamond",
        "table",
        "furnace",
    ]
    assert list(crafter_constants.materials) == expected_materials

    # Upstream SemanticView defines:
    # 0: None -> EMPTY
    # 1..12: materials
    # 13..18: [Player, Cow, Zombie, Skeleton, Arrow, Plant]
    mapping = NativeCrafterWrapper.CRAFTER_NATIVE_TO_OBJECT

    assert mapping[0] == int(CrafterObject.EMPTY)
    assert mapping[1] == int(CrafterObject.WATER)
    assert mapping[2] == int(CrafterObject.GRASS)
    assert mapping[3] == int(CrafterObject.STONE)
    assert mapping[4] == int(CrafterObject.PATH)
    assert mapping[5] == int(CrafterObject.SAND)
    assert mapping[6] == int(CrafterObject.TREE)
    assert mapping[7] == int(CrafterObject.LAVA)
    assert mapping[8] == int(CrafterObject.COAL)
    assert mapping[9] == int(CrafterObject.IRON)
    assert mapping[10] == int(CrafterObject.DIAMOND)
    assert mapping[11] == int(CrafterObject.CRAFTING_TABLE)
    assert mapping[12] == int(CrafterObject.FURNACE)
    assert mapping[13] == int(CrafterObject.PLAYER)
    assert mapping[14] == int(CrafterObject.COW)
    assert mapping[15] == int(CrafterObject.ZOMBIE)
    assert mapping[16] == int(CrafterObject.SKELETON)
    assert mapping[17] == int(CrafterObject.ARROW)
    assert mapping[18] == int(CrafterObject.PLANT)


def test_native_crafter_wrapper_lifecycle_and_vitals() -> None:
    """Verify NativeCrafterWrapper initializes correctly and live-tracks engine vitals."""
    env = make_crafter_env(seed=42, prefer_native=True)
    assert isinstance(env, NativeCrafterWrapper)
    assert env.is_native is True

    obs, info = env.reset(seed=42)

    # Verify initial vitals
    assert obs.vitals.health == 9
    assert obs.vitals.food == 9
    assert obs.vitals.drink == 9
    assert obs.vitals.energy == 9

    # Verify grid dimensions and player coordinate resolution
    assert len(obs.semantic_grid) == 64
    assert len(obs.semantic_grid[0]) == 64
    px, py = obs.player_pos
    assert obs.semantic_grid[py][px] == int(CrafterObject.PLAYER)

    # Step repeatedly and verify live vitals tracking
    # Upstream thirst drains every 20 non-sleep steps
    for _ in range(25):
        obs, r, term, trunc, info = env.step(CrafterAction.NOOP)
        if term or trunc:
            break

    # In native crafter, 25 steps drains thirst: drink drops from 9 to 8
    native_drink = env.native_env._player.inventory["drink"]
    assert obs.vitals.drink == native_drink
    assert obs.vitals.drink <= 8, f"Vitals drink should have drained from 9, got {obs.vitals.drink}"


def test_crafter_perception_adapter_raw_dict_ingest() -> None:
    """Verify CrafterPerceptionAdapter accepts raw Gymnasium/native observation dictionaries."""
    native_env = crafter.Env(seed=77)
    native_env.reset()
    raw_obs, reward, done, info = native_env.step(0)

    adapter = CrafterPerceptionAdapter()
    # Ingest info dictionary directly
    graph = adapter.ingest_observation(info)

    agent_node = graph.get_node("agent")
    assert agent_node is not None
    assert agent_node.properties["health"] == 9
    assert agent_node.properties["drink"] == 9
    assert "inventory" in agent_node.properties

    # Check that nearby resource entities were tracked
    nodes = [n for n in graph.all_nodes() if n.id.startswith("ent_")]
    assert len(nodes) > 0


def test_native_crafter_wood_and_table_milestones() -> None:
    """Execute PureHCIRCrafterAgent directly on real upstream crafter.Env."""
    env = make_crafter_env(seed=100, prefer_native=True)
    obs, _ = env.reset(seed=100)
    agent = PureHCIRCrafterAgent()
    goal = CrafterGoal(target_achievement=CrafterAchievement.PLACE_TABLE)

    for _ in range(50):
        act = agent.select_action(obs, goal)
        obs, r, term, trunc, info = env.step(act)
        if CrafterAchievement.PLACE_TABLE in obs.achievements:
            break
        if term or trunc:
            break

    assert CrafterAchievement.COLLECT_WOOD in obs.achievements
    assert CrafterAchievement.PLACE_TABLE in obs.achievements
