"""Unit and regression tests for ARC3SpatialCognitiveAgent (thin blackbox adapter)."""

import numpy as np

from hbllm.drivers import BaseDriver, DriverManager
from hbllm.drivers.base import DriverAction, DriverFeedback
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.world.morphology import ShapeArchetype
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from plugins.arc_agi_adapter.arc_driver import ArcadeDriver
from plugins.arc_agi_adapter.arc_memory import HCIRCrossGameMemory
from plugins.arc_agi_adapter.arc_spatial_agent import (
    ARC3SpatialCognitiveAgent,
)


def test_driver_management_registration() -> None:
    """Verify ArcadeDriver registers with DriverManager."""
    manager = DriverManager()
    driver = ArcadeDriver()
    assert isinstance(driver, BaseDriver)
    manager.register(driver)
    retrieved = manager.get_driver("arcade_driver")
    assert retrieved is driver
    assert manager.get_driver("arcade_driver").name == "arcade_driver"


def test_agent_creates_blackbox() -> None:
    """Verify ARC3SpatialCognitiveAgent wraps a CognitiveBlackbox."""
    agent = ARC3SpatialCognitiveAgent()
    assert isinstance(agent.blackbox, CognitiveBlackbox)


def test_agent_plan_returns_valid_action() -> None:
    """Verify plan_next_action returns an action from the available list."""
    agent = ARC3SpatialCognitiveAgent()
    grid = np.zeros((16, 16), dtype=np.uint8)
    avail = [1, 2, 3, 4, 5]
    act, conf = agent.plan_next_action(grid, avail)
    assert act in avail or act == 0
    assert 0.0 <= conf <= 1.0


def test_agent_update_does_not_crash() -> None:
    """Verify update_causal_dynamics handles grid diffs without error."""
    agent = ARC3SpatialCognitiveAgent()
    prev = np.zeros((16, 16), dtype=np.uint8)
    curr = np.zeros((16, 16), dtype=np.uint8)
    curr[5, 5] = 3  # Some change
    agent.update_causal_dynamics(1, prev, curr)


def test_agent_update_mismatched_shapes() -> None:
    """Verify update_causal_dynamics gracefully handles mismatched grid shapes."""
    agent = ARC3SpatialCognitiveAgent()
    prev = np.zeros((16, 16), dtype=np.uint8)
    curr = np.zeros((20, 20), dtype=np.uint8)
    agent.update_causal_dynamics(1, prev, curr)  # Should not crash


def test_agent_reset_episode() -> None:
    """Verify reset_episode creates fresh state."""
    agent = ARC3SpatialCognitiveAgent()
    # Do a plan to increment step count
    grid = np.zeros((16, 16), dtype=np.uint8)
    agent.plan_next_action(grid, [1, 2, 3, 4])
    agent.reset_episode()
    # After reset, blackbox state is fresh
    state = agent.blackbox.get_state("default")
    assert state.step_count == 0


def test_cross_game_memory_export_import() -> None:
    """Verify cross-game memory serialization round-trip."""
    memory = HCIRCrossGameMemory()
    memory.avatar_color = 14
    memory.step_size = 4
    memory.action_5_affordance = "PICKUP_DROP"
    memory.learned_item_colors.add(4)
    memory.learned_receptacle_colors.add(9)
    memory.learned_walkable_colors.update([0, 1])
    memory.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-4, delta_c=0, confidence=0.95, probes_tested=5
    )

    exported = memory.export_dict()
    assert exported["avatar_color"] == 14
    assert exported["step_size"] == 4

    new_memory = HCIRCrossGameMemory()
    new_memory.import_dict(exported)
    assert new_memory.avatar_color == 14
    assert new_memory.step_size == 4
    assert new_memory.action_5_affordance == "PICKUP_DROP"
    assert 4 in new_memory.learned_item_colors
    assert 1 in new_memory.action_models
    assert new_memory.action_models[1].confidence >= 0.9


def test_agent_record_episode_outcome() -> None:
    """Verify record_episode_outcome writes to cross-game memory."""
    memory = HCIRCrossGameMemory()
    agent = ARC3SpatialCognitiveAgent(shared_memory=memory)
    agent.record_episode_outcome(completed=True)
    assert len(memory.successful_episodes) == 1
    assert memory.successful_episodes[0]["score"] == 1.0


def test_shape_archetype_translation_and_rotational_orbits() -> None:
    """Verify ShapeArchetype canonicalization and 90/180/270 degree rotation orbit matching."""
    # Horizontal bar of 3 pixels at (5, 5), (5, 6), (5, 7)
    h_bar_coords = {(5, 5), (5, 6), (5, 7)}
    arch_h = ShapeArchetype.from_coords(h_bar_coords)
    assert arch_h.height == 1
    assert arch_h.width == 3
    assert arch_h.area == 3
    assert arch_h.canonical_id == ((0, 0), (0, 1), (0, 2))

    # Translated horizontal bar at (20, 10), (20, 11), (20, 12)
    h_bar_translated = {(20, 10), (20, 11), (20, 12)}
    arch_h_trans = ShapeArchetype.from_coords(h_bar_translated)
    assert arch_h == arch_h_trans  # Exactly equal canonical archetype

    # Vertical bar of 3 pixels (90 degree rotation of horizontal bar)
    v_bar_coords = {(8, 10), (9, 10), (10, 10)}
    arch_v = ShapeArchetype.from_coords(v_bar_coords)
    assert arch_v.height == 3
    assert arch_v.width == 1
    assert arch_h.is_rotation_of(arch_v)
    assert arch_v.is_rotation_of(arch_h)


def test_cognitive_blackbox_observe_decide_update_loop() -> None:
    """Verify the full observe→decide→update loop works end-to-end."""
    blackbox = CognitiveBlackbox()

    available = [DriverAction(action_id=a) for a in [1, 2, 3, 4]]
    action = blackbox.decide(available, source_id="test")
    assert isinstance(action, DriverAction)
    assert action.action_id in [0, 1, 2, 3, 4]

    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"observed_delta": [0, 1]},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert action.action_id in state.action_models


def test_cognitive_blackbox_learns_obstacles_from_feedback() -> None:
    """Verify blackbox learns obstacle features from collision feedback."""
    blackbox = CognitiveBlackbox()
    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"collision_feature": 5},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 5 in state.learned_obstacle_features


def test_cognitive_blackbox_learns_traversable_from_feedback() -> None:
    """Verify blackbox learns traversable features from successful movement."""
    blackbox = CognitiveBlackbox()
    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"traversed_feature": 3},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 3 in state.learned_traversable_features


def test_cognitive_blackbox_reset_preserves_memory() -> None:
    """Verify blackbox reset with retain_memory keeps learned knowledge."""
    blackbox = CognitiveBlackbox()

    # Teach it something
    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"collision_feature": 7, "observed_delta": [-2, 0]},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 7 in state.learned_obstacle_features
    assert 1 in state.action_models

    # Reset with memory retention
    blackbox.reset(source_id="test", retain_memory=True)

    state = blackbox.get_state("test")
    assert 7 in state.learned_obstacle_features
    assert 1 in state.action_models
    assert state.step_count == 0  # Step count reset


def test_cognitive_blackbox_reset_clears_memory() -> None:
    """Verify blackbox reset without retain_memory clears everything."""
    blackbox = CognitiveBlackbox()

    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"collision_feature": 7},
    )
    blackbox.update(action, feedback, source_id="test")

    blackbox.reset(source_id="test", retain_memory=False)

    state = blackbox.get_state("test")
    assert len(state.learned_obstacle_features) == 0
    assert len(state.action_models) == 0
