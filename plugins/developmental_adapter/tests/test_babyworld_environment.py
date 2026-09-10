"""Unit tests for BabyWorldEnvironment physics, occlusion, and interventions."""

from __future__ import annotations

from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.types import BabyActionType


def test_environment_reset_and_observation():
    env = BabyWorldEnvironment(seed=42)
    obs = env.reset("confounded_train_world")

    assert obs.step_index == 0
    assert len(obs.vision) >= 4
    assert "obj_red_ball" in env.objects
    assert "obj_blue_heavy_block" in env.objects or "obj_blue_block" in env.objects


def test_physics_push_mass_threshold():
    env = BabyWorldEnvironment(seed=42)
    env.reset("confounded_train_world")

    # Push light object (mass=1.2 < 5.0) -> Should move
    init_pos_light = (
        env.objects["obj_red_ball"].position.x,
        env.objects["obj_red_ball"].position.y,
    )
    _, _, _, cons_light = env.step(action=BabyActionType.PUSH, target_id="obj_red_ball")
    assert cons_light["moved"] is True
    new_pos_light = (env.objects["obj_red_ball"].position.x, env.objects["obj_red_ball"].position.y)
    assert new_pos_light != init_pos_light

    # Push heavy object (mass=12.0 >= 5.0) -> Should NOT move
    init_pos_heavy = (
        env.objects["obj_blue_ball"].position.x,
        env.objects["obj_blue_ball"].position.y,
    )
    _, _, _, cons_heavy = env.step(action=BabyActionType.PUSH, target_id="obj_blue_ball")
    assert cons_heavy["moved"] is False
    new_pos_heavy = (
        env.objects["obj_blue_ball"].position.x,
        env.objects["obj_blue_ball"].position.y,
    )
    assert new_pos_heavy == init_pos_heavy


def test_occlusion_and_object_permanence():
    env = BabyWorldEnvironment(seed=42)
    obs = env.reset("occlusion_permanence_world")

    # The hidden ball is behind the occluder screen
    assert "obj_hidden_ball" in obs.occluded_entity_ids
    # Hidden ball should not be in visible percepts
    visible_ids = [p["percept_id"] for p in obs.vision]
    assert "obj_hidden_ball" not in visible_ids
    assert "obj_occluder_screen" in visible_ids


def test_interventional_state_saving_and_rollback():
    env = BabyWorldEnvironment(seed=42)
    env.reset("confounded_train_world")

    # Snapshot
    snapshot_idx = env.save_state()
    initial_x = env.objects["obj_red_ball"].position.x

    # Mutate via action
    env.step(action=BabyActionType.PUSH, target_id="obj_red_ball")
    assert env.objects["obj_red_ball"].position.x != initial_x

    # Rollback
    env.restore_state(snapshot_idx)
    assert env.objects["obj_red_ball"].position.x == initial_x
