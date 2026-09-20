"""Unit tests for PhysicsPredictor spatial kinematics and topological mechanics."""

from __future__ import annotations

from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot


def test_spatial_translation_unobstructed() -> None:
    """Verify avatar moves from (2, 2) to (1, 2) with action 'MOVE_UP'."""
    snapshot = WorldStateSnapshot(
        world_id="spatial_w1",
        variables={
            "grid_shape": [8, 8],
            "spatial_entities": {
                "avatar": {
                    "position": [2, 2],
                    "is_avatar": True,
                    "movable": True,
                    "passable": False,
                },
                "goal": {
                    "position": [0, 2],
                    "is_goal": True,
                    "movable": False,
                    "passable": True,
                },
            },
        },
    )
    predictor = PhysicsPredictor()
    pred_state, conf = predictor.predict_state(snapshot, "MOVE_UP")

    assert conf >= 0.90
    outcome = pred_state["spatial_outcome"]
    assert outcome["success"] is True
    assert outcome["collision"] is False
    assert outcome["avatar_position"] == (1, 2)
    assert outcome["progress"] > 0.0


def test_spatial_barrier_collision() -> None:
    """Verify translation halts when walking into an obstacle barrier."""
    snapshot = WorldStateSnapshot(
        world_id="spatial_w2",
        variables={
            "grid_shape": [8, 8],
            "barrier_cells": [[1, 2]],
            "spatial_entities": {
                "avatar": {
                    "position": [2, 2],
                    "is_avatar": True,
                    "movable": True,
                    "passable": False,
                },
                "goal": {
                    "position": [0, 2],
                    "is_goal": True,
                    "movable": False,
                    "passable": True,
                },
            },
        },
    )
    predictor = PhysicsPredictor()
    pred_state, conf = predictor.predict_state(snapshot, "MOVE_UP")

    outcome = pred_state["spatial_outcome"]
    assert outcome["collision"] is True
    assert outcome["collision_type"] == "BARRIER_COLLISION"
    assert outcome["avatar_position"] == (2, 2)
    assert conf < 0.30


def test_spatial_boundary_collision() -> None:
    """Verify translation halts when walking into grid boundaries."""
    snapshot = WorldStateSnapshot(
        world_id="spatial_w3",
        variables={
            "grid_shape": [8, 8],
            "spatial_entities": {
                "avatar": {
                    "position": [0, 2],
                    "is_avatar": True,
                    "movable": True,
                    "passable": False,
                },
            },
        },
    )
    predictor = PhysicsPredictor()
    pred_state, conf = predictor.predict_state(snapshot, "MOVE_UP")

    outcome = pred_state["spatial_outcome"]
    assert outcome["collision"] is True
    assert outcome["collision_type"] == "BOUNDARY_COLLISION"
    assert outcome["avatar_position"] == (0, 2)


def test_tandem_push_propagation() -> None:
    """Verify avatar pushes a pushable box when destination behind box is clear."""
    snapshot = WorldStateSnapshot(
        world_id="spatial_w4",
        variables={
            "grid_shape": [8, 8],
            "spatial_entities": {
                "avatar": {
                    "position": [3, 2],
                    "is_avatar": True,
                    "movable": True,
                    "passable": False,
                },
                "box_1": {
                    "position": [2, 2],
                    "movable": True,
                    "passable": False,
                    "affordances": ["PUSHABLE"],
                },
                "goal": {
                    "position": [0, 2],
                    "is_goal": True,
                    "movable": False,
                    "passable": True,
                },
            },
        },
    )
    predictor = PhysicsPredictor()
    pred_state, conf = predictor.predict_state(snapshot, "MOVE_UP")

    outcome = pred_state["spatial_outcome"]
    assert outcome["success"] is True
    assert outcome["avatar_position"] == (2, 2)
    assert outcome["pushed_entities"] == ["box_1"]
    assert outcome["updated_entities"]["box_1"]["position"] == (1, 2)
    assert outcome["progress"] > 0.0


def test_tandem_push_into_corner_deadlock() -> None:
    """Verify push is detected as a corner deadlock when box ends up against two walls."""
    snapshot = WorldStateSnapshot(
        world_id="spatial_w5",
        variables={
            "grid_shape": [8, 8],
            # Wall on top (row 0) and left (col 0)
            "barrier_cells": [[0, c] for c in range(8)] + [[r, 0] for r in range(8)],
            "target_positions": [[5, 5]],  # target is far away
            "spatial_entities": {
                "avatar": {
                    "position": [2, 1],
                    "is_avatar": True,
                    "movable": True,
                    "passable": False,
                },
                # Box at (1, 1), pushing UP moves it to (0, 1) which is a barrier!
                # Or avatar at (1, 2), pushing LEFT moves box from (1, 1) to (1, 0)
                # Let's test box at (2, 1) and avatar at (3, 1), pushing UP moves box to (1, 1)
                # At (1, 1), UP (0, 1) is a wall and LEFT (1, 0) is a wall -> corner deadlock!
                "box_1": {
                    "position": [2, 1],
                    "movable": True,
                    "passable": False,
                    "affordances": ["PUSHABLE"],
                },
            },
        },
    )
    # Avatar at (3, 1)
    snapshot.variables["spatial_entities"]["avatar"]["position"] = [3, 1]

    predictor = PhysicsPredictor()
    pred_state, conf = predictor.predict_state(snapshot, "MOVE_UP")

    outcome = pred_state["spatial_outcome"]
    assert outcome["pushed_entities"] == ["box_1"]
    assert outcome["updated_entities"]["box_1"]["position"] == (1, 1)
    assert outcome["deadlock"] is True
    assert "box_1" in outcome["deadlocked_entities"]
    assert conf <= 0.05  # Severe penalty


def test_compute_geodesic_path() -> None:
    """Verify geodesic path correctly circumvents barriers."""
    barrier_cells = {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5)}
    path = PhysicsPredictor.compute_geodesic_path(
        start=(4, 4),
        goal=(2, 4),
        barrier_cells=barrier_cells,
        grid_shape=(8, 8),
    )
    assert len(path) > 1
    assert path[0] == (4, 4)
    assert path[-1] == (2, 4)
    for p in path:
        assert p not in barrier_cells


def test_find_solitaire_jump_sequence() -> None:
    """Verify solitaire jump sequence reduces pegs to target count."""
    pegs = {(0, 0), (0, 1), (0, 2)}
    holes = {(0, 0), (0, 1), (0, 2), (0, 3)}
    jumps = PhysicsPredictor.find_solitaire_jump_sequence(
        pegs=pegs,
        valid_holes=holes,
        step_delta=1,
        target_peg_count=2,
    )
    assert jumps is not None
    assert len(jumps) == 1
    assert jumps[0] == ((0, 1), (0, 2), (0, 3))


def test_simulate_telescopic_step() -> None:
    """Verify simulate_telescopic_step extends, retracts, and translates along rails."""
    base = (5, 5)
    axis = (0, 1)

    # Extension
    new_base, new_len = PhysicsPredictor.simulate_telescopic_step(
        base_pos=base, length=2, axis=axis, action_dir=(0, 1), max_length=5
    )
    assert new_base == base
    assert new_len == 3

    # Retraction
    new_base, new_len = PhysicsPredictor.simulate_telescopic_step(
        base_pos=base, length=3, axis=axis, action_dir=(0, -1), min_length=1
    )
    assert new_base == base
    assert new_len == 2

    # Translation along rails
    rails = {(4, 5), (5, 5), (6, 5)}
    new_base, new_len = PhysicsPredictor.simulate_telescopic_step(
        base_pos=base, length=2, axis=axis, action_dir=(1, 0), rails=rails
    )
    assert new_base == (6, 5)
    assert new_len == 2
