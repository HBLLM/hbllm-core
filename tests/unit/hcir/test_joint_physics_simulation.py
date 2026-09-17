"""Unit tests for line deadlock and joint multi-entity simulation in PhysicsPredictor."""

from hbllm.hcir.world.predictors.physics import PhysicsPredictor


def test_is_line_deadlock_detection() -> None:
    """Verify that is_line_deadlock flags a box pressed against a flat continuous barrier without targets."""
    grid_shape = (10, 10)
    # Continuous flat wall along row 0 from col 1 to 8, with corners at col 0 and 9
    barrier_cells = {(0, c) for c in range(10)}
    barrier_cells.add((1, 0))  # corner block at (1, 0)
    barrier_cells.add((1, 9))  # corner block at (1, 9)

    # Box at (1, 5), pressed against the wall at row 0
    # No targets along this wall
    target_positions = {(8, 8)}

    is_deadlocked = PhysicsPredictor.is_line_deadlock(
        box_pos=(1, 5),
        barrier_cells=barrier_cells,
        target_positions=target_positions,
        grid_shape=grid_shape,
        step_size=1,
    )
    assert is_deadlocked is True

    # If there IS a target along that wall at (1, 3), it should NOT be deadlocked
    target_positions_with_wall = {(1, 3)}
    not_deadlocked = PhysicsPredictor.is_line_deadlock(
        box_pos=(1, 5),
        barrier_cells=barrier_cells,
        target_positions=target_positions_with_wall,
        grid_shape=grid_shape,
        step_size=1,
    )
    assert not_deadlocked is False

    # Verify frozenset compatibility
    frozen_deadlocked = PhysicsPredictor.is_line_deadlock(
        box_pos=(1, 5),
        barrier_cells=frozenset(barrier_cells),
        target_positions=frozenset(target_positions),
        grid_shape=grid_shape,
        step_size=1,
    )
    assert frozen_deadlocked is True

    frozen_corner = PhysicsPredictor.is_corner_deadlock(
        box_pos=(1, 1),
        barrier_cells=frozenset({(0, 1), (1, 0)}),
        target_positions=frozenset(target_positions),
        grid_shape=grid_shape,
        step_size=1,
    )
    assert frozen_corner is True


def test_simulate_joint_displacement_cascade() -> None:
    """Verify that simulate_joint_displacement handles multi-body pushing chains and barrier blockage."""
    grid_shape = (10, 10)
    avatar_pos = (5, 2)
    movable_entities = {
        "box1": (5, 3),
        "box2": (5, 4),
    }
    barrier_cells = {(5, 6)}  # Barrier at (5, 6)

    # 1. Pushing east: avatar at (5,2) steps into box1 at (5,3), pushing box1 to (5,4), and box2 to (5,5)
    new_avatar, new_entities, is_blocked = PhysicsPredictor.simulate_joint_displacement(
        avatar_pos=avatar_pos,
        action_delta=(0, 1),
        movable_entities=movable_entities,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
    )

    assert not is_blocked
    assert new_avatar == (5, 3)
    assert new_entities["box1"] == (5, 4)
    assert new_entities["box2"] == (5, 5)

    # 2. Pushing again east: now box2 at (5,5) would be pushed into barrier at (5,6) -> must be blocked!
    blocked_avatar, blocked_entities, is_blocked2 = PhysicsPredictor.simulate_joint_displacement(
        avatar_pos=new_avatar,
        action_delta=(0, 1),
        movable_entities=new_entities,
        barrier_cells=barrier_cells,
        grid_shape=grid_shape,
    )

    assert is_blocked2 is True
    # State must remain unchanged
    assert blocked_avatar == (5, 3)
    assert blocked_entities["box1"] == (5, 4)
    assert blocked_entities["box2"] == (5, 5)
