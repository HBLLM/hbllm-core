"""Unit tests for general biological brain faculties in HBLLM Core.

Tests:
1. Saccadic Visual Attention & Foveal Saliency.
2. Spatiotemporal Hippocampal Hazard Tracking & Phase Precession.
3. Intuitive Physical Dynamics (2x2 Clump Deadlocks & Spatiotemporal Geodesic Pathfinding).
4. Prefrontal Working Memory & Hierarchical Subgoal Schemas.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.spatiotemporal_tracker import SpatiotemporalHazardTracker
from hbllm.perception.saccadic_attention import SaccadicAttentionSystem


def test_saccadic_visual_attention_foveation() -> None:
    """Verify SaccadicAttentionSystem identifies Gestalt centroids, contrast anomalies, and symmetry breaks."""
    system = SaccadicAttentionSystem(fovea_radius=1)

    # 10x10 background 0 grid with a rare focal item (color 3) at (5, 5)
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 3

    saliency = system.compute_saliency_map(grid, background_feature=0)
    assert saliency.shape == (10, 10)
    assert saliency[5, 5] > saliency[0, 0]

    fixations = system.extract_fixations(grid, background_feature=0, top_k=5)
    assert len(fixations) >= 1
    # The top fixation should be the focal entity at (5, 5)
    assert (fixations[0].r, fixations[0].c) == (5, 5)
    assert fixations[0].feature_id == 3


def test_saccadic_attention_symmetry_break() -> None:
    """Verify SaccadicAttentionSystem attends to local bilateral symmetry violations."""
    system = SaccadicAttentionSystem()

    # Bilaterally symmetric grid except for a single asymmetrical pixel at (3, 7)
    grid = np.zeros((8, 8), dtype=int)
    grid[2:6, 1] = 1
    grid[2:6, 6] = 1  # symmetric reflection
    grid[3, 7] = 2  # symmetry break!

    fixations = system.extract_fixations(grid, background_feature=0, top_k=8)
    fixation_coords = {(f.r, f.c) for f in fixations}
    assert (3, 7) in fixation_coords


def test_spatiotemporal_hazard_tracking_periodicity() -> None:
    """Verify SpatiotemporalHazardTracker infers cyclic periods T and predicts future lethal states."""
    tracker = SpatiotemporalHazardTracker(history_len=12, max_period=6)
    tracker.register_lethal_feature(9)  # 9 is lethal laser/hazard

    # Simulate 8 timesteps of an oscillating hazard at (3, 3) with period T=2
    # Even steps: 0 (safe background), Odd steps: 9 (lethal laser)
    grid_safe = np.zeros((6, 6), dtype=int)
    grid_haz = np.zeros((6, 6), dtype=int)
    grid_haz[3, 3] = 9

    for step in range(8):
        current_g = grid_haz if (step % 2 == 1) else grid_safe
        tracker.record_frame(step, current_g, background_feature=0)

    assert (3, 3) in tracker.periodic_cells
    phase_model = tracker.periodic_cells[(3, 3)]
    assert phase_model.period == 2

    # Predict hazard states into future relative steps
    # If last step was 7 (odd -> hazardous), relative +1 (step 8, even) is safe, +2 (step 9, odd) is hazardous
    is_haz_dt1 = tracker.is_hazard_at(3, 3, future_relative_step=1, background_feature=0)
    is_haz_dt2 = tracker.is_hazard_at(3, 3, future_relative_step=2, background_feature=0)
    assert is_haz_dt1 != is_haz_dt2


def test_spatiotemporal_pathfinding_with_temporal_waiting() -> None:
    """Verify PhysicsPredictor.compute_spatiotemporal_path utilizes waiting to bypass periodic hazards."""
    # 1D-like corridor: Start at (0, 0), Goal at (0, 2).
    # Chokepoint at (0, 1) is hazardous at t=1, but safe at t=2.
    hazard_schedule = {
        1: {(0, 1)},  # blocked at t=1
        2: set(),  # open at t=2
    }

    path = PhysicsPredictor.compute_spatiotemporal_path(
        start=(0, 0),
        goal=(0, 2),
        barrier_cells=set(),
        grid_shape=(3, 3),
        step_size=1,
        hazard_schedule=hazard_schedule,
        period=2,
        allow_wait=True,
    )
    assert path is not None
    # Path should hesitate/wait or navigate safely through (0, 1) at t=2
    # Format of path: [(0, 0, 0), (0, 0, 1) [wait], (0, 1, 2) [move], (0, 2, 3) [goal]]
    coords = [(step[0], step[1]) for step in path]
    assert coords[-1] == (0, 2)
    # Ensure it did not occupy (0, 1) at t=1
    for r, c, t in path:
        if (r, c) == (0, 1):
            assert t != 1


def test_intuitive_physics_2x2_and_tunnel_deadlocks() -> None:
    """Verify PhysicsPredictor detects 2x2 block clumps and dead-end tunnel traps."""
    # 2x2 clump: 2 blocks and 2 static barriers forming a solid block
    # (1, 1), (1, 2), (2, 1), (2, 2)
    barrier_cells = {(1, 1), (1, 2)}
    block_cells = {(2, 1), (2, 2)}
    goals = {(5, 5)}

    is_dead = PhysicsPredictor.is_2x2_deadlock(
        entity_pos=(2, 1),
        barrier_cells=barrier_cells,
        block_cells=block_cells,
        target_positions=goals,
        grid_shape=(10, 10),
    )
    assert is_dead is True

    # 1-wide dead-end tunnel: (1, 1) surrounded on 3 sides by barriers
    tunnel_barriers = {(0, 1), (1, 0), (2, 1)}  # Up, Left, Down blocked
    is_tunnel_dead = PhysicsPredictor.is_tunnel_deadlock(
        entity_pos=(1, 1),
        barrier_cells=tunnel_barriers,
        target_positions=goals,
        grid_shape=(10, 10),
    )
    assert is_tunnel_dead is True


def test_prefrontal_working_memory_schema_formulation() -> None:
    """Verify PrefrontalWorkingMemory tracks latent item possession and decomposes tool schemas."""
    wm = PrefrontalWorkingMemory()

    # Acquire key entity into working memory
    wm.acquire_item(
        feature_id=4,
        role="resource",
        step=5,
        position=(2, 2),
    )
    assert wm.is_holding(4)
    assert not wm.is_holding(9)

    # Formulate tool use schema
    avatar_pos = (1, 1)
    tools = [(2, 2, 4)]
    barriers = [(3, 4, 8)]
    goals = [(3, 5)]

    schema = wm.formulate_tool_use_schema(
        avatar_pos=avatar_pos,
        tools=tools,
        barriers=barriers,
        goals=goals,
    )
    assert schema is not None
    # Since key 4 is already held, Stage 1 is unlock barrier, Stage 2 is reach goal
    assert schema.current_stage is not None
    assert schema.current_stage.stage_id in ("ACQUIRE_TOOL", "UNLOCK_BARRIER")

    # Advancing schema
    next_stage = schema.advance()
    assert next_stage is not None

    # Expending item
    assert wm.expend_item(4) is True
    assert not wm.is_holding(4)
