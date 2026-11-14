from __future__ import annotations

"""Unit Tests for Round 7 Surviving Cognitive Architecture Enhancements:

Track 3: Frontopolar BA10 Recursive And-Or Subgoal Tree & Regret-Driven Backtracking
Track 4: Native Fast 2D Euclidean Distance Transform (EDT) & Obstacle Margins
"""

import math

import pytest

from hbllm.hcir.world import (
    AutonomousEpistemicEngine,
    EuclideanDistanceTransformResult,
    RecursiveSubgoalTree,
    SubgoalNodeType,
    SubgoalTreeStatus,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all
from hbllm.hcir.world.experimental.spatial_distance_transform import (
    SpatialDistanceTransformEngine,
)

# ─────────────────────────────────────────────────────────────────────────────
# Track 3: Frontopolar BA10 Recursive And-Or Subgoal Tree
# ─────────────────────────────────────────────────────────────────────────────


def test_subgoal_tree_and_satisfaction_propagation() -> None:
    """AND node satisfies only when all children are satisfied."""
    tree = RecursiveSubgoalTree(
        root_description="Mission Master", root_type=SubgoalNodeType.AND_NODE
    )

    child1 = tree.add_node(
        parent_id=tree.root_id,
        node_type=SubgoalNodeType.LEAF_ACTION,
        description="Pick up Red Key",
        target_coord=(1, 2),
    )
    child2 = tree.add_node(
        parent_id=tree.root_id,
        node_type=SubgoalNodeType.LEAF_ACTION,
        description="Unlock Red Door",
        target_coord=(4, 5),
    )

    frontier = tree.get_active_frontier()
    assert len(frontier) == 1
    assert frontier[0].node_id == child1.node_id

    # Satisfy child 1 -> root is still ACTIVE (not satisfied yet)
    tree.mark_satisfied(child1.node_id)
    assert child1.status == SubgoalTreeStatus.SATISFIED
    assert tree.nodes[tree.root_id].status == SubgoalTreeStatus.ACTIVE

    # Frontier now advances to child 2
    next_frontier = tree.get_active_frontier()
    assert len(next_frontier) == 1
    assert next_frontier[0].node_id == child2.node_id

    # Satisfy child 2 -> root becomes SATISFIED
    tree.mark_satisfied(child2.node_id)
    assert child2.status == SubgoalTreeStatus.SATISFIED
    assert tree.nodes[tree.root_id].status == SubgoalTreeStatus.SATISFIED


def test_subgoal_tree_or_satisfaction_prunes_siblings() -> None:
    """OR node satisfies when first alternative succeeds and prunes remaining siblings."""
    tree = RecursiveSubgoalTree(
        root_description="Navigation Master", root_type=SubgoalNodeType.OR_NODE
    )

    path_a = tree.add_node(
        parent_id=tree.root_id,
        node_type=SubgoalNodeType.LEAF_ACTION,
        description="North Bridge Path",
        target_coord=(0, 5),
    )
    path_b = tree.add_node(
        parent_id=tree.root_id,
        node_type=SubgoalNodeType.LEAF_ACTION,
        description="East Tunnel Path",
        target_coord=(5, 9),
    )

    # Satisfy path A
    tree.mark_satisfied(path_a.node_id)

    assert tree.nodes[tree.root_id].status == SubgoalTreeStatus.SATISFIED
    assert path_a.status == SubgoalTreeStatus.SATISFIED
    assert path_b.status == SubgoalTreeStatus.PRUNED


def test_subgoal_tree_lcva_regret_backtracking() -> None:
    """Lowest Common Viable Ancestor (LCVA) backtracks to nearest OR ancestor on branch failure."""
    # Root (AND)
    #  ├── Subgoal 1: Get Access (OR)
    #  │     ├── Option A: Hack Terminal (LEAF) -> FAILS
    #  │     └── Option B: Pick Lock (LEAF)    -> LCVA switches here!
    #  └── Subgoal 2: Enter Vault (LEAF)
    tree = RecursiveSubgoalTree(root_description="Infiltrate", root_type=SubgoalNodeType.AND_NODE)

    or_subgoal = tree.add_node(
        parent_id=tree.root_id,
        node_type=SubgoalNodeType.OR_NODE,
        description="Get Access",
    )
    opt_a = tree.add_node(
        parent_id=or_subgoal.node_id,
        node_type=SubgoalNodeType.LEAF_ACTION,
        description="Hack Terminal",
        cost_estimate=2.0,
    )
    opt_b = tree.add_node(
        parent_id=or_subgoal.node_id,
        node_type=SubgoalNodeType.LEAF_ACTION,
        description="Pick Lock",
        cost_estimate=3.0,
    )

    # Frontier initially picks Option A
    frontier = tree.get_active_frontier()
    assert len(frontier) == 1
    assert frontier[0].node_id == opt_a.node_id

    # Option A fails due to dead-end/regret
    fallback_node = tree.mark_failed(opt_a.node_id, regret=0.85)

    assert opt_a.status == SubgoalTreeStatus.FAILED
    assert opt_a.regret_score == 0.85
    # LCVA activates Option B
    assert fallback_node is not None
    assert fallback_node.node_id == opt_b.node_id
    assert fallback_node.status == SubgoalTreeStatus.ACTIVE

    # Now if Option B also fails, no more alternatives exist under this OR node -> Root fails
    fatal_fallback = tree.mark_failed(opt_b.node_id, regret=0.90)
    assert fatal_fallback is None
    assert tree.nodes[tree.root_id].status == SubgoalTreeStatus.FAILED

    # Check serialization
    pred = tree.to_causal_predicate()
    assert pred.variable == "frontopolar:subgoal_tree"
    assert pred.value == "FAILED"

    ops = tree.to_transaction_operations()
    assert len(ops) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# Track 4: Native Fast 2D Euclidean Distance Transform (EDT)
# ─────────────────────────────────────────────────────────────────────────────


def test_spatial_distance_transform_exact_distances() -> None:
    """Exact Euclidean distance values from obstacles match Pythagorean geometry."""
    engine = SpatialDistanceTransformEngine()
    grid_shape = (10, 10)
    # Single obstacle placed at (5, 5)
    obstacles = {(5, 5)}

    result = engine.compute_edt(grid_shape=grid_shape, obstacles=obstacles)

    assert isinstance(result, EuclideanDistanceTransformResult)
    assert result.grid_shape == (10, 10)
    assert result.obstacle_count == 1

    # Exact distances
    assert result.get_distance(5, 5) == pytest.approx(0.0)
    assert result.get_distance(5, 8) == pytest.approx(3.0)
    assert result.get_distance(2, 5) == pytest.approx(3.0)
    assert result.get_distance(2, 9) == pytest.approx(5.0)  # sqrt(3^2 + 4^2) = 5.0
    assert result.get_distance(1, 2) == pytest.approx(5.0)  # sqrt(4^2 + 3^2) = 5.0

    # Safety margin predicate
    assert result.is_safe(2, 9, min_margin=4.0) is True
    assert result.is_safe(5, 6, min_margin=2.0) is False

    # Safest cell is in one of the far corners (0,0), (0,9), (9,0), or (9,9)
    assert result.safest_cell in {(0, 0), (0, 9), (9, 0), (9, 9)}
    assert result.max_distance >= math.hypot(4, 4)


def test_spatial_distance_transform_gradient_and_repulsive_force() -> None:
    """Spatial gradients point away from obstacles towards clearance, generating repulsive vectors."""
    engine = SpatialDistanceTransformEngine()
    obstacles = {(5, 5)}
    result = engine.compute_edt(grid_shape=(11, 11), obstacles=obstacles)

    # At (5, 6), right of obstacle: clearance increases as column index increases
    # Gradient in column dimension must be positive (pointing to the right away from obstacle)
    assert result.gradient_c[5][6] > 0.0

    # At (5, 4), left of obstacle: clearance increases as column index decreases
    # Gradient in column dimension must be negative (pointing to the left away from obstacle)
    assert result.gradient_c[5][4] < 0.0

    # Repulsive force within buffer radius (1.5)
    force_r, force_c = engine.get_repulsive_force(5, 6, result, buffer_radius=2.0)
    assert force_c > 0.0  # pushes right away from obstacle at (5,5)
    assert math.hypot(force_r, force_c) > 0.0

    # Outside buffer radius (dist = 5.0 at (0, 5)), repulsive force is zero
    far_force_r, far_force_c = engine.get_repulsive_force(0, 5, result, buffer_radius=2.0)
    assert far_force_r == 0.0
    assert far_force_c == 0.0


def test_spatial_distance_transform_engine_integration() -> None:
    """Epistemic engine integrates distance transform and computes clearance field."""
    engine = AutonomousEpistemicEngine()
    _restore_all(engine)
    assert hasattr(engine, "distance_transform_engine")

    barriers = {(2, 2), (2, 3), (2, 4)}
    result = engine.compute_spatial_distance_field(obstacles=barriers, grid_shape=(10, 10))

    assert result.obstacle_count == 3
    assert result.get_distance(2, 2) == 0.0
    assert result.get_distance(8, 8) > 5.0
    assert any(
        "spatial:distance_transform" in p.variable for p in engine.symbolic_theory.causal_predicates
    )
