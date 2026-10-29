from __future__ import annotations

"""Comprehensive Unit Tests for Round 7 Cognitive Architecture Enhancements:

Track 1: Anterior Insula Interoceptive Allostasis & Energy Budgeting
Track 2: Claustrum Multimodal Salience Conductor & Perceptual Binding
Track 3: Frontopolar BA10 Recursive And-Or Subgoal Tree & Regret-Driven Backtracking
Track 4: Native Fast 2D Euclidean Distance Transform (EDT) & Obstacle Margins
"""

import math

import pytest

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world import (
    AllostaticRegime,
    AutonomousEpistemicEngine,
    ClaustrumSalienceConductor,
    EuclideanDistanceTransformResult,
    InteroceptiveAllostasisCortex,
    RecursiveSubgoalTree,
    SpatialDistanceTransformEngine,
    SubgoalNodeType,
    SubgoalTreeStatus,
    UnifiedPerceptualGestalt,
)

# ─────────────────────────────────────────────────────────────────────────────
# Track 1: Anterior Insula Interoceptive Allostasis & Energy Budgeting
# ─────────────────────────────────────────────────────────────────────────────


def test_allostasis_initial_optimal_state() -> None:
    """Anterior insula initializes in OPTIMAL regime with full energy and permits deliberation."""
    cortex = InteroceptiveAllostasisCortex(max_energy=100.0, initial_energy=100.0)
    state = cortex.get_state()

    assert state.energy_level == 100.0
    assert state.regime == AllostaticRegime.OPTIMAL
    assert state.is_exhausted is False
    assert state.fatigue_index == 0.0
    assert state.deliberation_multiplier == 1.0

    permitted, reason = cortex.should_permit_deliberation(estimated_cost=5.0)
    assert permitted is True
    assert "permitted" in reason.lower()

    pred = state.to_causal_predicate()
    assert pred.variable == "allostasis:interoceptive_state"
    assert pred.value == "OPTIMAL"

    ops = state.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "interoceptive_state"


def test_allostasis_fatigue_and_exhaustion_transitions() -> None:
    """Energy consumption induces regime shifts from OPTIMAL to FATIGUED to EXHAUSTED."""
    cortex = InteroceptiveAllostasisCortex(
        max_energy=100.0,
        initial_energy=100.0,
        exhaustion_threshold=30.0,
    )

    # Consume energy down to FATIGUED (e.g., 50.0)
    cortex.consume_energy("mental_simulation", scale=9.0)  # ~45 cost + basal
    state = cortex.get_state()
    assert state.regime == AllostaticRegime.FATIGUED
    assert state.is_exhausted is False
    assert 30.0 <= state.energy_level < 60.0

    # Deliberation still permitted if budget allows
    permitted, _ = cortex.should_permit_deliberation(estimated_cost=5.0)
    assert permitted is True

    # Consume energy further to EXHAUSTED (< 30.0)
    cortex.consume_energy("mental_simulation", scale=6.0)
    exhausted_state = cortex.get_state()
    assert exhausted_state.regime == AllostaticRegime.EXHAUSTED
    assert exhausted_state.is_exhausted is True
    assert exhausted_state.energy_level < 30.0

    # Forward deliberation is now actively inhibited by insula
    permitted_now, reason = cortex.should_permit_deliberation(estimated_cost=5.0)
    assert permitted_now is False
    assert "insular inhibition" in reason.lower()

    # Utility penalty scales down expensive actions under exhaustion
    utilities = {"deep_rollout": 10.0, "quick_macro": 4.0}
    adjusted = cortex.apply_allostatic_cost_penalty(utilities, is_deliberative_action=True)
    assert adjusted["deep_rollout"] < 10.0
    assert adjusted["deep_rollout"] == pytest.approx(10.0 * exhausted_state.deliberation_multiplier)


def test_allostasis_replenishment_and_engine_integration() -> None:
    """Engine correctly updates allostasis and replenishes energetic reserves."""
    engine = AutonomousEpistemicEngine()
    assert hasattr(engine, "allostasis_cortex")

    # Consume energy via engine delegate
    state_after = engine.consume_allostatic_energy("counterfactual_rollout", scale=2.0)
    assert state_after.energy_level < 100.0
    assert engine.symbolic_theory is not None
    assert any(
        "allostasis:interoceptive_state" in p.variable
        for p in engine.symbolic_theory.causal_predicates
    )

    # Replenish energy
    replenished = engine.allostasis_cortex.replenish_energy(50.0)
    assert replenished.energy_level == 100.0
    assert replenished.regime == AllostaticRegime.OPTIMAL


# ─────────────────────────────────────────────────────────────────────────────
# Track 2: Claustrum Multimodal Salience Conductor & Perceptual Binding
# ─────────────────────────────────────────────────────────────────────────────


def test_claustrum_multimodal_binding_and_salience_peak() -> None:
    """Claustrum fuses visual, hazard, and goal streams into unified spatial salience."""
    conductor = ClaustrumSalienceConductor(
        visual_weight=0.2,
        hazard_weight=0.5,
        goal_weight=0.3,
    )

    entities = [
        SpatialEntity(
            id="box_1",
            role=EntityRole.MANIPULABLE,
            grid_pos=(2, 2),
            color=3,
        )
    ]
    hazards = {(8, 8): 1.0, (8, 7): 0.8}
    goals = [(2, 2)]

    gestalt = conductor.bind_modalities(
        grid_shape=(10, 10),
        visual_entities=entities,
        hazard_map=hazards,
        goal_coords=goals,
    )

    assert isinstance(gestalt, UnifiedPerceptualGestalt)
    assert gestalt.grid_shape == (10, 10)
    assert (8, 8) in gestalt.salience_map
    assert (2, 2) in gestalt.salience_map

    # Hazard at (8, 8) has high weight (0.5 * 1.0 = 0.5)
    # Goal at (2, 2) has (0.2 * 0.8) + (0.3 * 1.0) = 0.46
    assert gestalt.focal_salience_peak == (8, 8)
    assert gestalt.dominant_modality == "threat_avoidance"

    # Transaction & Causal Predicate Serialization
    pred = gestalt.to_causal_predicate()
    assert pred.variable == "claustrum:gestalt_bound"
    assert "synchrony" in pred.value

    ops = gestalt.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "unified_perceptual_gestalt"
    assert ops[0].node_data["dominant_modality"] == "threat_avoidance"


def test_claustrum_goal_dominance_and_engine_binding() -> None:
    """Engine integrates claustrum multimodal binding and detects goal-directed dominance."""
    engine = AutonomousEpistemicEngine()
    assert hasattr(engine, "claustrum_conductor")

    # Only goals present
    goals = [(3, 4)]
    gestalt = engine.bind_multimodal_percept(
        grid_shape=(10, 10),
        visual_entities=[],
        hazard_map={},
        goal_coords=goals,
    )

    assert gestalt.dominant_modality == "goal_directed"
    assert gestalt.focal_salience_peak == (3, 4)
    assert gestalt.salience_map[(3, 4)] > 0.0
    assert any(
        "claustrum:gestalt_bound" in p.variable for p in engine.symbolic_theory.causal_predicates
    )


# ─────────────────────────────────────────────────────────────────────────────
# Track 3: Frontopolar BA10 Recursive And-Or Subgoal Tree & Regret Backtracking
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
# Track 4: Native Fast 2D Euclidean Distance Transform (EDT) & Obstacle Margins
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
    assert hasattr(engine, "distance_transform_engine")

    barriers = {(2, 2), (2, 3), (2, 4)}
    result = engine.compute_spatial_distance_field(obstacles=barriers, grid_shape=(10, 10))

    assert result.obstacle_count == 3
    assert result.get_distance(2, 2) == 0.0
    assert result.get_distance(8, 8) > 5.0
    assert any(
        "spatial:distance_transform" in p.variable for p in engine.symbolic_theory.causal_predicates
    )
