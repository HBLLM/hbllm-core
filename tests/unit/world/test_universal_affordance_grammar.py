"""Unit tests for Universal Declarative Affordance Grammar and Unified Induction."""

from __future__ import annotations

from hbllm.hcir.graph import GoalNode
from hbllm.hcir.spatial_planner import HCIRSpatialEntityPlanner
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.causal.affordances import AffordanceType
from hbllm.hcir.world.causal.theory import HCIRSymbolicWorldTheory
from hbllm.hcir.world.percept import TerminalSignal
from hbllm.hcir.world.predictors.physics import PhysicsPredictor


def test_affordance_type_enum() -> None:
    """Verify AffordanceType enum values correspond to universal physical roles."""
    assert AffordanceType.IMPASSABLE == "barrier"
    assert AffordanceType.PASSABLE == "walkable"
    assert AffordanceType.TELIC == "goal"
    assert AffordanceType.CANDIDATE_TELIC == "candidate_goal"
    assert AffordanceType.PUSHABLE == "cargo"
    assert AffordanceType.RECEPTACLE == "receptacle"
    assert AffordanceType.ACTUATABLE == "actuator"
    assert AffordanceType.HAZARD == "hazard"


def test_unified_induce_affordance() -> None:
    """Verify induce_affordance induces correct causal predicates and sets."""
    theory = HCIRSymbolicWorldTheory()

    # Induce impassable (barrier)
    assert theory.induce_affordance(5, AffordanceType.IMPASSABLE, evidence="collision")
    assert 5 in theory.barrier_features
    assert theory.has_affordance(5, AffordanceType.IMPASSABLE)
    assert theory.is_barrier(5)

    # Induce passable (walkable) on another feature
    assert theory.induce_affordance(0, AffordanceType.PASSABLE, evidence="traversal")
    assert 0 in theory.walkable_features
    assert theory.has_affordance(0, AffordanceType.PASSABLE)
    assert theory.is_walkable(0)

    # Induce telic (goal) with terminal reward evidence
    assert theory.induce_affordance(8, AffordanceType.TELIC, evidence=TerminalSignal.SUCCESS)
    assert 8 in theory.goal_features
    assert theory.has_affordance(8, AffordanceType.TELIC)
    assert theory.is_goal(8)

    # Induce pushable (cargo)
    assert theory.induce_affordance(3, AffordanceType.PUSHABLE, evidence="contact_displacement")
    assert 3 in theory.cargo_features
    assert theory.has_affordance(3, AffordanceType.PUSHABLE)
    assert theory.is_cargo(3)

    # Induce actuatable (actuator / switch)
    assert theory.induce_affordance(9, AffordanceType.ACTUATABLE, evidence="button_press")
    assert 9 in theory.actuator_features
    assert theory.has_affordance(9, AffordanceType.ACTUATABLE)
    assert theory.is_actuator(9)


def test_affordance_string_alias_interoperability() -> None:
    """Verify induce_affordance and has_affordance accept both string and enum values."""
    theory = HCIRSymbolicWorldTheory()

    theory.induce_affordance(12, "barrier", evidence="collision")
    assert theory.has_affordance(12, "barrier")
    assert theory.has_affordance(12, AffordanceType.IMPASSABLE)

    theory.induce_affordance(14, "walkable", evidence="traversal")
    assert theory.has_affordance(14, "walkable")
    assert theory.has_affordance(14, AffordanceType.PASSABLE)


def test_backward_compatibility_induce_wrappers() -> None:
    """Verify legacy induce_* wrappers continue to function identically via induce_affordance."""
    theory = HCIRSymbolicWorldTheory()

    assert theory.induce_barrier(7, evidence="collision")
    assert 7 in theory.barrier_features

    assert theory.induce_walkable(1, evidence="traversal")
    assert 1 in theory.walkable_features

    assert theory.induce_cargo(4, evidence="interaction")
    assert 4 in theory.cargo_features

    assert theory.induce_goal(6, evidence=TerminalSignal.SUCCESS)
    assert 6 in theory.goal_features


def test_physics_predictor_blocked_nodes_alias() -> None:
    """Verify PhysicsPredictor.compute_geodesic_path works identically with blocked_nodes."""
    start = (0, 0)
    goal = (4, 4)
    blocked = {(2, 0), (2, 1), (2, 2), (2, 3)}  # Wall with opening at (2, 4)
    grid_shape = (10, 10)

    # Path using blocked_nodes keyword argument
    path_blocked = PhysicsPredictor.compute_geodesic_path(
        start=start,
        goal=goal,
        blocked_nodes=blocked,
        grid_shape=grid_shape,
    )
    assert path_blocked
    assert path_blocked[0] == start
    assert path_blocked[-1] == goal
    # Assert path routes through the opening at column 4
    assert any(r == 2 and c == 4 for r, c in path_blocked)
    assert not any((r, c) in blocked for r, c in path_blocked)

    # Path using legacy barrier_cells keyword argument
    path_legacy = PhysicsPredictor.compute_geodesic_path(
        start=start,
        goal=goal,
        barrier_cells=blocked,
        grid_shape=grid_shape,
    )
    assert path_blocked == path_legacy


def test_topological_cut_set_blocked_nodes_alias() -> None:
    """Verify TopologicalCutSetAnalyzer accepts blocked_nodes."""
    start = (0, 0)
    goal = (5, 5)
    # Impassable dividing wall across all columns
    blocked = {(2, c) for c in range(10)}
    grid_shape = (10, 10)

    cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
        start=start,
        goal=goal,
        blocked_nodes=blocked,
        grid_shape=grid_shape,
    )
    assert cut_res.is_partitioned is True


def test_subgoal_decomposer_blocked_nodes_alias() -> None:
    """Verify HierarchicalGoalDecomposer accepts blocked_nodes."""
    workspace = HCIRWorkspaceState()
    decomposer = HierarchicalGoalDecomposer()

    primary_goal = GoalNode(
        id="goal_primary",
        description="Reach target",
        properties={"target_position": (8, 8)},
        priority=1.0,
    )
    avatar_pos = (0, 0)
    # Complete impenetrable wall blocking target
    blocked = {(4, c) for c in range(10)}
    grid_shape = (10, 10)

    # Candidate switch/milestone on the avatar's side of the wall
    candidate = {
        "id": "switch_1",
        "position": (1, 1),
        "feature_id": 9,
        "description": "Activate switch",
    }

    subgoal = decomposer.decompose_goal(
        workspace=workspace,
        primary_goal=primary_goal,
        avatar_pos=avatar_pos,
        blocked_nodes=blocked,
        grid_shape=grid_shape,
        candidate_subgoals=[candidate],
    )

    assert subgoal.id == "subgoal_switch_1"
    assert subgoal.properties.get("target_position") == (1, 1)


def test_spatial_planner_blocked_nodes_alias() -> None:
    """Verify HCIRSpatialEntityPlanner accepts blocked_nodes and impassable_nodes."""
    planner = HCIRSpatialEntityPlanner()
    start = (0, 0)
    goal = (3, 3)
    blocked = {(1, 0), (1, 1), (1, 2)}
    grid_shape = (5, 5)

    path = planner.compute_safe_path(
        start=start,
        goal=goal,
        blocked_nodes=blocked,
        grid_shape=grid_shape,
    )
    assert path
    assert path[0] == start
    assert path[-1] == goal
    assert not any((r, c) in blocked for r, c in path)


def test_vital_resource_affordance_and_induction() -> None:
    """Verify vital resource affordance induction and querying."""
    theory = HCIRSymbolicWorldTheory()
    assert theory.induce_vital_resource(15, evidence="timer_boost")
    assert 15 in theory.vital_resource_features
    assert theory.has_affordance(15, AffordanceType.HOMEOTROPIC)
    assert theory.has_affordance(15, "vital_resource")


def test_action_affordance_requires_contact_and_conditional_property() -> None:
    """Verify canonical ActionAffordance.requires_contact and ConditionalAffordance.required_property."""
    from hbllm.hcir.world.causal.affordances import ActionAffordance, ConditionalAffordance
    from hbllm.hcir.world.causal.predicates import CausalPredicate

    aff = ActionAffordance(action_id=5, requires_contact=True)
    assert aff.action_id == 5
    assert aff.requires_contact is True

    cond = ConditionalAffordance(
        feature=8,
        condition=CausalPredicate(variable="avatar_state", operator="==", value="color_match"),
        required_property="color_match",
    )
    assert cond.feature == 8
    assert cond.required_property == "color_match"
