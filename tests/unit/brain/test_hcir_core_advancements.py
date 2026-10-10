"""Unit tests for the 4 HCIR Core Cognitive Architecture Enhancements:
1. Causal Invariant Locking (Anti-Sussman Precondition Protection) in EmbodiedCausalOperator.
2. Hierarchical Goal Stacks with Resumption Checkpoints in TieredWorkspace.
3. Barrier-Aware Epistemic Frontier Scoring in EpistemicFrontierDetector.
4. Causal Action Prioritization and MCTS Branch Pruning in CounterfactualPlanner.
"""

from __future__ import annotations

import numpy as np

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
)
from hbllm.brain.reasoning.operators.embodied_causal import EmbodiedCausalOperator
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import (
    ActionModality,
    ActionNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)
from hbllm.hcir.subgoal_decomposer import EpistemicFrontierDetector
from hbllm.hcir.workspace_tiers import TieredWorkspace


def test_anti_sussman_causal_invariant_locking() -> None:
    """Test that EmbodiedCausalOperator locks already-satisfied goal conditions and rejects negating actions."""
    # Agent is currently holding the key
    agent_node = PhysicalEntityNode(
        id="agent",
        name="agent",
        properties={"position": (2, 2), "held_object": "key_gold", "reach_distance": 1.5},
    )
    key_node = PhysicalEntityNode(
        id="key_gold",
        name="key_gold",
        properties={"position": (2, 2), "is_carried": True},
    )
    rock_node = PhysicalEntityNode(
        id="rock",
        name="rock",
        properties={"position": (2, 3)},
    )
    exit_node = PhysicalEntityNode(
        id="door_exit",
        name="door_exit",
        properties={"position": (5, 5), "is_locked": True},
    )

    # Goal requires BOTH holding key and unlocking door
    goal_node = GoalNode(
        id="goal_escape",
        description="Escape the room",
        properties={"target_conditions": ["holds(key_gold)", "unlocked(door_exit)"]},
    )

    # Candidate 1: Pick up a rock (destroys holds(key_gold) by replacing held object)
    act_pick_rock = ActionNode(
        id="act_pick_rock",
        intent="pickup(rock)",
        modality=ActionModality.MANIPULATION,
        requirements=["adjacent_to(agent, rock)"],
        produces=["holds(rock)"],
        properties={"drops": "key_gold"},
    )

    # Candidate 2: Unlock door (requires holds(key_gold) and produces unlocked(door_exit))
    act_unlock = ActionNode(
        id="act_unlock",
        intent="unlock(door_exit)",
        modality=ActionModality.MANIPULATION,
        requirements=["holds(key_gold)"],
        produces=["unlocked(door_exit)"],
        properties={},
    )

    edge_holds = HCIREdge(
        sources=["agent"],
        targets=["key_gold"],
        edge_type=HCIREdgeType.DEPENDS_ON,
    )

    view = FrozenGraphView(
        nodes={
            "agent": agent_node,
            "key_gold": key_node,
            "rock": rock_node,
            "door_exit": exit_node,
            "goal_escape": goal_node,
            "act_pick_rock": act_pick_rock,
            "act_unlock": act_unlock,
        },
        edges={edge_holds.id: edge_holds},
    )

    operator = EmbodiedCausalOperator()
    problem = ReasoningProblem(
        problem_type=ProblemType.PLANNING,
        description="Resolve goal escape without dropping key",
        goal_node_ids=("goal_escape",),
    )
    context = CognitiveContext(problem=problem, graph_view=view)

    # 1. Verify that holds(key_gold) is satisfied
    assert operator.is_condition_satisfied("holds(key_gold)", view) is True

    # 2. Verify violates_invariant detects that act_pick_rock destroys holds(key_gold)
    violates, reason = operator.violates_invariant(act_pick_rock, {"holds(key_gold)"}, view)
    assert violates is True
    assert "drops protected held object" in (reason or "")

    # 3. Execute operator: it must select act_unlock, not act_pick_rock
    result = operator.execute(problem, context)
    assert result.status.value == "success"
    assert result.conclusions["best_action"] == "unlock(door_exit)"


def test_hierarchical_goal_stack_push_pop_resumption() -> None:
    """Test that TieredWorkspace correctly pushes and pops interruption checkpoints without state loss."""
    tiered = TieredWorkspace()

    # Create parent task frame for macro-crafting
    macro_frame = tiered.create_task_frame("craft_iron_pickaxe")
    assert macro_frame.is_active is True
    assert tiered.has_active_interruptions is False

    # Simulate emergency survival interrupt (thirst / low vital drink)
    interrupt_frame, checkpoint = tiered.push_interruption(
        parent_frame_id=macro_frame.frame_id,
        interrupt_goal_id="emergency_drink",
        target_conditions=["has(drink, 1)"],
        unsatisfied_conditions=["has(drink, 1)"],
        in_flight_action="act_approach_water",
        step_index=125,
        context_data={"prior_wood": 2, "prior_stone": 4},
    )

    assert tiered.has_active_interruptions is True
    assert len(tiered.interruption_stack) == 1
    assert interrupt_frame.is_active is True
    assert interrupt_frame.goal_id == "emergency_drink"

    # Query active checkpoint
    active_ckpt = tiered.get_active_checkpoint("craft_iron_pickaxe")
    assert active_ckpt is not None
    assert active_ckpt.step_index == 125
    assert active_ckpt.parent_goal_id == "craft_iron_pickaxe"
    assert active_ckpt.context_data["prior_stone"] == 4

    # Resolve and pop the interrupt
    popped_ckpt = tiered.pop_interruption(interrupt_frame.frame_id, reason="completed")
    assert popped_ckpt is not None
    assert popped_ckpt.checkpoint_id == checkpoint.checkpoint_id
    assert interrupt_frame.is_active is False  # interrupt frame is closed
    assert tiered.has_active_interruptions is False


def test_barrier_aware_epistemic_frontier_scoring() -> None:
    """Test that EpistemicFrontierDetector discounts wall barriers and respects physical accessibility."""
    grid_shape = (10, 10)
    avatar_pos = (2, 2)

    # Everything is unobserved except a 5x5 room around avatar
    unobserved_mask = np.ones(grid_shape, dtype=bool)
    for r in range(1, 4):
        for c in range(1, 4):
            unobserved_mask[r, c] = False

    # Solid wall at row 3 (blocking South), except door at (3, 2)
    wall_cells = {(3, 0), (3, 1), (3, 3), (3, 4), (3, 5)}

    # Detect frontiers with barrier awareness
    frontiers = EpistemicFrontierDetector.detect_frontiers(
        avatar_pos=avatar_pos,
        unobserved_mask=unobserved_mask,
        barrier_cells=wall_cells,
        grid_shape=grid_shape,
    )

    assert len(frontiers) > 0

    # The frontier cells must be reachable and not inside wall_cells
    for (fr, fc), score in frontiers:
        assert (fr, fc) not in wall_cells
        assert score > 0.0


def test_counterfactual_mcts_causal_prioritization() -> None:
    """Test that CounterfactualPlanner prioritizes causal actions advancing goal preconditions."""
    goal = GoalNode(
        id="goal_craft_table",
        description="Craft and place crafting table",
        properties={"target_conditions": ["has(table)"]},
    )

    act_idle = ActionNode(
        id="act_idle",
        intent="noop",
        produces=[],
    )
    act_turn_left = ActionNode(
        id="act_turn_left",
        intent="turn_left",
        produces=["facing(west)"],
    )
    act_craft_table = ActionNode(
        id="act_craft_table",
        intent="craft_table",
        produces=["has(table)"],
    )

    candidates = [act_idle, act_turn_left, act_craft_table]

    # Rank actions using the causal filter
    ranked = CounterfactualPlanner._rank_and_filter_causal_actions(
        candidates, goal, causal_pruning=True
    )

    # act_craft_table must be ranked first because it produces has(table)
    assert len(ranked) == 3
    assert ranked[0].id == "act_craft_table"
