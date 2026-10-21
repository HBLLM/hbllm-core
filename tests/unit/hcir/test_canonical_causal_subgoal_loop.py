"""Canonical Neuro-Symbolic Causal & Subgoal Loop Unit Tests.

Validates the full cognitive architecture invariants from AGENTS.md:
1. Effector Action Locus Binding: Action coordinates from spatial interventions are correctly
   bound in Pearl's do(a) causal attribution without losing click/actuation coordinates.
2. Tool-Barrier Resonance Induction: Possessing a held tool while a barrier clears induces
   formal tool-barrier resonance in ExtendedBodySchema and PrefrontalWorkingMemory.
3. Hierarchical Subgoal Resolution (BA10 Subgoal Stack): When reachability to primary goals
   is blocked by barrier bottlenecks, the planner queries CausalInductionCortex to formulate
   explicit FrontopolarSubgoals targeting the prerequisite triggers/tools.
4. Domain-Agnostic Invariant: Zero level-specific or game-specific constants, 100% LLM-independent.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_planner import MentalSimulationPlanner
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAffordance


def test_spatial_effector_causal_binding_and_remote_attribution():
    """Verify that clicking a distal trigger binds the action locus to distal barrier clearance."""
    engine = AutonomousEpistemicEngine()

    # 10x10 world: Avatar at (2, 2), Trigger button at (2, 8), Locked Door at (6, 5), Goal at (8, 5)
    grid_0 = np.zeros((10, 10), dtype=int)
    grid_0[2, 2] = 1  # Avatar
    grid_0[2, 8] = 5  # Remote switch / button
    grid_0[6, 5] = 4  # Barrier / door
    grid_0[8, 5] = 2  # Goal

    engine.avatar_feature = 1
    engine.avatar_pos = (2, 2)
    engine.symbolic_theory.induce_barrier(4)
    engine.symbolic_theory.induce_goal(2)

    avail_actions = [1, 2, 3, 4, 6]  # 6 = CLICK_CELL

    # Agent executes click on switch at (r=2, c=8) -> {'x': 8, 'y': 2}
    engine.decide(grid_0, avail_actions)
    engine.last_action = 6
    engine.last_action_data = {"x": 8, "y": 2}

    # Frame 1: Door at (6, 5) opens (becomes background 0)
    grid_1 = grid_0.copy()
    grid_1[6, 5] = 0

    engine.assimilate_feedback(
        grid_1,
        avail_actions,
        action=6,
        action_data={"x": 8, "y": 2},
    )

    # Verify vmPFC distal causal link was formed between (2, 8) and (6, 5)
    affordance = engine.causal_cortex.causal_attributor.get_trigger_for_barrier(
        (6, 5), barrier_feature=4
    )
    assert affordance is not None, (
        "Remote causal attributor failed to discover trigger-barrier link"
    )
    assert affordance.trigger_pos == (2, 8), (
        f"Expected trigger at (2, 8), got {affordance.trigger_pos}"
    )
    assert affordance.confidence >= 0.50


def test_tool_barrier_resonance_learning():
    """Verify that possessing a key when a door unlocks induces tool-barrier resonance."""
    engine = AutonomousEpistemicEngine()

    # Avatar picks up tool (feat=7) at (3, 3)
    engine.avatar_feature = 1
    engine.avatar_pos = (3, 3)
    engine.working_memory.acquire_item(feature_id=7, role="RESOURCE", step=1, position=(3, 3))
    assert engine.working_memory.is_holding(7)
    assert engine.body_schema.is_holding(7)

    # Initial frame with door at (5, 5) of feat=8
    grid_pre = np.zeros((10, 10), dtype=int)
    grid_pre[3, 3] = 1
    grid_pre[5, 5] = 8
    engine.symbolic_theory.induce_barrier(8)
    engine.prev_grid = grid_pre.copy()

    # Step: Door at (5, 5) unlocks
    grid_post = grid_pre.copy()
    grid_post[5, 5] = 0

    engine.assimilate_feedback(
        grid_post,
        [1, 2, 3, 4],
        action=1,
    )

    # Verify tool-barrier resonance was learned
    assert 8 in engine.body_schema.tool_barrier_affinities.get(7, set())
    assert engine.body_schema.is_barrier_permeable(8)


def test_hierarchical_subgoal_decomposition_with_causal_prerequisite():
    """Verify BA10 subgoal stack pushes prerequisite unlock subgoal when goal is blocked."""
    engine = AutonomousEpistemicEngine()

    # 7x7 grid: Avatar at (1, 1), Goal at (5, 5), Horizontal Wall at row 3 with door at (3, 3)
    # Switch at (1, 5) that unlocks door at (3, 3)
    grid = np.zeros((7, 7), dtype=int)
    grid[1, 1] = 1  # Avatar
    grid[5, 5] = 2  # Goal
    grid[1, 5] = 9  # Switch / button
    for c in range(7):
        grid[3, c] = 4  # Wall
    grid[3, 3] = 4  # Door

    engine.avatar_feature = 1
    engine.avatar_pos = (1, 1)
    engine.symbolic_theory.induce_walkable(0)
    engine.symbolic_theory.induce_barrier(4)
    engine.symbolic_theory.induce_goal(2)

    # Register known directional movements
    engine.action_dynamics = {
        1: type(
            "Dyn",
            (),
            {
                "is_displacement_action": lambda s: True,
                "get_displacement": lambda s: (-1, 0),
                "delta_r": -1,
                "delta_c": 0,
                "confidence": 1.0,
            },
        )(),
        2: type(
            "Dyn",
            (),
            {
                "is_displacement_action": lambda s: True,
                "get_displacement": lambda s: (1, 0),
                "delta_r": 1,
                "delta_c": 0,
                "confidence": 1.0,
            },
        )(),
        3: type(
            "Dyn",
            (),
            {
                "is_displacement_action": lambda s: True,
                "get_displacement": lambda s: (0, -1),
                "delta_r": 0,
                "delta_c": -1,
                "confidence": 1.0,
            },
        )(),
        4: type(
            "Dyn",
            (),
            {
                "is_displacement_action": lambda s: True,
                "get_displacement": lambda s: (0, 1),
                "delta_r": 0,
                "delta_c": 1,
                "confidence": 1.0,
            },
        )(),
    }

    # Feed causal knowledge that switch at (1, 5) clears door at (3, 3)
    engine.causal_cortex.causal_attributor.co_occurrence_evidence[((1, 5), (3, 3))] = 2
    aff = RemoteCausalAffordance(
        trigger_pos=(1, 5),
        trigger_feature=9,
        remote_pos=(3, 3),
        original_remote_feature=4,
        mutated_remote_feature=0,
        activation_count=2,
        confidence=0.98,
    )
    engine.causal_cortex.causal_attributor.remote_actuators[(3, 3)].append(aff)
    engine.causal_cortex.causal_attributor.trigger_registry[(1, 5)].append(aff)

    # 1. Forward simulation should find complete plan through trigger
    plan = MentalSimulationPlanner.simulate_in_mind(
        engine,
        grid,
        available_actions=[1, 2, 3, 4],
    )
    assert plan is not None, "Mental simulation should find plan via trigger"
    assert len(plan) > 0
    # Must visit trigger (1, 5) and then goal (5, 5)
    visited_positions = [step.predicted_avatar_pos for step in plan]
    assert (1, 5) in visited_positions
    assert (5, 5) in visited_positions

    # 2. Hierarchical Subgoal Decomposition directly registers Frontopolar Subgoal
    movable = [
        (1, -1, 0),
        (2, 1, 0),
        (3, 0, -1),
        (4, 0, 1),
    ]
    subgoal_plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
        engine=engine,
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        start_pos=(1, 1),
        goals={(5, 5)},
        static_barriers={(3, c) for c in range(7)},
        mutation_triggers={(1, 5): {(3, 3)}},
        movable_actions=movable,
    )
    assert subgoal_plan is not None
    assert engine.subgoal_stack.has_pending_goals
    top_sg = engine.subgoal_stack.current_subgoal
    assert top_sg is not None
    assert top_sg.subgoal_type == SubgoalType.UNLOCK_REMOTE_MECHANISM
    assert top_sg.target_destination == (1, 5)
