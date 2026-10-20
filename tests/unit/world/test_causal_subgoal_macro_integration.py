"""Neuro-Symbolic Unit Tests for Option 2 (Active Causal Affordance Discovery) and Option 3 (BA10 Subgoal Stack).

Validates:
1. Active Causal Induction of Tool-Barrier Incompatibility on failed intervention.
2. Active Causal Induction of Distal Actuation when Interacting with allocentric triggers.
3. Brodmann Area 10 (aPFC) Recursive Subgoal Decomposition (nested goals, prerequisites).
4. Synchronized advancement and popping of the frontopolar subgoal stack during plan execution.
5. Incompatibility-guided pruning of ineffective tools during cognitive branching.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
from hbllm.hcir.world.macro_actions import (
    InteractWith,
    MacroPlanSynthesizer,
    PickUpCargo,
    UseToolOnBarrier,
    attribute_macro_action_outcome,
)
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel


@pytest.fixture
def calibrated_engine() -> AutonomousEpistemicEngine:
    engine = AutonomousEpistemicEngine()
    engine.action_dynamics = {
        1: ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=1.0),
        2: ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=1.0),
        3: ActionDynamicsModel(action_id=3, delta_r=0, delta_c=-1, confidence=1.0),
        4: ActionDynamicsModel(action_id=4, delta_r=0, delta_c=1, confidence=1.0),
    }
    engine.avatar_pos = (1, 1)
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.bg_feature = 0
    engine.symbolic_theory.walkable_features.add(0)
    return engine


def test_tool_barrier_incompatibility_learning(calibrated_engine: AutonomousEpistemicEngine):
    """Option 2: When an acquired tool fails to clear a barrier, record causal incompatibility."""
    engine = calibrated_engine
    grid_pre = np.zeros((10, 10), dtype=int)
    grid_pre[1, 1] = 1  # Avatar
    grid_pre[5, 5] = 4  # Barrier feature 4
    engine.symbolic_theory.barrier_features.add(4)

    # Avatar holds tool 7
    engine.working_memory.body_schema.acquire_tool(7, step=1, role="key")
    assert engine.working_memory.body_schema.is_holding(7)

    # Use tool 7 on barrier 4 at (5, 5)
    action_op = UseToolOnBarrier(barrier_pos=(5, 5), tool_feature=7, barrier_feature=4)

    # Step: Action fails to clear the barrier (grid remains unchanged)
    attribute_macro_action_outcome(
        engine=engine,
        macro_action=action_op,
        prev_grid=grid_pre,
        curr_grid=grid_pre,
        is_success=False,
    )

    # Verify causal incompatibility recorded in symbolic theory
    assert 4 in engine.symbolic_theory.tool_barrier_incompatibilities.get(7, set())
    assert not engine.symbolic_theory.is_tool_compatible(7, 4)
    # Untested barrier 8 is optimistically compatible
    assert engine.symbolic_theory.is_tool_compatible(7, 8)

    # Verify transaction serialization
    tx_ops = engine.symbolic_theory.to_transaction_operations()
    incomp_nodes = [
        op
        for op in tx_ops
        if "causal_incompatibility_tool_7_barrier_4" in getattr(op, "node_id", "")
    ]
    assert len(incomp_nodes) == 1
    props = incomp_nodes[0].node_data.get("properties", {})
    assert props["tool_feature"] == 7
    assert props["barrier_feature"] == 4


def test_distal_actuation_learning_and_barrier_unblocking(
    calibrated_engine: AutonomousEpistemicEngine,
):
    """Option 2: Interacting with an allocentric switch distally clears barriers and records causal rule."""
    engine = calibrated_engine

    grid_0 = np.zeros((10, 10), dtype=int)
    grid_0[1, 1] = 1  # Avatar
    grid_0[2, 8] = 5  # Allocentric Switch
    grid_0[6, 5] = 4  # Obstructing Door Barrier
    engine.symbolic_theory.barrier_features.add(4)
    engine.learned_barriers.add((6, 5))

    # Avatar interacts with switch at (2, 8)
    interact_op = InteractWith(target_coord=(2, 8))

    # Grid transitions: Door at (6, 5) opens (becomes 0)
    grid_1 = grid_0.copy()
    grid_1[6, 5] = 0

    attribute_macro_action_outcome(
        engine=engine,
        macro_action=interact_op,
        prev_grid=grid_0,
        curr_grid=grid_1,
        is_success=True,
    )

    # 1. Obstructing barrier coordinate is unblocked
    assert (6, 5) not in engine.learned_barriers

    # 2. Distal actuation causal rule recorded
    assert 4 in engine.symbolic_theory.distal_actuations.get(5, set())
    assert (6, 5) in engine.symbolic_theory.distal_actuation_coords.get(5, set())

    # 3. Transaction serialization
    tx_ops = engine.symbolic_theory.to_transaction_operations()
    distal_nodes = [
        op for op in tx_ops if "causal_distal_actuation_trig_5_aff_4" in getattr(op, "node_id", "")
    ]
    assert len(distal_nodes) == 1
    d_props = distal_nodes[0].node_data.get("properties", {})
    assert d_props["trigger_feature"] == 5
    assert d_props["affected_feature"] == 4


def test_ba10_recursive_subgoal_decomposition_and_plan_synthesis(
    calibrated_engine: AutonomousEpistemicEngine,
):
    """Option 3: BA10 FrontopolarSubgoalStack decomposes goal blocked by door requiring tool."""
    engine = calibrated_engine

    # 10x10 world:
    # Avatar at (1, 1)
    # Key tool at (2, 2) (feat=7)
    # Gate at (5, 5) (feat=8)
    # Target Exit Goal at (8, 8) (feat=2)
    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1] = 1
    grid[2, 2] = 7
    grid[5, 5] = 8
    grid[8, 8] = 2

    engine.symbolic_theory.goal_features.add(2)
    engine.symbolic_theory.barrier_features.add(8)
    engine.symbolic_theory.cargo_features.add(7)

    # Ingest knowledge via MacroPlanSynthesizer
    objects = [
        {"id": "obj_avatar", "pos": (1, 1), "feature_id": 1, "role": "avatar"},
        {"id": "obj_key", "pos": (2, 2), "feature_id": 7, "role": "cargo"},
        {"id": "obj_gate", "pos": (5, 5), "feature_id": 8, "role": "receptacle"},
        {"id": "obj_goal", "pos": (8, 8), "feature_id": 2, "role": "goal"},
    ]
    MacroPlanSynthesizer.ingest_knowledge(engine, objects=objects)

    # 1. Synthesize plan triggers BA10 decomposition
    plan = MacroPlanSynthesizer.synthesize_plan(
        engine=engine,
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        objects=objects,
    )
    assert plan is not None
    assert not plan.is_empty()

    # 2. Verify BA10 Subgoal Stack was populated with 3 nested subgoals
    assert engine.subgoal_stack.has_pending_goals
    assert engine.subgoal_stack.depth == 3

    # Top of stack must be immediate tool acquisition
    tos = engine.subgoal_stack.current_subgoal
    assert tos is not None
    assert tos.subgoal_type == SubgoalType.ACQUIRE_TOOL
    assert tos.target_entity_pos == (2, 2)
    assert tos.required_feature == 7

    # 3. Macro plan begins with PickUpCargo for tool at (2, 2)
    cur_action = plan.current()
    assert isinstance(cur_action, PickUpCargo)
    assert cur_action.cargo_pos == (2, 2)


def test_ba10_synchronized_stack_popping_across_phases(
    calibrated_engine: AutonomousEpistemicEngine,
):
    """Option 3: Executing macro actions seamlessly pops and resumes parent subgoals."""
    engine = calibrated_engine

    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1] = 1
    grid[2, 2] = 7
    grid[5, 5] = 8
    grid[8, 8] = 2

    engine.symbolic_theory.goal_features.add(2)
    engine.symbolic_theory.barrier_features.add(8)
    engine.symbolic_theory.cargo_features.add(7)

    objects = [
        {"id": "obj_avatar", "pos": (1, 1), "feature_id": 1, "role": "avatar"},
        {"id": "obj_key", "pos": (2, 2), "feature_id": 7, "role": "cargo"},
        {"id": "obj_gate", "pos": (5, 5), "feature_id": 8, "role": "receptacle"},
        {"id": "obj_goal", "pos": (8, 8), "feature_id": 2, "role": "goal"},
    ]
    plan = MacroPlanSynthesizer.synthesize_plan(
        engine=engine,
        curr_grid=grid,
        available_actions=[1, 2, 3, 4],
        objects=objects,
    )
    assert plan is not None
    assert engine.subgoal_stack.depth == 3

    # Phase 1: Avatar reaches adjacent to (2, 2) and acquires tool 7
    engine.avatar_pos = (2, 1)
    engine.working_memory.body_schema.acquire_tool(7, step=1, role="key")

    # Advance plan: PickUpCargo is satisfied
    plan.advance_if_satisfied(engine=engine, curr_grid=grid)

    # Subgoal ACQUIRE_TOOL popped! Current subgoal is now CLEAR_OBSTACLE at (5, 5)
    assert engine.subgoal_stack.depth == 2
    cur_sg = engine.subgoal_stack.current_subgoal
    assert cur_sg is not None
    assert cur_sg.subgoal_type == SubgoalType.CLEAR_OBSTACLE
    assert cur_sg.target_destination == (5, 5)

    # Phase 2: Avatar moves adjacent to (5, 5) and clears gate
    engine.avatar_pos = (5, 4)
    grid_cleared = grid.copy()
    grid_cleared[5, 5] = 0  # Gate clears to background

    plan.advance_if_satisfied(engine=engine, curr_grid=grid_cleared, prev_grid=grid)

    # Subgoal CLEAR_OBSTACLE popped! Current subgoal is now master DELIVER_TO_GOAL at (8, 8)
    assert engine.subgoal_stack.depth == 1
    master_sg = engine.subgoal_stack.current_subgoal
    assert master_sg is not None
    assert master_sg.subgoal_type == SubgoalType.DELIVER_TO_GOAL
    assert master_sg.target_destination == (8, 8)

    # Phase 3: Avatar reaches (8, 8)
    engine.avatar_pos = (8, 8)
    plan.advance_if_satisfied(engine=engine, curr_grid=grid_cleared)

    # All subgoals popped and complete!
    assert engine.subgoal_stack.is_empty()
    assert plan.is_empty()


def test_incompatibility_prunes_ineffective_tool_during_decomposition(
    calibrated_engine: AutonomousEpistemicEngine,
):
    """Option 2 & 3: Causal incompatibility knowledge prunes the wrong tool during decomposition."""
    engine = calibrated_engine

    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1] = 1  # Avatar
    grid[2, 2] = 6  # Incompatible Key (feat 6)
    grid[2, 3] = 7  # Compatible Key (feat 7)
    grid[5, 5] = 8  # Gate (feat 8)
    grid[8, 8] = 2  # Goal

    engine.symbolic_theory.goal_features.add(2)
    engine.symbolic_theory.barrier_features.add(8)

    # Agent has previously learned that Tool 6 does NOT work on Barrier 8
    engine.symbolic_theory.record_tool_barrier_incompatibility(tool_feature=6, barrier_feature=8)
    assert not engine.symbolic_theory.is_tool_compatible(6, 8)

    # Decompose target goal (8, 8)
    tools = [
        type("Entity", (), {"grid_pos": (2, 2), "feature_id": 6})(),
        type("Entity", (), {"grid_pos": (2, 3), "feature_id": 7})(),
    ]
    receptacles = [type("Entity", (), {"grid_pos": (5, 5), "feature_id": 8})()]

    decomposed = engine.subgoal_stack.decompose_target_goal(
        target_goal=(8, 8),
        grid=grid,
        avatar_pos=(1, 1),
        tools=tools,
        receptacles_or_gates=receptacles,
        theory=engine.symbolic_theory,
    )

    # The chosen tool subgoal MUST be tool 7 at (2, 3), skipping incompatible tool 6!
    tool_sg = next((sg for sg in decomposed if sg.subgoal_type == SubgoalType.ACQUIRE_TOOL), None)
    assert tool_sg is not None
    assert tool_sg.required_feature == 7
    assert tool_sg.target_entity_pos == (2, 3)
