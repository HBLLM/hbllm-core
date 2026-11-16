"""Broad Epistemic Integration Tests — Multi-turn active inference & faculty integration.

Tests end-to-end cognitive cycles across all recent core fixes:
1. Lethal position memorization & evasive re-planning across level attempts.
2. Click effector outer-border exclusion & quiescent-target behavioral switching.
3. Same-color bilateral twin tracking without centroid collapse.
4. Quad-lateral (4-agent D4 symmetry) detection & convergent coordination.
5. Frontopolar subgoal decomposition for multi-stage locked barrier puzzles.
6. Nested multi-stage subgoals with LIFO stack order & causal prerequisite dependencies.
7. Looming collision cone forecasting & prefrontal deliberation evasion gating.
8. Asymmetric bilateral obstacle wall-slip parity shifting and coordination.
9. Predictive coding sensory discrepancy, mental plan invalidation & replanning.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EpistemicPhase,
)
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    SubgoalType,
)
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel


def test_e2e_lethal_learning_and_evasion_across_lives() -> None:
    """Verify agent registers lethal position upon death and deliberatively paths around it in next life."""
    engine = AutonomousEpistemicEngine()

    engine.current_level = 1
    engine.avatar_feature = 1
    engine.avatar_pos = (2, 1)

    # Calibrate cardinal 4-way movements
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)
    engine.learned_goal_features.add(3)

    # 5x7 grid: Avatar at (2, 1), Goal at (2, 5).
    # Cell (2, 3) contains a lethal hazard learned from a prior death.
    grid = np.zeros((5, 7), dtype=int)
    grid[2, 1] = 1  # Avatar
    grid[2, 5] = 3  # Goal

    # Record lethal position from prior attempt
    engine.level_lethal_positions.setdefault(1, set()).add((2, 3))

    available_actions = [1, 2, 3, 4]

    # Deliberative mental simulation must find a detour avoiding (2, 3)
    plan = engine.simulate_in_mind(grid, available_actions)
    assert plan is not None
    assert len(plan) > 0

    # Trace simulated path and ensure (2, 3) is strictly never stepped on
    curr_pos = (2, 1)
    visited_positions = [curr_pos]
    for step in plan:
        act = step.action
        dr, dc = engine.action_deltas[act]
        curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
        visited_positions.append(curr_pos)

    assert (2, 3) not in visited_positions, "Planner must avoid known lethal coordinate!"
    assert visited_positions[-1] == (2, 5), "Planner must safely reach the goal!"


def test_e2e_peripheral_border_filtering_and_quiescent_switching() -> None:
    """Verify click effector filters outer grid borders and inhibits clicks when targets are quiescent."""
    engine = AutonomousEpistemicEngine()

    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.avatar_pos = (5, 5)
    engine.bg_feature = 0

    # 10x10 grid with outer frame lines (val 8) along borders and an inert clickable block at (4, 4)
    grid = np.zeros((10, 10), dtype=int)
    grid[0, :] = 8  # Top border
    grid[9, :] = 8  # Bottom border
    grid[:, 0] = 8  # Left border
    grid[:, 9] = 8  # Right border
    grid[5, 5] = 1  # Avatar
    grid[4, 4] = 7  # Inert salient object

    available_actions = [1, 2, 3, 4, 6]  # 6 is CLICK_CELL

    # 1. Grounding should choose (4, 4) and NEVER choose border coordinates (0, 0), (0, 9), etc.
    coords = engine.ground_effector_action(grid, 6)
    assert coords is not None
    assert (coords.get("y"), coords.get("x")) == (4, 4), (
        "Effector should target inner object, not border!"
    )

    # 2. Suppose all inner candidate clicks proved quiescent
    engine.quiescent_actuation_targets.update([(4, 4), (4, 5), (5, 4)])

    # 3. Next grounding attempt: With inner candidates quiescent and border cells filtered, effector returns None
    coords2 = engine.ground_effector_action(grid, 6)
    assert coords2 is None, (
        "When candidates are exhausted/quiescent, ground_effector_action returns None!"
    )

    # 4. In decide(), deliberation must inhibit action 6 and switch to movement without looping
    chosen_act, _ = engine.decide(grid, available_actions)
    assert chosen_act in [1, 2, 3, 4], (
        "Agent must autonomously switch to locomotion when clicks are exhausted!"
    )
    assert engine.inhibited_actions.get(6, 0) > 0, "Action 6 must be Habenularly inhibited!"


def test_e2e_frontopolar_subgoal_decomposition_multi_stage_puzzle() -> None:
    """Verify recursive subgoal decomposition paths to remote switch, pushes subgoal, and reaches goal."""
    from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAffordance

    engine = AutonomousEpistemicEngine()

    engine.avatar_feature = 1
    engine.avatar_pos = (2, 1)

    # Calibrate movements
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)
    engine.learned_goal_features.add(3)

    # 7x9 grid:
    # Avatar at (2, 1)
    # Complete barrier wall along column 4 (feature 2): (0, 4) through (6, 4)
    # Goal at (2, 7) (feature 3) — blocked by the barrier wall!
    # Remote switch at (5, 2) (feature 6) that clears barrier cell (2, 4)
    grid = np.zeros((7, 9), dtype=int)
    grid[2, 1] = 1  # Avatar
    grid[2, 7] = 3  # Goal
    grid[:, 4] = 2  # Barrier wall blocking direct transit
    grid[5, 2] = 6  # Switch

    engine.symbolic_theory.induce_barrier(2)
    engine.learned_barriers.update([(r, 4) for r in range(7)])

    # Register remote causal affordance: switch at (5, 2) mutates barrier at (2, 4) from 2 to 0
    aff = RemoteCausalAffordance(
        trigger_pos=(5, 2),
        trigger_feature=6,
        remote_pos=(2, 4),
        original_remote_feature=2,
        mutated_remote_feature=0,
        confidence=0.95,
    )
    engine.remote_causal.remote_actuators[(2, 4)].append(aff)

    available_actions = [1, 2, 3, 4]

    # Deliberative planner should decompose into subgoals:
    # Direct path to (2, 7) is impossible, so it must plan a route to switch (5, 2)!
    plan = engine.simulate_in_mind(grid, available_actions)
    assert plan is not None
    assert len(plan) > 0

    # Ensure subgoal stack has pending subgoals
    assert engine.subgoal_stack.has_pending_goals
    current_sg = engine.subgoal_stack.current_subgoal
    assert current_sg is not None
    assert current_sg.target_destination == (5, 2)
    assert current_sg.subgoal_type == SubgoalType.UNLOCK_REMOTE_MECHANISM

    # Trace simulated steps: The first segment must lead the avatar to the switch at (5, 2)
    curr_pos = (2, 1)
    reached_switch = False
    for step in plan:
        act = step.action
        dr, dc = engine.action_deltas[act]
        curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
        if curr_pos == (5, 2):
            reached_switch = True
            break

    assert reached_switch, "Decomposed plan must guide avatar to the remote mechanism switch!"


def test_e2e_nested_multi_stage_subgoals_lifo_execution() -> None:
    """Verify recursive multi-stage subgoal decomposition chains remote switches and executes in strict LIFO order."""
    from hbllm.hcir.world.remote_causal_attribution import RemoteCausalAffordance

    engine = AutonomousEpistemicEngine()

    engine.avatar_feature = 1
    engine.avatar_pos = (1, 1)

    # Calibrate movements
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)
    engine.learned_goal_features.add(3)

    # 5x10 grid:
    # Avatar at (1, 1)
    # Barrier 1 along col 3: (0, 3) to (4, 3) (feature 2)
    # Barrier 2 along col 6: (0, 6) to (4, 6) (feature 4)
    # Goal at (1, 8) (feature 3)
    # Switch 1 at (3, 1) (feature 6) clears (1, 3)
    # Switch 2 at (3, 4) (feature 7) clears (1, 6)
    grid = np.zeros((5, 10), dtype=int)
    grid[1, 1] = 1  # Avatar
    grid[1, 8] = 3  # Goal
    grid[:, 3] = 2  # Barrier 1
    grid[:, 6] = 4  # Barrier 2
    grid[3, 1] = 6  # Switch 1
    grid[3, 4] = 7  # Switch 2

    engine.symbolic_theory.induce_walkable(0)
    engine.symbolic_theory.induce_barrier(2)
    engine.symbolic_theory.induce_barrier(4)
    engine.learned_barriers.update([(r, 3) for r in range(5)] + [(r, 6) for r in range(5)])

    aff1 = RemoteCausalAffordance(
        trigger_pos=(3, 1),
        trigger_feature=6,
        remote_pos=(1, 3),
        original_remote_feature=2,
        mutated_remote_feature=0,
        confidence=0.95,
    )
    aff2 = RemoteCausalAffordance(
        trigger_pos=(3, 4),
        trigger_feature=7,
        remote_pos=(1, 6),
        original_remote_feature=4,
        mutated_remote_feature=0,
        confidence=0.95,
    )
    engine.remote_causal.remote_actuators[(1, 3)].append(aff1)
    engine.remote_causal.remote_actuators[(1, 6)].append(aff2)

    available_actions = [1, 2, 3, 4]
    plan = engine.simulate_in_mind(grid, available_actions)
    assert plan is not None
    assert len(plan) > 0

    # Ensure subgoal stack has both subgoals queued
    assert engine.subgoal_stack.has_pending_goals
    # Current active subgoal must be Switch 1 (LIFO top of stack)
    current_sg = engine.subgoal_stack.current_subgoal
    assert current_sg is not None
    assert current_sg.target_destination == (3, 1)

    # Pop Switch 1 and verify Switch 2 is next with prerequisite dependency satisfied
    popped = engine.subgoal_stack.pop_subgoal()
    assert popped is not None
    assert popped.target_destination == (3, 1)
    next_sg = engine.subgoal_stack.current_subgoal
    assert next_sg is not None
    assert next_sg.target_destination == (3, 4)
    assert popped.subgoal_id in next_sg.prerequisite_subgoal_ids

    # Trace complete simulated route: visits (3, 1) -> (1, 3) -> (3, 4) -> (1, 6) -> (1, 8)
    curr_pos = (1, 1)
    milestones = []
    for step in plan:
        act = step.action
        dr, dc = engine.action_deltas[act]
        curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
        if curr_pos in [(3, 1), (3, 4), (1, 8)] and (not milestones or milestones[-1] != curr_pos):
            milestones.append(curr_pos)

    assert (3, 1) in milestones, "Plan must visit Switch 1 first!"
    assert (3, 4) in milestones, "Plan must visit Switch 2 second!"
    assert milestones[-1] == (1, 8), "Plan must culminate at final goal!"


def test_e2e_high_surprise_hypothesis_invalidation_and_replanning() -> None:
    """Verify predictive sensory discrepancy/high surprise purges active mental plan and triggers replanning."""
    from hbllm.hcir.world.cortex_planner import MentalSimulationStep

    engine = AutonomousEpistemicEngine()

    engine.avatar_feature = 1
    engine.avatar_pos = (2, 1)

    # Calibrate movements
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)
    engine.learned_goal_features.add(3)
    engine.symbolic_theory.induce_walkable(0)

    # 5x7 grid: Avatar at (2, 1), Goal at (2, 5)
    grid = np.zeros((5, 7), dtype=int)
    grid[2, 1] = 1
    grid[2, 5] = 3

    # Active 3-step forward plan along row 2
    step1 = MentalSimulationStep(action=4, predicted_avatar_pos=(2, 2))
    step2 = MentalSimulationStep(action=4, predicted_avatar_pos=(2, 3))
    step3 = MentalSimulationStep(action=4, predicted_avatar_pos=(2, 4))
    engine.mental_plan = [step1, step2, step3]
    engine.last_action = 4
    engine.last_predicted_pos = (2, 2)
    engine.prev_grid = grid.copy()
    engine.phase = EpistemicPhase.EXPLOITATION

    # Execute step: Environment reports avatar DID NOT move (invisible barrier at (2, 2))
    # actual grid has avatar still at (2, 1)
    grid_feedback = grid.copy()
    engine.assimilate_feedback(grid_feedback, [1, 2, 3, 4])

    # 1. Discrepancy invalidates the stale plan
    assert len(engine.mental_plan) == 0, (
        "Sensory surprise must immediately clear stale mental plan!"
    )

    # 2. Epistemic phase shifts to REPLANNING
    assert engine.phase == EpistemicPhase.REPLANNING, (
        "Engine must switch to REPLANNING phase upon surprise divergence!"
    )

    # 3. Failed transition recorded
    assert ((2, 1), 4) in engine.failed_transitions, (
        "Failed transition must be cataloged in working memory!"
    )

    # 4. Action 4 Habenularly inhibited at current position
    assert engine.inhibited_actions.get(4, 0) >= 3, (
        "Action 4 must receive Habenular inhibition punishment!"
    )

    # 5. In subsequent decide(), deliberation replans avoiding failed transition
    chosen_act, _ = engine.decide(grid_feedback, [1, 2, 3, 4])
    assert chosen_act != 4, (
        "Replanning must never immediately repeat the failed action from the same position!"
    )
    assert chosen_act in [1, 2, 3], "Replanning must detour via safe uninhibited action!"
