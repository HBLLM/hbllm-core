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
from hbllm.hcir.world.bilateral_coordination import (
    BilateralCoordinateIntegrator,
    SymmetryAxis,
)
from hbllm.hcir.world.cortex_motor import MotorCortexEffector
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
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
    engine.quiescent_click_targets.update([(4, 4), (4, 5), (5, 4)])

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


def test_e2e_same_color_bilateral_twins_no_collapse() -> None:
    """Verify identical-color bilateral twins maintain distinct coordinates across multi-turn displacements."""
    engine = AutonomousEpistemicEngine()
    engine.bg_feature = 0

    # 10x12 grid with two identical cyan twins (feature 5) at (4, 2) and (4, 9)
    grid_0 = np.zeros((10, 12), dtype=int)
    grid_0[4, 2] = 5
    grid_0[4, 9] = 5
    engine.prev_grid = grid_0.copy()

    # Step 1: Twins move under vertical symmetry (Agent 1 moves right to (4, 3), Agent 2 moves left to (4, 8))
    grid_1 = np.zeros((10, 12), dtype=int)
    grid_1[4, 3] = 5
    grid_1[4, 8] = 5
    engine.last_action = 4
    engine.assimilate_feedback(grid_1, [1, 2, 3, 4])

    assert engine.bilateral_state is not None
    assert engine.bilateral_state.symmetry_axis == SymmetryAxis.VERTICAL
    assert engine.bilateral_state.features == {5}

    # Step 2: Twins advance further towards center: (4, 4) and (4, 7)
    grid_2 = np.zeros((10, 12), dtype=int)
    grid_2[4, 4] = 5
    grid_2[4, 7] = 5
    engine.prev_grid = grid_1.copy()
    engine.last_action = 4
    engine.assimilate_feedback(grid_2, [1, 2, 3, 4])

    assert engine.bilateral_state is not None
    # Crucial assertion: Twin positions must NOT collapse to midpoint (4, 5)
    p1 = engine.bilateral_state.pos1
    p2 = engine.bilateral_state.pos2
    assert p1 != p2, "Bilateral twins sharing same color must never collapse to identical centroid!"
    assert (p1 == (4, 4) and p2 == (4, 7)) or (p1 == (4, 7) and p2 == (4, 4))


def test_e2e_quad_lateral_4_agent_symmetry_and_convergence() -> None:
    """Verify 4-agent quad-lateral D4 reflection detection and convergent joint planning."""
    integrator = BilateralCoordinateIntegrator()

    # 4 agents in 4 quadrants moving synchronously under reflection:
    # Action right (+dc):
    # - Top-Left at (2, 2) moves right -> delta = (0, 1)
    # - Top-Right at (2, 8) moves left -> delta = (0, -1)
    # - Bottom-Left at (8, 2) moves right -> delta = (0, 1)
    # - Bottom-Right at (8, 8) moves left -> delta = (0, -1)
    moved_entities = [
        ((2, 2), (0, 1), 4),
        ((2, 8), (0, -1), 4),
        ((8, 2), (0, 1), 4),
        ((8, 8), (0, -1), 4),
    ]

    state = integrator.detect_bilateral_pairing(moved_entities, (11, 11))
    assert state is not None
    assert state.symmetry_axis == SymmetryAxis.QUAD
    assert len(state.quad_positions) == 4

    # Plan convergence sequence to center dock targets:
    # Targets for the 4 agents: (5, 4), (5, 6), (6, 4), (6, 6)
    passable = np.ones((11, 11), dtype=bool)
    action_deltas = {
        1: (-1, 0),  # UP
        2: (1, 0),  # DOWN
        3: (0, -1),  # LEFT
        4: (0, 1),  # RIGHT
    }
    targets = [(5, 4), (5, 6), (6, 4), (6, 6)]

    plan = integrator.plan_convergence_sequence(
        pos1=state.quad_positions[0],
        pos2=state.quad_positions[1],
        axis=SymmetryAxis.QUAD,
        passable_mask=passable,
        action_deltas=action_deltas,
        target_positions=targets,
    )

    assert plan is not None
    assert len(plan) > 0

    # Simulate plan execution with 4-way wall slip physics
    curr_pts = tuple(state.quad_positions)
    for act in plan:
        d = action_deltas[act]
        deltas = integrator.transform_actions_for_quad(d)
        next_pts = []
        for p, ad in zip(curr_pts, deltas):
            np_pos = (p[0] + ad[0], p[1] + ad[1])
            next_pts.append(
                np_pos if (0 <= np_pos[0] < 11 and 0 <= np_pos[1] < 11 and passable[np_pos]) else p
            )
        curr_pts = tuple(next_pts)

    assert all(p in targets for p in curr_pts), (
        "All 4 agents must successfully converge into docking targets!"
    )


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

    engine.symbolic_theory.barrier_features.add(2)
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
    engine.causal_cortex.causal_attributor.remote_actuators[(2, 4)].append(aff)

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

    engine.symbolic_theory.walkable_features.add(0)
    engine.symbolic_theory.barrier_features.update({2, 4})
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
    engine.causal_cortex.causal_attributor.remote_actuators[(1, 3)].append(aff1)
    engine.causal_cortex.causal_attributor.remote_actuators[(1, 6)].append(aff2)

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


def test_e2e_dynamic_threat_pacing_and_looming_cone_avoidance() -> None:
    """Verify deliberation safety gating and premotor reflex evade dynamic kinetic threat cones."""
    from hbllm.hcir.world.cortex_deliberation import SpatiotemporalCollisionConePredicate
    from hbllm.hcir.world.cortex_planner import MentalSimulationStep
    from hbllm.hcir.world.kinetic_stream import KineticEntity

    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 1
    engine.avatar_pos = (3, 3)

    # Calibrate movements
    engine.action_dynamics[1] = ActionDynamicsModel(1, delta_r=-1, delta_c=0, confidence=1.0)
    engine.action_dynamics[2] = ActionDynamicsModel(2, delta_r=1, delta_c=0, confidence=1.0)
    engine.action_dynamics[3] = ActionDynamicsModel(3, delta_r=0, delta_c=-1, confidence=1.0)
    engine.action_dynamics[4] = ActionDynamicsModel(4, delta_r=0, delta_c=1, confidence=1.0)

    # 7x7 grid: Avatar at (3, 3)
    grid = np.zeros((7, 7), dtype=int)
    grid[3, 3] = 1

    # Kinetic threat at (1, 3) heading downwards with velocity (1.0, 0.0)
    # At t=1: enters (2, 3)
    # At t=2: enters (3, 3) (directly striking stationary avatar!)
    threat = KineticEntity(
        centroid=(1.0, 3.0),
        velocity=(1.0, 0.0),
        cells=[(1, 3)],
        bounding_box=(1, 1, 3, 3),
        area=1,
        features={9},
        is_self=False,
    )
    engine.collision_cones.update_trajectories(
        kinetic_entities=[threat],
        static_barriers=set(),
        grid_shape=(7, 7),
        horizon=6,
    )

    # 1. Spatiotemporal hazard verification
    assert engine.collision_cones.is_collision_hazard(2, 3, 1), "Threat must occupy (2, 3) at t=1!"
    assert engine.collision_cones.is_collision_hazard(3, 3, 2), "Threat must strike (3, 3) at t=2!"

    # 2. Premotor Looming Threat Reflex: Avatar at (3, 3) detects incoming threat and chooses safe evasion
    disp_actions = [(act, engine.action_deltas[act]) for act in [1, 2, 3, 4]]
    evasion_act = MotorCortexEffector._probe_looming_threat_evasion(
        engine, grid, disp_actions, bg=0, bg_is_impassable_void=False
    )
    assert evasion_act is not None
    # Action 1 steps UP into (2, 3), which is in the collision cone at t=1!
    assert evasion_act != 1, "Reflex must never step directly into oncoming collision cone!"
    assert evasion_act in [2, 3, 4], "Reflex must evade away from threat trajectory!"

    # 3. Deliberation Gating: Rejects candidate forward step heading into collision cone
    cone_predicate = SpatiotemporalCollisionConePredicate()
    dangerous_step = MentalSimulationStep(action=1, predicted_avatar_pos=(2, 3))
    safety_eval = cone_predicate.evaluate(engine, grid, dangerous_step, [], [1, 2, 3, 4])
    assert not safety_eval.is_safe, "Deliberation predicate must reject step into looming cone!"
    assert "looming collision cone" in safety_eval.reason

    safe_step = MentalSimulationStep(action=4, predicted_avatar_pos=(3, 4))
    safe_eval = cone_predicate.evaluate(engine, grid, safe_step, [], [1, 2, 3, 4])
    assert safe_eval.is_safe, "Lateral step outside collision cone must be accepted!"


def test_e2e_asymmetric_bilateral_wall_slip_synchronization() -> None:
    """Verify bilateral coordinate planning exploits asymmetric obstacle wall-slip to shift parity."""
    integrator = BilateralCoordinateIntegrator()

    # 9x9 room:
    # Twins start at (2, 2) and (2, 6) under vertical reflection symmetry (column diff = 4, even).
    # Target positions: (5, 4) and (5, 5) (column diff = 1, odd parity!).
    # Without wall slip, pure vertical reflection delta always preserves parity: dc2 = -dc1.
    # An obstacle at (2, 1) allows Agent 1 to slip/stall when moving left while Agent 2 advances,
    # breaking symmetric parity and enabling joint convergence!
    passable = np.ones((9, 9), dtype=bool)
    passable[2, 1] = False  # Slip obstacle for Agent 1

    action_deltas = {
        1: (-1, 0),  # UP
        2: (1, 0),  # DOWN
        3: (0, -1),  # LEFT
        4: (0, 1),  # RIGHT
    }
    targets = [(5, 4), (5, 5)]

    plan = integrator.plan_convergence_sequence(
        pos1=(2, 2),
        pos2=(2, 6),
        axis=SymmetryAxis.VERTICAL,
        passable_mask=passable,
        action_deltas=action_deltas,
        target_positions=targets,
    )
    assert plan is not None
    assert len(plan) > 0

    # Simulate joint twin physics with wall slip
    p1, p2 = (2, 2), (2, 6)
    for act in plan:
        d1 = action_deltas[act]
        d2 = integrator.transform_action_for_agent2(d1, SymmetryAxis.VERTICAL)
        np1 = (p1[0] + d1[0], p1[1] + d1[1])
        np2 = (p2[0] + d2[0], p2[1] + d2[1])
        p1 = np1 if (0 <= np1[0] < 9 and 0 <= np1[1] < 9 and passable[np1]) else p1
        p2 = np2 if (0 <= np2[0] < 9 and 0 <= np2[1] < 9 and passable[np2]) else p2

    assert {p1, p2} == {(5, 4), (5, 5)}, (
        "Asymmetric wall-slip must successfully dock both twins at odd-parity targets!"
    )


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
    engine.symbolic_theory.walkable_features.add(0)

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
