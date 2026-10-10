"""Integration Test Suite for Autonomous Epistemic Engine Closed Cognitive Loop (Criterion B).

Validates that all 4 interactive world-modeling modules:
- interactive_action_discovery (W161-W165, W171, W187, W188)
- interactive_goal_evaluator (W168-W170, W174, W175, W181, W182)
- temporal_dependency_tracker (W167, W172, W173, W176-W180, W189, W190)
- legacy capabilities (W014, W040, W079)
operate synchronously within AutonomousEpistemicEngine during the full Observe -> Predict -> Act -> Evaluate cycle.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


def test_autonomous_epistemic_engine_interactive_loop_integration() -> None:
    """Verifies Criterion B: End-to-end integration of interactive faculties in AutonomousEpistemicEngine."""
    engine = AutonomousEpistemicEngine(exploration_budget=50, enable_logging=False)

    # 1. Verify subsystem presence
    assert hasattr(engine, "action_discovery")
    assert hasattr(engine, "unknown_mechanics")
    assert hasattr(engine, "active_experimenter")
    assert hasattr(engine, "budget_governor")
    assert hasattr(engine, "goal_induction")
    assert hasattr(engine, "model_deadlock_detector")
    assert hasattr(engine, "delayed_effects")
    assert hasattr(engine, "dual_memory")
    assert hasattr(engine, "change_detector")
    assert hasattr(engine, "efficiency_tracker")

    # 2. Simulate 5-step synthetic environment interaction loop
    grid_0 = np.zeros((10, 10), dtype=int)
    grid_0[1, 1] = 1  # Avatar feature 1 at (1, 1)
    grid_0[5, 5] = 2  # Consumable target feature 2

    # Step 1: Agent perceives initial grid
    engine.assimilate_feedback(
        curr_grid=grid_0,
        available_actions=[1, 2, 3, 4],
        action=None,
    )
    assert engine.budget_governor.steps_consumed == 0

    # Step 2: Agent takes action 1 (e.g. displacement right)
    grid_1 = grid_0.copy()
    grid_1[1, 1] = 0
    grid_1[1, 2] = 1  # Avatar moved to (1, 2)

    engine.avatar_pos = (1, 2)
    engine.assimilate_feedback(
        curr_grid=grid_1,
        available_actions=[1, 2, 3, 4],
        action=1,
    )

    # Verify budget and action discovery updated
    assert engine.budget_governor.steps_consumed == 1
    prof = engine.action_discovery.action_profiles[1]
    assert prof.total_observations == 1

    # Step 3: Multiple steps to calibrate action semantics dynamically
    for step in range(3):
        engine.assimilate_feedback(
            curr_grid=grid_1,
            available_actions=[1, 2, 3, 4],
            action=1,
        )

    assert prof.total_observations == 4

    # Step 4: Simulate win/level completion event
    grid_win = grid_1.copy()
    grid_win[5, 5] = 0  # Target cleared

    engine.assimilate_feedback(
        curr_grid=grid_win,
        available_actions=[1, 2, 3, 4],
        is_win=True,
        action=1,
    )

    # Verify Bayesian goal induction updated
    dominant_goal = engine.goal_induction.get_dominant_goal()
    assert dominant_goal is not None

    # Step 5: Test episodic reset and dual-store memory preservation
    engine.dual_memory.commit_invariant_schema(
        schema_id="action_1_is_horizontal_motor",
        rule_signature="Act(1) => dr=0, dc>0",
        provenance_source="level_0",
        confidence=0.90,
    )
    engine.dual_memory.episodic_facts["level_0_obstacle"] = (3, 3)

    engine.reset_episode(is_new_level=True, level=1)

    # Budget reset for new level
    assert engine.budget_governor.steps_consumed == 0
    # Invariant schema preserved across levels
    assert "action_1_is_horizontal_motor" in engine.dual_memory.invariant_schemas
    # Transient episodic facts purged
    assert "level_0_obstacle" not in engine.dual_memory.episodic_facts
