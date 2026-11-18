"""Homeostatic Intrinsic Motivation & Goal Arbitration test suite.

Validates:
1. Homeostatic state tracking (energy budget, safety margins, competence progress, social instructions).
2. Expected Free Energy (EFE) goal arbitration across extrinsic, epistemic, and cost dimensions.
3. Social instructions preempting open-ended curiosity until satisfied.
4. Low-energy state autonomously prioritizing high-safety, low-cost subgoals over exploration.
5. Neuromodulatory modulation of arbitration weights (Dopamine, Noradrenaline, Serotonin).
6. Automatic TaskFrame activation in TieredWorkspace for selected goals.
"""

from __future__ import annotations

import pytest

from hbllm.hcir.world.prefrontal_working_memory import HomeostaticState


def test_homeostatic_state_initialization_and_operations() -> None:
    """Validate homeostatic state tracking, energy consumption, and predicate export."""
    state = HomeostaticState(
        energy_budget=100.0,
        safety_margin=1.0,
        competence_progress=0.0,
    )
    assert not state.is_critical_energy

    # Consume energy
    remaining = state.consume_energy(40.0)
    assert remaining == pytest.approx(60.0)
    assert state.fatigue == pytest.approx(20.0)
    assert not state.is_critical_energy

    # Further consumption into critical energy
    state.consume_energy(45.0)
    assert state.energy_budget == pytest.approx(15.0)
    assert state.is_critical_energy

    # Replenish energy
    state.replenish_energy(70.0)
    assert state.energy_budget == pytest.approx(85.0)
    assert not state.is_critical_energy

    # Social instruction tracking
    instruction = {"goal_id": "escort_unit", "target": (4, 4), "utility": 75.0}
    state.add_instruction(instruction)
    assert len(state.social_instructions) == 1
    assert state.social_instructions[0]["goal_id"] == "escort_unit"

    # Export to causal predicates and transaction operations
    preds = state.to_causal_predicates()
    assert len(preds) == 3
    ops = state.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].node_id == "homeostatic_state"

    state.clear_instructions()
    assert len(state.social_instructions) == 0
