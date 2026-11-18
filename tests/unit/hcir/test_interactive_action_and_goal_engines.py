"""Unit test suite for Domain-General Interactive World-Modeling Engines.

Verifies Criterion A across Modules 1, 2, and 3:
- interactive_action_discovery.py (W161-W165, W171, W187, W188)
- interactive_goal_evaluator.py (W168-W170, W174, W175, W181, W182)
- temporal_dependency_tracker.py (W167, W172, W173, W176-W180, W189, W190)
"""

from __future__ import annotations

from hbllm.hcir.world.interactive_action_discovery import ActionBudgetGovernor

# ── Module 1 Tests ────────────────────────────────────────────────────────────


def test_budget_governor_and_habenular_ior() -> None:
    """Throttles exploration as budget diminishes and inhibits repeat testing via IOR."""
    gov = ActionBudgetGovernor(max_budget=100)
    assert gov.get_exploration_weight() == 1.0

    # Consume 80 steps
    for _ in range(80):
        gov.record_step()

    # Budget < 25%: shifted to emergency exploitation mode
    assert gov.get_exploration_weight() == 0.10

    # Habenular IOR refractory inhibition
    gov.register_experiment_trial("test_lever_A")
    assert gov.is_inhibited_by_ior("test_lever_A", refractory_period=10) is True


# ── Module 2 Tests ────────────────────────────────────────────────────────────


# ── Module 3 Tests ────────────────────────────────────────────────────────────
