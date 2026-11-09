"""Tests for Learning-Progress Signals and Recursive Value Induction Engine.

Verifies:
1. Oudeyer/Schmidhuber learning-progress signal (-dPE/dt) in HierarchicalPrecisionController.
2. Epistemic gain modulation driven by active learning progress.
3. Multi-hop backward recursive value induction across causal enablement chains.
4. Confidence-weighted value attenuation.
5. GoalArbiter integration: extraction of recursively-induced subgoals and weight scaling.
6. Causal predicate and transaction export.
"""

from __future__ import annotations

from hbllm.hcir.world.goal_arbiter import GoalArbiter
from hbllm.hcir.world.hierarchical_precision_control import (
    HierarchicalPrecisionController,
    PrecisionState,
)
from hbllm.hcir.world.prefrontal_working_memory import HomeostaticState
from hbllm.hcir.world.recursive_value_induction import (
    RecursiveValueInducer,
)


def test_oudeyer_schmidhuber_learning_progress_detection() -> None:
    """Validate that diminishing prediction error produces positive learning progress."""
    controller = HierarchicalPrecisionController(base_learning_rate=0.5)

    # Phase 1: High initial errors (unfamiliar task, PE ~ 0.85)
    for step in range(12):
        controller.update_with_prediction_error(step=step, error_magnitude=0.85)

    # Phase 2: Mastering the task (PE rapidly drops to 0.10)
    final_state: PrecisionState = controller.current_state
    for step in range(12, 24):
        final_state = controller.update_with_prediction_error(
            step=step,
            error_magnitude=0.15,
        )

    # Learning progress must be positive: long window error > short window error
    assert final_state.learning_progress > 0.10

    # Epistemic exploration gain must be boosted above the baseline for steady-state error
    steady_controller = HierarchicalPrecisionController(base_learning_rate=0.5)
    for step in range(24):
        steady_controller.update_with_prediction_error(step=step, error_magnitude=0.15)

    assert (
        final_state.epistemic_exploration_gain
        > steady_controller.current_state.epistemic_exploration_gain
    )


def test_learning_progress_stagnation_and_regression() -> None:
    """Validate that constant or increasing error yields zero or negative learning progress."""
    controller = HierarchicalPrecisionController()

    # Flat unlearnable noise or already-mastered task (constant PE)
    for step in range(20):
        controller.update_with_prediction_error(step=step, error_magnitude=0.50)

    # When error is stagnant, learning progress is approximately zero
    assert abs(controller.current_state.learning_progress) < 0.05

    # Regressing task (error increases)
    for step in range(20, 30):
        controller.update_with_prediction_error(step=step, error_magnitude=0.90)

    assert controller.current_state.learning_progress < 0.0


def test_recursive_value_induction_backward_propagation() -> None:
    """Validate that intermediate keys and doors inherit value from the terminal goal."""
    inducer = RecursiveValueInducer(discount_factor=0.90)

    # Register terminal reward
    inducer.register_terminal_value("goal_exit", value=1.0)

    # Build causal enablement chain: key -> door -> goal_exit
    inducer.record_causal_enablement(source_key="door_barrier", target_key="goal_exit")
    inducer.record_causal_enablement(source_key="golden_key", target_key="door_barrier")

    # Propagate values backwards
    values = inducer.propagate_values()

    # Values must satisfy Bellman causal induction:
    # V(goal_exit) = 1.0
    # V(door_barrier) = 0.90 * 1.0 = 0.90
    # V(golden_key) = 0.90 * 0.90 = 0.81
    assert values["goal_exit"] == 1.0
    assert abs(values["door_barrier"] - 0.90) < 1e-3
    assert abs(values["golden_key"] - 0.81) < 1e-3

    assert abs(inducer.get_induced_value("golden_key") - 0.81) < 1e-3


def test_recursive_value_induction_confidence_discounting() -> None:
    """Validate that uncertain causal links attenuate backed-up value proportionally."""
    inducer = RecursiveValueInducer(discount_factor=0.90)

    inducer.register_terminal_value("target_chest", value=1.0)

    # Partial confidence in lever effect (conf = 0.50)
    inducer.record_causal_enablement(
        source_key="mysterious_lever",
        target_key="target_chest",
        confidence=0.50,
    )

    inducer.propagate_values()
    # Expected: gamma * target_val * confidence = 0.90 * 1.0 * 0.50 = 0.45
    lever_val = inducer.get_induced_value("mysterious_lever")
    assert abs(lever_val - 0.45) < 1e-3


def test_goal_arbiter_incorporates_learning_progress_and_induced_subgoals() -> None:
    """Validate that GoalArbiter boosts epistemic weights with learning progress and extracts induced goals."""
    arbiter = GoalArbiter()

    # 1. Test weight scaling with learning progress
    p_state_stagnant = PrecisionState(
        sensory_precision=1.0,
        prior_precision=1.0,
        environmental_volatility=0.0,
        adaptive_learning_rate=0.5,
        epistemic_exploration_gain=1.0,
        mean_prediction_error=0.5,
        timestamp_step=1,
        learning_progress=0.0,
    )
    weights_stagnant = arbiter.compute_arbitration_weights(precision_state=p_state_stagnant)

    p_state_learning = PrecisionState(
        sensory_precision=1.0,
        prior_precision=1.0,
        environmental_volatility=0.0,
        adaptive_learning_rate=0.5,
        epistemic_exploration_gain=1.0,
        mean_prediction_error=0.2,
        timestamp_step=1,
        learning_progress=0.50,
    )
    weights_learning = arbiter.compute_arbitration_weights(precision_state=p_state_learning)

    # Epistemic curiosity weight must be noticeably higher when actively learning
    assert weights_learning["w_epistemic"] > weights_stagnant["w_epistemic"]

    # 2. Test extraction of recursively induced subgoals
    class MockEngine:
        def __init__(self) -> None:
            self.avatar_pos = (0, 0)
            self.goals = [(5, 5)]
            self.homeostatic_state = HomeostaticState()
            self.recursive_value_inducer = RecursiveValueInducer()
            self.recursive_value_inducer.register_terminal_value((5, 5), value=1.0)
            self.recursive_value_inducer.record_causal_enablement((2, 2), (5, 5))

    engine = MockEngine()
    candidates = arbiter.extract_candidate_goals(engine=engine)

    # Candidate list must contain the recursively induced intermediate waypoint at (2, 2)
    induced_goals = [c for c in candidates if c.goal_type == "recursively_induced"]
    assert len(induced_goals) > 0
    assert induced_goals[0].target == (2, 2)
    assert induced_goals[0].extrinsic_utility > 0.0


def test_recursive_value_export_to_predicates_and_transactions() -> None:
    """Validate export to formal CausalPredicates and TransactionOperations."""
    inducer = RecursiveValueInducer()
    inducer.register_terminal_value("gem", value=1.0)
    inducer.record_causal_enablement("pickaxe", "gem")

    preds = inducer.to_causal_predicates()
    assert len(preds) >= 2
    assert any("gem" in p.variable for p in preds)
    assert any("pickaxe" in p.variable for p in preds)

    ops = inducer.to_transaction_operations()
    assert len(ops) >= 2
    assert any("gem" in op.node_id for op in ops)
