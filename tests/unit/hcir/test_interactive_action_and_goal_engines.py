"""Unit test suite for Domain-General Interactive World-Modeling Engines.

Verifies Criterion A across Modules 1, 2, and 3:
- interactive_action_discovery.py (W161-W165, W171, W187, W188)
- interactive_goal_evaluator.py (W168-W170, W174, W175, W181, W182)
- temporal_dependency_tracker.py (W167, W172, W173, W176-W180, W189, W190)
"""

from __future__ import annotations

from hbllm.hcir.world.interactive_action_discovery import (
    ActionBudgetGovernor,
    ActionEffectProfile,
    ActionEffectType,
    ActionPreconditionLearner,
    ConstrainedActiveExperimenter,
    UngroundedActionDiscoveryEngine,
    UnknownMechanicsLedger,
)
from hbllm.hcir.world.interactive_goal_evaluator import (
    BayesianGoalInductionEngine,
    DynamicProgressEstimator,
    GoalConfirmationGate,
    ModelBasedDeadlockDetector,
    TerminalStateCategory,
)
from hbllm.hcir.world.temporal_dependency_tracker import (
    DelayedEffectModeler,
    DualStoreMemoryManager,
    EnvironmentChangeDetector,
    LongHorizonDependencyGraph,
    ObjectiveLearningEfficiencyTracker,
)

# ── Module 1 Tests ────────────────────────────────────────────────────────────


def test_ungrounded_action_discovery_and_uncertainty() -> None:
    """Action tokens have unknown initial meanings; dynamics are inferred with Bayesian uncertainty."""
    engine = UngroundedActionDiscoveryEngine(action_space=["act_alpha", "act_beta"])
    prof_alpha = engine.register_action("act_alpha")
    assert prof_alpha.uncertainty == 1.0

    # Observe act_alpha producing spatial movement (+1, 0)
    for _ in range(5):
        engine.record_transition(
            action_id="act_alpha",
            prior_state={"avatar_pos": (2, 2)},
            next_state={"avatar_pos": (3, 2)},
        )

    assert prof_alpha.effect_type == ActionEffectType.SPATIAL_DISPLACEMENT
    assert prof_alpha.mean_displacement == (1.0, 0.0)
    assert prof_alpha.uncertainty < 0.50


def test_unknown_mechanics_ledger_lifecycle() -> None:
    """Tracks open-set mechanics hypotheses through interventional probing."""
    ledger = UnknownMechanicsLedger()
    key = ledger.register_untested_affordance("gold_switch", 3, "Toggles door barrier")
    assert len(ledger.get_pending_hypotheses()) == 1

    ledger.record_probe_outcome(key, produced_expected_effect=True)
    ledger.record_probe_outcome(key, produced_expected_effect=True)
    assert ledger.entries[key].confirmed is True
    assert len(ledger.get_pending_hypotheses()) == 0


def test_constrained_active_experimentation() -> None:
    """Selects action maximizing Expected Free Energy balancing EIG against cost and hazard risk."""
    exp = ConstrainedActiveExperimenter(cost_weight=0.1, risk_weight=1.0, goal_weight=0.5)
    prof_safe = ActionEffectProfile(action_id="safe_probe", uncertainty=0.8)
    prof_dangerous = ActionEffectProfile(action_id="risky_probe", uncertainty=0.9)

    val_safe = exp.evaluate_experiment("safe_probe", prof_safe, predicted_hazard_prob=0.0)
    val_dangerous = exp.evaluate_experiment(
        "risky_probe", prof_dangerous, predicted_hazard_prob=0.8
    )

    # Risk penalty significantly degrades net value despite higher uncertainty
    assert val_safe.net_value > val_dangerous.net_value


def test_action_precondition_induction() -> None:
    """Induces contrastive feature preconditions enabling action effects."""
    learner = ActionPreconditionLearner()
    # Positive examples: requires key=True
    learner.record_outcome("open_gate", "unlocked", {"has_key": True, "stamina": 10}, True)
    learner.record_outcome("open_gate", "unlocked", {"has_key": True, "stamina": 5}, True)
    # Negative example: no key
    learner.record_outcome("open_gate", "unlocked", {"has_key": False, "stamina": 10}, False)

    rule = learner.induce_preconditions("open_gate", "unlocked")
    assert rule is not None
    assert rule.required_features == {"has_key": True}
    assert rule.confidence > 0.60


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


def test_bayesian_goal_induction_under_feedback() -> None:
    """Updates Bayesian posteriors over open-set goal criteria upon level completion feedback."""
    engine = BayesianGoalInductionEngine()
    initial_dominant = engine.get_dominant_goal()
    assert initial_dominant is not None

    # Environment level increment achieved when all items cleared
    state_win = {"remaining_items": 0, "avatar_pos": (1, 1)}
    confirmed = engine.observe_feedback(state_win, level_incremented=True)

    assert len(confirmed) >= 1
    cleared_h = engine.hypotheses["clear_all_targets"]
    assert cleared_h.is_confirmed is True
    assert cleared_h.posterior_probability > 0.50


def test_model_based_deadlock_detection_and_recovery() -> None:
    """Detects absorbing non-goal states via forward reachability and finds recovery branch."""
    detector = ModelBasedDeadlockDetector(max_search_depth=5)

    # Deterministic transition model
    # State: 1D position x in [0, 4]. Goal is at x=4. State x=0 is a trap (cannot move).
    def transition(s: int, a: str) -> int:
        if s == 0:
            return 0  # Absorbing trap
        if a == "right":
            return min(4, s + 1)
        if a == "left":
            return max(0, s - 1)
        return s

    def is_goal(s: int) -> bool:
        return s == 4

    # Trap state x=0 has no path to goal
    res_trap = detector.analyze_reachability(0, transition, ["left", "right"], is_goal)
    assert res_trap.is_deadlocked is True

    # State x=2 can reach goal
    res_valid = detector.analyze_reachability(2, transition, ["left", "right"], is_goal)
    assert res_valid.is_deadlocked is False
    assert res_valid.goal_reachable is True


def test_goal_confirmation_and_progress_estimation() -> None:
    """Terminal state classification and metric progress estimation."""
    cat_win = GoalConfirmationGate.classify_terminal_event(True, 1, False, 100)
    assert cat_win == TerminalStateCategory.WIN

    cat_death = GoalConfirmationGate.classify_terminal_event(False, 0, True, 50)
    assert cat_death == TerminalStateCategory.DEATH_OR_LOSS

    prog = DynamicProgressEstimator.estimate_progress(
        current_distance=2.0, initial_distance=10.0, subgoals_achieved=1, total_subgoals=2
    )
    assert 0.60 <= prog <= 0.80


# ── Module 3 Tests ────────────────────────────────────────────────────────────


def test_delayed_effect_modeler() -> None:
    """Discovers consequences occurring with temporal lag tau > 0."""
    modeler = DelayedEffectModeler(max_lag=5)
    modeler.record_step(1, "pull_lever", {"lever": "down"})
    modeler.record_step(2, "walk_forward", {"avatar": (1, 1)})
    # At step 3, gate opens (delayed by 2 steps from pull_lever)
    delayed = modeler.record_step(
        3, "walk_forward", {"avatar": (1, 2)}, observed_mutations={"gate_opened": 1}
    )

    assert len(delayed) >= 1
    assert any(d.trigger_action == "pull_lever" and d.observed_lag_steps == 2 for d in delayed)


def test_long_horizon_dependency_dag() -> None:
    """Topological prerequisite satisfaction across long-horizon milestones."""
    graph = LongHorizonDependencyGraph()
    graph.add_milestone("obtain_key", "Collect silver key")
    graph.add_milestone("unlock_door", "Open silver door", prerequisites=["obtain_key"])

    exec_initial = graph.get_executable_milestones()
    assert len(exec_initial) == 1
    assert exec_initial[0].node_id == "obtain_key"

    graph.mark_satisfied("obtain_key")
    exec_next = graph.get_executable_milestones()
    assert len(exec_next) == 1
    assert exec_next[0].node_id == "unlock_door"


def test_dual_store_memory_provenance_and_validation_gate() -> None:
    """Decouples invariant causal schemas from episodic facts with provenance validation gate."""
    mem = DualStoreMemoryManager()
    mem.commit_invariant_schema(
        schema_id="key_unlocks_matching_door",
        rule_signature="ColorMatch(Key, Door) => Open",
        provenance_source="world_gamma",
        confidence=0.85,
    )
    mem.episodic_facts["current_room"] = "dungeon_10"

    # Reset episode clears local facts but preserves invariant schemas
    mem.reset_episodic_memory()
    assert "current_room" not in mem.episodic_facts
    assert "key_unlocks_matching_door" in mem.invariant_schemas
    schema = mem.invariant_schemas["key_unlocks_matching_door"]
    assert schema.validation_status_in_target == "PENDING_TARGET_VALIDATION"

    # Validation gate: verify against target world evidence
    is_valid = mem.validate_schema_in_target(
        "key_unlocks_matching_door", is_consistent_with_target_evidence=True
    )
    assert is_valid is True
    assert schema.validation_status_in_target == "VALIDATED"


def test_environment_change_detection_and_efficiency_tracking() -> None:
    """Monitors predictive surprise and computes objective sample efficiency."""
    change_detector = EnvironmentChangeDetector(surprise_threshold=2.0)
    # Expected match
    regime_shift, surprise = change_detector.record_prediction_error(
        {"a": 1, "b": 2}, {"a": 1, "b": 2}
    )
    assert regime_shift is False
    assert surprise == 0.0

    # Massive mismatch triggers surprise
    regime_shift, surprise = change_detector.record_prediction_error(
        {"a": 1, "b": 2, "c": 3}, {"a": 9, "b": 8, "c": 7}
    )
    assert surprise >= 3.0

    # Objective learning efficiency tracker
    tracker = ObjectiveLearningEfficiencyTracker()
    tracker.record_step_evaluation(action_optimal_value=1.0, action_chosen_value=1.0, info_bits=0.5)
    tracker.record_step_evaluation(action_optimal_value=1.0, action_chosen_value=0.5, info_bits=0.2)
    summary = tracker.compute_summary_metrics(task_solved=True)

    assert summary.sample_complexity_steps == 2
    assert summary.cumulative_regret == 0.5
    assert summary.information_gain_rate == 0.35
    assert summary.zero_shot_transfer_success is True
