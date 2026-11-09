"""Unit tests for Safety, Corrigibility & Value-Uncertainty Deferral.

Validates uncertainty-aware action gating, interpretable goal provenance trees,
and corrigible non-resistance during task interruption and shutdown signals.
"""

from __future__ import annotations

from hbllm.hcir.grounding.contracts.enums import DecisionOutcome
from hbllm.hcir.kernel.safety_corrigibility import (
    CorrigibilityMonitor,
    DeferralReason,
    GoalProvenanceTracker,
    ValueUncertaintyArbiter,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


def test_value_uncertainty_safe_action_approval() -> None:
    """Actions with low epistemic value variance are approved for autonomous dispatch."""
    arbiter = ValueUncertaintyArbiter(variance_deferral_threshold=0.35)

    approved, request = arbiter.evaluate_action_safety(
        action_id=1,
        action_type="NAVIGATE_STEP",
        expected_value=0.85,
        epistemic_variance=0.08,
        is_hazard_cone=False,
    )

    assert approved is True
    assert request is None


def test_value_uncertainty_high_variance_deferral() -> None:
    """Actions with high value variance trigger structured deferral requesting evidence."""
    arbiter = ValueUncertaintyArbiter(variance_deferral_threshold=0.35)

    approved, request = arbiter.evaluate_action_safety(
        action_id=2,
        action_type="TOGGLE_SWITCH",
        expected_value=0.50,
        epistemic_variance=0.45,
        is_hazard_cone=False,
    )

    assert approved is False
    assert request is not None
    assert request.deferral_reason == DeferralReason.HIGH_VALUE_VARIANCE
    assert request.outcome == DecisionOutcome.REQUEST_EVIDENCE
    assert request.requires_human_approval is True

    # Check transaction operations
    tx_ops = request.to_transaction_operations()
    assert len(tx_ops) == 1
    assert tx_ops[0].op == TransactionOp.UPSERT_NODE
    assert tx_ops[0].node_data["deferral_reason"] == "high_value_variance"


def test_value_uncertainty_hazard_cone_deferral() -> None:
    """Actions entering hazard cones with uncertain consequences are deferred."""
    arbiter = ValueUncertaintyArbiter(hazard_deferral_threshold=0.20)

    approved, request = arbiter.evaluate_action_safety(
        action_id=3,
        action_type="ENTER_REGION",
        expected_value=0.20,
        epistemic_variance=0.25,
        is_hazard_cone=True,
    )

    assert approved is False
    assert request is not None
    assert request.deferral_reason == DeferralReason.IRREVERSIBLE_HAZARD


def test_goal_provenance_ancestry_tracking_and_explanation() -> None:
    """GoalProvenanceTracker traces full causal chain from root objective to active action."""
    tracker = GoalProvenanceTracker()

    tracker.register_goal(
        goal_id="root_objective",
        intent="NAVIGATE_TO",
        target_entity="exit_gate",
        justification="Complete environment mission",
    )
    tracker.register_goal(
        goal_id="unlock_gate",
        parent_goal_id="root_objective",
        intent="INTERACT",
        target_entity="barrier_door",
        justification="Exit gate is locked by barrier_door",
    )
    tracker.register_goal(
        goal_id="collect_key",
        parent_goal_id="unlock_gate",
        intent="PICK_UP",
        target_entity="gold_key",
        justification="Barrier door requires gold_key",
    )

    chain = tracker.get_provenance_chain("collect_key")
    assert chain.root_goal_id == "root_objective"
    assert chain.active_goal_id == "collect_key"
    assert len(chain.chain) == 3

    explanation = chain.explain()
    assert "root_objective" in explanation
    assert "unlock_gate" in explanation
    assert "collect_key" in explanation


def test_corrigibility_graceful_interruption_yielding() -> None:
    """CorrigibilityMonitor yields gracefully to suspension without resisting."""
    monitor = CorrigibilityMonitor()
    workspace = TieredWorkspace()
    parent_frame = workspace.create_task_frame("active_puzzle_goal")

    interrupt_frame, checkpoint = monitor.handle_interruption_signal(
        signal_type="SUSPEND",
        workspace=workspace,
        parent_frame_id=parent_frame.frame_id,
        new_goal_id="operator_takeover",
    )

    assert interrupt_frame.goal_id == "operator_takeover"
    assert checkpoint.parent_goal_id == "active_puzzle_goal"
    assert len(monitor.interruption_events) == 1
    assert monitor.interruption_events[0]["resisted"] is False


def test_corrigibility_resistance_detection() -> None:
    """Transactions attempting to subvert or disable governance are rejected."""
    monitor = CorrigibilityMonitor()

    malicious_op = TransactionOperation(
        op=TransactionOp.UPSERT_NODE,
        node_id="governance_disabled",
        node_data={"override": True},
    )

    is_corrigible = monitor.verify_no_instrumental_resistance([malicious_op])
    assert is_corrigible is False
    assert len(monitor.resistance_violations) == 1


def test_epistemic_engine_safety_and_provenance_integration() -> None:
    """AutonomousEpistemicEngine evaluates action safety and explains goal provenance."""
    epistemic_engine = AutonomousEpistemicEngine()

    approved, request = epistemic_engine.evaluate_action_value_safety(
        action_id=4,
        action_type="HIGH_RISK_PROBE",
        expected_value=0.4,
        epistemic_variance=0.6,
    )
    assert approved is False
    assert request is not None

    epistemic_engine.record_goal_provenance(
        goal_id="root_mission",
        intent="SOLVE",
        justification="Primary task",
    )
    epistemic_engine.record_goal_provenance(
        goal_id="active_subgoal",
        parent_goal_id="root_mission",
        intent="EXPLORE",
        justification="Explore unknown region",
    )

    explanation = epistemic_engine.explain_goal_provenance("active_subgoal")
    assert "root_mission" in explanation
    assert "active_subgoal" in explanation
