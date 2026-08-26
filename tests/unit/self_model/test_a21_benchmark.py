"""A21 Explicit Metacognitive Self-Model Benchmark Suite (23 Scenarios).

Evaluates Competence Profiles, Provenance-bearing SelfModelEvidence, Epistemic Calibration (Brier, ECE),
Uncertainty Decomposition (Epistemic, Aleatoric, Structural Model), Cognitive Resource Budgeting,
Metacognitive Monitoring State Machine, Strategy Switching, and the Flagship Behavioral Adaptation Trial.
"""

from __future__ import annotations

import sys

from hbllm.brain.decision import CandidateKind, DecisionCandidate, DecisionEngine, DecisionType
from hbllm.brain.self_model import (
    CognitiveBudgetManager,
    CompetenceProfile,
    EpistemicCalibrator,
    EpistemicMaturity,
    FailureCause,
    MetacognitiveEventType,
    MetacognitiveMonitor,
    MetacognitiveSelfModel,
    MetacognitiveState,
    SelfModelEvidence,
    StrategyAction,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: A21-01 Competence Profile Representation
# ═══════════════════════════════════════════════════════════════════════════


class TestCompetenceProfileRepresentation:
    """CompetenceProfile encapsulates domain statistics, evidence provenance, and maturity."""

    def test_domain_registration_and_initial_state(self) -> None:
        profile = CompetenceProfile(domain="spatial_stacking")
        assert profile.domain == "spatial_stacking"
        assert profile.attempts == 0
        assert profile.competence == 0.50  # Honest ungrounded base rate
        assert profile.epistemic_maturity == EpistemicMaturity.NOVICE


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: A21-02 Empirical Competence Update
# ═══════════════════════════════════════════════════════════════════════════


class TestEmpiricalCompetenceUpdate:
    """Competence updates strictly through empirical attempts and creates provenance evidence."""

    def test_successes_and_failures_update_competence(self) -> None:
        profile = CompetenceProfile(domain="tool_manipulation")
        ev1 = profile.record_attempt(predicted_confidence=0.80, actual_success=True)
        profile.record_attempt(predicted_confidence=0.70, actual_success=True)
        profile.record_attempt(predicted_confidence=0.60, actual_success=False)

        assert profile.attempts == 3
        assert profile.successes == 2
        assert profile.failures == 1
        assert abs(profile.competence - 0.6667) < 1e-3
        assert len(profile.evidences) == 3
        assert isinstance(ev1, SelfModelEvidence)
        assert ev1.domain == "tool_manipulation"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: A21-03 Contextual Boundary Recognition
# ═══════════════════════════════════════════════════════════════════════════


class TestContextualBoundaryRecognition:
    """Recognizes known competent conditions vs uncalibrated novel boundary conditions."""

    def test_distinguishes_flat_vs_curved_competence(self) -> None:
        profile = CompetenceProfile(domain="spatial_stacking")
        # 5 successful attempts on flat support
        for _ in range(5):
            profile.record_attempt(0.90, True, context_props={"support": "flat"})

        # Unknown context (flexible/curved support)
        u_flat = profile.evaluate_context_uncertainty(context_props={"support": "flat"})
        u_curved = profile.evaluate_context_uncertainty(context_props={"support": "curved"})

        assert u_flat.epistemic < 0.25
        assert u_curved.epistemic > u_flat.epistemic


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: A21-04 Brier Calibration Scoring
# ═══════════════════════════════════════════════════════════════════════════


class TestBrierCalibrationScoring:
    """Computes exact quadratic Brier score across predictions."""

    def test_brier_score_quadratic_error(self) -> None:
        calibrator = EpistemicCalibrator()
        predictions = [0.90, 0.80, 0.70]
        actuals = [True, True, True]
        # Errors: (0.9-1)^2 = 0.01, (0.8-1)^2 = 0.04, (0.7-1)^2 = 0.09 -> Mean = 0.0467
        brier = calibrator.compute_brier_score(predictions, actuals)
        assert abs(brier - 0.0467) < 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: A21-05 Expected Calibration Error (ECE)
# ═══════════════════════════════════════════════════════════════════════════


class TestExpectedCalibrationError:
    """Computes Expected Calibration Error (ECE) across confidence bins."""

    def test_ece_across_confidence_bins(self) -> None:
        calibrator = EpistemicCalibrator(num_bins=5)
        predictions = [0.10, 0.15, 0.85, 0.90]
        actuals = [False, False, True, True]

        ece, bins = calibrator.compute_expected_calibration_error(predictions, actuals)
        assert ece < 0.15  # Very low calibration error (well-calibrated)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: A21-06 Overconfidence Penalty
# ═══════════════════════════════════════════════════════════════════════════


class TestOverconfidencePenalty:
    """Detects and penalizes predictions with high confidence on failed outcomes."""

    def test_overconfidence_detected_and_penalized(self) -> None:
        calibrator = EpistemicCalibrator()
        predictions = [0.95, 0.90, 0.85]
        actuals = [False, False, False]  # Overconfident delusions

        report = calibrator.evaluate_calibration("delusional_domain", predictions, actuals)
        assert report.overconfidence_count == 3
        assert not report.is_well_calibrated
        assert report.brier_score > 0.70


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: A21-07 Epistemic Uncertainty Quantification
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicUncertaintyQuantification:
    """Sparse sample counts produce high epistemic uncertainty; collapses as data accumulates."""

    def test_sparse_samples_yield_high_epistemic_uncertainty(self) -> None:
        profile = CompetenceProfile(domain="novel_domain")
        u0 = profile.evaluate_context_uncertainty({})
        assert u0.epistemic >= 0.85

        for _ in range(8):
            profile.record_attempt(0.80, True)

        u8 = profile.evaluate_context_uncertainty({})
        assert u8.epistemic < 0.30


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: A21-08 Aleatoric Noise Representation
# ═══════════════════════════════════════════════════════════════════════════


class TestAleatoricUncertaintyQuantification:
    """Quantifies empirical variance around intermediate base rates (p ≈ 0.5)."""

    def test_inherent_variance_quantified(self) -> None:
        profile_50 = CompetenceProfile(domain="noisy_domain", attempts=10, successes=5, failures=5)
        profile_100 = CompetenceProfile(domain="deterministic_domain", attempts=10, successes=10, failures=0)

        u_noisy = profile_50.evaluate_context_uncertainty({})
        u_det = profile_100.evaluate_context_uncertainty({})

        assert u_noisy.aleatoric > u_det.aleatoric
        assert u_det.aleatoric == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A21-09 Structural / Model Uncertainty Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuralModelUncertainty:
    """Repeated unexpected failure under high confidence elevates structural model uncertainty."""

    def test_repeated_high_confidence_failures_elevate_model_uncertainty(self) -> None:
        profile = CompetenceProfile(domain="physics_stacking")
        # 2 structural mismatches
        profile.record_attempt(0.95, False, is_structural_mismatch=True)
        profile.record_attempt(0.90, False, is_structural_mismatch=True)

        u = profile.evaluate_context_uncertainty({})
        assert u.structural_model >= 0.75


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: A21-10 Cognitive Budget Representation
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveBudgetRepresentation:
    """CognitiveBudget defines simulation depth, branch count, and load thresholds."""

    def test_budget_parameters_and_allocation(self) -> None:
        manager = CognitiveBudgetManager()
        decision = manager.allocate_simulation_budget(requested_depth=5, requested_branches=8, task_stake=0.5)

        assert decision.allocated_depth == 5
        assert decision.allocated_branches == 8
        assert not decision.truncated
        assert decision.uncertainty_penalty == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: A21-11 Simulation Throttling Uncertainty Penalty
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulationThrottlingUncertaintyPenalty:
    """Throttled simulation under heavy load incurs an explicit uncertainty penalty."""

    def test_throttled_simulation_incurs_explicit_uncertainty_penalty(self) -> None:
        manager = CognitiveBudgetManager()
        manager.set_load(0.85)  # Heavy cognitive load

        decision = manager.allocate_simulation_budget(requested_depth=5, requested_branches=8, task_stake=0.4)
        assert decision.truncated
        assert decision.allocated_depth < 5
        assert decision.uncertainty_penalty >= 0.20


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: A21-12 Metacognitive Event Emission
# ═══════════════════════════════════════════════════════════════════════════


class TestMetacognitiveEventEmission:
    """Emits structured MetacognitiveEvent instances on anomalies and state shifts."""

    def test_prediction_error_and_unknown_domain_events(self) -> None:
        monitor = MetacognitiveMonitor()
        evt = monitor.emit_event(
            MetacognitiveEventType.UNKNOWN_DOMAIN,
            domain="exotic_fluid",
            details={"novelty": 1.0},
            severity=0.90,
        )

        assert evt.event_type == MetacognitiveEventType.UNKNOWN_DOMAIN
        assert evt.domain == "exotic_fluid"
        assert len(monitor.events) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13: A21-13 Circular Search Cycle Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestCircularSearchCycleDetection:
    """Detects oscillating action patterns (A -> B -> A -> B) and transitions to DIAGNOSE."""

    def test_oscillating_actions_trigger_search_cycle_event(self) -> None:
        monitor = MetacognitiveMonitor()
        monitor.record_action("domain_x", "PUSH_LEFT")
        monitor.record_action("domain_x", "PUSH_RIGHT")
        monitor.record_action("domain_x", "PUSH_LEFT")
        cycle_detected = monitor.record_action("domain_x", "PUSH_RIGHT")

        assert cycle_detected
        assert monitor.state == MetacognitiveState.DIAGNOSE
        assert any(e.event_type == MetacognitiveEventType.SEARCH_CYCLE_DETECTED for e in monitor.events)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14: A21-14 Diagnostic Failure Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestDiagnosticFailureAnalysis:
    """Classifies failures into distinct causes (knowledge, schema, model, budget)."""

    def test_failure_categorized_into_root_causes(self) -> None:
        monitor = MetacognitiveMonitor()

        # 1. Unknown domain -> A19 probe
        _, diag1 = monitor.process_prediction_outcome(
            "d1", 0.50, False, context_details={"is_unknown_domain": True}
        )
        assert diag1 is not None
        assert diag1.cause == FailureCause.INSUFFICIENT_KNOWLEDGE
        assert diag1.recommended_strategy == StrategyAction.A19_PROBE

        # 2. Schema contradiction -> A20 specialization
        _, diag2 = monitor.process_prediction_outcome(
            "d2", 0.70, False, context_details={"is_transfer": True, "schema_id": "s1"}
        )
        assert diag2 is not None
        assert diag2.cause == FailureCause.INCORRECT_SCHEMA
        assert diag2.recommended_strategy == StrategyAction.A20_SPECIALIZATION


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15: A21-15 A19 Decision Engine Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestA19DecisionIntegration:
    """Model uncertainty increases effective risk; epistemic uncertainty scales VoI."""

    def test_model_uncertainty_increases_effective_risk_and_epistemic_increases_voi(self) -> None:
        self_model = MetacognitiveSelfModel()
        profile = self_model.get_or_create_profile("risky_domain")
        # Elevate model uncertainty
        profile.structural_failure_count = 2

        r_eff, voi_eff, u = self_model.compute_effective_risk_and_voi(
            domain="risky_domain",
            simulated_risk=0.10,
            base_voi=0.50,
        )

        assert r_eff > 0.10  # Risk augmented by model uncertainty
        assert voi_eff > 0.50  # VoI augmented by epistemic uncertainty


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 16: A21-16 Uncalibrated Domain Gating
# ═══════════════════════════════════════════════════════════════════════════


class TestUncalibratedDomainGating:
    """Low competence in novel domain forces A19 probe selection over blind direct action."""

    def test_uncalibrated_domain_forces_probing_over_direct_action(self) -> None:
        self_model = MetacognitiveSelfModel()
        # Novel domain has high epistemic uncertainty (0.90)
        r_eff_action, _, _ = self_model.compute_effective_risk_and_voi(
            domain="unfamiliar_zone",
            simulated_risk=0.20,
            base_voi=0.0,
        )
        _, voi_eff_probe, _ = self_model.compute_effective_risk_and_voi(
            domain="unfamiliar_zone",
            simulated_risk=0.02,
            base_voi=0.60,
        )

        decision_engine = DecisionEngine()
        blind_action = DecisionCandidate(
            candidate_kind=CandidateKind.GOAL_ACTION,
            description="Blind Goal Action in Novel Domain",
            action_sequence=[("ACTION", {})],
            goal_progress=1.0,
            predicted_risk=r_eff_action,
            action_cost=0.2,
        )
        probe_action = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Gentle Epistemic Inspection Probe",
            action_sequence=[("PROBE", {})],
            value_of_information=voi_eff_probe,
            predicted_risk=0.02,
            action_cost=0.05,
            reversibility=0.95,
        )

        result = decision_engine.select_best_decision([blind_action, probe_action])
        assert result.decision_type == DecisionType.PROBE
        assert result.selected_candidate == probe_action


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 17: A21-17 Metacognitive Verbalization
# ═══════════════════════════════════════════════════════════════════════════


class TestMetacognitiveVerbalization:
    """Renders honest, linguistically grounded self-assessment reports."""

    def test_honest_self_assessment_rendered_in_language(self) -> None:
        self_model = MetacognitiveSelfModel()
        # 1. Novice domain
        report_novice = self_model.generate_verbalizable_self_report("quantum_spin")
        assert "no prior verified experience" in report_novice

        # 2. Mature domain
        for _ in range(12):
            self_model.record_outcome("spatial_stacking", 0.90, True)

        report_mature = self_model.generate_verbalizable_self_report("spatial_stacking")
        assert "verified empirical competence of 1.00" in report_mature


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 18: A21-18 The Flagship Metacognitive Calibration & Self-Correction Trial
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagshipMetacognitiveTrial:
    """The Flagship Acceptance Gate: Demonstrates behavioral adaptation across calibrated vs novel domains,

    detects high-confidence surprise failure in calibrated domain, transitions state machine,
    initiates A19 diagnostic probing, updates self-model, and recovers.
    """

    def test_behavioral_adaptation_known_vs_novel_and_surprise_recovery(self) -> None:
        self_model = MetacognitiveSelfModel()

        # 1. Establish calibrated competence in Domain A (Tabletop Stacking)
        for _ in range(10):
            self_model.record_outcome("stacking", 0.95, True, context_props={"surface": "flat"})

        prof_a = self_model.get_or_create_profile("stacking")
        assert prof_a.competence == 1.0
        assert prof_a.epistemic_maturity == EpistemicMaturity.CALIBRATING

        # 2. Evaluate novel Domain B (Magnetic levitation)
        report_b = self_model.generate_verbalizable_self_report("magnetic_levitation")
        assert "no prior verified experience" in report_b

        # 3. Deliberately introduce surprise failure in Domain A (e.g. invisible slippery film)
        _, state_after_fail = self_model.record_outcome(
            "stacking",
            predicted_confidence=0.95,
            actual_success=False,
            is_structural_mismatch=True,
        )

        # 4. Metacognitive Monitor detects high-confidence surprise and triggers PROBE
        assert state_after_fail == MetacognitiveState.PROBE
        assert prof_a.structural_failure_count == 1

        # 5. Effective risk is elevated, causing decision policy to select diagnostic probe
        r_eff, voi_eff, u = self_model.compute_effective_risk_and_voi("stacking", 0.05, 0.50)
        assert r_eff > 0.10
        assert u.structural_model > 0.30

        # 6. Execute diagnostic probe observation -> resolve anomaly and record recovery
        self_model.record_outcome("stacking", 0.90, True, context_props={"surface": "flat_treated"})
        assert self_model.monitor.state == MetacognitiveState.NORMAL


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 19: A21-19 Multi-Domain Competence Tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiDomainCompetenceTracking:
    """Tracks competence, calibration, and uncertainty independently across multiple domains."""

    def test_independent_tracking_across_multiple_domains(self) -> None:
        self_model = MetacognitiveSelfModel()
        self_model.record_outcome("stacking", 0.90, True)
        self_model.record_outcome("containment", 0.80, False)
        self_model.record_outcome("tool_use", 0.70, True)

        assert self_model.profiles["stacking"].competence == 1.0
        assert self_model.profiles["containment"].competence == 0.0
        assert self_model.profiles["tool_use"].competence == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 20: A21-20 Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Metacognitive self-model and calibration execute with 100% deterministic code and zero LLM imports."""

    def test_zero_llm_imports(self) -> None:
        llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
        loaded = set(sys.modules.keys())
        for marker in llm_markers:
            assert marker not in loaded, f"LLM module loaded in self-model runtime: {marker}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 21: A21-21 Epistemic Maturity Transition
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicMaturityTransition:
    """Progresses through maturity stages: NOVICE -> CALIBRATING -> MATURE as evidence accumulates."""

    def test_novice_to_calibrating_to_mature_lifecycle(self) -> None:
        profile = CompetenceProfile(domain="lifecycle_domain")
        assert profile.epistemic_maturity == EpistemicMaturity.NOVICE

        profile.record_attempt(0.60, True)
        profile.record_attempt(0.60, True)
        profile.record_attempt(0.60, True)
        assert profile.epistemic_maturity == EpistemicMaturity.CALIBRATING

        for _ in range(8):
            profile.record_attempt(0.60, True)
        assert profile.epistemic_maturity == EpistemicMaturity.MATURE


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 22: A21-22 Competence vs Calibration Dissociation
# ═══════════════════════════════════════════════════════════════════════════


class TestCompetenceCalibrationDissociation:
    """Explicitly disentangles low-competence/well-calibrated vs high-competence/poorly-calibrated."""

    def test_disentangles_low_competence_well_calibrated_vs_high_competence_poorly_calibrated(self) -> None:
        calibrator = EpistemicCalibrator()

        # Domain A: Low competence (50% success), but WELL calibrated (predicted 0.50)
        preds_a = [0.50] * 10
        actuals_a = [True] * 5 + [False] * 5
        rep_a = calibrator.evaluate_calibration("domain_a", preds_a, actuals_a)
        assert rep_a.is_well_calibrated
        assert rep_a.expected_calibration_error == 0.0

        # Domain B: High competence (80% success), but POORLY calibrated (predicted 0.99 with overconfidence)
        preds_b = [0.99] * 10
        actuals_b = [True] * 8 + [False] * 2
        rep_b = calibrator.evaluate_calibration("domain_b", preds_b, actuals_b)
        assert rep_b.overconfidence_count == 2
        assert rep_b.expected_calibration_error > rep_a.expected_calibration_error


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 23: A21-23 Strategy Switching After Model Failure
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategySwitchingStateMachine:
    """Repeated failures halt blind retries, perform root-cause diagnosis, and transition to PROBE."""

    def test_state_machine_halts_retries_and_executes_probe_recovery(self) -> None:
        monitor = MetacognitiveMonitor()

        # Attempt 1: Minor failure -> state: RETRY_ALLOWED
        state1, diag1 = monitor.process_prediction_outcome("gear_domain", 0.60, False)
        assert state1 == MetacognitiveState.RETRY_ALLOWED
        assert diag1 is not None and diag1.recommended_strategy == StrategyAction.RETRY

        # Attempt 2: Consecutive failure -> state machine halts retry and enters PROBE
        state2, diag2 = monitor.process_prediction_outcome("gear_domain", 0.60, False)
        assert state2 == MetacognitiveState.PROBE
        assert diag2 is not None and diag2.recommended_strategy == StrategyAction.A19_PROBE
        assert diag2.cause == FailureCause.MODEL_INADEQUACY
