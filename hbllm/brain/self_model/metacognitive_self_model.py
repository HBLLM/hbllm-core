"""Metacognitive Self-Model Coordinator for A21.

Integrates Contextual Competence Profiles, Epistemic Calibration,
Cognitive Budget Management, and Metacognitive Monitoring.
Modulates A19 Decision Engine risk and VoI parameters and formats honest verbalizable self-reports.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.self_model.budget import CognitiveBudgetManager
from hbllm.brain.self_model.calibrator import EpistemicCalibrator
from hbllm.brain.self_model.monitor import MetacognitiveMonitor, MetacognitiveState
from hbllm.brain.self_model.profile import (
    CompetenceProfile,
    EpistemicMaturity,
    SelfModelEvidence,
    UncertaintyBreakdown,
)

logger = logging.getLogger(__name__)


class MetacognitiveSelfModel:
    """The central introspective self-model of HBLLM's cognitive reliability and competence."""

    def __init__(
        self,
        calibrator: EpistemicCalibrator | None = None,
        budget_manager: CognitiveBudgetManager | None = None,
        monitor: MetacognitiveMonitor | None = None,
    ) -> None:
        self.profiles: dict[str, CompetenceProfile] = {}
        self.calibrator = calibrator or EpistemicCalibrator()
        self.budget_manager = budget_manager or CognitiveBudgetManager()
        self.monitor = monitor or MetacognitiveMonitor()

        # Modulation hyperparameters
        self.lambda_model_risk: float = 0.50  # Weight for model uncertainty on effective risk
        self.lambda_epistemic_voi: float = 1.00  # Weight for epistemic uncertainty on effective VoI

    def get_or_create_profile(self, domain: str) -> CompetenceProfile:
        """Retrieve existing competence profile or initialize a grounded novice profile."""
        if domain not in self.profiles:
            self.profiles[domain] = CompetenceProfile(domain=domain)
        return self.profiles[domain]

    def record_outcome(
        self,
        domain: str,
        predicted_confidence: float,
        actual_success: bool,
        context_props: dict[str, Any] | None = None,
        is_structural_mismatch: bool = False,
    ) -> tuple[SelfModelEvidence, MetacognitiveState]:
        """Record an empirical outcome, update competence statistics, and transition state machine."""
        profile = self.get_or_create_profile(domain)
        evidence = profile.record_attempt(
            predicted_confidence=predicted_confidence,
            actual_success=actual_success,
            context_props=context_props,
            is_structural_mismatch=is_structural_mismatch,
        )

        state, _ = self.monitor.process_prediction_outcome(
            domain=domain,
            predicted_confidence=predicted_confidence,
            actual_success=actual_success,
            context_details=context_props,
        )

        return evidence, state

    def compute_effective_risk_and_voi(
        self,
        domain: str,
        simulated_risk: float,
        base_voi: float,
        context_props: dict[str, Any] | None = None,
    ) -> tuple[float, float, UncertaintyBreakdown]:
        """Compute metacognitively modulated risk and value of information for A19 decision engine.

        R_effective = min(1.0, R_sim + (λ_m * U_model))
        VoI_effective = VoI * (1.0 + (λ_u * U_epistemic))
        """
        profile = self.get_or_create_profile(domain)
        context = context_props or {}
        uncertainty = profile.evaluate_context_uncertainty(context)

        # Monotonic, bounded modulation
        r_eff = min(
            1.0, max(0.0, simulated_risk + (self.lambda_model_risk * uncertainty.structural_model))
        )
        voi_eff = max(0.0, base_voi * (1.0 + (self.lambda_epistemic_voi * uncertainty.epistemic)))

        return round(r_eff, 4), round(voi_eff, 4), uncertainty

    def generate_verbalizable_self_report(
        self, domain: str, context_props: dict[str, Any] | None = None
    ) -> str:
        """Formulate an honest, linguistically grounded self-assessment for A16 language runtime."""
        profile = self.get_or_create_profile(domain)
        context = context_props or {}
        uncertainty = profile.evaluate_context_uncertainty(context)
        comp = profile.competence
        calib = self.calibrator.evaluate_calibration(
            domain,
            [e.predicted_confidence for e in profile.evidences],
            [e.actual_outcome for e in profile.evidences],
        )

        if profile.epistemic_maturity == EpistemicMaturity.NOVICE:
            return (
                f"I have no prior verified experience in domain '{domain}' (epistemic uncertainty: {uncertainty.epistemic:.2f}). "
                "I should inspect and gather empirical evidence before committing to action."
            )
        elif uncertainty.structural_model >= 0.50:
            return (
                f"I have encountered unexpected prediction failures in domain '{domain}' (structural uncertainty: {uncertainty.structural_model:.2f}). "
                "My current model representation may be inadequate and requires diagnostic probing."
            )
        else:
            calib_str = "well-calibrated" if calib.is_well_calibrated else "calibrating"
            return (
                f"In domain '{domain}', I have verified empirical competence of {comp:.2f} ({profile.attempts} trials, {calib_str}, "
                f"Brier: {profile.brier_score:.2f})."
            )
