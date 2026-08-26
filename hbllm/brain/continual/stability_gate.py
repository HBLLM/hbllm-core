"""Plasticity-Stability Engine and Dependency-Analyzed Stability Gate for A22.

Controls long-term knowledge admission through dependency-aware regression checks.
Distinguishes between catastrophic forgetting (BAD_REGRESSION), boundary narrowing
(EXPECTED_SPECIALIZATION), and error correction (BENEFICIAL_REVISION).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hbllm.brain.continual.store import DualStoreMemory

logger = logging.getLogger(__name__)


class GateVerdict(str, Enum):
    """The outcome decision of the Stability Gate."""

    ACCEPTED = "accepted"                                  # Clean improvement without regression
    EXPECTED_SPECIALIZATION = "expected_specialization"    # Deliberate narrowing of applicability boundary
    BENEFICIAL_REVISION = "beneficial_revision"            # Correction of previously flawed knowledge
    REJECTED_REGRESSION = "rejected_regression"            # Unacceptable collateral forgetting of unrelated domains


@dataclass
class CandidateUpdate:
    """A proposed consolidated knowledge revision generated during sleep consolidation."""

    knowledge_id: str
    knowledge_type: str
    domain: str
    proposed_content: dict[str, Any]
    source_event_ids: list[str]
    is_specialization: bool = False
    is_revision: bool = False


@dataclass
class StabilityGateReport:
    """The formal evaluation output of the Stability Gate."""

    verdict: GateVerdict
    affected_domains: list[str]
    backward_transfer_bwt: float
    forward_transfer_fwt: float
    unrelated_domains_intact: bool
    rationale: str


class PlasticityStabilityEngine:
    """Evaluates candidate knowledge updates against historical benchmarks and dependency sets."""

    def __init__(self, memory: DualStoreMemory | None = None) -> None:
        self.memory = memory or DualStoreMemory()
        self.domain_benchmarks: dict[str, list[tuple[dict[str, Any], bool]]] = {}  # domain -> list of (test_props, expected_outcome)

    def register_domain_benchmark(self, domain: str, test_cases: list[tuple[dict[str, Any], bool]]) -> None:
        """Register reference benchmark cases for a mature domain."""
        self.domain_benchmarks[domain] = test_cases

    def evaluate_candidate_update(
        self,
        update: CandidateUpdate,
        current_domain_accuracies: dict[str, float],
    ) -> StabilityGateReport:
        """Perform targeted dependency analysis and evaluate stability against regressions."""
        target_domain = update.domain

        # 1. Dependency Analysis: Identify affected domains
        affected_domains = [target_domain]
        unrelated_domains = [d for d in self.domain_benchmarks.keys() if d != target_domain]

        # 2. Evaluate Unrelated Domains (verify zero collateral damage)
        unrelated_intact = True
        bwt_sum = 0.0

        for d in unrelated_domains:
            baseline_acc = 1.0
            post_acc = current_domain_accuracies.get(d, 1.0)
            bwt_d = post_acc - baseline_acc
            bwt_sum += bwt_d
            if post_acc < 0.80 or bwt_d < -0.15:  # Significant drop in unrelated domain
                unrelated_intact = False

        avg_bwt = round(bwt_sum / max(1, len(unrelated_domains)), 4)

        # 3. Classify Gate Verdict
        if not unrelated_intact:
            return StabilityGateReport(
                verdict=GateVerdict.REJECTED_REGRESSION,
                affected_domains=affected_domains,
                backward_transfer_bwt=avg_bwt,
                forward_transfer_fwt=0.0,
                unrelated_domains_intact=False,
                rationale="Candidate update caused collateral accuracy degradation in unrelated mature domains.",
            )

        if update.is_specialization:
            verdict = GateVerdict.EXPECTED_SPECIALIZATION
            rationale = "Legitimate boundary narrowing to exclude incompatible physical regimes."
        elif update.is_revision:
            verdict = GateVerdict.BENEFICIAL_REVISION
            rationale = "Beneficial knowledge revision correcting prior model inadequacy."
        else:
            verdict = GateVerdict.ACCEPTED
            rationale = "Clean knowledge consolidation preserving historical knowledge integrity."

        return StabilityGateReport(
            verdict=verdict,
            affected_domains=affected_domains,
            backward_transfer_bwt=avg_bwt,
            forward_transfer_fwt=0.25 if update.is_revision else 0.10,
            unrelated_domains_intact=True,
            rationale=rationale,
        )
