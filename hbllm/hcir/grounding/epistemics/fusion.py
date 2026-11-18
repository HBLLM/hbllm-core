from __future__ import annotations

import math
from dataclasses import dataclass, field

from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord


@dataclass(frozen=True)
class FusionResult:
    """The result of multi-source Bayesian evidence fusion across dependency groups."""

    posterior_probability: float
    effective_evidence_strength: float
    independent_group_count: int
    total_evidence_count: int
    group_summaries: dict[str, dict[str, float]] = field(default_factory=dict)


class DependencyGroupEvidenceFusion:
    """Bayesian evidence fusion engine enforcing strict dependency-group isolation.

    Prevents artificial Bayesian inflation when multiple perception models evaluate
    the same raw sensory data or share underlying neural backbones. Aggregates evidence
    intra-group first, then applies Bayesian updating across independent groups.
    """

    def __init__(self, default_prior: float = 0.5) -> None:
        if not (0.0 < default_prior < 1.0):
            raise ValueError(f"default_prior must be strictly in (0.0, 1.0), got {default_prior}")
        self._default_prior = default_prior

    def fuse(
        self,
        evidence_records: list[EvidenceRecord],
        prior_probability: float | None = None,
    ) -> FusionResult:
        """Executes dependency-group clustered Bayesian updating."""
        if not evidence_records:
            prior = prior_probability if prior_probability is not None else self._default_prior
            return FusionResult(
                posterior_probability=prior,
                effective_evidence_strength=0.0,
                independent_group_count=0,
                total_evidence_count=0,
            )

        prior = prior_probability if prior_probability is not None else self._default_prior
        prior_clamped = max(0.001, min(0.999, prior))
        odds = prior_clamped / (1.0 - prior_clamped)

        # 1. Cluster by dependency group
        groups: dict[str, list[EvidenceRecord]] = {}
        for record in evidence_records:
            groups.setdefault(record.dependency_group, []).append(record)

        group_summaries: dict[str, dict[str, float]] = {}
        one_minus_strength_product = 1.0

        # 2. Intra-group aggregation & inter-group Bayesian multiplication
        for group_id, records in groups.items():
            total_weight = sum(r.evidence_strength for r in records)
            if total_weight > 0.0:
                # Precision-weighted log-likelihood-ratio average
                weighted_log_lr = (
                    sum(
                        math.log(
                            max(
                                0.001,
                                min(
                                    1000.0,
                                    r.likelihood_ratio
                                    if r.likelihood_ratio is not None
                                    else (
                                        max(0.001, min(0.999, r.calibrated_likelihood))
                                        / (1.0 - max(0.001, min(0.999, r.calibrated_likelihood)))
                                    ),
                                ),
                            )
                        )
                        * r.evidence_strength
                        for r in records
                    )
                    / total_weight
                )
                group_lr = math.exp(weighted_log_lr)
                group_likelihood = group_lr / (1.0 + group_lr)
            else:
                group_likelihood = sum(r.calibrated_likelihood for r in records) / len(records)
                p_clamped = max(0.001, min(0.999, group_likelihood))
                group_lr = p_clamped / (1.0 - p_clamped)

            group_strength = max(r.evidence_strength for r in records)
            one_minus_strength_product *= 1.0 - group_strength

            group_summaries[group_id] = {
                "records_count": float(len(records)),
                "aggregated_likelihood": group_likelihood,
                "group_strength": group_strength,
                "group_likelihood_ratio": group_lr,
            }

            # 3. Inter-group Bayesian update (with likelihood damping by sensor strength)
            # Dampen likelihood ratio by sensor strength SNR: LR^S
            damped_ratio = math.pow(group_lr, group_strength)
            odds *= damped_ratio

        # Convert odds back to posterior probability
        posterior = odds / (1.0 + odds)
        posterior = max(0.001, min(0.999, posterior))
        effective_strength = 1.0 - one_minus_strength_product

        return FusionResult(
            posterior_probability=posterior,
            effective_evidence_strength=max(0.0, min(1.0, effective_strength)),
            independent_group_count=len(groups),
            total_evidence_count=len(evidence_records),
            group_summaries=group_summaries,
        )
