"""
Predictor Disagreement Analyzer — Measures Predictor Variance & Triggers Attention Salience.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DisagreementReport:
    """Analysis report measuring variance across component predictions."""

    disagreement_score: float
    high_disagreement: bool
    component_confidences: dict[str, float] = field(default_factory=dict)


class PredictorDisagreementAnalyzer:
    """Analyzes variance across ensemble predictor outputs."""

    def analyze_disagreement(
        self, component_predictions: dict[str, tuple[dict[str, Any], float]]
    ) -> DisagreementReport:
        """Calculate variance across predictor confidences and output states."""
        if not component_predictions:
            return DisagreementReport(disagreement_score=0.0, high_disagreement=False)

        confidences = {name: res[1] for name, res in component_predictions.items()}
        conf_values = list(confidences.values())
        mean_conf = sum(conf_values) / len(conf_values)
        variance = sum((c - mean_conf) ** 2 for c in conf_values) / len(conf_values)
        disagreement_score = min(1.0, variance * 10.0)

        high_disagreement = disagreement_score > 0.15
        logger.debug(
            "DisagreementAnalyzer calculated variance=%.4f score=%.4f (high=%s)",
            variance,
            disagreement_score,
            high_disagreement,
        )
        return DisagreementReport(
            disagreement_score=disagreement_score,
            high_disagreement=high_disagreement,
            component_confidences=confidences,
        )

    @staticmethod
    def detect_ambiguity(
        hypotheses: list[dict[str, Any]],
        confidence_delta_threshold: float = 0.05,
    ) -> tuple[bool, float]:
        """W126: Ambiguity detection when multiple competing hypotheses share near-identical confidence."""
        if len(hypotheses) < 2:
            return False, 0.0

        confs = [float(h.get("confidence", 0.5)) for h in hypotheses]
        confs.sort(reverse=True)
        top1, top2 = confs[0], confs[1]
        delta = abs(top1 - top2)
        is_ambiguous = delta <= confidence_delta_threshold
        ambiguity_level = max(0.0, 1.0 - delta)
        return is_ambiguous, ambiguity_level

    @staticmethod
    def detect_contradiction(
        pred_a: dict[str, Any],
        pred_b: dict[str, Any],
    ) -> list[str]:
        """W128: Contradiction detection identifying direct conflicts on categorical variables."""
        contradictions: list[str] = []
        for k in set(pred_a.keys()) & set(pred_b.keys()):
            va, vb = pred_a[k], pred_b[k]
            if va != vb:
                contradictions.append(f"Conflict on variable '{k}': {va} vs {vb}")
        return contradictions
