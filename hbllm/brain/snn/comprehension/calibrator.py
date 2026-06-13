"""
SNN Parameter Calibrator.

Adjusts SNN neuron parameters based on downstream success signals
from the EvaluationNode.  Uses reinforcement-style parameter nudging
correlated with response quality — NOT backpropagation through spikes.
"""

from __future__ import annotations

from hbllm.brain.snn.comprehension.ensemble import DOMAIN_PARAMS


class SNNCalibrator:
    """Adjusts SNN parameters based on downstream success signals.

    Uses EvaluationNode feedback:
      - "Was segmentation helpful?" (response quality improved)
      - "Did memory retrieval improve response?" (memory relevance)
      - "Did concept boundaries align with successful reasoning?"

    Tuning method: simple parameter nudging based on outcome correlation.
    NOT backpropagation through spikes.
    """

    def __init__(self) -> None:
        # Per-domain parameter history
        self._param_history: dict[str, list[dict]] = {}
        # EMA of success signal per domain
        self._domain_success: dict[str, float] = {}
        self._alpha = 0.1  # EMA update rate

    def record_outcome(
        self,
        domain: str,
        params_used: dict[str, float],
        num_concepts: int,
        response_quality: float,
        memory_relevance: float,
    ) -> None:
        """Record outcome for parameter tuning.

        Args:
            domain: The domain of the query.
            params_used: SNN parameters that were active during this query.
            num_concepts: Number of concept boundaries detected.
            response_quality: Quality score from EvaluationNode (0.0-1.0).
            memory_relevance: Memory hit rate / relevance score (0.0-1.0).
        """
        success = (response_quality + memory_relevance) / 2.0

        current = self._domain_success.get(domain, 0.5)
        self._domain_success[domain] = current * (1 - self._alpha) + success * self._alpha

        if domain not in self._param_history:
            self._param_history[domain] = []

        self._param_history[domain].append(
            {
                "params": params_used,
                "concepts": num_concepts,
                "success": success,
            }
        )

        # Keep last 100 entries
        if len(self._param_history[domain]) > 100:
            self._param_history[domain] = self._param_history[domain][-100:]

    def suggest_params(self, domain: str) -> dict[str, float]:
        """Suggest adjusted parameters based on history.

        Returns the default domain params if insufficient history
        is available, otherwise nudges toward parameters that
        correlated with highest success.
        """
        history = self._param_history.get(domain, [])
        if len(history) < 10:
            return dict(DOMAIN_PARAMS.get(domain, DOMAIN_PARAMS["general"]))

        # Find params that correlated with highest success
        recent = history[-20:]
        best = max(recent, key=lambda h: h["success"])
        worst = min(recent, key=lambda h: h["success"])

        # Nudge current params toward best, away from worst
        current = dict(DOMAIN_PARAMS.get(domain, DOMAIN_PARAMS["general"]))

        for key in current:
            if key in best["params"] and key in worst["params"]:
                best_val = best["params"][key]
                worst_val = worst["params"][key]
                direction = best_val - worst_val
                current[key] += direction * 0.05  # Small nudge
                current[key] = max(0.3, min(2.0, current[key]))  # Clamp

        return current

    def get_domain_success(self, domain: str) -> float:
        """Get the current EMA success signal for a domain."""
        return self._domain_success.get(domain, 0.5)
