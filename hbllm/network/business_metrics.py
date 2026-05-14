"""
Business Metrics Tracker for HBLLM Core.

Provides per-tenant cost tracking, token usage, and satisfaction scores
to be integrated with OpenTelemetry or custom dashboarding.
"""

import logging
import time

logger = logging.getLogger(__name__)


class BusinessMetrics:
    def __init__(self):
        # tenant_id -> metrics dict
        self._metrics: dict[str, dict[str, float]] = {}

    def _ensure_tenant(self, tenant_id: str) -> None:
        if tenant_id not in self._metrics:
            self._metrics[tenant_id] = {
                "total_tokens": 0.0,
                "total_cost": 0.0,
                "satisfaction_score": 0.0,
                "satisfaction_count": 0.0,
                "api_calls": 0.0,
            }

    def record_inference_cost(self, tenant_id: str, tokens: int, model: str) -> None:
        self._ensure_tenant(tenant_id)
        self._metrics[tenant_id]["total_tokens"] += tokens

        # Simple mocked cost calculation
        cost_per_1k = 0.002
        if "gpt-4" in model:
            cost_per_1k = 0.03

        cost = (tokens / 1000.0) * cost_per_1k
        self._metrics[tenant_id]["total_cost"] += cost
        self._metrics[tenant_id]["api_calls"] += 1

        logger.debug("Recorded %d tokens for tenant %s (model: %s)", tokens, tenant_id, model)

    def record_satisfaction(self, tenant_id: str, score: float) -> None:
        self._ensure_tenant(tenant_id)
        self._metrics[tenant_id]["satisfaction_score"] += score
        self._metrics[tenant_id]["satisfaction_count"] += 1

    def record_response_quality(self, tenant_id: str, rating: str) -> None:
        """Map categorical rating (good/bad/neutral) to numerical score."""
        score_map = {"good": 1.0, "neutral": 0.5, "bad": 0.0}
        score = score_map.get(rating.lower(), 0.5)
        self.record_satisfaction(tenant_id, score)

    def get_tenant_dashboard(self, tenant_id: str) -> dict[str, float]:
        if tenant_id not in self._metrics:
            return {}

        data = self._metrics[tenant_id]
        avg_sat = 0.0
        if data["satisfaction_count"] > 0:
            avg_sat = data["satisfaction_score"] / data["satisfaction_count"]

        return {
            "total_tokens": data["total_tokens"],
            "total_cost_usd": data["total_cost"],
            "api_calls": data["api_calls"],
            "average_satisfaction": avg_sat,
        }
