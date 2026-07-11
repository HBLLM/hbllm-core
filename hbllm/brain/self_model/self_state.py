"""SelfStateEngine: Introspective Cognition and Epistemic Calibration."""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.governance.governance import CognitiveGovernanceEngine

logger = logging.getLogger(__name__)


class ToolReliabilityTracker:
    """Tracks tool reliability using an Exponentially Weighted Moving Average (EWMA)."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self._scores: dict[str, float] = {}

    def get_reliability(self, tool_name: str) -> float:
        """Get the current reliability of a tool (0.0 to 1.0, default 0.8)."""
        return self._scores.get(tool_name, 0.8)

    def record_execution(self, tool_name: str, success: bool) -> None:
        """Update EWMA reliability score based on a success or failure."""
        current = self.get_reliability(tool_name)
        new_val = 1.0 if success else 0.0

        # Slow decay on failure, slow repair on success
        updated = (self.alpha * new_val) + ((1.0 - self.alpha) * current)
        self._scores[tool_name] = max(0.0, min(1.0, updated))


class EpistemicCalibrationTracker:
    """Measures prediction divergence from verified reality."""

    def __init__(self) -> None:
        # Dictionary mapping domain/tool to a calibration score (0.0 to 1.0)
        self._calibration_scores: dict[str, float] = {}

    def get_calibration(self, domain: str) -> float:
        """Get epistemic calibration for a domain (default 0.7)."""
        return self._calibration_scores.get(domain, 0.7)

    def record_verification(
        self, domain: str, predicted_outcome: Any, verified_outcome: Any, match: bool
    ) -> None:
        """Update epistemic calibration based on how well the simulation matched reality."""
        current = self.get_calibration(domain)
        # EWMA approach to epistemic calibration
        alpha = 0.15
        new_val = 1.0 if match else 0.0

        updated = (alpha * new_val) + ((1.0 - alpha) * current)
        self._calibration_scores[domain] = max(0.0, min(1.0, updated))


class CognitiveStressMonitor:
    """Tracks active plans, context saturation, and memory pressure."""

    def __init__(self, governance: CognitiveGovernanceEngine) -> None:
        self.governance = governance
        self.active_plans = 0
        self.queue_backlog = 0
        self.memory_pressure = 0.0

    def update_stress(self, active_plans: int, queue_backlog: int, memory_pressure: float) -> float:
        """Update metrics and return the aggregate cognitive pressure."""
        self.active_plans = active_plans
        self.queue_backlog = queue_backlog
        self.memory_pressure = memory_pressure

        return self.governance.get_cognitive_pressure(
            memory_pressure=self.memory_pressure,
            active_goals=self.active_plans,
            queue_depth=self.queue_backlog,
        )


class SelfStateEngine:
    """The central introspective subsystem."""

    def __init__(self, governance: CognitiveGovernanceEngine | None = None) -> None:
        self.governance = governance or CognitiveGovernanceEngine()
        self.tools = ToolReliabilityTracker()
        self.calibration = EpistemicCalibrationTracker()
        self.stress = CognitiveStressMonitor(self.governance)

    def get_cognitive_pressure(self) -> float:
        """Get the current cognitive stress level (0.0 to 1.0)."""
        return self.stress.update_stress(
            self.stress.active_plans, self.stress.queue_backlog, self.stress.memory_pressure
        )
