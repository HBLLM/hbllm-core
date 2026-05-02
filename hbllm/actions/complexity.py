"""
Complexity Detector — auto-detect whether a task needs multi-agent orchestration.
"""

from __future__ import annotations

import re

COMPLEX_INDICATORS = [
    r"\b(analyze|audit|refactor|implement|build|create|design|compare)\b",
    r"\b(step by step|multi-step|multiple files|across files)\b",
    r"\b(and then|after that|first .+ then)\b",
    r"\b(write .+ test|review .+ code|debug .+ fix)\b",
]


class ComplexityDetector:
    """Detect whether a task requires multi-agent orchestration."""

    @staticmethod
    def needs_multi_agent(message: str) -> bool:
        """Return True if the message is complex enough for multi-agent."""
        score = 0
        msg_lower = message.lower()

        # Length heuristic
        if len(message.split()) > 30:
            score += 1

        # Pattern matching
        for pattern in COMPLEX_INDICATORS:
            if re.search(pattern, msg_lower):
                score += 1

        # Question marks suggest simple queries
        if message.strip().endswith("?") and len(message.split()) < 20:
            score -= 2

        return score >= 2
