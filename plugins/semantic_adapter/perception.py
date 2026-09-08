"""
Semantic Ambiguity Perception Adapter.

Parses natural language directives, estimates linguistic grounding confidence,
and detects epistemic ambiguity requiring LLM reasoning guidance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .types import SemanticObservation

logger = logging.getLogger(__name__)

AMBIGUITY_MARKERS = [
    r"\bscratch that\b",
    r"\bwait\b",
    r"\bactually\b",
    r"\btidy up\b",
    r"\bprepare\b",
    r"\bclean\b",
    r"\bmake ready\b",
]


class SemanticPerceptionAdapter:
    """Evaluates linguistic ambiguity and intent clarity of human instructions."""

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """Reset internal buffers."""
        pass

    def process_observation(self, obs: SemanticObservation) -> dict[str, Any]:
        """Classify instruction clarity and extract candidate entities."""
        text = obs.instruction.lower()

        ambiguity_detected = False
        for pattern in AMBIGUITY_MARKERS:
            if re.search(pattern, text):
                ambiguity_detected = True
                break

        # Check for direct canonical grammar
        is_canonical = (
            bool(re.search(r"\b(pick up|unlock|place)\b", text)) and not ambiguity_detected
        )

        confidence = 1.0 if is_canonical else (0.4 if ambiguity_detected else 0.8)

        return {
            "instruction": obs.instruction,
            "is_canonical": is_canonical,
            "ambiguity_detected": ambiguity_detected,
            "grounding_confidence": confidence,
            "scene_objects": dict(obs.scene_objects),
            "step_count": obs.step_count,
        }
