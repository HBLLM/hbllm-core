"""Epistemic Realization Policy for A16.

Translates rich cognitive EpistemicState into calibrated natural language verbalization
policies, selecting appropriate hedges and certainty markers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EpistemicVerbalizationLevel(StrEnum):
    """Calibrated levels of epistemic certainty for surface verbalization."""

    CERTAIN = "certain"                  # Direct assertion ("The cup is on the table.")
    PROBABLE = "probable"                # Probable hedge ("The cup is probably on the table.")
    PLAUSIBLE = "plausible"              # Plausible hedge ("I think the cup may be on the table.")
    UNCERTAIN = "uncertain"              # Explicit uncertainty ("I am not certain, but it might be on the table.")
    CONTRADICTED = "contradicted"        # Expressing conflict ("There is conflicting evidence about whether the cup is on the table.")
    INSUFFICIENT_EVIDENCE = "unknown"    # Complete knowledge gap ("I do not have enough evidence to determine whether the cup is on the table.")


@dataclass
class CognitiveEpistemicState:
    """Rich multi-factor epistemic evaluation object from A11."""

    target_predicate: str
    target_subject: str
    target_object: str | None = None
    confidence: float = 0.0          # 0.0 to 1.0
    uncertainty: float = 1.0         # 0.0 to 1.0
    support_count: int = 0           # Number of supporting evidence nodes
    contradiction_count: int = 0     # Number of contradicting evidence nodes
    freshness: float = 1.0           # 0.0 to 1.0 (recency of evidence)
    source_quality: float = 1.0      # Provider reliability score
    is_known: bool = True            # False if knowledge gap / unobserved
    raw_belief_value: Any = None
    provenance: str = ""


class EpistemicRealizationPolicy:
    """Evaluates multi-factor EpistemicState and assigns verbalization policy.

    Usage::

        policy = EpistemicRealizationPolicy()
        level = policy.evaluate(epistemic_state)
        # -> EpistemicVerbalizationLevel.PROBABLE
    """

    def __init__(
        self,
        certain_threshold: float = 0.92,
        probable_threshold: float = 0.70,
        plausible_threshold: float = 0.40,
    ) -> None:
        self._certain_threshold = certain_threshold
        self._probable_threshold = probable_threshold
        self._plausible_threshold = plausible_threshold

    def evaluate(self, state: CognitiveEpistemicState) -> EpistemicVerbalizationLevel:
        """Determine verbalization level from multi-dimensional epistemic state."""
        # 1. Unknown / Knowledge gap
        if not state.is_known or (state.support_count == 0 and state.confidence < 0.20):
            return EpistemicVerbalizationLevel.INSUFFICIENT_EVIDENCE

        # 2. Contradiction detected
        if state.contradiction_count > 0 and state.contradiction_count >= state.support_count:
            return EpistemicVerbalizationLevel.CONTRADICTED

        # 3. Freshness / Staleness decay
        effective_confidence = state.confidence * state.freshness * state.source_quality

        # 4. Certainty bands
        if effective_confidence >= self._certain_threshold and state.contradiction_count == 0:
            return EpistemicVerbalizationLevel.CERTAIN
        elif effective_confidence >= self._probable_threshold:
            return EpistemicVerbalizationLevel.PROBABLE
        elif effective_confidence >= self._plausible_threshold:
            return EpistemicVerbalizationLevel.PLAUSIBLE
        else:
            return EpistemicVerbalizationLevel.UNCERTAIN
