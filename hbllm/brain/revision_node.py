"""
Revision Node — iterative self-improvement through critique-and-revise loops.

Implements: Generate → Critique → Revise → Verify cycle.
If confidence is below threshold after critique, the response is
regenerated with targeted feedback for improvement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RevisionResult:
    """Result of a revision cycle."""
    original: str
    revised: str
    revision_count: int
    confidence: float
    critique_notes: list[str] = field(default_factory=list)
    improved: bool = False
    elapsed_ms: float = 0.0


class RevisionNode:
    """
    Iterative self-revision node for cognitive pipeline.

    Flow:
    1. Receive initial response from brain
    2. Run confidence estimator
    3. If confidence < threshold → critique and revise
    4. Repeat up to max_revisions times
    5. Return best version

    Integration:
    - Uses CriticNode for evaluation
    - Uses ConfidenceEstimator for scoring
    - Feeds revision pairs to RewardModel
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_revisions: int = 3,
        improvement_threshold: float = 0.05,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_revisions = max_revisions
        self.improvement_threshold = improvement_threshold
        self._total_revisions = 0
        self._total_processed = 0

    async def revise(
        self,
        query: str,
        response: str,
        critique_fn: Any = None,
        generate_fn: Any = None,
        confidence_fn: Any = None,
    ) -> RevisionResult:
        """
        Run the revision loop on a response.

        Args:
            query: Original user query
            response: Initial response to potentially revise
            critique_fn: async fn(query, response) -> dict with 'issues', 'score'
            generate_fn: async fn(query, feedback) -> str (revised response)
            confidence_fn: fn(query, response) -> float (0-1 confidence score)
        """
        start = time.monotonic()
        self._total_processed += 1

        current = response
        best = response
        best_confidence = 0.0
        notes: list[str] = []

        # Use default confidence if no estimator provided
        if confidence_fn is None:
            confidence_fn = self._default_confidence

        best_confidence = confidence_fn(query, current)

        if best_confidence >= self.confidence_threshold:
            return RevisionResult(
                original=response,
                revised=current,
                revision_count=0,
                confidence=best_confidence,
                improved=False,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )

        for i in range(self.max_revisions):
            # Step 1: Critique
            if critique_fn:
                critique = await critique_fn(query, current)
                issues = critique.get("issues", [])
                score = critique.get("score", 0.5)
                notes.append(f"Rev {i+1}: score={score:.2f}, issues={issues}")
            else:
                issues = self._default_critique(query, current)
                notes.append(f"Rev {i+1}: {len(issues)} issues found")

            if not issues:
                break

            # Step 2: Revise
            if generate_fn:
                feedback = f"Improve this response. Issues: {'; '.join(issues)}"
                current = await generate_fn(query, feedback)
            else:
                # Without LLM, we can't regenerate — return with critique notes
                break

            # Step 3: Re-evaluate confidence
            new_confidence = confidence_fn(query, current)
            self._total_revisions += 1

            if new_confidence > best_confidence + self.improvement_threshold:
                best = current
                best_confidence = new_confidence

            if best_confidence >= self.confidence_threshold:
                break

        elapsed = (time.monotonic() - start) * 1000
        revision_count = len(notes)

        return RevisionResult(
            original=response,
            revised=best,
            revision_count=revision_count,
            confidence=best_confidence,
            critique_notes=notes,
            improved=best != response,
            elapsed_ms=elapsed,
        )

    def _default_confidence(self, query: str, response: str) -> float:
        """Heuristic confidence scoring when no ML model is available."""
        score = 0.5

        # Length check
        words = response.split()
        if len(words) < 3:
            score -= 0.3
        elif len(words) > 10:
            score += 0.1

        # Relevance check (query-response word overlap)
        q_words = set(query.lower().split())
        r_words = set(response.lower().split())
        if q_words:
            overlap = len(q_words & r_words) / len(q_words)
            score += overlap * 0.3

        # Hedging language (low confidence indicators)
        hedges = {"maybe", "perhaps", "might", "possibly", "i think", "not sure"}
        if any(h in response.lower() for h in hedges):
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _default_critique(self, query: str, response: str) -> list[str]:
        """Basic heuristic critique when no LLM critique is available."""
        issues = []
        if len(response.split()) < 5:
            issues.append("Response too short")
        if query.lower().endswith("?") and "?" in response:
            issues.append("Response contains questions instead of answers")
        if response == response.upper() and len(response) > 10:
            issues.append("Response is all caps")
        return issues

    def stats(self) -> dict[str, Any]:
        """Return revision statistics."""
        return {
            "total_processed": self._total_processed,
            "total_revisions": self._total_revisions,
            "avg_revisions_per_query": (
                self._total_revisions / max(self._total_processed, 1)
            ),
        }
