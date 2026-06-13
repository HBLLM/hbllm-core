"""
RewardEvaluator — scores generated thought fragments.

Evaluates each LLM-generated fragment against its source thought goal
and the comprehension context.  No LLM call — uses embedding similarity
and lexical heuristics for fast (~1ms) scoring.

Scoring dimensions:
    relevance  — does the fragment address the goal?
    coherence  — does it connect smoothly to the previous fragment?
    completeness — does it cover key terms from the goal?
    conciseness  — is it appropriately sized for its token budget?

Each dimension is [0.0, 1.0]; the final reward is a weighted blend.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from hbllm.brain.snn.expression.models import ThoughtFragment, ThoughtGoal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias for the ONNX encoder function
EncoderFn = Callable[[str], np.ndarray]


class RewardEvaluator:
    """Scores generated fragments against their goals.

    Uses embedding cosine similarity (if encoder is available) and
    lexical overlap heuristics for fast evaluation.

    Args:
        encoder: Optional ONNX encoder function (text → embedding).
            If None, falls back to lexical-only scoring.
        relevance_weight: Weight for the relevance dimension.
        coherence_weight: Weight for the coherence dimension.
        completeness_weight: Weight for the completeness dimension.
        conciseness_weight: Weight for the conciseness dimension.
        min_acceptable_reward: Threshold below which a fragment should
            be revised.
    """

    def __init__(
        self,
        encoder: EncoderFn | None = None,
        relevance_weight: float = 0.4,
        coherence_weight: float = 0.2,
        completeness_weight: float = 0.25,
        conciseness_weight: float = 0.15,
        min_acceptable_reward: float = 0.4,
    ) -> None:
        self.encoder = encoder
        self.relevance_weight = relevance_weight
        self.coherence_weight = coherence_weight
        self.completeness_weight = completeness_weight
        self.conciseness_weight = conciseness_weight
        self.min_acceptable_reward = min_acceptable_reward

    def evaluate(
        self,
        fragment_text: str,
        goal: ThoughtGoal,
        prev_fragment_text: str | None = None,
    ) -> ThoughtFragment:
        """Evaluate a generated fragment against its goal.

        Args:
            fragment_text: The LLM-generated text.
            goal: The thought goal this fragment addresses.
            prev_fragment_text: The previous fragment (for coherence).

        Returns:
            A ThoughtFragment with all scores populated.
        """
        # 1. Relevance: how well does the fragment address the goal?
        relevance = self._score_relevance(fragment_text, goal)

        # 2. Coherence: does it connect to previous fragment?
        coherence = self._score_coherence(fragment_text, prev_fragment_text)

        # 3. Completeness: does it cover key terms?
        completeness = self._score_completeness(fragment_text, goal)

        # 4. Conciseness: is it appropriately sized?
        conciseness = self._score_conciseness(fragment_text, goal)

        # Weighted blend
        reward = (
            relevance * self.relevance_weight
            + coherence * self.coherence_weight
            + completeness * self.completeness_weight
            + conciseness * self.conciseness_weight
        )

        return ThoughtFragment(
            goal_id=goal.id,
            text=fragment_text,
            reward_score=reward,
            coherence_score=coherence,
            relevance_score=relevance,
            metadata={
                "completeness": completeness,
                "conciseness": conciseness,
            },
        )

    def should_revise(self, fragment: ThoughtFragment) -> bool:
        """Check whether a fragment should be revised.

        Args:
            fragment: The evaluated fragment.

        Returns:
            True if the reward is below the acceptable threshold.
        """
        return fragment.reward_score < self.min_acceptable_reward

    def _score_relevance(self, text: str, goal: ThoughtGoal) -> float:
        """Score relevance using embedding similarity or lexical fallback."""
        if self.encoder is not None:
            try:
                text_emb = self.encoder(text)
                goal_emb = self.encoder(goal.source_concept_text or goal.text)

                # Cosine similarity
                dot = float(np.dot(text_emb, goal_emb))
                norms = float(np.linalg.norm(text_emb) * np.linalg.norm(goal_emb) + 1e-9)
                sim = dot / norms

                # Map [-1, 1] → [0, 1]
                return max(0.0, min(1.0, (sim + 1.0) / 2.0))
            except Exception:
                logger.debug("Encoder failed, falling back to lexical relevance")

        # Lexical fallback: word overlap
        return self._lexical_relevance(text, goal)

    def _lexical_relevance(self, text: str, goal: ThoughtGoal) -> float:
        """Compute relevance from word overlap (fallback)."""
        text_words = set(text.lower().split())
        goal_words = set((goal.source_concept_text or goal.text).lower().split())

        if not goal_words:
            return 0.5

        overlap = len(text_words & goal_words)
        return min(1.0, overlap / max(1, len(goal_words)))

    def _score_coherence(self, text: str, prev_text: str | None) -> float:
        """Score coherence with the previous fragment."""
        if prev_text is None:
            return 1.0  # First fragment is always coherent

        # Check for transitional continuity
        text_words = set(text.lower().split()[:15])
        prev_words = set(prev_text.lower().split()[-15:])

        overlap = len(text_words & prev_words)
        continuity = min(1.0, overlap * 0.15)

        # Check if fragment starts with a connector
        connectors = {
            "additionally",
            "furthermore",
            "moreover",
            "also",
            "however",
            "but",
            "yet",
            "therefore",
            "thus",
            "consequently",
            "this",
            "that",
            "these",
            "here",
            "in",
            "for",
            "the",
            "as",
            "with",
            "to",
        }
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        connector_bonus = 0.2 if first_word in connectors else 0.0

        return min(1.0, 0.5 + continuity + connector_bonus)

    def _score_completeness(self, text: str, goal: ThoughtGoal) -> float:
        """Score how completely the fragment covers the goal's key terms."""
        # Extract key terms from goal (non-stopword words > 3 chars)
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "in",
            "of",
            "for",
            "on",
            "with",
            "it",
            "this",
            "that",
            "and",
            "also",
            "address",
            "establish",
            "context",
            "briefly",
        }
        goal_text = goal.source_concept_text or goal.text
        key_terms = {
            w.lower().strip(".,!?\"'()[]{}:;")
            for w in goal_text.split()
            if len(w) > 3 and w.lower() not in stopwords
        }

        if not key_terms:
            return 0.7  # No specific terms to check

        text_lower = text.lower()
        covered = sum(1 for term in key_terms if term in text_lower)
        return min(1.0, covered / max(1, len(key_terms)))

    def _score_conciseness(self, text: str, goal: ThoughtGoal) -> float:
        """Score whether the fragment is appropriately sized.

        Too short (< 20% of budget) or too long (> 200% of budget) gets
        penalized.
        """
        # Rough token estimate: 4 chars per token
        estimated_tokens = max(1, len(text) // 4)
        budget = max(1, goal.max_tokens)

        ratio = estimated_tokens / budget

        if ratio < 0.1:
            return 0.2  # Way too short
        elif ratio < 0.2:
            return 0.5
        elif ratio <= 1.5:
            return 1.0  # Good range
        elif ratio <= 2.0:
            return 0.7  # Slightly long
        else:
            return 0.4  # Way too long
