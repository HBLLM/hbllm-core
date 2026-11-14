"""Recognition Policy — HBLLM Grounded Perception §V0.

Thresholds and rules for visual recognition live here, not hardcoded
in perception methods.  Policies are configurable and can vary by
model, embedding space, object category, environment, etc.

This prevents magic numbers like ``0.7`` from becoming embedded
throughout the codebase.
"""

from __future__ import annotations

from pydantic import BaseModel

from hbllm.perception.providers.evidence import CandidateRanking


class RecognitionPolicy(BaseModel):
    """Recognition thresholds and decision rules.

    Configurable rather than hardcoded.  Can vary by:
        - Model and embedding space
        - Object category
        - Number of exemplars
        - Environment and camera quality

    Attributes:
        minimum_similarity: Below this, the observation is novel.
        ambiguity_margin: Below this margin between best and second-best,
            the recognition is ambiguous.
        minimum_supporting_observations: Require at least this many
            matching observations for a confident recognition.
        novelty_threshold: Below this best_score, treat as completely novel.
        exemplar_limit: Maximum exemplars stored per concept.
        exemplar_diversity_threshold: Don't store near-duplicate
            exemplars (above this similarity to existing exemplars).

    """

    minimum_similarity: float = 0.7
    ambiguity_margin: float = 0.1
    minimum_supporting_observations: int = 1
    novelty_threshold: float = 0.5
    exemplar_limit: int = 20
    exemplar_diversity_threshold: float = 0.95

    def is_match(self, ranking: CandidateRanking) -> bool:
        """Clear match: high similarity with sufficient margin."""
        return (
            ranking.best_score >= self.minimum_similarity
            and ranking.margin >= self.ambiguity_margin
        )

    def is_ambiguous(self, ranking: CandidateRanking) -> bool:
        """Ambiguous: high similarity but low margin between candidates."""
        return (
            ranking.best_score >= self.minimum_similarity and ranking.margin < self.ambiguity_margin
        )

    def is_novel(self, ranking: CandidateRanking) -> bool:
        """Novel: best candidate below novelty threshold."""
        return ranking.best_score < self.novelty_threshold
