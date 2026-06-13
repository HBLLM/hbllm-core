"""
Data models for the expression-side Cognitive Stream.

ThoughtGoal:     A single item in the symbolic thought outline.
ThoughtFragment: A generated text fragment for one thought goal.
ExpressionResult: The assembled final output with per-thought traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtGoal:
    """A single thought goal derived from comprehension concepts.

    The ThoughtPlanner decomposes UnderstandingState.concepts into a
    sequence of ThoughtGoals.  Each goal is a constrained reasoning
    target for the LLM.

    Attributes:
        id: Unique identifier for this goal.
        text: Natural-language description of what the LLM should address.
        source_concept_text: The comprehension concept that spawned this goal.
        salience: Inherited salience from the source concept.
        domain: Primary domain for this goal (e.g. "coding", "general").
        memory_hints: Memory snippets relevant to this goal (from comprehension).
        constraints: Constraint metadata (e.g. "only", "except") detected
            during comprehension.
        priority: Ordering priority (lower = generate first).
        max_tokens: Soft token budget for this thought's LLM generation.
    """

    id: str = ""
    text: str = ""
    source_concept_text: str = ""
    salience: float = 1.0
    domain: str = "general"
    memory_hints: list[str] = field(default_factory=list)
    constraints: dict[str, float] = field(default_factory=dict)
    priority: int = 0
    max_tokens: int = 512


@dataclass
class ThoughtFragment:
    """A generated text fragment for one thought goal.

    Attributes:
        goal_id: The ThoughtGoal this fragment addresses.
        text: The generated text.
        reward_score: Score from the RewardEvaluator (0.0–1.0).
        coherence_score: How well this fragment connects to previous fragments.
        relevance_score: How well this addresses the goal.
        revision_count: Number of times this was revised.
        metadata: Additional metadata (tokens used, latency, etc.).
    """

    goal_id: str = ""
    text: str = ""
    reward_score: float = 0.0
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    revision_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpressionResult:
    """Assembled result from the expression pipeline.

    Attributes:
        text: The final assembled response text.
        fragments: Per-thought fragments with individual scores.
        mean_reward: Average reward across all fragments.
        total_tokens: Total tokens consumed during generation.
        thought_count: Number of thought goals processed.
        revision_count: Total revisions across all fragments.
    """

    text: str = ""
    fragments: list[ThoughtFragment] = field(default_factory=list)
    mean_reward: float = 0.0
    total_tokens: int = 0
    thought_count: int = 0
    revision_count: int = 0
