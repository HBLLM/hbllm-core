"""
Shallow Renderer — reduces LLM to text rendering from pre-reasoned conclusions.

This is the final piece of the v3 architecture:

    v1-v2: LLM does constrained reasoning per thought unit
    v3:    LLM does shallow rendering, SNN+memory do most reasoning  ← this
    v4+:   LLM is genuinely just Broca's area

The ShallowRenderer receives a ``RenderingContext`` that contains all
conclusions from the SNN pipeline (concepts, associations, causal chains,
PRM confidence) and generates minimal prompts that ask the LLM to
*render text* — not *reason*.

Cost model:
    Deep prompt:    ~500-800 tokens (reasoning instructions + context)
    Shallow prompt: ~200-350 tokens (conclusions + rendering instructions)
    Savings:        40-60% token reduction per goal

Components:
    RenderingContext    — pre-reasoned data for the LLM to render
    RenderPromptBuilder — generates minimal rendering prompts
    ShallowRenderer     — orchestrates shallow rendering with fallback

Usage::

    renderer = ShallowRenderer()
    context = renderer.build_context(understanding, goals, "user query", "base thought")

    if renderer.should_use_shallow(context):
        prompt = renderer.render_prompt(context, goal)
        # Send to LLM — prompt is ~60% shorter, no reasoning needed
    else:
        # Fall back to deep reasoning path
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.snn.expression.models import ThoughtGoal

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# RenderingContext — pre-reasoned data payload
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RenderingContext:
    """Everything the LLM needs to render a response — no reasoning required.

    Assembled from the full SNN pipeline output:

    - **concepts**: Key concept texts from comprehension
    - **associations**: Concept relationships from AssociationLayer
    - **causal_chains**: Reasoning paths from CausalReasoner
    - **goals**: Thought goals from ThoughtPlanner
    - **memory_hints**: Relevant memories from comprehension
    - **confidence**: Overall PRM confidence in the pre-reasoning
    - **constraints**: Constraint metadata from channel signals
    - **domain_context**: Primary domain and activation scores
    - **original_query**: The user's original input
    - **base_thought**: The raw thought from workspace consensus

    Attributes:
        concepts: List of concept texts (strings).
        associations: List of association dicts (type, strength, texts).
        causal_chains: List of causal chain dicts (source, conclusion, confidence).
        goals: ThoughtGoal objects from the planner.
        memory_hints: Relevant memory snippets.
        confidence: Overall confidence score [0, 1].
        constraints: Constraint metadata dict.
        domain_context: Primary domain and scores.
        original_query: The user's question.
        base_thought: Raw thought text.
    """

    concepts: list[str] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)
    causal_chains: list[dict[str, Any]] = field(default_factory=list)
    goals: list[ThoughtGoal] = field(default_factory=list)
    memory_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    constraints: dict[str, float] = field(default_factory=dict)
    domain_context: dict[str, float] = field(default_factory=dict)
    original_query: str = ""
    base_thought: str = ""

    @property
    def concept_count(self) -> int:
        """Number of concepts in the context."""
        return len(self.concepts)

    @property
    def has_associations(self) -> bool:
        """Whether any concept associations were found."""
        return len(self.associations) > 0

    @property
    def has_causal_chains(self) -> bool:
        """Whether any causal reasoning chains were found."""
        return len(self.causal_chains) > 0

    @property
    def primary_domain(self) -> str:
        """The highest-scoring domain, or 'general'."""
        if not self.domain_context:
            return "general"
        return max(self.domain_context, key=lambda k: self.domain_context[k])

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "concept_count": self.concept_count,
            "association_count": len(self.associations),
            "causal_chain_count": len(self.causal_chains),
            "goal_count": len(self.goals),
            "confidence": round(self.confidence, 4),
            "primary_domain": self.primary_domain,
            "has_memory_hints": len(self.memory_hints) > 0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# RenderPromptBuilder — minimal rendering prompts
# ═══════════════════════════════════════════════════════════════════════════


class RenderPromptBuilder:
    """Builds LLM prompts that only ask for text rendering.

    The prompts contain:
    1. Pre-computed conclusions (from SNN reasoning)
    2. Tone/style guidance based on domain
    3. Token budget
    4. **NO** reasoning instructions

    This is the key difference from deep prompts: the LLM is told
    *what to say*, not *what to figure out*.

    Args:
        max_conclusion_chars: Maximum characters for conclusions section.
        include_associations: Whether to include association hints.
        include_causal: Whether to include causal chain summaries.
    """

    def __init__(
        self,
        max_conclusion_chars: int = 1500,
        include_associations: bool = True,
        include_causal: bool = True,
    ) -> None:
        self.max_conclusion_chars = max_conclusion_chars
        self.include_associations = include_associations
        self.include_causal = include_causal

    def build_full_render(self, context: RenderingContext) -> str:
        """Build a single prompt to render the entire response.

        Args:
            context: The pre-reasoned RenderingContext.

        Returns:
            A rendering prompt string.
        """
        parts: list[str] = []

        # Header: tell the LLM this is a rendering task
        parts.append(
            "RENDER the following pre-analyzed conclusions into a clear, "
            "natural response. Do NOT add new reasoning — just express "
            "these points fluently."
        )
        parts.append("")

        # User query for context
        parts.append(f"User asked: {context.original_query}")
        parts.append("")

        # Pre-computed conclusions
        parts.append("CONCLUSIONS to render:")
        for i, concept in enumerate(context.concepts):
            parts.append(f"  {i + 1}. {concept}")

        # Association hints
        if self.include_associations and context.associations:
            parts.append("")
            parts.append("RELATIONSHIPS found:")
            for assoc in context.associations[:5]:
                a_type = assoc.get("association_type", "related")
                source = assoc.get("source_text", "")
                target = assoc.get("target_text", "")
                parts.append(f"  - {source} [{a_type}] {target}")

        # Causal chain summaries
        if self.include_causal and context.causal_chains:
            parts.append("")
            parts.append("CAUSAL reasoning:")
            for chain in context.causal_chains[:3]:
                source = chain.get("source_concept", "")
                conclusion = chain.get("conclusion", "")
                conf = chain.get("snn_confidence", 0.0)
                parts.append(f"  - {source} → {conclusion} (confidence: {conf:.0%})")

        # Memory hints
        if context.memory_hints:
            parts.append("")
            parts.append("CONTEXT from memory:")
            for hint in context.memory_hints[:3]:
                parts.append(f"  - {hint[:200]}")

        # Constraint guidance
        if context.constraints:
            parts.append("")
            constraint_items = [f"{k} (strength: {v:.1f})" for k, v in context.constraints.items()]
            parts.append(f"CONSTRAINTS to respect: {', '.join(constraint_items)}")

        # Style guidance based on domain
        parts.append("")
        domain = context.primary_domain
        tone = self._domain_tone(domain)
        parts.append(f"Tone: {tone}")

        # Enforce rendering-only
        parts.append("")
        parts.append(
            "IMPORTANT: Express these conclusions as natural prose. "
            "Do not add analysis, speculation, or new information. "
            "The reasoning is already complete."
        )

        prompt = "\n".join(parts)
        return prompt[: self.max_conclusion_chars + 500]

    def build_per_goal_render(
        self,
        context: RenderingContext,
        goal: ThoughtGoal,
        prev_text: str | None = None,
    ) -> str:
        """Build a per-goal rendering prompt.

        Args:
            context: The full RenderingContext.
            goal: The specific ThoughtGoal to render.
            prev_text: Previous fragment text for continuity.

        Returns:
            A rendering prompt for one goal.
        """
        parts: list[str] = []

        parts.append(
            "RENDER one section of a response. The analysis is already done — "
            "just express this conclusion naturally."
        )
        parts.append("")
        parts.append(f"User asked: {context.original_query}")
        parts.append("")

        # The specific conclusion to render
        parts.append(f"SECTION TOPIC: {goal.source_concept_text}")
        parts.append(f"GOAL: {goal.text}")

        # Relevant associations for this goal
        if self.include_associations and context.associations:
            relevant = [
                a
                for a in context.associations
                if goal.source_concept_text
                in (
                    a.get("source_text", ""),
                    a.get("target_text", ""),
                )
            ]
            if relevant:
                parts.append("")
                parts.append("Related findings:")
                for a in relevant[:3]:
                    parts.append(
                        f"  - [{a.get('association_type', 'related')}] {a.get('target_text', '')}"
                    )

        # Memory hints for this goal
        if goal.memory_hints:
            parts.append("")
            parts.append("Context:")
            for hint in goal.memory_hints[:2]:
                parts.append(f"  - {hint[:150]}")

        # Continuity with previous section
        if prev_text:
            parts.append("")
            parts.append(f"Previous section ended with: ...{prev_text[-150:]}")

        # Constraint guidance
        if goal.constraints:
            parts.append("")
            parts.append(f"Constraints: {goal.constraints}")

        parts.append("")
        parts.append(
            f"Write ONLY this section ({goal.max_tokens} tokens max). "
            f"Do not reason — the conclusion is provided above. "
            f"Just render it as clear prose."
        )

        return "\n".join(parts)

    @staticmethod
    def _domain_tone(domain: str) -> str:
        """Map domain to a tone/style guidance string."""
        tones = {
            "coding": "Technical and precise. Use code terminology where appropriate.",
            "general": "Clear and conversational. Be helpful and direct.",
            "creative": "Engaging and expressive. Use vivid language.",
            "academic": "Formal and well-structured. Cite reasoning clearly.",
            "business": "Professional and concise. Focus on actionable insights.",
        }
        return tones.get(domain, tones["general"])


# ═══════════════════════════════════════════════════════════════════════════
# ShallowRenderer — orchestrates rendering with fallback
# ═══════════════════════════════════════════════════════════════════════════


class ShallowRenderer:
    """Renders pre-reasoned conclusions into natural language.

    Replaces the full reasoning pipeline when SNN confidence is
    sufficient.  When confidence is too low, signals that the deep
    path should be used instead.

    The rendering flow:

    1. ``build_context()`` — assemble ``RenderingContext`` from
       ``UnderstandingState`` + associations + causal chains
    2. ``should_use_shallow()`` — check confidence threshold
    3. ``render_prompt()`` — generate the minimal LLM prompt

    Args:
        prompt_builder: The RenderPromptBuilder to use.
        min_confidence: Minimum confidence for shallow rendering.
            Below this, falls back to deep reasoning. Default 0.3.
    """

    def __init__(
        self,
        prompt_builder: RenderPromptBuilder | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        self._builder = prompt_builder or RenderPromptBuilder()
        self._min_confidence = min_confidence
        self._render_count = 0
        self._fallback_count = 0

    def build_context(
        self,
        understanding: Any,
        goals: list[ThoughtGoal],
        original_query: str,
        base_thought: str,
    ) -> RenderingContext:
        """Assemble RenderingContext from SNN pipeline output.

        Args:
            understanding: UnderstandingState from ComprehensionStream.
            goals: ThoughtGoals from ThoughtPlanner.
            original_query: The user's original query text.
            base_thought: The raw thought from workspace consensus.

        Returns:
            A RenderingContext ready for rendering.
        """
        # Extract concepts
        concepts = []
        if hasattr(understanding, "concepts"):
            concepts = [c.text for c in understanding.concepts]

        # Extract associations
        associations = []
        if hasattr(understanding, "associations"):
            for assoc in understanding.associations:
                if hasattr(assoc, "to_dict"):
                    associations.append(assoc.to_dict())
                elif isinstance(assoc, dict):
                    associations.append(assoc)

        # Extract causal chains
        causal_chains = []
        if hasattr(understanding, "causal_chains"):
            for chain in understanding.causal_chains:
                if hasattr(chain, "to_dict"):
                    causal_chains.append(chain.to_dict())
                elif isinstance(chain, dict):
                    causal_chains.append(chain)

        # Extract memory hints
        memory_hints = []
        if hasattr(understanding, "all_memories"):
            memory_hints = [m.content[:200] for m in understanding.all_memories[:5]]

        # Compute overall confidence from multiple signals
        confidence = self._compute_confidence(understanding, associations, causal_chains)

        # Extract constraints from goals
        constraints: dict[str, float] = {}
        for goal in goals:
            for k, v in goal.constraints.items():
                constraints[k] = max(constraints.get(k, 0.0), v)

        # Domain context
        domain_context: dict[str, float] = {}
        if hasattr(understanding, "domain_activations"):
            domain_context = dict(understanding.domain_activations)

        return RenderingContext(
            concepts=concepts,
            associations=associations,
            causal_chains=causal_chains,
            goals=goals,
            memory_hints=memory_hints,
            confidence=confidence,
            constraints=constraints,
            domain_context=domain_context,
            original_query=original_query,
            base_thought=base_thought,
        )

    def should_use_shallow(self, context: RenderingContext) -> bool:
        """Check if confidence is high enough for shallow rendering.

        Args:
            context: The assembled RenderingContext.

        Returns:
            True if shallow rendering should be used;
            False if the deep reasoning path is needed.
        """
        if context.confidence < self._min_confidence:
            self._fallback_count += 1
            logger.debug(
                "ShallowRenderer: confidence %.3f < %.3f, falling back to deep",
                context.confidence,
                self._min_confidence,
            )
            return False

        # Also require at least one concept
        if not context.concepts:
            self._fallback_count += 1
            return False

        return True

    def render_prompt(
        self,
        context: RenderingContext,
        goal: ThoughtGoal | None = None,
        prev_text: str | None = None,
    ) -> str:
        """Generate a rendering prompt for the LLM.

        Args:
            context: The pre-reasoned RenderingContext.
            goal: Specific ThoughtGoal for per-goal rendering.
                If None, generates a full-response prompt.
            prev_text: Previous fragment for continuity.

        Returns:
            A minimal rendering prompt string.
        """
        self._render_count += 1

        if goal is not None:
            return self._builder.build_per_goal_render(context, goal, prev_text)
        return self._builder.build_full_render(context)

    def _compute_confidence(
        self,
        understanding: Any,
        associations: list[dict],
        causal_chains: list[dict],
    ) -> float:
        """Compute overall confidence for shallow rendering.

        Factors:
        - Number of concepts (more = more context)
        - Association coverage (found relationships = better understanding)
        - Causal chain confidence (high SNN scores = good reasoning)
        - Salience coverage (high-salience concepts processed)
        """
        score = 0.0
        weights_total = 0.0

        # Concept coverage: at least 1 concept = base confidence
        concept_count = 0
        if hasattr(understanding, "concepts"):
            concept_count = len(understanding.concepts)
        if concept_count > 0:
            score += min(1.0, concept_count / 3.0) * 0.3
            weights_total += 0.3

        # Association signal
        if associations:
            avg_strength = sum(a.get("strength", 0.5) for a in associations) / len(associations)
            score += avg_strength * 0.2
            weights_total += 0.2

        # Causal chain confidence
        if causal_chains:
            avg_conf = sum(c.get("snn_confidence", 0.5) for c in causal_chains) / len(causal_chains)
            score += avg_conf * 0.3
            weights_total += 0.3

        # Salience coverage
        if hasattr(understanding, "salience_map") and understanding.salience_map:
            avg_salience = sum(understanding.salience_map) / len(understanding.salience_map)
            score += min(1.0, avg_salience) * 0.2
            weights_total += 0.2

        # Normalize
        if weights_total > 0:
            return score / weights_total

        # Fallback: no signals at all → low confidence
        return 0.2

    @property
    def render_count(self) -> int:
        """Number of shallow renders performed."""
        return self._render_count

    @property
    def fallback_count(self) -> int:
        """Number of times we fell back to deep rendering."""
        return self._fallback_count

    @property
    def shallow_rate(self) -> float:
        """Fraction of renders that used shallow mode."""
        total = self._render_count + self._fallback_count
        if total == 0:
            return 0.0
        return self._render_count / total

    def reset_stats(self) -> None:
        """Reset rendering statistics."""
        self._render_count = 0
        self._fallback_count = 0
