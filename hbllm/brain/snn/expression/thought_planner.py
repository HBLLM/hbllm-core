"""
ThoughtPlanner — symbolic outline generator.

Takes an ``UnderstandingState`` (from the comprehension pipeline) and
decomposes it into an ordered list of ``ThoughtGoal`` items.  Each goal
represents a single reasoning target for constrained LLM generation.

The planner is *symbolic* — no ML, no LLM calls.  It maps concepts to
goals using salience, domain activation, constraints, and memory density.

Design principles:
    * One concept → one or more goals (constraint concepts expand to two:
      the constraint context + the constrained assertion).
    * Goals are ordered by priority: high-salience + constrained first.
    * Each goal carries a soft token budget proportional to salience.
    * Memory hints from comprehension are forwarded to goals so the LLM
      can reference prior context without duplicate retrieval.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from hbllm.brain.snn.expression.models import ThoughtGoal

if TYPE_CHECKING:
    from hbllm.brain.snn.comprehension.models import (
        ComprehensionUnit,
        UnderstandingState,
    )

logger = logging.getLogger(__name__)


class ThoughtPlanner:
    """Decomposes ``UnderstandingState`` into an ordered thought outline.

    Args:
        base_token_budget: Default token budget per goal when salience == 1.0.
        constraint_expansion: Whether constraint concepts produce two goals.
        min_salience_for_goal: Concepts below this salience are merged into
            an "other" catch-all goal rather than getting their own.
    """

    def __init__(
        self,
        base_token_budget: int = 512,
        constraint_expansion: bool = True,
        min_salience_for_goal: float = 0.3,
    ) -> None:
        self.base_token_budget = base_token_budget
        self.constraint_expansion = constraint_expansion
        self.min_salience_for_goal = min_salience_for_goal

    def plan(self, understanding: UnderstandingState) -> list[ThoughtGoal]:
        """Generate an ordered list of ThoughtGoals from comprehension output.

        Args:
            understanding: The UnderstandingState from ComprehensionStream.

        Returns:
            Ordered list of ThoughtGoals ready for the ThoughtController.
        """
        if not understanding.concepts:
            return []

        goals: list[ThoughtGoal] = []
        low_salience_concepts: list[ComprehensionUnit] = []
        priority_counter = 0

        for concept in understanding.concepts:
            if concept.salience < self.min_salience_for_goal:
                low_salience_concepts.append(concept)
                continue

            # Determine primary domain from concept's domain_activation
            primary_domain = "general"
            if concept.domain_activation:
                primary_domain = max(
                    concept.domain_activation,
                    key=lambda k: concept.domain_activation[k],
                )

            # Extract memory hints from activated memories
            memory_hints = [
                m.content[:200] for m in concept.activated_memories
            ]

            # Extract constraint metadata from channel_metadata
            constraints = {
                k: v
                for k, v in concept.channel_metadata.items()
                if k in ("constraint", "surprise") and v > 0.3
            }

            # Determine token budget proportional to salience
            token_budget = int(self.base_token_budget * concept.salience)

            has_constraint = constraints.get("constraint", 0.0) > 0.3

            if has_constraint and self.constraint_expansion:
                # Constraint expansion: split into context + assertion
                # Goal 1: Set up the context
                goals.append(
                    ThoughtGoal(
                        id=self._make_id(concept.text, "ctx"),
                        text=f"Establish context for: {concept.text}",
                        source_concept_text=concept.text,
                        salience=concept.salience,
                        domain=primary_domain,
                        memory_hints=memory_hints,
                        constraints={},  # No constraints on context setup
                        priority=priority_counter,
                        max_tokens=token_budget // 3,  # Context is brief
                    )
                )
                priority_counter += 1

                # Goal 2: Apply the constraint
                goals.append(
                    ThoughtGoal(
                        id=self._make_id(concept.text, "cst"),
                        text=f"Address constraint: {concept.text}",
                        source_concept_text=concept.text,
                        salience=concept.salience * 1.2,  # Boost constrained
                        domain=primary_domain,
                        memory_hints=memory_hints,
                        constraints=constraints,
                        priority=priority_counter,
                        max_tokens=token_budget * 2 // 3,
                    )
                )
                priority_counter += 1
            else:
                # Standard single goal
                goals.append(
                    ThoughtGoal(
                        id=self._make_id(concept.text, "std"),
                        text=f"Address: {concept.text}",
                        source_concept_text=concept.text,
                        salience=concept.salience,
                        domain=primary_domain,
                        memory_hints=memory_hints,
                        constraints=constraints,
                        priority=priority_counter,
                        max_tokens=token_budget,
                    )
                )
                priority_counter += 1

        # Merge low-salience concepts into a catch-all goal
        if low_salience_concepts:
            merged_text = "; ".join(c.text for c in low_salience_concepts)
            merged_memories: list[str] = []
            for c in low_salience_concepts:
                merged_memories.extend(m.content[:200] for m in c.activated_memories)

            goals.append(
                ThoughtGoal(
                    id=self._make_id(merged_text, "mrg"),
                    text=f"Also briefly address: {merged_text}",
                    source_concept_text=merged_text,
                    salience=0.3,
                    domain="general",
                    memory_hints=merged_memories[:3],  # Cap hints
                    constraints={},
                    priority=priority_counter,
                    max_tokens=self.base_token_budget // 2,
                )
            )

        # Sort by priority (already ordered, but enforce)
        goals.sort(key=lambda g: g.priority)

        logger.debug(
            "ThoughtPlanner produced %d goals from %d concepts",
            len(goals),
            len(understanding.concepts),
        )

        return goals

    @staticmethod
    def _make_id(text: str, suffix: str) -> str:
        """Generate a short deterministic ID from text + suffix."""
        h = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"tg_{h}_{suffix}"
