"""Reference & Discourse Resolver for A16.

Maintains discourse context and resolves anaphoric references (pronouns like "it",
"that", "this") to the most salient, compatible entity in recent conversational history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hbllm.brain.language.core.semantic_frame import EntityReference

logger = logging.getLogger(__name__)


@dataclass
class DiscourseEntity:
    """An entity active in the recent conversational discourse."""

    entity_id: str
    concept_name: str
    properties: dict[str, str] = field(default_factory=dict)
    salience: float = 1.0  # Decays with subsequent turns
    mention_turn: int = 0


class ReferenceResolver:
    """Tracks discourse state and resolves pronouns to recent entities.

    Usage::

        resolver = ReferenceResolver()
        resolver.register_mention(entity_id="e1", concept_name="ball", properties={"color": "red"})

        # Later turn: "Move it"
        ref = EntityReference(specifier="anaphoric", raw_text="it")
        resolved = resolver.resolve_anaphor(ref)
        # -> returns DiscourseEntity with entity_id="e1"
    """

    def __init__(self, max_history: int = 10, decay_rate: float = 0.8) -> None:
        self._history: list[DiscourseEntity] = []
        self._current_turn: int = 0
        self._max_history = max_history
        self._decay_rate = decay_rate

    def next_turn(self) -> None:
        """Advance the conversational turn and decay salience of previous mentions."""
        self._current_turn += 1
        for item in self._history:
            item.salience *= self._decay_rate
        # Prune low-salience or old items
        self._history = [item for item in self._history if item.salience > 0.1][: self._max_history]

    def register_mention(
        self,
        entity_id: str,
        concept_name: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Record an entity mention in the current turn with highest salience."""
        # If already in history, boost salience to front
        self._history = [item for item in self._history if item.entity_id != entity_id]
        discourse_item = DiscourseEntity(
            entity_id=entity_id,
            concept_name=concept_name,
            properties=properties or {},
            salience=1.0,
            mention_turn=self._current_turn,
        )
        self._history.insert(0, discourse_item)

    def resolve_anaphor(self, ref: EntityReference) -> DiscourseEntity | None:
        """Resolve a pronoun or anaphoric reference to a DiscourseEntity."""
        if not self._history:
            return None

        # Filter by compatibility if reference has constraints
        for item in self._history:
            if ref.concept_name and ref.concept_name != item.concept_name:
                continue
            if ref.properties:
                props_match = all(
                    item.properties.get(k) == v
                    for k, v in ref.properties.items()
                )
                if not props_match:
                    continue
            return item

        # Fallback to most salient entity if general pronoun ("it", "this", "that")
        return self._history[0] if self._history else None

    def clear(self) -> None:
        """Clear discourse history."""
        self._history.clear()
        self._current_turn = 0
