"""
World Belief Graph — Cognitive World Belief Container.

Maintains cognitive beliefs, evidence sources, and certainty levels about the world.
Strict invariant: WorldBeliefGraph CANNOT directly mutate DigitalTwinRegistry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.world_state_interpreter import InterpretedBeliefHypothesis

logger = logging.getLogger(__name__)


@dataclass
class WorldBeliefNode:
    """Individual cognitive belief held by HCIR about a world subject."""

    belief_id: str
    subject: str
    predicate: str
    value: Any
    confidence: float = 0.9
    evidence_sources: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class WorldBeliefGraph:
    """Graph container for higher-order cognitive world beliefs.

    Usage::

        belief_graph = WorldBeliefGraph(world_id="factory_a")
        belief_graph.add_belief(WorldBeliefNode(...))
    """

    def __init__(self, world_id: str = "default_world") -> None:
        self.world_id = world_id
        self._beliefs: dict[str, WorldBeliefNode] = {}

    def add_belief(self, belief: WorldBeliefNode) -> None:
        """Add or update a cognitive belief. Does NOT mutate DigitalTwinRegistry."""
        self._beliefs[belief.belief_id] = belief
        logger.debug(
            "WorldBeliefGraph [%s] added belief '%s' (%s %s %s)",
            self.world_id,
            belief.belief_id,
            belief.subject,
            belief.predicate,
            belief.value,
        )

    def ingest_hypotheses(self, hypotheses: list[InterpretedBeliefHypothesis]) -> None:
        """Incorporate interpreted hypotheses into belief graph."""
        for hyp in hypotheses:
            belief_id = f"b_{hyp.subject}_{hyp.predicate}"
            existing = self._beliefs.get(belief_id)
            sources = [hyp.evidence_source]
            if existing and hyp.evidence_source not in existing.evidence_sources:
                sources = existing.evidence_sources + [hyp.evidence_source]

            self._beliefs[belief_id] = WorldBeliefNode(
                belief_id=belief_id,
                subject=hyp.subject,
                predicate=hyp.predicate,
                value=hyp.value,
                confidence=hyp.confidence,
                evidence_sources=sources,
                last_updated=time.time(),
            )

    def get_belief(self, belief_id: str) -> WorldBeliefNode | None:
        """Retrieve belief by ID."""
        return self._beliefs.get(belief_id)

    def get_beliefs_for_subject(self, subject: str) -> list[WorldBeliefNode]:
        """Retrieve all beliefs targeting a specific subject."""
        return [b for b in self._beliefs.values() if b.subject == subject]

    def all_beliefs(self) -> list[WorldBeliefNode]:
        """Retrieve all beliefs in the graph."""
        return list(self._beliefs.values())
