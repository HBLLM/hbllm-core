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
    predicate: str = "has_state"
    value: Any = None
    confidence: float = 0.9
    evidence_sources: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    is_latent: bool = False
    distribution: dict[str, float] = field(default_factory=dict)

    def update_distribution(self, likelihoods: dict[str, float]) -> dict[str, float]:
        """Perform normalized Bayesian posterior update on discrete state distribution."""
        if not self.distribution:
            self.distribution = {str(k): float(v) for k, v in likelihoods.items()}
            total = sum(self.distribution.values())
            if total > 0.0:
                self.distribution = {k: v / total for k, v in self.distribution.items()}
            return self.distribution

        unnormalized = {}
        for state, prior in self.distribution.items():
            lh = likelihoods.get(state, 1.0)
            unnormalized[state] = prior * lh

        total = sum(unnormalized.values())
        if total > 0.0:
            self.distribution = {k: v / total for k, v in unnormalized.items()}
            # Update value to maximum a posteriori (MAP) state
            map_state = max(self.distribution.items(), key=lambda x: x[1])
            self.value = map_state[0]
            self.confidence = map_state[1]
        self.last_updated = time.time()
        return self.distribution


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

    def get_latent_beliefs(self) -> list[WorldBeliefNode]:
        """Retrieve all active latent variable beliefs."""
        return [b for b in self._beliefs.values() if b.is_latent]

    def all_beliefs(self) -> list[WorldBeliefNode]:
        """Retrieve all beliefs in the graph."""
        return list(self._beliefs.values())
