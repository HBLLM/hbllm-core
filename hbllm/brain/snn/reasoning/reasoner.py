"""
Causal Reasoner — multi-hop graph traversal with SNN evaluation.

Given a set of concept IDs (from comprehension), discovers causal
chains by traversing the ``CausalGraph`` up to a configurable depth,
evaluates each chain with the ``ReasoningNetwork``, and returns the
top-K most confident chains.

This is the bridge between the passive causal storage layer
(``brain/causality/``) and active SNN-driven reasoning.

Usage::

    from hbllm.brain.snn.reasoning.reasoner import CausalReasoner
    from hbllm.brain.causality.causal_graph import CausalGraph

    graph = CausalGraph(data_dir="data")
    reasoner = CausalReasoner(graph)

    chains = reasoner.reason(["concept_1", "concept_2"])
    for chain in chains:
        print(f"{chain.source_concept} → {chain.conclusion} "
              f"(conf={chain.snn_confidence:.2f})")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.causality.causal_graph import CausalGraph, CausalLink
from hbllm.brain.snn.reasoning.reasoning_network import ReasoningNetwork

logger = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """A discovered reasoning path through the causal graph.

    Attributes:
        links: Ordered sequence of CausalLink objects forming the chain.
        depth: Number of hops in the chain.
        combined_probability: Product of all link probabilities.
        snn_confidence: ReasoningNetwork's confidence score [0,1].
        source_concept: Text/ID of the starting concept.
        conclusion: Text/ID of the final inferred concept.
    """

    links: list[CausalLink] = field(default_factory=list)
    depth: int = 0
    combined_probability: float = 0.0
    snn_confidence: float = 0.0
    source_concept: str = ""
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in UnderstandingState."""
        return {
            "depth": self.depth,
            "combined_probability": round(self.combined_probability, 4),
            "snn_confidence": round(self.snn_confidence, 4),
            "source_concept": self.source_concept,
            "conclusion": self.conclusion,
            "links": [link.to_dict() for link in self.links],
        }


class CausalReasoner:
    """Multi-hop causal graph traversal with SNN-based evaluation.

    Given concept IDs (from comprehension spikes), discovers causal
    chains by:

    1. **BFS traversal** of ``CausalGraph`` following ``get_effects()``
       links up to ``max_depth`` hops.
    2. **Probability pruning**: discard chains whose combined probability
       falls below ``min_probability``.
    3. **SNN evaluation**: score each surviving chain with the
       ``ReasoningNetwork`` for structural quality.
    4. **Ranking**: return top-K chains sorted by SNN confidence.

    Args:
        causal_graph: The CausalGraph to traverse.
        reasoning_network: SNN for evaluating chain quality.
            If None, a default ReasoningNetwork is created.
        max_depth: Maximum chain depth (hops). Default 3.
        min_probability: Minimum combined probability to keep a chain.
            Default 0.3.
        top_k: Maximum number of chains to return. Default 5.
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        reasoning_network: ReasoningNetwork | None = None,
        max_depth: int = 3,
        min_probability: float = 0.3,
        top_k: int = 5,
    ) -> None:
        self._graph = causal_graph
        self._network = reasoning_network or ReasoningNetwork()
        self._max_depth = max_depth
        self._min_probability = min_probability
        self._top_k = top_k

    def reason(self, concept_ids: list[str]) -> list[CausalChain]:
        """Find and evaluate causal chains from concept IDs.

        Traverses forward from each concept ID, discovering what
        effects they cause.  Returns the top-K most confident chains
        across all starting concepts.

        Args:
            concept_ids: List of concept IDs to reason from.

        Returns:
            List of CausalChain objects, sorted by SNN confidence
            descending.
        """
        all_chains: list[CausalChain] = []

        for concept_id in concept_ids:
            raw_paths = self._traverse_forward(concept_id, self._max_depth)

            for path in raw_paths:
                if not path:
                    continue

                # Calculate combined probability
                combined_prob = 1.0
                for link in path:
                    combined_prob *= link.probability

                # Prune weak chains
                if combined_prob < self._min_probability:
                    continue

                # Evaluate with SNN
                features = self._extract_features(path)
                snn_confidence = self._network.evaluate(features)

                chain = CausalChain(
                    links=path,
                    depth=len(path),
                    combined_probability=combined_prob,
                    snn_confidence=snn_confidence,
                    source_concept=path[0].source_id,
                    conclusion=path[-1].target_id,
                )
                all_chains.append(chain)

        # Sort by SNN confidence, take top-K
        all_chains.sort(key=lambda c: c.snn_confidence, reverse=True)
        result = all_chains[: self._top_k]

        if result:
            logger.info(
                "CausalReasoner found %d chains from %d concepts (top confidence: %.3f)",
                len(result),
                len(concept_ids),
                result[0].snn_confidence,
            )

        return result

    def reason_between(self, source_id: str, target_id: str) -> list[CausalChain]:
        """Find causal paths between two specific concepts.

        Uses BFS to find all paths from source_id that reach target_id
        within max_depth hops.

        Args:
            source_id: Starting concept ID.
            target_id: Target concept ID to reach.

        Returns:
            List of CausalChain objects connecting source to target.
        """
        paths = self._traverse_forward(source_id, self._max_depth)

        matching: list[CausalChain] = []
        for path in paths:
            if not path:
                continue
            # Check if the chain ends at target_id
            if path[-1].target_id != target_id:
                continue

            combined_prob = 1.0
            for link in path:
                combined_prob *= link.probability

            if combined_prob < self._min_probability:
                continue

            features = self._extract_features(path)
            snn_confidence = self._network.evaluate(features)

            chain = CausalChain(
                links=path,
                depth=len(path),
                combined_probability=combined_prob,
                snn_confidence=snn_confidence,
                source_concept=source_id,
                conclusion=target_id,
            )
            matching.append(chain)

        matching.sort(key=lambda c: c.snn_confidence, reverse=True)
        return matching[: self._top_k]

    def _traverse_forward(self, start_id: str, max_depth: int) -> list[list[CausalLink]]:
        """BFS forward traversal of the causal graph.

        Discovers all paths from start_id by following ``get_effects()``
        links up to max_depth hops.  Avoids cycles by tracking visited
        nodes per path.

        Args:
            start_id: Starting node ID.
            max_depth: Maximum traversal depth.

        Returns:
            List of paths, where each path is a list of CausalLinks.
        """
        all_paths: list[list[CausalLink]] = []

        # BFS queue: (current_node_id, current_path, visited_set)
        queue: list[tuple[str, list[CausalLink], set[str]]] = [(start_id, [], {start_id})]

        while queue:
            current_id, current_path, visited = queue.pop(0)

            if len(current_path) >= max_depth:
                continue

            # Get outgoing links
            effects = self._graph.get_effects(current_id)

            for link in effects:
                next_id = link.target_id

                # Avoid cycles
                if next_id in visited:
                    continue

                new_path = current_path + [link]
                new_visited = visited | {next_id}

                # Record this path
                all_paths.append(new_path)

                # Continue BFS if we haven't hit max depth
                if len(new_path) < max_depth:
                    queue.append((next_id, new_path, new_visited))

        return all_paths

    def _extract_features(self, chain: list[CausalLink]) -> dict[str, float]:
        """Extract SNN input features from a causal chain.

        Features are all normalized to [0, 1]:
        - chain_probability: product of link probabilities
        - chain_length: 1.0 / depth (shorter = higher)
        - recency: based on most recent link timestamp
        - diversity: fraction of unique nodes relative to chain length

        Args:
            chain: List of CausalLink objects.

        Returns:
            Dict of normalized features for ReasoningNetwork.
        """
        if not chain:
            return {
                "chain_probability": 0.0,
                "chain_length": 0.0,
                "recency": 0.0,
                "diversity": 0.0,
            }

        # Combined probability
        combined_prob = 1.0
        for link in chain:
            combined_prob *= link.probability

        # Chain length (inverted: shorter = higher)
        chain_length = 1.0 / len(chain)

        # Recency: how recent is the most recent link?
        now = time.time()
        most_recent = max(link.created_at for link in chain)
        age_seconds = max(0.0, now - most_recent)
        # Normalize: links within 1 hour are "recent"
        recency = max(0.0, 1.0 - (age_seconds / 3600.0))

        # Diversity: unique nodes / total possible nodes
        all_nodes: set[str] = set()
        for link in chain:
            all_nodes.add(link.source_id)
            all_nodes.add(link.target_id)
        max_unique = len(chain) + 1  # perfect chain has all unique
        diversity = len(all_nodes) / max(1, max_unique)

        return {
            "chain_probability": combined_prob,
            "chain_length": chain_length,
            "recency": recency,
            "diversity": min(1.0, diversity),
        }

    @property
    def graph(self) -> CausalGraph:
        """Access the underlying CausalGraph."""
        return self._graph

    @property
    def reasoning_network(self) -> ReasoningNetwork:
        """Access the underlying ReasoningNetwork."""
        return self._network
