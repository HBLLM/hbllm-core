"""Explanation Engine — 'Why do you believe this?' via graph traversal.

Not generated text — structured provenance chains through HCIR nodes.
Every belief is traceable back to observations through evidence and
experiments.

Architecture::

    Belief("X causes Y", confidence=0.82)
        ├── Evidence(experiment_42, quality=0.90, source="arxiv:2301.12345")
        │     └── Experiment(discriminative, reality=DIGITAL)
        │           └── Prediction(confirmed, hypothesis_17)
        ├── Evidence(observation_99, quality=0.60, source="sensor_log")
        └── Counter-Evidence(claim_12, quality=0.30, source="blog_post")
              └── Source Reputation: 0.34 (low trust)

This is a graph walk, not text generation.  One of HBLLM's strongest
potential features.

Usage::

    engine = ExplanationEngine(graph=graph)
    chain = await engine.explain_belief(belief_id)
    for step in chain.steps:
        print(f"{step.node_type}: {step.label} [{step.edge_type}]")
"""

from __future__ import annotations

import logging

from hbllm.brain.epistemics.interfaces import ExplanationChain, ExplanationStep
from hbllm.hcir.graph import (
    AudioObservationNode,
    BeliefNode,
    ClaimNode,
    CognitiveGraph,
    EvidenceNode,
    ExperimentNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNode,
    HypothesisNode,
    ObservationNode,
    PerceptualEvidenceNode,
    PredictionNode,
    VisualObservationNode,
)

logger = logging.getLogger(__name__)


# Edge types that carry evidential weight in explanation chains.
_SUPPORTING_EDGES = frozenset(
    {
        HCIREdgeType.SUPPORTS,
        HCIREdgeType.STRENGTHENS,
        HCIREdgeType.DERIVED_FROM,
        HCIREdgeType.TESTS,
        HCIREdgeType.PREDICTS,
        HCIREdgeType.REPLICATES,
    }
)

_COUNTER_EDGES = frozenset(
    {
        HCIREdgeType.CONTRADICTS,
        HCIREdgeType.WEAKENS,
        HCIREdgeType.FALSIFIES,
    }
)


class ExplanationEngine:
    """Answers 'Why do you believe this?' via HCIR graph traversal.

    Implements the ``IExplanationEngine`` protocol.

    The engine walks the cognitive graph starting from a belief node,
    following evidential edges (SUPPORTS, STRENGTHENS, DERIVED_FROM,
    TESTS, PREDICTS) to build a structured explanation chain.

    It also collects counter-evidence through CONTRADICTS, WEAKENS,
    and FALSIFIES edges.

    The engine is domain-neutral — it traverses edge types and node
    types, never interpreting content semantically.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        max_depth: int = 6,
    ) -> None:
        """Initialize the explanation engine.

        Args:
            graph: The shared HCIR cognitive graph.
            max_depth: Maximum traversal depth to prevent cycles.
        """
        self._graph = graph
        self._max_depth = max_depth

    async def explain_belief(self, belief_id: str) -> ExplanationChain:
        """Trace a belief back to its supporting and counter evidence.

        Walks the graph following evidential edges to build a structured
        explanation chain.  Each step includes the edge type and
        confidence contribution.

        Args:
            belief_id: The HCIR BeliefNode ID to explain.

        Returns:
            An ExplanationChain with supporting steps and counter-evidence.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            logger.warning("Node %s is not a BeliefNode", belief_id)
            return ExplanationChain(belief_id=belief_id)

        # Collect supporting and counter evidence chains
        supporting: list[ExplanationStep] = []
        counter: list[ExplanationStep] = []
        visited: set[str] = {belief_id}

        self._walk_support(belief_id, supporting, visited, depth=0)
        self._walk_counter(belief_id, counter, visited, depth=0)

        return ExplanationChain(
            belief_id=belief_id,
            belief_claim=node.claim,
            derived_confidence=node.belief_confidence.derived_confidence,
            steps=supporting,
            counter_evidence=counter,
        )

    async def explain_confidence(
        self,
        belief_id: str,
    ) -> dict[str, float]:
        """Return the confidence decomposition for a belief.

        Returns all ``BeliefConfidence`` dimensions plus the derived score.

        Args:
            belief_id: The HCIR BeliefNode ID.

        Returns:
            Dict with dimension names → float values.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            return {}
        return node.belief_confidence.to_dict()

    async def trace_to_observations(
        self,
        belief_id: str,
    ) -> list[str]:
        """Follow the full provenance chain to original observations.

        Returns all ObservationNode IDs reachable from the belief
        through supporting evidential edges.

        Args:
            belief_id: The HCIR BeliefNode ID.

        Returns:
            List of ObservationNode IDs at the leaves of the chain.
        """
        observations: list[str] = []
        visited: set[str] = {belief_id}
        self._find_observations(belief_id, observations, visited, depth=0)
        return observations

    # ── Internal Traversal ─────────────────────────────────────────────

    def _walk_support(
        self,
        node_id: str,
        steps: list[ExplanationStep],
        visited: set[str],
        depth: int,
    ) -> None:
        """Recursively walk supporting edges to build the explanation chain."""
        if depth >= self._max_depth:
            return

        edges = self._graph.edges_from(node_id) + self._graph.edges_to(node_id)
        for edge in edges:
            if edge.edge_type not in _SUPPORTING_EDGES:
                continue

            # Follow edges where the current node is a target
            # (i.e., something supports/strengthens this node)
            source_ids = self._get_supporting_sources(edge, node_id)
            for source_id in source_ids:
                if source_id in visited:
                    continue
                visited.add(source_id)

                source_node = self._graph.get_node(source_id)
                if source_node is None:
                    continue

                step = self._node_to_step(
                    source_node,
                    edge.edge_type,
                )
                steps.append(step)

                # Recurse into deeper provenance
                self._walk_support(source_id, steps, visited, depth + 1)

    def _walk_counter(
        self,
        node_id: str,
        steps: list[ExplanationStep],
        visited: set[str],
        depth: int,
    ) -> None:
        """Walk counter-evidence edges."""
        if depth >= self._max_depth:
            return

        edges = self._graph.edges_from(node_id) + self._graph.edges_to(node_id)
        for edge in edges:
            if edge.edge_type not in _COUNTER_EDGES:
                continue

            source_ids = self._get_supporting_sources(edge, node_id)
            for source_id in source_ids:
                if source_id in visited:
                    continue
                visited.add(source_id)

                source_node = self._graph.get_node(source_id)
                if source_node is None:
                    continue

                step = self._node_to_step(
                    source_node,
                    edge.edge_type,
                )
                steps.append(step)

    def _find_observations(
        self,
        node_id: str,
        observations: list[str],
        visited: set[str],
        depth: int,
    ) -> None:
        """Find all ObservationNode leaves in the provenance chain."""
        if depth >= self._max_depth:
            return

        node = self._graph.get_node(node_id)
        if isinstance(
            node,
            (
                ObservationNode,
                PerceptualEvidenceNode,
                VisualObservationNode,
                AudioObservationNode,
                EvidenceNode,
            ),
        ):
            observations.append(node_id)
            return

        edges = self._graph.edges_from(node_id) + self._graph.edges_to(node_id)
        for edge in edges:
            if edge.edge_type not in _SUPPORTING_EDGES:
                continue

            source_ids = self._get_supporting_sources(edge, node_id)
            for source_id in source_ids:
                if source_id in visited:
                    continue
                visited.add(source_id)
                self._find_observations(
                    source_id,
                    observations,
                    visited,
                    depth + 1,
                )

    def _get_supporting_sources(
        self,
        edge: HCIREdge,
        target_id: str,
    ) -> list[str]:
        """Get source node IDs that support/affect the target."""
        # For edges like SUPPORTS, sources support the targets
        if target_id in edge.targets:
            return [s for s in edge.sources if s != target_id]
        # For edges like DERIVED_FROM, the node derives from sources
        if target_id in edge.sources:
            return [t for t in edge.targets if t != target_id]
        return []

    def _node_to_step(
        self,
        node: HCIRNode,
        edge_type: HCIREdgeType | str,
    ) -> ExplanationStep:
        """Convert an HCIR node into an ExplanationStep."""
        label = self._get_node_label(node)
        source_uri = ""
        confidence_contribution = node.uncertainty.confidence

        if isinstance(node, EvidenceNode):
            source_uri = node.source_uri
            confidence_contribution = node.strength
        elif isinstance(node, PerceptualEvidenceNode):
            confidence_contribution = node.strength
        elif isinstance(node, ExperimentNode):
            label = f"Experiment: {node.design[:60]}" if node.design else label
        elif isinstance(node, PredictionNode):
            label = f"Prediction: {node.claim[:60]}" if node.claim else label
        elif isinstance(node, HypothesisNode):
            label = f"Hypothesis: {node.claim[:60]}" if node.claim else label

        return ExplanationStep(
            node_id=node.id,
            node_type=node.node_type.value,
            edge_type=edge_type.value if hasattr(edge_type, "value") else str(edge_type),
            label=label,
            confidence_contribution=confidence_contribution,
            source_uri=source_uri,
        )

    def _get_node_label(self, node: HCIRNode) -> str:
        """Generate a human-readable label for any HCIR node."""
        if isinstance(node, (BeliefNode, HypothesisNode, PredictionNode, ClaimNode)):
            claim = getattr(node, "claim", "") or getattr(node, "statement", "")
            return f"{node.node_type.value}: {claim[:80]}" if claim else node.node_type.value
        if isinstance(node, PerceptualEvidenceNode):
            return f"perceptual_evidence ({node.modality}): {node.proposition.subject} {node.proposition.predicate} {node.proposition.object_value}"
        if isinstance(node, EvidenceNode):
            return f"evidence: {node.methodology[:60]}" if node.methodology else "evidence"
        if isinstance(node, ExperimentNode):
            return f"experiment: {node.design[:60]}" if node.design else "experiment"
        if isinstance(node, ObservationNode):
            desc = getattr(node, "description", "") or ""
            return f"observation: {desc[:60]}" if desc else "observation"
        return node.node_type.value
