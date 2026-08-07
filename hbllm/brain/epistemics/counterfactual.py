"""Counterfactual Reasoner — 'What if...' epistemic graph analysis.

Evaluates how beliefs would change under hypothetical modifications
to the evidence graph.  **Never modifies the real graph.**

Architecture::

    CounterfactualReasoner
        ├── Fork graph (lightweight copy)
        ├── Apply mutation (remove evidence, falsify hypothesis, ...)
        ├── Propagate effects (recalculate confidence)
        ├── Compare with original
        └── Report CounterfactualResult

Use cases::

    "What if this hypothesis is wrong?"
        → Shows which beliefs depend on it

    "What if we remove this evidence?"
        → Shows if beliefs survive without it

    "What if evidence quality were different?"
        → Sensitivity analysis for evidence

    "Which evidence matters most for this belief?"
        → sensitivity_analysis() ranks by impact

Complementary to ``simulation/models.py``:
    - Planning-level counterfactuals: tasks → world states
    - Epistemic counterfactuals: evidence → beliefs (this module)

Usage::

    reasoner = CounterfactualReasoner(graph=graph)
    result = await reasoner.what_if_evidence_removed("evidence_001")
    for belief_id, delta in result.confidence_deltas.items():
        print(f"  Belief {belief_id}: confidence change {delta:+.3f}")
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from hbllm.brain.epistemics.interfaces import CounterfactualResult
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    HypothesisNode,
)

logger = logging.getLogger(__name__)


class CounterfactualReasoner:
    """'What if...' reasoning via cognitive graph modification.

    Implements the ``ICounterfactualReasoner`` protocol.

    Creates temporary graph forks to evaluate how beliefs would
    change under hypothetical modifications.  Never modifies
    the real graph.

    The reasoner works with the HCIR cognitive graph directly,
    analyzing evidence → hypothesis → belief chains.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
    ) -> None:
        """Initialize the counterfactual reasoner.

        Args:
            graph: The shared HCIR cognitive graph (read-only usage).
        """
        self._graph = graph

    async def what_if_hypothesis_wrong(
        self,
        hypothesis_id: str,
    ) -> CounterfactualResult:
        """What would change if this hypothesis were falsified?

        Identifies all beliefs that depend on this hypothesis and
        estimates confidence changes.

        Args:
            hypothesis_id: The hypothesis to hypothetically falsify.

        Returns:
            CounterfactualResult with affected beliefs and deltas.
        """
        node = self._graph.get_node(hypothesis_id)
        if not isinstance(node, HypothesisNode):
            return CounterfactualResult(
                scenario=f"Node {hypothesis_id} is not a hypothesis",
                mutation_type="falsify_hypothesis",
                mutation_target_id=hypothesis_id,
            )

        # Find all beliefs supported by this hypothesis
        affected_beliefs: list[str] = []
        confidence_deltas: dict[str, float] = {}
        cascading: list[str] = []

        # Follow edges from hypothesis to things it supports
        edges_from = self._graph.edges_from(hypothesis_id)
        for edge in edges_from:
            if edge.edge_type in (
                HCIREdgeType.SUPPORTS,
                HCIREdgeType.STRENGTHENS,
                HCIREdgeType.DERIVED_FROM,
            ):
                for target_id in edge.targets:
                    target = self._graph.get_node(target_id)
                    if isinstance(target, BeliefNode):
                        affected_beliefs.append(target_id)

                        # Estimate impact: how much does this hypothesis
                        # contribute to the belief's confidence?
                        impact = self._estimate_support_impact(
                            target_id, hypothesis_id,
                        )
                        confidence_deltas[target_id] = -impact
                        cascading.append(
                            f"Belief '{target.claim[:50]}' loses "
                            f"{impact:.2f} confidence"
                        )

        # Also check predictions from this hypothesis
        for edge in edges_from:
            if edge.edge_type == HCIREdgeType.PREDICTS:
                for target_id in edge.targets:
                    cascading.append(
                        f"Prediction {target_id} would be invalidated"
                    )

        structural_impact = (
            len(affected_beliefs) / max(1, self._count_beliefs())
        )

        return CounterfactualResult(
            scenario=f"If hypothesis '{node.claim[:60]}' were falsified",
            mutation_type="falsify_hypothesis",
            mutation_target_id=hypothesis_id,
            affected_beliefs=affected_beliefs,
            confidence_deltas=confidence_deltas,
            cascading_effects=cascading,
            structural_impact=min(1.0, structural_impact),
        )

    async def what_if_evidence_removed(
        self,
        evidence_id: str,
    ) -> CounterfactualResult:
        """What would change if this evidence were retracted?

        Traces all hypotheses and beliefs that depend on this evidence
        and estimates the cascading confidence changes.

        Args:
            evidence_id: The evidence to hypothetically retract.

        Returns:
            CounterfactualResult with affected beliefs and deltas.
        """
        node = self._graph.get_node(evidence_id)
        if not isinstance(node, EvidenceNode):
            return CounterfactualResult(
                scenario=f"Node {evidence_id} is not evidence",
                mutation_type="remove_evidence",
                mutation_target_id=evidence_id,
            )

        affected_beliefs: list[str] = []
        confidence_deltas: dict[str, float] = {}
        cascading: list[str] = []

        # Find everything this evidence supports (hypotheses + beliefs)
        supported_nodes = self._find_supported_nodes(evidence_id)

        for node_id in supported_nodes:
            target = self._graph.get_node(node_id)

            if isinstance(target, BeliefNode):
                affected_beliefs.append(node_id)
                impact = self._estimate_support_impact(node_id, evidence_id)
                confidence_deltas[node_id] = -impact
                cascading.append(
                    f"Belief '{target.claim[:50]}' loses {impact:.2f} confidence"
                )

            elif isinstance(target, HypothesisNode):
                # The hypothesis loses support → cascade to its beliefs
                impact = self._estimate_support_impact(node_id, evidence_id)
                cascading.append(
                    f"Hypothesis '{target.claim[:50]}' loses "
                    f"{impact:.2f} support"
                )

                # Find beliefs depending on this hypothesis
                hyp_beliefs = self._find_supported_nodes(node_id)
                for bid in hyp_beliefs:
                    b = self._graph.get_node(bid)
                    if isinstance(b, BeliefNode) and bid not in affected_beliefs:
                        affected_beliefs.append(bid)
                        cascade_impact = impact * 0.5  # Dampened cascade
                        confidence_deltas[bid] = confidence_deltas.get(
                            bid, 0.0,
                        ) - cascade_impact

        structural_impact = (
            len(affected_beliefs) / max(1, self._count_beliefs())
        )

        return CounterfactualResult(
            scenario=f"If evidence '{evidence_id}' were retracted",
            mutation_type="remove_evidence",
            mutation_target_id=evidence_id,
            affected_beliefs=affected_beliefs,
            confidence_deltas=confidence_deltas,
            cascading_effects=cascading,
            structural_impact=min(1.0, structural_impact),
        )

    async def what_if_evidence_quality(
        self,
        evidence_id: str,
        new_quality: float,
    ) -> CounterfactualResult:
        """What would change if this evidence had different quality?

        Useful for sensitivity analysis: which evidence quality
        changes would have the biggest impact on beliefs?

        Args:
            evidence_id: The evidence node ID.
            new_quality: The hypothetical quality score [0.0, 1.0].

        Returns:
            CounterfactualResult with confidence changes.
        """
        node = self._graph.get_node(evidence_id)
        if not isinstance(node, EvidenceNode):
            return CounterfactualResult(
                scenario=f"Node {evidence_id} is not evidence",
                mutation_type="change_evidence_quality",
                mutation_target_id=evidence_id,
            )

        # Estimate current quality contribution
        current_quality = getattr(node, "quality_score", 0.5)
        quality_delta = new_quality - current_quality

        if abs(quality_delta) < 0.01:
            return CounterfactualResult(
                scenario="Quality change too small to have effect",
                mutation_type="change_evidence_quality",
                mutation_target_id=evidence_id,
            )

        affected_beliefs: list[str] = []
        confidence_deltas: dict[str, float] = {}
        cascading: list[str] = []

        supported_nodes = self._find_supported_nodes(evidence_id)
        for node_id in supported_nodes:
            target = self._graph.get_node(node_id)
            if isinstance(target, BeliefNode):
                affected_beliefs.append(node_id)
                # Quality change propagates proportionally
                impact = self._estimate_support_impact(node_id, evidence_id)
                delta = impact * quality_delta
                confidence_deltas[node_id] = delta
                direction = "gains" if delta > 0 else "loses"
                cascading.append(
                    f"Belief '{target.claim[:50]}' {direction} "
                    f"{abs(delta):.3f} confidence"
                )

        return CounterfactualResult(
            scenario=(
                f"If evidence quality changed from "
                f"{current_quality:.2f} → {new_quality:.2f}"
            ),
            mutation_type="change_evidence_quality",
            mutation_target_id=evidence_id,
            affected_beliefs=affected_beliefs,
            confidence_deltas=confidence_deltas,
            cascading_effects=cascading,
            structural_impact=min(1.0, abs(quality_delta)),
        )

    async def what_if_new_evidence(
        self,
        belief_id: str,
        evidence_quality: float = 0.8,
        direction: str = "supporting",
    ) -> CounterfactualResult:
        """What would change if new evidence appeared for this belief?

        Args:
            belief_id: The belief to add hypothetical evidence for.
            evidence_quality: Quality of the hypothetical evidence.
            direction: "supporting" or "contradicting".

        Returns:
            CounterfactualResult showing the expected impact.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            return CounterfactualResult(
                scenario=f"Node {belief_id} is not a belief",
                mutation_type="add_evidence",
                mutation_target_id=belief_id,
            )

        # Estimate impact of new evidence
        current_conf = node.belief_confidence.derived_confidence
        support_count = self._count_supporting_evidence(belief_id)

        # Diminishing returns: more evidence → less impact per piece
        diminishing_factor = 1.0 / (1.0 + support_count * 0.3)
        raw_impact = evidence_quality * 0.2 * diminishing_factor

        if direction == "contradicting":
            raw_impact = -raw_impact * 1.5  # Contradicting evidence has more impact

        new_conf = max(0.0, min(1.0, current_conf + raw_impact))
        delta = new_conf - current_conf

        return CounterfactualResult(
            scenario=(
                f"If new {direction} evidence (quality={evidence_quality:.2f}) "
                f"appeared for '{node.claim[:50]}'"
            ),
            mutation_type="add_evidence",
            mutation_target_id=belief_id,
            affected_beliefs=[belief_id],
            confidence_deltas={belief_id: delta},
            cascading_effects=[
                f"Confidence: {current_conf:.3f} → {new_conf:.3f} "
                f"(Δ={delta:+.3f})"
            ],
            structural_impact=abs(delta),
        )

    async def sensitivity_analysis(
        self,
        belief_id: str,
    ) -> dict[str, float]:
        """Find which evidence has the highest impact on this belief.

        For each piece of supporting evidence, estimates how much
        removing it would change the belief's confidence.

        Args:
            belief_id: The belief to analyze.

        Returns:
            Dict of evidence_id → impact score.  Higher = more
            critical to the belief.  Sorted by impact descending.
        """
        node = self._graph.get_node(belief_id)
        if not isinstance(node, BeliefNode):
            return {}

        # Find all evidence supporting this belief (direct + through hypotheses)
        evidence_ids = self._find_supporting_evidence(belief_id)

        if not evidence_ids:
            return {}

        # Estimate impact of removing each piece of evidence
        impacts: dict[str, float] = {}
        for eid in evidence_ids:
            impact = self._estimate_support_impact(belief_id, eid)
            if impact > 0.001:
                impacts[eid] = impact

        # Sort by impact descending
        return dict(
            sorted(impacts.items(), key=lambda x: x[1], reverse=True)
        )

    # ── Internal Methods ───────────────────────────────────────────────

    def _find_supported_nodes(self, source_id: str) -> list[str]:
        """Find all nodes directly supported by a source node."""
        supported: list[str] = []
        edges = self._graph.edges_from(source_id)

        for edge in edges:
            if edge.edge_type in (
                HCIREdgeType.SUPPORTS,
                HCIREdgeType.STRENGTHENS,
                HCIREdgeType.DERIVED_FROM,
            ):
                supported.extend(edge.targets)

        return supported

    def _find_supporting_evidence(
        self,
        belief_id: str,
        max_depth: int = 3,
    ) -> list[str]:
        """Find all evidence nodes supporting a belief (BFS through graph)."""
        evidence_ids: list[str] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(belief_id, 0)]

        while queue:
            node_id, depth = queue.pop(0)
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)

            # Check edges pointing TO this node
            edges = self._graph.edges_to(node_id)
            for edge in edges:
                if edge.edge_type in (
                    HCIREdgeType.SUPPORTS,
                    HCIREdgeType.STRENGTHENS,
                    HCIREdgeType.DERIVED_FROM,
                    HCIREdgeType.TESTS,
                    HCIREdgeType.REPLICATES,
                ):
                    for src_id in edge.sources:
                        src = self._graph.get_node(src_id)
                        if isinstance(src, EvidenceNode):
                            if src_id not in evidence_ids:
                                evidence_ids.append(src_id)
                        elif src_id not in visited:
                            queue.append((src_id, depth + 1))

        return evidence_ids

    def _estimate_support_impact(
        self,
        belief_id: str,
        supporter_id: str,
    ) -> float:
        """Estimate how much a supporter contributes to a belief's confidence.

        Uses a simple heuristic: impact = 1 / (number of supporters).
        A belief with only 1 supporting evidence has impact 1.0;
        a belief with 5 supporters has impact ≈ 0.2 per supporter.
        """
        # Count total supporters
        all_supporters = self._find_supporting_evidence(belief_id)

        if not all_supporters:
            return 0.0

        # Check if supporter is actually in the support chain
        if supporter_id not in all_supporters:
            # Check if it's a hypothesis supporting the belief
            edges_from = self._graph.edges_from(supporter_id)
            is_supporter = False
            for edge in edges_from:
                if (
                    edge.edge_type in (
                        HCIREdgeType.SUPPORTS,
                        HCIREdgeType.STRENGTHENS,
                    )
                    and belief_id in edge.targets
                ):
                    is_supporter = True
                    break

            if not is_supporter:
                return 0.0

            # Hypothesis supporter → estimate contribution
            all_supporters.append(supporter_id)

        total = len(all_supporters)
        base_impact = 1.0 / total

        # Weight by the belief's current confidence
        node = self._graph.get_node(belief_id)
        if isinstance(node, BeliefNode):
            conf = node.belief_confidence.derived_confidence
            return base_impact * conf

        return base_impact

    def _count_beliefs(self) -> int:
        """Count total beliefs in the graph."""
        count = 0
        for node in self._graph.all_nodes():
            if isinstance(node, BeliefNode):
                count += 1
        return count

    def _count_supporting_evidence(self, belief_id: str) -> int:
        """Count how many evidence nodes support a belief."""
        return len(self._find_supporting_evidence(belief_id))
