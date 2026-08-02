"""Evidence Evaluator — scores evidence quality, weight, bias, and trust.

Every piece of evidence must be evaluated before it touches beliefs.
The evaluation produces an ``EvidenceAssessment`` whose dimensions
map directly to ``BeliefConfidence`` dimensions during belief revision.

Design principle: evidence quality is not one number.  It decomposes
into methodology, sample size, reproducibility, source trust, and
reality level weight.  This prevents a single weak source from
inflating overall belief confidence.

Architecture::

    Evidence
        │
        ├── Quality   ← methodology + sample size + controls
        ├── Weight    ← evidence strength × reality level
        ├── Bias      ← confirmation, selection, publication bias
        ├── Trust     ← source reputation adjustment
        └── Reproducibility ← independent replication status
                │
                ▼
        EvidenceAssessment → feeds BeliefConfidence dimensions

Usage::

    evaluator = EvidenceEvaluator(graph=graph, reputation=tracker)
    assessment = await evaluator.evaluate(evidence_id)
    # assessment.quality_score, .weight, .bias_flags, .trust_adjustment
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.epistemics.interfaces import EvidenceAssessment
from hbllm.hcir.graph import (
    CognitiveGraph,
    EvidenceNode,
    ExperimentNode,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.types import EvidenceStrength, ExperimentRealityLevel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Evidence Weight Tables — domain-neutral scoring
# ═══════════════════════════════════════════════════════════════════════════

#: Base quality weights by evidence strength classification.
EVIDENCE_STRENGTH_WEIGHTS: dict[EvidenceStrength, float] = {
    EvidenceStrength.ANECDOTAL: 0.10,
    EvidenceStrength.OBSERVATIONAL: 0.30,
    EvidenceStrength.CORRELATIONAL: 0.45,
    EvidenceStrength.EXPERIMENTAL: 0.70,
    EvidenceStrength.META_ANALYTIC: 0.85,
    EvidenceStrength.REPLICATED: 0.95,
}

#: Weight multipliers by experiment reality level.
REALITY_LEVEL_WEIGHTS: dict[ExperimentRealityLevel, float] = {
    ExperimentRealityLevel.SIMULATION: 0.2,
    ExperimentRealityLevel.DIGITAL: 0.4,
    ExperimentRealityLevel.OBSERVATIONAL: 0.6,
    ExperimentRealityLevel.CONTROLLED: 0.8,
    ExperimentRealityLevel.PHYSICAL: 1.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Evidence Evaluator
# ═══════════════════════════════════════════════════════════════════════════


class EvidenceEvaluator:
    """Scores evidence quality, weight, bias, and trust.

    Implements the ``IEvidenceEvaluator`` protocol.

    The evaluator is domain-neutral — it scores based on methodology,
    evidence type, reproducibility, and source reputation.  It never
    knows what the evidence is *about* (medicine, robotics, etc.).

    The evaluator uses two modes:
    - **Structural scoring**: Based on evidence metadata (type, sample
      size, reproducibility, reality level).
    - **LLM-assisted scoring**: Optional deep analysis for bias
      detection and methodology assessment (when ``llm`` is provided).
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        reputation_tracker: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize the evidence evaluator.

        Args:
            graph: The shared HCIR cognitive graph.
            reputation_tracker: Optional SourceReputationTracker for
                trust adjustment.
            llm: Optional LLM instance for bias detection.
        """
        self._graph = graph
        self._reputation = reputation_tracker
        self._llm = llm

    async def evaluate(self, evidence_id: str) -> EvidenceAssessment:
        """Full evidence evaluation — quality, bias, trust.

        Produces an ``EvidenceAssessment`` whose dimensions feed directly
        into ``BeliefConfidence`` during belief revision.

        Args:
            evidence_id: The HCIR EvidenceNode ID to evaluate.

        Returns:
            Complete evidence assessment.
        """
        node = self._graph.get_node(evidence_id)
        if not isinstance(node, EvidenceNode):
            logger.warning("Node %s is not an EvidenceNode", evidence_id)
            return EvidenceAssessment(evidence_id=evidence_id)

        quality = self._compute_quality(node)
        reality_weight = self._compute_reality_level_weight(node)
        reproducibility = self._assess_reproducibility(node)
        trust = await self._compute_trust(node)
        bias_flags = await self.detect_bias(evidence_id)

        # Final weight: quality × reality × trust, penalized by bias
        bias_penalty = max(0.5, 1.0 - 0.1 * len(bias_flags))
        weight = min(1.0, quality * reality_weight * trust * bias_penalty)

        return EvidenceAssessment(
            evidence_id=evidence_id,
            quality_score=quality,
            weight=weight,
            bias_flags=bias_flags,
            trust_adjustment=trust,
            reality_level_weight=reality_weight,
            reproducibility_status=reproducibility,
            reasoning=self._build_reasoning(
                node, quality, reality_weight, trust, bias_flags,
            ),
        )

    async def compute_weight(self, evidence_id: str) -> float:
        """Compute the evidence weight for belief revision.

        Convenience method that returns just the weight value.
        """
        assessment = await self.evaluate(evidence_id)
        return assessment.weight

    async def detect_bias(self, evidence_id: str) -> list[str]:
        """Detect potential biases in the evidence.

        Uses structural heuristics and optional LLM analysis.

        Common biases detected:
        - **confirmation_bias**: Evidence only supports prior beliefs
        - **selection_bias**: Non-representative sampling
        - **publication_bias**: Only positive results published
        - **survivorship_bias**: Missing failure cases
        - **single_source**: Only one source/study
        """
        node = self._graph.get_node(evidence_id)
        if not isinstance(node, EvidenceNode):
            return []

        flags: list[str] = []

        # Structural heuristics
        if node.sample_size is not None and node.sample_size < 30:
            flags.append("small_sample_size")

        if not node.limitations:
            flags.append("no_limitations_reported")

        if not node.dataset_refs:
            flags.append("no_data_references")

        if node.evidence_type == EvidenceStrength.ANECDOTAL:
            flags.append("anecdotal_evidence")

        if not node.reproducible:
            flags.append("not_reproduced")

        # Check if evidence only has supporting links (no counter)
        edges = self._graph.edges_from(evidence_id) + self._graph.edges_to(evidence_id)
        support_count = sum(
            1 for e in edges if e.edge_type == HCIREdgeType.SUPPORTS
        )
        contradict_count = sum(
            1 for e in edges if e.edge_type in (
                HCIREdgeType.CONTRADICTS, HCIREdgeType.WEAKENS,
            )
        )
        if support_count > 0 and contradict_count == 0:
            flags.append("one_directional_evidence")

        # LLM-assisted bias detection (if available)
        if self._llm is not None and node.methodology:
            try:
                llm_flags = await self._llm_bias_analysis(node)
                flags.extend(llm_flags)
            except Exception as exc:
                logger.warning(
                    "LLM bias analysis failed for %s: %s",
                    evidence_id, exc,
                )

        return flags

    # ── Internal Scoring Methods ───────────────────────────────────────

    def _compute_quality(self, node: EvidenceNode) -> float:
        """Compute evidence quality from methodology metadata."""
        base = EVIDENCE_STRENGTH_WEIGHTS.get(
            node.evidence_type, 0.3,
        )

        # Boost for large sample sizes
        sample_boost = 0.0
        if node.sample_size is not None:
            if node.sample_size >= 1000:
                sample_boost = 0.15
            elif node.sample_size >= 100:
                sample_boost = 0.10
            elif node.sample_size >= 30:
                sample_boost = 0.05

        # Penalty for missing methodology description
        methodology_factor = 1.0 if node.methodology else 0.8

        # Boost for documented limitations (intellectual honesty)
        limitation_boost = min(0.1, 0.03 * len(node.limitations))

        quality = (base + sample_boost + limitation_boost) * methodology_factor
        return min(1.0, max(0.0, quality))

    def _compute_reality_level_weight(self, node: EvidenceNode) -> float:
        """Compute weight from experiment reality level (if linked)."""
        # Check if evidence is linked to an experiment
        edges = self._graph.edges_from(node.id) + self._graph.edges_to(node.id)
        for edge in edges:
            if edge.edge_type == HCIREdgeType.TESTS:
                for target_id in edge.targets:
                    exp_node = self._graph.get_node(target_id)
                    if isinstance(exp_node, ExperimentNode):
                        return REALITY_LEVEL_WEIGHTS.get(
                            exp_node.reality_level, 0.5,
                        )

        # Default: use evidence strength as proxy
        if node.evidence_type in (
            EvidenceStrength.EXPERIMENTAL,
            EvidenceStrength.META_ANALYTIC,
            EvidenceStrength.REPLICATED,
        ):
            return 0.7
        return 0.5

    def _assess_reproducibility(self, node: EvidenceNode) -> str:
        """Assess the reproducibility status of evidence."""
        if node.reproducible:
            return "reproduced"

        # Check for replication edges in graph
        edges = self._graph.edges_from(node.id) + self._graph.edges_to(node.id)
        has_replication = any(
            e.edge_type == HCIREdgeType.REPLICATES for e in edges
        )
        if has_replication:
            return "replication_attempted"

        return "not_attempted"

    async def _compute_trust(self, node: EvidenceNode) -> float:
        """Compute source trust adjustment."""
        if self._reputation is None or not node.source_uri:
            return 1.0  # No adjustment without reputation data

        try:
            reputation = await self._reputation.get_reputation(node.source_uri)
            return max(0.1, reputation)
        except Exception:
            return 1.0

    async def _llm_bias_analysis(self, node: EvidenceNode) -> list[str]:
        """Use LLM to detect subtle biases in methodology."""
        prompt = (
            f"Analyze this evidence methodology for potential biases.\n"
            f"Evidence type: {node.evidence_type}\n"
            f"Methodology: {node.methodology}\n"
            f"Limitations: {', '.join(node.limitations) if node.limitations else 'None reported'}\n"
            f"Sample size: {node.sample_size}\n\n"
            f"List only the bias types found (one per line). "
            f"Valid types: confirmation_bias, selection_bias, publication_bias, "
            f"survivorship_bias, experimenter_bias, funding_bias, "
            f"recency_bias, authority_bias.\n"
            f"If no biases detected, respond with 'none'."
        )
        response = await self._llm.generate(prompt)
        text = response if isinstance(response, str) else str(response)

        if "none" in text.lower():
            return []

        # Parse bias flags from LLM response
        valid_biases = {
            "confirmation_bias", "selection_bias", "publication_bias",
            "survivorship_bias", "experimenter_bias", "funding_bias",
            "recency_bias", "authority_bias",
        }
        found = []
        for line in text.strip().split("\n"):
            cleaned = line.strip().lower().replace(" ", "_")
            if cleaned in valid_biases:
                found.append(cleaned)
        return found

    def _build_reasoning(
        self,
        node: EvidenceNode,
        quality: float,
        reality_weight: float,
        trust: float,
        bias_flags: list[str],
    ) -> str:
        """Build a human-readable reasoning string for the assessment."""
        parts = [
            f"Evidence type: {node.evidence_type.value} "
            f"(base quality={quality:.2f})",
            f"Reality level weight: {reality_weight:.2f}",
            f"Source trust: {trust:.2f}",
        ]
        if node.sample_size is not None:
            parts.append(f"Sample size: {node.sample_size}")
        if node.reproducible:
            parts.append("Independently reproduced: yes")
        if bias_flags:
            parts.append(f"Detected biases: {', '.join(bias_flags)}")
        else:
            parts.append("No biases detected")
        return "; ".join(parts)
