"""Idea Generator — raw creative generation from LLM.

Deliberately unconstrained — quantity over quality.  The Idea Generator
produces ``RawIdea`` objects from knowledge gaps, contradictions,
anomalies, and analogies.  Filtering, validation, and deduplication
are the HypothesisBuilder's responsibility.

Architecture::

    IdeaGenerator (this module)
        │
        ├── generate_from_unknown     → 5-30 raw ideas
        ├── generate_from_contradiction → 5-20 raw ideas
        ├── generate_from_analogy     → 3-15 raw ideas
        └── generate_from_anomaly     → 5-20 raw ideas
                │
                ▼
        list[RawIdea]  →  HypothesisBuilder (filter funnel)

Design principle: the LLM is creative; the validator is strict.
Separating these concerns prevents good ideas from being filtered
too early and bad ideas from bypassing validation.

Usage::

    generator = IdeaGenerator(graph=graph, llm=llm)
    ideas = await generator.generate_from_unknown(unknown_id)
    # → [RawIdea(claim="...", plausibility=0.7), ...]
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.epistemics.interfaces import RawIdea
from hbllm.hcir.graph import (
    CognitiveGraph,
    ContradictionNode,
    HCIRNodeType,
    ObservationNode,
    UnknownNode,
)
from hbllm.hcir.types import DiscoveryTrigger

logger = logging.getLogger(__name__)


class IdeaGenerator:
    """Generates raw ideas from knowledge gaps, contradictions, and anomalies.

    Implements the ``IIdeaGenerator`` protocol.

    The generator is deliberately unconstrained — it asks the LLM for
    many possible explanations without filtering.  This maximizes
    creative coverage.  Quality control happens downstream in the
    HypothesisBuilder.

    If no LLM is available, falls back to template-based generation
    (structural ideas derived from graph topology).
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        llm: Any | None = None,
        max_ideas_per_generation: int = 15,
    ) -> None:
        """Initialize the idea generator.

        Args:
            graph: The shared HCIR cognitive graph.
            llm: Optional LLM instance for creative reasoning.
            max_ideas_per_generation: Maximum ideas per generation call.
        """
        self._graph = graph
        self._llm = llm
        self._max_ideas = max_ideas_per_generation

    async def generate_from_unknown(
        self,
        unknown_id: str,
        context: dict[str, Any] | None = None,
    ) -> list[RawIdea]:
        """Generate raw ideas to explain a knowledge gap.

        Args:
            unknown_id: The HCIR UnknownNode ID.
            context: Optional additional context for the LLM.

        Returns:
            List of unvalidated RawIdea objects.
        """
        node = self._graph.get_node(unknown_id)
        if not isinstance(node, UnknownNode):
            logger.warning("Node %s is not an UnknownNode", unknown_id)
            return []

        if self._llm is not None:
            return await self._llm_generate_from_unknown(node, context)

        # Fallback: template-based generation
        return self._template_generate_from_unknown(node)

    async def generate_from_contradiction(
        self,
        contradiction_id: str,
    ) -> list[RawIdea]:
        """Generate raw ideas to resolve a contradiction.

        Args:
            contradiction_id: The HCIR ContradictionNode ID.

        Returns:
            List of unvalidated RawIdea objects.
        """
        node = self._graph.get_node(contradiction_id)
        if not isinstance(node, ContradictionNode):
            logger.warning("Node %s is not a ContradictionNode", contradiction_id)
            return []

        if self._llm is not None:
            return await self._llm_generate_from_contradiction(node)

        return self._template_generate_from_contradiction(node)

    async def generate_from_analogy(
        self,
        source_domain: str,
        target_domain: str,
        structural_pattern: str,
    ) -> list[RawIdea]:
        """Generate raw ideas by cross-domain analogical reasoning.

        Args:
            source_domain: The domain where the pattern was observed.
            target_domain: The domain to apply the pattern to.
            structural_pattern: Description of the structural similarity.

        Returns:
            List of unvalidated RawIdea objects.
        """
        if self._llm is not None:
            return await self._llm_generate_from_analogy(
                source_domain, target_domain, structural_pattern,
            )

        return [
            RawIdea(
                claim=f"The pattern '{structural_pattern}' from "
                      f"{source_domain} may apply to {target_domain}",
                plausibility=0.4,
                origin_trigger=DiscoveryTrigger.ANALOGY,
                reasoning="Cross-domain structural transfer (template)",
            )
        ]

    async def generate_from_anomaly(
        self,
        observation_id: str,
    ) -> list[RawIdea]:
        """Generate raw ideas to explain an anomalous observation.

        Args:
            observation_id: The HCIR ObservationNode ID.

        Returns:
            List of unvalidated RawIdea objects.
        """
        node = self._graph.get_node(observation_id)
        if not isinstance(node, ObservationNode):
            logger.warning("Node %s is not an ObservationNode", observation_id)
            return []

        if self._llm is not None:
            return await self._llm_generate_from_anomaly(node)

        return self._template_generate_from_anomaly(node)

    # ── LLM-Powered Generation ─────────────────────────────────────────

    async def _llm_generate_from_unknown(
        self,
        node: UnknownNode,
        context: dict[str, Any] | None,
    ) -> list[RawIdea]:
        """Use LLM to generate ideas from a knowledge gap."""
        context_str = ""
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        elif node.context:
            context_str = node.context

        prompt = (
            f"A knowledge gap has been identified:\n"
            f"Question: {node.question}\n"
            f"Domain: {node.domain or 'unspecified'}\n"
            f"Context: {context_str or 'None'}\n\n"
            f"Generate {self._max_ideas} possible explanations or mechanisms "
            f"that could address this gap.  Be creative and diverse.  "
            f"Include unlikely but interesting possibilities.\n\n"
            f"For each idea, provide:\n"
            f"1. A one-sentence claim\n"
            f"2. Your plausibility estimate (0.0 to 1.0)\n"
            f"3. Brief reasoning\n\n"
            f"Format each as: CLAIM: ... | PLAUSIBILITY: ... | REASONING: ..."
        )

        return await self._parse_llm_response(
            prompt, DiscoveryTrigger.KNOWLEDGE_GAP, node.id,
        )

    async def _llm_generate_from_contradiction(
        self,
        node: ContradictionNode,
    ) -> list[RawIdea]:
        """Use LLM to generate resolution ideas for a contradiction."""
        # Get the conflicting claims
        claim_a = self._get_node_description(node.claim_a_id)
        claim_b = self._get_node_description(node.claim_b_id)

        prompt = (
            f"Two claims contradict each other:\n"
            f"Claim A: {claim_a}\n"
            f"Claim B: {claim_b}\n"
            f"Contradiction type: {node.contradiction_type}\n\n"
            f"Generate {self._max_ideas} possible explanations for why "
            f"these claims conflict.  Consider:\n"
            f"- Hidden variables that could explain both\n"
            f"- Context dependencies (both could be true in different contexts)\n"
            f"- Methodological differences\n"
            f"- Measurement errors\n"
            f"- Scale or scope differences\n\n"
            f"Format each as: CLAIM: ... | PLAUSIBILITY: ... | REASONING: ..."
        )

        return await self._parse_llm_response(
            prompt, DiscoveryTrigger.CONTRADICTION, node.id,
        )

    async def _llm_generate_from_analogy(
        self,
        source_domain: str,
        target_domain: str,
        structural_pattern: str,
    ) -> list[RawIdea]:
        """Use LLM for cross-domain analogical reasoning."""
        prompt = (
            f"A structural pattern has been observed:\n"
            f"Source domain: {source_domain}\n"
            f"Target domain: {target_domain}\n"
            f"Pattern: {structural_pattern}\n\n"
            f"Generate {self._max_ideas} hypotheses about how this pattern "
            f"might manifest in the target domain.  Consider direct "
            f"transfers, adapted mechanisms, and surprising implications.\n\n"
            f"Format each as: CLAIM: ... | PLAUSIBILITY: ... | REASONING: ..."
        )

        return await self._parse_llm_response(
            prompt, DiscoveryTrigger.ANALOGY, "",
        )

    async def _llm_generate_from_anomaly(
        self,
        node: ObservationNode,
    ) -> list[RawIdea]:
        """Use LLM to explain an anomalous observation."""
        description = getattr(node, "description", "") or ""

        prompt = (
            f"An anomalous observation has been recorded:\n"
            f"Observation: {description}\n"
            f"Tags: {', '.join(node.tags) if node.tags else 'None'}\n\n"
            f"This observation doesn't fit existing models.  "
            f"Generate {self._max_ideas} possible explanations.  "
            f"Include mundane causes, measurement errors, AND genuinely "
            f"novel mechanisms.\n\n"
            f"Format each as: CLAIM: ... | PLAUSIBILITY: ... | REASONING: ..."
        )

        return await self._parse_llm_response(
            prompt, DiscoveryTrigger.ANOMALY, node.id,
        )

    async def _parse_llm_response(
        self,
        prompt: str,
        trigger: DiscoveryTrigger,
        origin_id: str,
    ) -> list[RawIdea]:
        """Send prompt to LLM and parse the response into RawIdeas."""
        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            return []

        ideas: list[RawIdea] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "CLAIM:" not in line.upper():
                continue

            try:
                parts = line.split("|")
                claim = ""
                plausibility = 0.5
                reasoning = ""

                for part in parts:
                    part = part.strip()
                    upper = part.upper()
                    if upper.startswith("CLAIM:"):
                        claim = part[6:].strip()
                    elif upper.startswith("PLAUSIBILITY:"):
                        try:
                            plausibility = float(part[13:].strip())
                            plausibility = min(1.0, max(0.0, plausibility))
                        except ValueError:
                            plausibility = 0.5
                    elif upper.startswith("REASONING:"):
                        reasoning = part[10:].strip()

                if claim:
                    ideas.append(RawIdea(
                        claim=claim,
                        plausibility=plausibility,
                        origin_trigger=trigger,
                        origin_id=origin_id,
                        reasoning=reasoning,
                    ))
            except Exception:
                continue

        return ideas[:self._max_ideas]

    # ── Template-Based Fallback ────────────────────────────────────────

    def _template_generate_from_unknown(
        self, node: UnknownNode,
    ) -> list[RawIdea]:
        """Generate structural ideas without LLM."""
        ideas = []

        # Look at related observations for context
        for obs_id in node.related_observations[:3]:
            obs = self._graph.get_node(obs_id)
            if obs is not None:
                ideas.append(RawIdea(
                    claim=f"The observation '{obs_id}' may directly "
                          f"explain: {node.question}",
                    plausibility=0.3,
                    origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
                    origin_id=node.id,
                    reasoning="Direct observation-to-question link (template)",
                ))

        # Default structural ideas
        ideas.append(RawIdea(
            claim=f"An unknown mechanism may explain: {node.question}",
            plausibility=0.3,
            origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
            origin_id=node.id,
            reasoning="Default gap-filling hypothesis (template)",
        ))

        return ideas

    def _template_generate_from_contradiction(
        self, node: ContradictionNode,
    ) -> list[RawIdea]:
        """Generate structural ideas from contradiction topology."""
        return [
            RawIdea(
                claim="A hidden variable may explain the contradiction "
                      f"between {node.claim_a_id} and {node.claim_b_id}",
                plausibility=0.4,
                origin_trigger=DiscoveryTrigger.CONTRADICTION,
                origin_id=node.id,
                reasoning="Hidden variable hypothesis (template)",
            ),
            RawIdea(
                claim="The contradiction may be context-dependent — "
                      "both claims could be true under different conditions",
                plausibility=0.4,
                origin_trigger=DiscoveryTrigger.CONTRADICTION,
                origin_id=node.id,
                reasoning="Context dependency hypothesis (template)",
            ),
        ]

    def _template_generate_from_anomaly(
        self, node: ObservationNode,
    ) -> list[RawIdea]:
        """Generate structural ideas from anomalous observation."""
        return [
            RawIdea(
                claim=f"The anomaly in {node.id} may be a measurement artifact",
                plausibility=0.3,
                origin_trigger=DiscoveryTrigger.ANOMALY,
                origin_id=node.id,
                reasoning="Measurement error hypothesis (template)",
            ),
            RawIdea(
                claim=f"The anomaly in {node.id} may indicate a genuinely "
                      "novel mechanism not captured by existing models",
                plausibility=0.3,
                origin_trigger=DiscoveryTrigger.ANOMALY,
                origin_id=node.id,
                reasoning="Novel mechanism hypothesis (template)",
            ),
        ]

    def _get_node_description(self, node_id: str) -> str:
        """Get a human-readable description of any node."""
        node = self._graph.get_node(node_id)
        if node is None:
            return f"(unknown node: {node_id})"
        claim = getattr(node, "claim", None) or getattr(node, "statement", None)
        if claim:
            return claim
        return getattr(node, "description", node_id) or node_id
