"""Hypothesis Builder — filters, validates, deduplicates ideas into HypothesisNodes.

The funnel::

    30 raw ideas
        → Validator (plausibility, testability, coherence)
        → 4 plausible candidates
        → Deduplicator (semantic similarity against existing hypotheses)
        → 2 novel candidates
        → promote_to_node()
        → HypothesisNode in HCIR graph

This separation ensures LLM creativity isn't constrained by
validation logic.  The IdeaGenerator is deliberately unconstrained;
the HypothesisBuilder is deliberately strict.

Usage::

    builder = HypothesisBuilder(graph=graph, llm=llm)
    ideas = await idea_generator.generate_from_unknown(unknown_id)
    candidates = await builder.validate(ideas)
    novel = await builder.deduplicate(candidates)
    for candidate in novel:
        node_id = await builder.promote_to_node(candidate, program_id)
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.epistemics.interfaces import HypothesisCandidate, RawIdea
from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    HypothesisNode,
    UnknownNode,
)
from hbllm.hcir.types import (
    DiscoveryTrigger,
    EpistemicLifecycle,
    FalsificationStatus,
    KnowledgeValue,
)

logger = logging.getLogger(__name__)


class HypothesisBuilder:
    """Filters, validates, and deduplicates raw ideas into HypothesisNodes.

    Implements the ``IHypothesisBuilder`` protocol.

    The builder applies a multi-stage funnel:
    1. **Validate**: Score ideas by plausibility, testability, coherence
    2. **Deduplicate**: Reject ideas semantically equivalent to existing hypotheses
    3. **Promote**: Create HCIR HypothesisNodes with proper lifecycle and edges

    The builder is domain-neutral — it scores structural properties
    (testability, specificity) not domain-specific content.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        llm: Any | None = None,
        min_plausibility: float = 0.3,
        min_testability: float = 0.2,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize the hypothesis builder.

        Args:
            graph: The shared HCIR cognitive graph.
            llm: Optional LLM instance for semantic validation.
            min_plausibility: Minimum plausibility to pass validation.
            min_testability: Minimum testability to pass validation.
            similarity_threshold: Semantic similarity threshold for dedup.
        """
        self._graph = graph
        self._llm = llm
        self._min_plausibility = min_plausibility
        self._min_testability = min_testability
        self._similarity_threshold = similarity_threshold

    async def validate(
        self,
        ideas: list[RawIdea],
    ) -> list[HypothesisCandidate]:
        """Filter raw ideas for plausibility, testability, and coherence.

        Args:
            ideas: Raw ideas from the IdeaGenerator.

        Returns:
            List of validated HypothesisCandidate objects.
        """
        if not ideas:
            return []

        candidates: list[HypothesisCandidate] = []

        for idea in ideas:
            # Skip empty claims
            if not idea.claim.strip():
                continue

            # Score the idea
            scores = await self._score_idea(idea)
            plausibility = scores.get("plausibility", idea.plausibility)
            testability = scores.get("testability", 0.5)
            novelty = scores.get("novelty", 0.5)
            impact = scores.get("impact", 0.5)

            # Apply minimum thresholds
            if plausibility < self._min_plausibility:
                logger.debug(
                    "Idea rejected (low plausibility=%.2f): %s",
                    plausibility, idea.claim[:60],
                )
                continue

            if testability < self._min_testability:
                logger.debug(
                    "Idea rejected (low testability=%.2f): %s",
                    testability, idea.claim[:60],
                )
                continue

            candidates.append(HypothesisCandidate(
                claim=idea.claim,
                novelty=novelty,
                plausibility=plausibility,
                predicted_impact=impact,
                testability=testability,
                origin=idea.origin_trigger if isinstance(idea.origin_trigger, str) else str(idea.origin_trigger),
                reasoning=idea.reasoning or scores.get("reasoning", ""),
            ))

        # Sort by composite score (plausibility + novelty + impact)
        candidates.sort(
            key=lambda c: c.plausibility * 0.4 + c.novelty * 0.3 + c.predicted_impact * 0.3,
            reverse=True,
        )

        logger.info(
            "Validated %d/%d ideas into candidates",
            len(candidates), len(ideas),
        )
        return candidates

    async def deduplicate(
        self,
        candidates: list[HypothesisCandidate],
        existing_hypothesis_ids: list[str] | None = None,
    ) -> list[HypothesisCandidate]:
        """Remove candidates semantically equivalent to existing hypotheses.

        Args:
            candidates: Validated hypothesis candidates.
            existing_hypothesis_ids: Optional list of existing hypothesis
                IDs to check against.  If None, checks all hypotheses
                in the graph.

        Returns:
            Deduplicated list of novel candidates.
        """
        if not candidates:
            return []

        # Collect existing hypothesis claims
        existing_claims = self._get_existing_claims(existing_hypothesis_ids)

        if not existing_claims:
            return candidates  # Nothing to deduplicate against

        novel: list[HypothesisCandidate] = []
        for candidate in candidates:
            is_duplicate = await self._is_duplicate(candidate.claim, existing_claims)
            if is_duplicate:
                logger.debug(
                    "Candidate deduplicated (similar to existing): %s",
                    candidate.claim[:60],
                )
            else:
                novel.append(candidate)

        logger.info(
            "Deduplicated %d/%d candidates (novel: %d)",
            len(candidates) - len(novel), len(candidates), len(novel),
        )
        return novel

    async def promote_to_node(
        self,
        candidate: HypothesisCandidate,
        program_id: str = "",
    ) -> str:
        """Create an HCIR HypothesisNode from a validated candidate.

        Sets the initial ``EpistemicLifecycle``, ``KnowledgeValue``,
        and graph edges.

        Args:
            candidate: A validated, deduplicated candidate.
            program_id: Optional research program ID.

        Returns:
            The new HypothesisNode ID.
        """
        # Map origin to DiscoveryTrigger
        trigger = self._map_origin_to_trigger(candidate.origin)

        node = HypothesisNode(
            claim=candidate.claim,
            supporting_evidence=candidate.supporting_evidence,
            counter_evidence=[],
            epistemic_lifecycle=EpistemicLifecycle.HYPOTHESIZED,
            falsification_status=FalsificationStatus.UNTESTED,
            novelty=candidate.novelty,
            plausibility=candidate.plausibility,
            predicted_impact=candidate.predicted_impact,
            testability=candidate.testability,
            origin=candidate.origin,
            trigger=trigger,
            research_program_id=program_id,
            knowledge_value=KnowledgeValue(
                novelty=candidate.novelty,
                impact=candidate.predicted_impact,
                strategic_relevance=0.5,
            ),
        )

        self._graph.upsert_node(node)

        logger.info(
            "Promoted hypothesis to graph: %s (id=%s, program=%s)",
            candidate.claim[:60], node.id, program_id or "none",
        )
        return node.id

    # ── Scoring ────────────────────────────────────────────────────────

    async def _score_idea(self, idea: RawIdea) -> dict[str, Any]:
        """Score a raw idea on plausibility, testability, novelty, impact."""
        if self._llm is not None:
            return await self._llm_score(idea)
        return self._structural_score(idea)

    def _structural_score(self, idea: RawIdea) -> dict[str, Any]:
        """Score an idea using structural heuristics (no LLM)."""
        plausibility = idea.plausibility

        # Testability: ideas with specific mechanisms are more testable
        testability = 0.5
        claim_lower = idea.claim.lower()
        if any(w in claim_lower for w in ("cause", "mechanism", "leads to", "results in")):
            testability += 0.2
        if any(w in claim_lower for w in ("correlat", "associat")):
            testability += 0.1
        if any(w in claim_lower for w in ("may", "might", "could", "possibly")):
            testability -= 0.1
        testability = min(1.0, max(0.0, testability))

        # Novelty: rough heuristic
        novelty = 0.5

        # Impact: use plausibility as proxy without LLM
        impact = min(1.0, plausibility + 0.1)

        return {
            "plausibility": plausibility,
            "testability": testability,
            "novelty": novelty,
            "impact": impact,
            "reasoning": "Structural scoring (no LLM)",
        }

    async def _llm_score(self, idea: RawIdea) -> dict[str, Any]:
        """Use LLM to score an idea."""
        prompt = (
            f"Evaluate this hypothesis candidate:\n"
            f"Claim: {idea.claim}\n"
            f"Reasoning: {idea.reasoning}\n\n"
            f"Score each dimension from 0.0 to 1.0:\n"
            f"PLAUSIBILITY: (is this claim reasonable given current knowledge?)\n"
            f"TESTABILITY: (can this be tested with experiments or observations?)\n"
            f"NOVELTY: (does this offer a new perspective?)\n"
            f"IMPACT: (what would the implications be if confirmed?)\n\n"
            f"Format: PLAUSIBILITY: X | TESTABILITY: X | NOVELTY: X | IMPACT: X"
        )

        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
            return self._parse_scores(text, idea)
        except Exception as exc:
            logger.warning("LLM scoring failed: %s", exc)
            return self._structural_score(idea)

    def _parse_scores(
        self, text: str, idea: RawIdea,
    ) -> dict[str, Any]:
        """Parse LLM scoring response."""
        scores: dict[str, Any] = {
            "plausibility": idea.plausibility,
            "testability": 0.5,
            "novelty": 0.5,
            "impact": 0.5,
            "reasoning": "LLM scoring",
        }

        for part in text.split("|"):
            part = part.strip().upper()
            for key in ("PLAUSIBILITY", "TESTABILITY", "NOVELTY", "IMPACT"):
                if part.startswith(f"{key}:"):
                    try:
                        val = float(part.split(":")[1].strip())
                        scores[key.lower()] = min(1.0, max(0.0, val))
                    except (ValueError, IndexError):
                        pass

        return scores

    # ── Deduplication ──────────────────────────────────────────────────

    def _get_existing_claims(
        self, hypothesis_ids: list[str] | None,
    ) -> list[str]:
        """Collect existing hypothesis claims for deduplication."""
        claims: list[str] = []

        if hypothesis_ids is not None:
            for hid in hypothesis_ids:
                node = self._graph.get_node(hid)
                if isinstance(node, HypothesisNode) and node.claim:
                    claims.append(node.claim)
        else:
            # Scan all hypotheses in graph
            for node in self._graph.all_nodes():
                if isinstance(node, HypothesisNode) and node.claim:
                    claims.append(node.claim)

        return claims

    async def _is_duplicate(
        self, claim: str, existing: list[str],
    ) -> bool:
        """Check if a claim is semantically similar to any existing claim."""
        if self._llm is not None:
            return await self._llm_similarity_check(claim, existing)

        # Fallback: simple string containment / overlap
        claim_words = set(claim.lower().split())
        for existing_claim in existing:
            existing_words = set(existing_claim.lower().split())
            if not claim_words or not existing_words:
                continue
            overlap = len(claim_words & existing_words)
            similarity = overlap / max(len(claim_words), len(existing_words))
            if similarity >= self._similarity_threshold:
                return True
        return False

    async def _llm_similarity_check(
        self, claim: str, existing: list[str],
    ) -> bool:
        """Use LLM to check semantic similarity."""
        existing_str = "\n".join(f"- {c}" for c in existing[:10])
        prompt = (
            f"Is this new claim semantically equivalent to any existing claim?\n\n"
            f"New claim: {claim}\n\n"
            f"Existing claims:\n{existing_str}\n\n"
            f"Answer YES or NO only."
        )

        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
            return "yes" in text.lower()
        except Exception:
            return False

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _map_origin_to_trigger(origin: str) -> DiscoveryTrigger:
        """Map a string origin to a DiscoveryTrigger enum."""
        mapping = {
            "contradiction": DiscoveryTrigger.CONTRADICTION,
            "analogy": DiscoveryTrigger.ANALOGY,
            "anomaly": DiscoveryTrigger.ANOMALY,
            "gap": DiscoveryTrigger.KNOWLEDGE_GAP,
            "knowledge_gap": DiscoveryTrigger.KNOWLEDGE_GAP,
            "curiosity": DiscoveryTrigger.CURIOSITY,
            "unexpected_success": DiscoveryTrigger.UNEXPECTED_SUCCESS,
            "unexpected_failure": DiscoveryTrigger.UNEXPECTED_FAILURE,
            "novel_observation": DiscoveryTrigger.NOVEL_OBSERVATION,
        }
        return mapping.get(origin, DiscoveryTrigger.KNOWLEDGE_GAP)
