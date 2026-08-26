"""Grounded Lexicon, Asymmetric Grounding/Realization & Compositionality for A17.

Manages active hypothesis sets and committed LexicalSenses in HCIR.
Supports multilingual convergence, polysemy resolution, compositional grounding,
and epistemically honest realization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.brain.language.acquisition.lexical_hypothesis import (
    LexicalCandidateStatus,
    LexicalHypothesisSet,
    LexicalSense,
    LexicalTargetType,
)
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of attempting to ground a lexical token into HCIR semantics."""

    token: str
    target_type: LexicalTargetType
    target_id: str
    target_value: Any = None
    confidence: float = 0.0
    status: LexicalCandidateStatus = LexicalCandidateStatus.HYPOTHESIS
    is_grounded: bool = False
    is_tentative: bool = False
    runner_up_id: str | None = None
    runner_up_type: LexicalTargetType | None = None
    margin: float = 0.0


@dataclass
class RealizationResult:
    """Result of realizing an HCIR semantic structure into surface language."""

    target_id: str
    token: str
    language: str
    confidence: float = 0.0
    is_produced: bool = False


class GroundedLexicon:
    """Central repository of learned lexical mappings in HCIR."""

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph
        # token:language -> LexicalHypothesisSet
        self._hypothesis_sets: dict[str, LexicalHypothesisSet] = {}
        # sense_id -> LexicalSense
        self._committed_senses: dict[str, LexicalSense] = {}
        # (language, target_type, target_id) -> list[sense_id] for realization
        self._target_to_senses: dict[tuple[str, str, str], list[str]] = {}

    def get_or_create_hypothesis_set(
        self, token: str, language: str = "en"
    ) -> LexicalHypothesisSet:
        """Get or initialize the competing hypothesis set for a token."""
        key = f"{token.lower()}:{language}"
        if key not in self._hypothesis_sets:
            self._hypothesis_sets[key] = LexicalHypothesisSet(
                token=token.lower(), language=language
            )
        return self._hypothesis_sets[key]

    def all_hypothesis_sets(self) -> list[LexicalHypothesisSet]:
        return list(self._hypothesis_sets.values())

    def all_committed_senses(self) -> list[LexicalSense]:
        return list(self._committed_senses.values())

    # ── Lexical Commitment ────────────────────────────────────────────

    def commit_sense(
        self,
        token: str,
        target_type: LexicalTargetType,
        target_id: str,
        language: str = "en",
        target_value: Any = None,
        supporting_evidence_ids: list[str] | None = None,
        contradicting_evidence_ids: list[str] | None = None,
        contrast_ids: list[str] | None = None,
        comprehension_confidence: float = 0.85,
        generation_confidence: float = 0.80,
        status: LexicalCandidateStatus = LexicalCandidateStatus.GROUNDED,
        timestamp: float = 0.0,
    ) -> LexicalSense:
        """Commit an authoritative LexicalSense to HCIR with full provenance."""
        sense = LexicalSense(
            token=token.lower(),
            language=language,
            target_type=target_type,
            target_id=target_id,
            target_value=target_value,
            supporting_evidence_ids=supporting_evidence_ids or [],
            contradicting_evidence_ids=contradicting_evidence_ids or [],
            contrast_ids=contrast_ids or [],
            comprehension_confidence=comprehension_confidence,
            generation_confidence=generation_confidence,
            status=status,
            first_observed_event=timestamp,
            last_updated_event=timestamp,
        )
        self._committed_senses[sense.id] = sense

        rel_key = (language, target_type.value, target_id)
        if rel_key not in self._target_to_senses:
            self._target_to_senses[rel_key] = []
        if sense.id not in self._target_to_senses[rel_key]:
            self._target_to_senses[rel_key].append(sense.id)

        logger.debug(
            "GroundedLexicon: Committed sense %s ('%s' -> %s:%s)",
            sense.id,
            token,
            target_type.value,
            target_id,
        )
        return sense

    # ── Lexical Grounding (Comprehension: Token -> Meaning) ───────────

    def ground_token(
        self,
        token: str,
        language: str = "en",
        context_sense: str | None = None,
    ) -> GroundingResult:
        """Resolve a surface language token into its grounded HCIR semantic target."""
        norm_token = token.lower()

        # 1. First check committed senses
        matching_senses = [
            s
            for s in self._committed_senses.values()
            if s.token == norm_token
            and s.language == language
            and s.status == LexicalCandidateStatus.GROUNDED
        ]

        if matching_senses:
            # If polysemous, filter by context_sense if provided
            chosen = matching_senses[0]
            if context_sense and len(matching_senses) > 1:
                for s in matching_senses:
                    if s.target_id == context_sense:
                        chosen = s
                        break

            return GroundingResult(
                token=norm_token,
                target_type=chosen.target_type,
                target_id=chosen.target_id,
                target_value=chosen.target_value,
                confidence=chosen.comprehension_confidence,
                status=LexicalCandidateStatus.GROUNDED,
                is_grounded=True,
                is_tentative=False,
            )

        # 2. Check active hypothesis sets
        key = f"{norm_token}:{language}"
        hyp_set = self._hypothesis_sets.get(key)
        if not hyp_set or not hyp_set.candidates:
            return GroundingResult(
                token=norm_token,
                target_type=LexicalTargetType.CONCEPT,
                target_id="",
                confidence=0.0,
                status=LexicalCandidateStatus.UNKNOWN,
                is_grounded=False,
                is_tentative=False,
            )

        winner = hyp_set.winner
        runner_up = hyp_set.runner_up

        if winner and winner.status == LexicalCandidateStatus.GROUNDED:
            return GroundingResult(
                token=norm_token,
                target_type=winner.target_type,
                target_id=winner.target_id,
                target_value=winner.target_value,
                confidence=winner.confidence,
                status=LexicalCandidateStatus.GROUNDED,
                is_grounded=True,
                is_tentative=False,
                runner_up_id=runner_up.target_id if runner_up else None,
                runner_up_type=runner_up.target_type if runner_up else None,
                margin=hyp_set.margin_of_victory,
            )
        elif winner:
            return GroundingResult(
                token=norm_token,
                target_type=winner.target_type,
                target_id=winner.target_id,
                target_value=winner.target_value,
                confidence=winner.confidence,
                status=winner.status,
                is_grounded=False,
                is_tentative=True,
                runner_up_id=runner_up.target_id if runner_up else None,
                runner_up_type=runner_up.target_type if runner_up else None,
                margin=hyp_set.margin_of_victory,
            )

        return GroundingResult(
            token=norm_token,
            target_type=LexicalTargetType.CONCEPT,
            target_id="",
            confidence=0.0,
            status=LexicalCandidateStatus.UNKNOWN,
            is_grounded=False,
            is_tentative=False,
        )

    # ── Lexical Realization (Generation: Meaning -> Token) ────────────

    def realize_target(
        self,
        target_id: str,
        target_type: LexicalTargetType = LexicalTargetType.CONCEPT,
        language: str = "en",
        min_confidence: float = 0.50,
    ) -> RealizationResult:
        """Produce the surface language token corresponding to an HCIR semantic target."""
        rel_key = (language, target_type.value, target_id)
        sense_ids = self._target_to_senses.get(rel_key, [])

        # Check committed senses first
        for sid in sense_ids:
            sense = self._committed_senses.get(sid)
            if sense and sense.generation_confidence >= min_confidence:
                return RealizationResult(
                    target_id=target_id,
                    token=sense.token,
                    language=language,
                    confidence=sense.generation_confidence,
                    is_produced=True,
                )

        # Check candidate hypothesis sets for high-confidence winners
        for hyp_set in self._hypothesis_sets.values():
            if hyp_set.language != language:
                continue
            winner = hyp_set.winner
            if winner and winner.target_type == target_type and winner.target_id == target_id:
                if winner.confidence >= min_confidence and winner.status in (
                    LexicalCandidateStatus.GROUNDED,
                    LexicalCandidateStatus.TENTATIVE,
                ):
                    return RealizationResult(
                        target_id=target_id,
                        token=hyp_set.token,
                        language=language,
                        confidence=winner.confidence,
                        is_produced=True,
                    )

        return RealizationResult(
            target_id=target_id,
            token="",
            language=language,
            confidence=0.0,
            is_produced=False,
        )

    # ── Compositional Grounding ───────────────────────────────────────

    def ground_compositional_phrase(
        self,
        tokens: list[str],
        language: str = "en",
    ) -> dict[str, Any]:
        """Compose multiple learned lexical items into a structured semantic representation.

        Example: ['red', 'dax'] -> {'modifier': 'property:color:red', 'head': 'cylinder'}
                 ['zog', 'the', 'dax'] -> {'action': 'transition:push', 'theme': 'cylinder'}
        """
        parsed_components: dict[str, Any] = {
            "modifiers": [],
            "head": None,
            "action": None,
            "relation": None,
            "unresolved": [],
        }

        for tok in tokens:
            if tok.lower() in ("the", "a", "an", "this", "that", "it"):
                continue

            grounded = self.ground_token(tok, language=language)
            if not grounded.is_grounded and not grounded.is_tentative:
                parsed_components["unresolved"].append(tok)
                continue

            if grounded.target_type == LexicalTargetType.PROPERTY:
                parsed_components["modifiers"].append(grounded.target_id)
            elif grounded.target_type == LexicalTargetType.CONCEPT:
                parsed_components["head"] = grounded.target_id
            elif grounded.target_type in (LexicalTargetType.ACTION, LexicalTargetType.EVENT):
                parsed_components["action"] = grounded.target_id
            elif grounded.target_type == LexicalTargetType.RELATION:
                parsed_components["relation"] = grounded.target_id

        return parsed_components
