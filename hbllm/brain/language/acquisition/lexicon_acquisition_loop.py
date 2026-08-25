"""Unified Lexicon Acquisition Loop for A17.

Orchestrates cross-situational observation, ostensive teaching, contrastive learning,
A14 error feedback, and A15 predictive evaluation to dynamically expand the GroundedLexicon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.brain.concepts.grounded_concept_registry import GroundedConceptRegistry
from hbllm.brain.language.acquisition.contrastive_learner import ContrastiveLearner
from hbllm.brain.language.acquisition.cross_situational_learner import CrossSituationalLearner
from hbllm.brain.language.acquisition.grounded_lexicon import (
    GroundedLexicon,
    GroundingResult,
    RealizationResult,
)
from hbllm.brain.language.acquisition.lexical_hypothesis import (
    LexicalCandidateStatus,
    LexicalEvidence,
    LexicalTargetType,
)
from hbllm.brain.language.acquisition.ostensive_teacher import OstensiveTeacher
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


@dataclass
class AcquisitionCycleResult:
    """Summary metrics of a lexical acquisition cycle."""

    evidences_processed: int = 0
    hypotheses_updated: int = 0
    senses_committed: int = 0
    senses_contradicted: int = 0


class LexiconAcquisitionLoop:
    """Top-level A17 learning loop connecting language acquisition to HCIR."""

    def __init__(
        self,
        graph: CognitiveGraph,
        concept_registry: GroundedConceptRegistry | None = None,
    ) -> None:
        self._graph = graph
        self._concept_registry = concept_registry or GroundedConceptRegistry(graph)
        self._lexicon = GroundedLexicon(graph)
        self._cross_situational = CrossSituationalLearner(graph, self._concept_registry)
        self._contrastive = ContrastiveLearner()
        self._ostensive = OstensiveTeacher(graph)
        self._event_log: list[LexicalEvidence] = []

    @property
    def lexicon(self) -> GroundedLexicon:
        return self._lexicon

    @property
    def cross_situational_learner(self) -> CrossSituationalLearner:
        return self._cross_situational

    @property
    def contrastive_learner(self) -> ContrastiveLearner:
        return self._contrastive

    @property
    def ostensive_teacher(self) -> OstensiveTeacher:
        return self._ostensive

    @property
    def event_history(self) -> list[LexicalEvidence]:
        return list(self._event_log)

    # ── High-Level Learning APIs ──────────────────────────────────────

    def observe_scene(
        self,
        utterance_tokens: list[str],
        visible_entity_ids: list[str],
        spatial_edges: list[tuple[str, str, str]] | None = None,
        state_delta: dict[str, Any] | None = None,
        language: str = "en",
        timestamp: float = 0.0,
    ) -> list[LexicalEvidence]:
        """Observe tokens spoken in a situated scene."""
        all_evidence: list[LexicalEvidence] = []

        for token in utterance_tokens:
            if token.lower() in ("the", "a", "an", "this", "that", "it", "is", "look", "at"):
                continue

            hyp_set = self._lexicon.get_or_create_hypothesis_set(token, language=language)
            evidences = self._cross_situational.observe_situated_token(
                token=token,
                hypothesis_set=hyp_set,
                visible_entity_ids=visible_entity_ids,
                spatial_edges=spatial_edges,
                state_delta=state_delta,
                language=language,
                timestamp=timestamp,
            )
            all_evidence.extend(evidences)
            self._event_log.extend(evidences)

            # Check if winner has reached grounded threshold
            winner = hyp_set.winner
            if winner and winner.status == LexicalCandidateStatus.GROUNDED:
                # Commit sense if margin of victory is sufficient
                if hyp_set.margin_of_victory >= 0.15:
                    self._lexicon.commit_sense(
                        token=token,
                        target_type=winner.target_type,
                        target_id=winner.target_id,
                        language=language,
                        target_value=winner.target_value,
                        supporting_evidence_ids=winner.evidence_ids,
                        contradicting_evidence_ids=winner.contradiction_ids,
                        comprehension_confidence=winner.confidence,
                        generation_confidence=max(0.0, winner.confidence - 0.05),
                        status=LexicalCandidateStatus.GROUNDED,
                        timestamp=timestamp,
                    )

        return all_evidence

    def teach_ostensive(
        self,
        token: str,
        entity_id: str,
        language: str = "en",
        timestamp: float = 0.0,
    ) -> LexicalEvidence:
        """Explicit teacher presentation: 'This is a <token>'."""
        hyp_set = self._lexicon.get_or_create_hypothesis_set(token, language=language)
        ev = self._ostensive.teach_positive(
            token=token,
            hypothesis_set=hyp_set,
            entity_id=entity_id,
            language=language,
            timestamp=timestamp,
        )
        self._event_log.append(ev)

        winner = hyp_set.winner
        if winner and winner.status == LexicalCandidateStatus.GROUNDED:
            self._lexicon.commit_sense(
                token=token,
                target_type=winner.target_type,
                target_id=winner.target_id,
                language=language,
                target_value=winner.target_value,
                supporting_evidence_ids=winner.evidence_ids,
                comprehension_confidence=winner.confidence,
                generation_confidence=max(0.0, winner.confidence - 0.05),
                status=LexicalCandidateStatus.GROUNDED,
                timestamp=timestamp,
            )

        return ev

    def correct_mistake(
        self,
        token: str,
        incorrect_target: str,
        correct_token: str | None = None,
        correct_target: str | None = None,
        language: str = "en",
        timestamp: float = 0.0,
    ) -> list[LexicalEvidence]:
        """Explicit negative teacher feedback: 'No, that is not an <incorrect_target>'."""
        hyp_set = self._lexicon.get_or_create_hypothesis_set(token, language=language)
        ev_list = self._ostensive.teach_negative_correction(
            token=token,
            incorrect_target=incorrect_target,
            hypothesis_set=hyp_set,
            correct_token=correct_token,
            correct_target=correct_target,
            language=language,
            timestamp=timestamp,
        )
        self._event_log.extend(ev_list)
        return ev_list

    # ── Inference / Query APIs ────────────────────────────────────────

    def understand(self, token: str, language: str = "en") -> GroundingResult:
        """Ground token to meaning (comprehension)."""
        return self._lexicon.ground_token(token, language=language)

    def produce(
        self,
        target_id: str,
        target_type: LexicalTargetType = LexicalTargetType.CONCEPT,
        language: str = "en",
    ) -> RealizationResult:
        """Realize meaning to language token (generation)."""
        return self._lexicon.realize_target(target_id, target_type=target_type, language=language)
