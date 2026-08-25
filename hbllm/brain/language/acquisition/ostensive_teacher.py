"""Ostensive Teacher and Error Correction Interface for A17.

Handles high-weight ostensive teaching ("This is a cup") and negative corrections
("No, that is not an apple. That is a ball"), updating candidate hypothesis sets.
"""

from __future__ import annotations

from hbllm.brain.language.acquisition.lexical_hypothesis import (
    EvidenceSourceType,
    LexicalEvidence,
    LexicalHypothesisSet,
    LexicalTargetType,
)
from hbllm.brain.language.acquisition.scoring import apply_evidence_to_candidate
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode


class OstensiveTeacher:
    """Processes explicit teacher demonstrations and corrections."""

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph

    def teach_positive(
        self,
        token: str,
        hypothesis_set: LexicalHypothesisSet,
        entity_id: str,
        language: str = "en",
        speaker: str = "teacher",
        timestamp: float = 0.0,
    ) -> LexicalEvidence:
        """Process explicit ostensive teaching: 'This is a <token>'."""
        entity = self._graph.get_node(entity_id)
        cat_name = entity.entity_type if isinstance(entity, PhysicalEntityNode) else "object"

        candidate = hypothesis_set.add_or_get_candidate(
            target_type=LexicalTargetType.CONCEPT,
            target_id=cat_name,
            timestamp=timestamp,
        )

        evidence = LexicalEvidence(
            source_type=EvidenceSourceType.OSTENSIVE_POSITIVE,
            token=token,
            language=language,
            target_type=LexicalTargetType.CONCEPT,
            target_value=cat_name,
            is_positive=True,
            context_entities=[entity_id],
            speaker=speaker,
            timestamp=timestamp,
        )
        apply_evidence_to_candidate(candidate, evidence)
        return evidence

    def teach_negative_correction(
        self,
        token: str,
        incorrect_target: str,
        hypothesis_set: LexicalHypothesisSet,
        correct_token: str | None = None,
        correct_target: str | None = None,
        language: str = "en",
        speaker: str = "teacher",
        timestamp: float = 0.0,
    ) -> list[LexicalEvidence]:
        """Process teacher correction: 'No, that is not an <incorrect_target>'."""
        evidence_list: list[LexicalEvidence] = []

        # 1. Apply negative contradiction to incorrect candidate
        c_incorrect = hypothesis_set.get_candidate(LexicalTargetType.CONCEPT, incorrect_target)
        if c_incorrect is not None:
            ev_neg = LexicalEvidence(
                source_type=EvidenceSourceType.OSTENSIVE_NEGATIVE,
                token=token,
                language=language,
                target_type=LexicalTargetType.CONCEPT,
                target_value=incorrect_target,
                is_positive=False,
                speaker=speaker,
                timestamp=timestamp,
            )
            apply_evidence_to_candidate(c_incorrect, ev_neg)
            evidence_list.append(ev_neg)

        # 2. If correct target provided, strengthen correct alternative
        if correct_target:
            c_correct = hypothesis_set.add_or_get_candidate(
                target_type=LexicalTargetType.CONCEPT,
                target_id=correct_target,
                timestamp=timestamp,
            )
            ev_pos = LexicalEvidence(
                source_type=EvidenceSourceType.OSTENSIVE_POSITIVE,
                token=token,
                language=language,
                target_type=LexicalTargetType.CONCEPT,
                target_value=correct_target,
                is_positive=True,
                speaker=speaker,
                timestamp=timestamp,
            )
            apply_evidence_to_candidate(c_correct, ev_pos)
            evidence_list.append(ev_pos)

        return evidence_list
