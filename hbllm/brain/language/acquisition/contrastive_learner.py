"""Contrastive Learner and Lexical Contrast Graph for A17.

Identifies distinguishing feature dimensions between competing concepts
and maintains the Lexical Contrast Graph (cup != bowl, apple != ball).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.language.acquisition.lexical_hypothesis import (
    EvidenceSourceType,
    LexicalEvidence,
    LexicalHypothesisSet,
    LexicalTargetType,
)
from hbllm.brain.language.acquisition.scoring import apply_evidence_to_candidate


@dataclass
class ContrastiveRelation:
    """A directed contrast relation between two concepts or lexical items."""

    source_token: str
    target_token: str
    relation_type: str  # "DIFFERENT_FROM" | "SIMILAR_TO"
    distinguishing_features: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    confidence: float = 0.8
    timestamp: float = 0.0


class ContrastiveLearner:
    """Discovers boundaries between similar concepts and builds Lexical Contrast Graphs."""

    def __init__(self) -> None:
        self._contrast_graph: list[ContrastiveRelation] = []

    @property
    def contrast_relations(self) -> list[ContrastiveRelation]:
        return list(self._contrast_graph)

    def learn_contrast(
        self,
        token_a: str,
        token_b: str,
        proto_a: dict[str, Any],
        proto_b: dict[str, Any],
        hyp_set_a: LexicalHypothesisSet | None = None,
        hyp_set_b: LexicalHypothesisSet | None = None,
        timestamp: float = 0.0,
    ) -> ContrastiveRelation:
        """Extract the minimal distinguishing feature delta between two concepts and record contrast."""
        distinguishing: dict[str, tuple[Any, Any]] = {}

        # Scan all keys in proto_a and proto_b
        all_keys = set(proto_a.keys()) | set(proto_b.keys())
        for k in all_keys:
            val_a = proto_a.get(k)
            val_b = proto_b.get(k)
            if val_a != val_b:
                distinguishing[k] = (val_a, val_b)

        relation = ContrastiveRelation(
            source_token=token_a,
            target_token=token_b,
            relation_type="DIFFERENT_FROM",
            distinguishing_features=distinguishing,
            confidence=0.85,
            timestamp=timestamp,
        )
        self._contrast_graph.append(relation)

        # Strengthen contrastive support on candidate hypotheses if available
        if hyp_set_a:
            for c in hyp_set_a.candidates:
                if c.target_type == LexicalTargetType.CONCEPT:
                    ev = LexicalEvidence(
                        source_type=EvidenceSourceType.CONTRASTIVE,
                        token=token_a,
                        target_type=LexicalTargetType.CONCEPT,
                        target_value=distinguishing,
                        is_positive=True,
                        timestamp=timestamp,
                    )
                    apply_evidence_to_candidate(c, ev)

        return relation

    def get_contrasts_for_token(self, token: str) -> list[ContrastiveRelation]:
        """Find all recorded contrasts for a given lexical token."""
        return [
            r for r in self._contrast_graph if r.source_token == token or r.target_token == token
        ]
