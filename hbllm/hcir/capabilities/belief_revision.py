"""Belief Conflict Detection & Confidence-Weighted Revision for HCIR Memory.

Checks newly proposed BeliefNodes against existing persistent beliefs using
domain-agnostic structural contradiction detection (zero LLM dependency).
All revisions (CONTRADICTS edges, confidence adjustments, supersession metadata)
are returned as pure TransactionOperations so they commit atomically through
TransactionManager and ConstitutionalVerifier (State Ownership Invariant).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import BeliefNode, CognitiveGraph, HCIREdge, HCIREdgeType, HCIRNodeType
from hbllm.hcir.reasoning.contradiction_utils import detect_structural_contradiction
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

logger = logging.getLogger(__name__)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "not",
        "no",
        "never",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "that",
        "this",
        "it",
        "and",
        "or",
        "but",
        "as",
        "is_a",
        "causes",
        "requires",
    }
)


def _content_tokens(text: str) -> set[str]:
    """Extract normalized non-stopword content tokens for fast candidate filtering."""
    raw = re.findall(r"[a-z0-9_]+", text.lower())
    return {tok for tok in raw if len(tok) > 2 and tok not in _STOP_WORDS}


@dataclass(frozen=True)
class BeliefConflict:
    """Record of a detected contradiction between a new belief and an existing belief."""

    existing_belief_id: str
    existing_statement: str
    new_belief_id: str
    new_statement: str
    explanation: str
    conflict_confidence: float
    resolution: str  # "superseded_existing" | "contested" | "rejected_lower_confidence"


@dataclass
class BeliefRevisionReport:
    """Result of checking a proposed belief against existing persistent beliefs."""

    conflicts: list[BeliefConflict] = field(default_factory=list)
    operations: list[TransactionOperation] = field(default_factory=list)
    adjusted_confidence: float = 0.80
    contested: bool = False

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


class BeliefConflictResolver:
    """Detects contradictions and computes transactional belief-revision operations.

    Invariants:
      1. Pure calculation: never mutates the CognitiveGraph directly.
      2. Zero LLM dependency: relies strictly on structural contradiction detection.
      3. Preserves provenance: superseded beliefs are marked and down-weighted rather
         than silently deleted.
    """

    def __init__(self, supersede_margin: float = 0.10) -> None:
        self._supersede_margin = supersede_margin

    def evaluate_belief(
        self,
        new_belief: BeliefNode,
        graph: CognitiveGraph | None,
    ) -> BeliefRevisionReport:
        """Compare a proposed BeliefNode against existing BeliefNodes in ``graph``.

        Returns a ``BeliefRevisionReport`` containing:
          - ``conflicts``: list of detected ``BeliefConflict`` records
          - ``operations``: transactional ``ADD_EDGE`` (``CONTRADICTS``) and
            ``UPSERT_NODE`` (revised existing belief) operations
          - ``adjusted_confidence``: calibrated confidence for ``new_belief``
          - ``contested``: whether the new belief is actively contested
        """
        new_conf = float(new_belief.confidence if new_belief.confidence is not None else 0.80)
        report = BeliefRevisionReport(adjusted_confidence=new_conf)
        if graph is None or not new_belief.statement:
            return report

        new_tokens = _content_tokens(new_belief.statement)
        if not new_tokens:
            return report

        existing_edges = getattr(graph, "edges", {})

        for node in graph.nodes_by_type(HCIRNodeType.BELIEF):
            if not isinstance(node, BeliefNode):
                continue
            if node.id == new_belief.id or not node.statement:
                continue

            # Fast token-overlap pre-filter to avoid O(N) full checks on unrelated beliefs
            existing_tokens = _content_tokens(node.statement)
            if not (new_tokens & existing_tokens):
                continue

            is_conflict, explanation, conflict_conf = detect_structural_contradiction(
                node.statement,
                new_belief.statement,
            )
            if not is_conflict:
                continue

            existing_conf = float(node.confidence if node.confidence is not None else 0.75)
            if new_conf > existing_conf + self._supersede_margin:
                resolution = "superseded_existing"
                revised_existing_conf = max(0.05, round(existing_conf * 0.5, 4))
                updated_props: dict[str, Any] = dict(node.properties)
                updated_props["superseded_by"] = new_belief.id
                updated_props["contested"] = True
                revised_existing = node.model_copy(
                    update={
                        "confidence": revised_existing_conf,
                        "properties": updated_props,
                    }
                )
                report.operations.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=revised_existing.id,
                        node_data=revised_existing.model_dump(),
                    )
                )
            else:
                resolution = "contested"
                report.contested = True
                report.adjusted_confidence = max(
                    0.10, round(min(report.adjusted_confidence, new_conf * 0.70), 4)
                )
                updated_props = dict(node.properties)
                updated_props["contested"] = True
                updated_props["contested_by"] = new_belief.id
                revised_existing = node.model_copy(
                    update={
                        "confidence": max(0.10, round(existing_conf * 0.85, 4)),
                        "properties": updated_props,
                    }
                )
                report.operations.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=revised_existing.id,
                        node_data=revised_existing.model_dump(),
                    )
                )

            edge_id = f"contradicts_{new_belief.id}__{node.id}"
            if edge_id not in existing_edges:
                contra_edge = HCIREdge(
                    id=edge_id,
                    edge_type=HCIREdgeType.CONTRADICTS,
                    sources=[new_belief.id],
                    targets=[node.id],
                    weight=round(conflict_conf, 4),
                    properties={
                        "explanation": explanation,
                        "resolution": resolution,
                    },
                )
                report.operations.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_EDGE,
                        edge_data=contra_edge.model_dump(),
                    )
                )

            report.conflicts.append(
                BeliefConflict(
                    existing_belief_id=node.id,
                    existing_statement=node.statement,
                    new_belief_id=new_belief.id,
                    new_statement=new_belief.statement,
                    explanation=explanation,
                    conflict_confidence=conflict_conf,
                    resolution=resolution,
                )
            )
            logger.info(
                "BeliefConflictResolver: %s between '%s' and '%s' (%s)",
                resolution,
                new_belief.statement[:60],
                node.statement[:60],
                explanation,
            )

        return report
