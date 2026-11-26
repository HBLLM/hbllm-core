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

from hbllm.hcir.fuzzy import (
    BipolarEvidenceState,
    BipolarObservation,
    aggregate_bipolar_evidence,
)
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
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
    bipolar_evidence: dict[str, Any] | None = None


@dataclass
class BeliefRevisionReport:
    """Result of checking a proposed belief against existing persistent beliefs."""

    conflicts: list[BeliefConflict] = field(default_factory=list)
    operations: list[TransactionOperation] = field(default_factory=list)
    adjusted_confidence: float = 0.80
    contested: bool = False
    bipolar_state: BipolarEvidenceState | None = None

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

    def __init__(
        self,
        supersede_margin: float = 0.10,
        ontology_store: Any | None = None,
    ) -> None:
        self._supersede_margin = supersede_margin
        self._ontology_store = ontology_store

    @staticmethod
    def _build_bipolar_observations(
        belief: BeliefNode,
        polarity: str,
        graph: CognitiveGraph,
        fallback_conf: float,
    ) -> tuple[list[BipolarObservation], bool]:
        """Extract correlation-tagged `BipolarObservation`s from a BeliefNode's evidence provenance."""
        ev_ids = list(getattr(belief, "evidence_sources", ()) or ())
        obs_list: list[BipolarObservation] = []
        has_provenance = False
        for ev_id in ev_ids:
            ev_node = graph.get_node(str(ev_id))
            if isinstance(ev_node, EvidenceNode):
                has_provenance = True
                props = ev_node.properties if isinstance(ev_node.properties, dict) else {}
                strength = max(
                    0.0,
                    min(
                        1.0,
                        float(props.get("strength", ev_node.confidence or fallback_conf)),
                    ),
                )
                src_w = max(0.0, min(1.0, float(props.get("source_weight", 1.0))))
                dep_grp = str(props.get("dependency_group", f"ev:{ev_node.id}"))
                chan_id = str(props.get("source_channel_id", ev_node.source_uri or "default"))
                epoch = int(props.get("tracklet_epoch", 0))
                obs_list.append(
                    BipolarObservation(
                        evidence_id=ev_node.id,
                        polarity="support" if polarity == "support" else "refute",
                        strength=strength,
                        source_weight=src_w,
                        dependency_group=dep_grp,
                        source_channel_id=chan_id,
                        tracklet_epoch=epoch,
                    )
                )
        if not obs_list:
            props = belief.properties if isinstance(belief.properties, dict) else {}
            obs_list.append(
                BipolarObservation(
                    evidence_id=f"{polarity}_{belief.id}",
                    polarity="support" if polarity == "support" else "refute",
                    strength=max(0.0, min(1.0, fallback_conf)),
                    dependency_group=str(props.get("dependency_group", f"belief:{belief.id}")),
                    source_channel_id=str(props.get("source_channel_id", "belief_node")),
                    tracklet_epoch=int(props.get("tracklet_epoch", 0)),
                )
            )
        return obs_list, has_provenance

    def evaluate_belief(
        self,
        new_belief: BeliefNode,
        graph: CognitiveGraph | None,
        *,
        ontology_store: Any | None = None,
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
        all_support_obs, new_has_prov = self._build_bipolar_observations(
            new_belief, "support", graph, new_conf
        )
        all_refute_obs: list[BipolarObservation] = []
        any_prov = new_has_prov

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
            store = ontology_store or self._ontology_store
            if not is_conflict and store is not None and hasattr(store, "are_antonyms"):
                diff_new = sorted(new_tokens - existing_tokens)
                diff_old = sorted(existing_tokens - new_tokens)
                if 1 <= len(diff_new) <= 4 and 1 <= len(diff_old) <= 4:
                    for tok_new in diff_new:
                        for tok_old in diff_old:
                            if store.are_antonyms(tok_new, tok_old):
                                is_conflict = True
                                explanation = f"Antonym conflict: '{tok_new}' vs '{tok_old}'"
                                conflict_conf = 0.80
                                break
                        if is_conflict:
                            break
            if not is_conflict:
                continue

            existing_conf = float(node.confidence if node.confidence is not None else 0.75)
            refute_obs, ex_has_prov = self._build_bipolar_observations(
                node, "refute", graph, existing_conf
            )
            any_prov = any_prov or ex_has_prov
            all_refute_obs.extend(refute_obs)
            bipolar = aggregate_bipolar_evidence([*all_support_obs, *refute_obs])
            bipolar_dict = bipolar.model_dump()

            use_cluster_supersede = (new_has_prov or ex_has_prov) and (
                bipolar.independent_support_clusters > 1
                or bipolar.independent_refute_clusters > 1
                or len(all_support_obs) > 1
                or len(refute_obs) > 1
            )
            if use_cluster_supersede:
                should_supersede = bipolar.support_share > bipolar.refutation_share + (
                    self._supersede_margin * 0.5
                )
            else:
                should_supersede = new_conf > existing_conf + self._supersede_margin

            if should_supersede:
                resolution = "superseded_existing"
                revised_existing_conf = max(0.05, round(existing_conf * 0.5, 4))
                updated_props: dict[str, Any] = dict(node.properties)
                updated_props["superseded_by"] = new_belief.id
                updated_props["contested"] = True
                updated_props["bipolar_evidence"] = bipolar_dict
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
                updated_props["bipolar_evidence"] = bipolar_dict
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
                        "bipolar_evidence": bipolar_dict,
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
                    bipolar_evidence=bipolar_dict,
                )
            )
            logger.info(
                "BeliefConflictResolver: %s between '%s' and '%s' (%s)",
                resolution,
                new_belief.statement[:60],
                node.statement[:60],
                explanation,
            )

        if report.conflicts or any_prov:
            report.bipolar_state = aggregate_bipolar_evidence([*all_support_obs, *all_refute_obs])

        return report
