"""
Induction Operator — pattern → generalization over HCIR observations.

Discovers statistical regularities from observed facts, beliefs, and
evidence in the frozen HCIR view, and proposes confidence-weighted
generalizations as new beliefs.

Example::

    Observations:
        entity_12 is spherical AND entity_12 rolls
        entity_34 is spherical AND entity_34 rolls
        entity_56 is spherical AND entity_56 rolls

    Induction:
        "spherical entities tend to roll" (confidence=0.85, n=3)

Independence Level: L1 (no LLM execution)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.hcir.graph import (
    BeliefNode,
    FactNode,
    HCIRNodeType,
    ObservationNode,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


class InductionOperator:
    """Pattern → generalization over HCIR observations.

    Scans the frozen view for co-occurring properties across entities
    and proposes generalized beliefs when patterns exceed a confidence
    threshold.

    Methodology:
        1. Extract property assertions: (subject, predicate, value)
        2. Group by predicate-value pairs
        3. Find frequently co-occurring property pairs
        4. Propose generalizations for pairs above threshold
    """

    @property
    def operator_id(self) -> str:
        return "induction"

    @property
    def operator_name(self) -> str:
        return "Statistical Induction Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        """Score applicability for induction."""
        type_scores: dict[ProblemType, float] = {
            ProblemType.GENERALIZATION: 0.9,
            ProblemType.CLASSIFICATION: 0.6,
            ProblemType.EXPLANATION: 0.4,
            ProblemType.PREDICTION: 0.5,
            ProblemType.CAUSAL: 0.3,
        }
        base = type_scores.get(problem.problem_type, 0.15)

        view = context.graph_view
        n_observations = len(view.nodes_by_type(HCIRNodeType.OBSERVATION))
        n_beliefs = len(view.nodes_by_type(HCIRNodeType.BELIEF))
        n_facts = len(view.nodes_by_type(HCIRNodeType.FACT))

        total = n_observations + n_beliefs + n_facts
        if total < 3:
            return 0.0  # Need enough data for induction

        # More data → better induction
        data_boost = min(0.3, total * 0.01)
        return min(1.0, base + data_boost)

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        """Induction is moderately cheap — statistical analysis."""
        n = context.graph_view.node_count
        return ResourceCost(
            wall_clock_ms=max(1.0, n * 0.5),
            nodes_read=n,
            edges_read=context.graph_view.edge_count,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Run inductive generalization over HCIR view.

        1. Extract property triples (subject, predicate, value).
        2. Find co-occurrence patterns.
        3. Propose generalizations exceeding confidence threshold.
        """
        start = time.time()
        view = context.graph_view

        # ── Extract property assertions ──────────────────────────────
        entity_properties = self._extract_entity_properties(view)

        if len(entity_properties) < 2:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "Insufficient entities for induction"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Find co-occurrence patterns ──────────────────────────────
        generalizations = self._find_generalizations(
            entity_properties,
            min_support=max(2, len(entity_properties) // 5),
            min_confidence=1.0 - context.budget.uncertainty_tolerance,
        )

        if not generalizations:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"patterns_found": 0},
                metadata={"reason": "No patterns above threshold"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Build result ─────────────────────────────────────────────
        proposed_ops: list[TransactionOperation] = []
        provenance_chains: list[ProvenanceChain] = []
        evidence_refs: list[str] = []

        for gen in generalizations:
            claim = gen["claim"]
            confidence = gen["confidence"]
            support = gen["support"]
            sources = gen["source_ids"]

            new_belief = BeliefNode(
                claim=claim,
                belief_type="factual",
                evidence_sources=sources,
                provenance=Provenance(
                    created_by=self.operator_id,
                    source_type="inferred",
                    reason=(f"Induced from {support} observations: co-occurrence of properties"),
                ),
            )
            new_belief.uncertainty.confidence = confidence

            proposed_ops.append(
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=new_belief.model_dump(),
                )
            )

            provenance_chains.append(
                ProvenanceChain(
                    conclusion=claim,
                    evidence_node_ids=sources,
                    operator_id=self.operator_id,
                    reasoning_steps=[
                        f"Observed in {support}/{gen['total']} entities",
                        f"Confidence: {confidence:.3f}",
                    ],
                    assumptions=[
                        "IID assumption: instances are independently observed",
                        "Closed-world: no contradicting evidence in view",
                    ],
                    confidence=confidence,
                )
            )
            evidence_refs.extend(sources)

        evidence_refs = list(dict.fromkeys(evidence_refs))
        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "patterns_found": len(generalizations),
                "generalizations": [g["claim"] for g in generalizations],
            },
            confidence=min(g["confidence"] for g in generalizations),
            evidence_refs=evidence_refs,
            assumptions=[
                "IID assumption on observations",
                "Closed-world on available evidence",
            ],
            proposed_transitions=proposed_ops,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Internal Methods ─────────────────────────────────────────────

    @staticmethod
    def _extract_entity_properties(
        view: Any,  # FrozenGraphView
    ) -> dict[str, set[str]]:
        """Extract per-entity property sets from the frozen view.

        Scans beliefs, facts, and observations for property assertions.
        Returns: {entity_id_or_subject: {property1, property2, ...}}
        """
        entity_props: dict[str, set[str]] = defaultdict(set)

        # From beliefs: "entity_X is Y" or "entity_X has Y"
        for node in view.nodes_by_type(HCIRNodeType.BELIEF):
            if not isinstance(node, BeliefNode):
                continue
            claim = node.claim.strip().lower()
            # Simple patterns: "X is Y", "X has Y", "X can Y"
            for pattern in [" is ", " has ", " can ", " are "]:
                if pattern in claim:
                    parts = claim.split(pattern, 1)
                    subject = parts[0].strip()
                    prop = parts[1].strip()
                    if subject and prop:
                        entity_props[subject].add(prop)
                    break

        # From facts
        for node in view.nodes_by_type(HCIRNodeType.FACT):
            if not isinstance(node, FactNode):
                continue
            claim = node.claim.strip().lower()
            for pattern in [" is ", " has ", " can ", " are "]:
                if pattern in claim:
                    parts = claim.split(pattern, 1)
                    subject = parts[0].strip()
                    prop = parts[1].strip()
                    if subject and prop:
                        entity_props[subject].add(prop)
                    break

        # From observations with payload
        for node in view.nodes_by_type(HCIRNodeType.OBSERVATION):
            if not isinstance(node, ObservationNode):
                continue
            if node.payload:
                subject = node.payload.get("subject", node.id)
                for key, value in node.payload.items():
                    if key != "subject" and isinstance(value, (str, bool, int, float)):
                        entity_props[str(subject)].add(f"{key}:{value}")

        return dict(entity_props)

    @staticmethod
    def _find_generalizations(
        entity_properties: dict[str, set[str]],
        min_support: int = 2,
        min_confidence: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find property co-occurrence patterns.

        For each pair of properties (A, B), compute:
            support = # entities having both A and B
            confidence = support / # entities having A

        Return generalizations: "entities with A tend to have B"
        where support >= min_support and confidence >= min_confidence.
        """
        # Collect all properties and their entity sets
        prop_entities: dict[str, set[str]] = defaultdict(set)
        for entity, props in entity_properties.items():
            for prop in props:
                prop_entities[prop].add(entity)

        generalizations: list[dict[str, Any]] = []
        properties = list(prop_entities.keys())

        for i, prop_a in enumerate(properties):
            entities_a = prop_entities[prop_a]
            if len(entities_a) < min_support:
                continue

            for prop_b in properties[i + 1 :]:
                entities_b = prop_entities[prop_b]
                if len(entities_b) < min_support:
                    continue

                # Co-occurrence
                both = entities_a & entities_b
                support = len(both)

                if support < min_support:
                    continue

                # Confidence A → B
                conf_ab = support / len(entities_a)
                # Confidence B → A
                conf_ba = support / len(entities_b)

                # Use the higher confidence direction
                if conf_ab >= conf_ba and conf_ab >= min_confidence:
                    generalizations.append(
                        {
                            "claim": (f"entities with '{prop_a}' tend to have '{prop_b}'"),
                            "confidence": conf_ab,
                            "support": support,
                            "total": len(entities_a),
                            "source_ids": list(both),
                        }
                    )
                elif conf_ba >= min_confidence:
                    generalizations.append(
                        {
                            "claim": (f"entities with '{prop_b}' tend to have '{prop_a}'"),
                            "confidence": conf_ba,
                            "support": support,
                            "total": len(entities_b),
                            "source_ids": list(both),
                        }
                    )

        # Sort by confidence × support
        generalizations.sort(
            key=lambda g: g["confidence"] * g["support"],
            reverse=True,
        )

        return generalizations
