"""Concept Formation Loop — orchestrates the A15 sleep/consolidation cycle.

Periodic consolidation cycle::

    WAKE / ONLINE
    ─────────────
    observe, update entities, collect features,
    record behavior, record prediction errors

                 ↓

    CONSOLIDATION / SLEEP
    ─────────────────────
    1. FeatureAccumulator.accumulate_all()
    2. ConceptHypothesisGenerator.generate(features)
    3. ConceptConsolidator.consolidate(hypotheses)
       → with predictive utility test (decisive)
    4. GroundedConceptRegistry.register(confirmed concepts)
    5. ConceptPredictionBridge.generate_predictions(new concepts)
    6. detect_heterogeneity(existing concepts)
    7. detect_degradation(existing concepts)

                 ↓

    ONLINE
    ──────
    new concepts become predictive abstractions → A14

This prevents concept formation from constantly restructuring
the graph while perception is happening.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hbllm.brain.concepts.concept_consolidator import (
    ConceptConsolidator,
    ConceptDegradationSignal,
    ConceptRefinementSignal,
    ConsolidationDecision,
    PredictiveUtilityTest,
)
from hbllm.brain.concepts.concept_hypothesis_generator import (
    ConceptHypothesisGenerator,
)
from hbllm.brain.concepts.concept_prediction_bridge import (
    ConceptPredictionBridge,
)
from hbllm.brain.concepts.feature_accumulator import (
    FeatureAccumulator,
)
from hbllm.brain.concepts.grounded_concept_registry import GroundedConceptRegistry
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Consolidation Cycle Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ConsolidationCycleResult:
    """Result of a single consolidation cycle."""

    entities_processed: int = 0
    hypotheses_generated: int = 0
    concepts_formed: int = 0
    concepts_rejected: int = 0
    concepts_deferred: int = 0
    concepts_merged: int = 0
    refinement_signals: list[ConceptRefinementSignal] = field(default_factory=list)
    degradation_signals: list[ConceptDegradationSignal] = field(default_factory=list)
    predictions_generated: int = 0
    formed_concept_ids: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Concept Formation Loop
# ═══════════════════════════════════════════════════════════════════════════


class ConceptFormationLoop:
    """Orchestrates the A15 concept formation cycle.

    Runs during consolidation/sleep phases to discover and
    validate grounded concepts from accumulated experience.

    Usage::

        loop = ConceptFormationLoop(graph)

        # Run during consolidation
        result = loop.run_consolidation(
            utility_tests={"hyp_abc": PredictiveUtilityTest(...)},
        )
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        similarity_threshold: float = 0.6,
        min_utility_gain: float = 0.05,
        min_exemplars: int = 2,
    ) -> None:
        self._graph = graph
        self._accumulator = FeatureAccumulator(graph)
        self._generator = ConceptHypothesisGenerator(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_exemplars,
        )
        self._consolidator = ConceptConsolidator(
            min_exemplars=min_exemplars,
            min_utility_gain=min_utility_gain,
        )
        self._registry = GroundedConceptRegistry(graph)
        self._bridge = ConceptPredictionBridge(self._registry)

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def accumulator(self) -> FeatureAccumulator:
        return self._accumulator

    @property
    def generator(self) -> ConceptHypothesisGenerator:
        return self._generator

    @property
    def consolidator(self) -> ConceptConsolidator:
        return self._consolidator

    @property
    def registry(self) -> GroundedConceptRegistry:
        return self._registry

    @property
    def bridge(self) -> ConceptPredictionBridge:
        return self._bridge

    # ── Consolidation Cycle ───────────────────────────────────────────

    def run_consolidation(
        self,
        utility_tests: dict[str, PredictiveUtilityTest] | None = None,
        member_outcomes: dict[str, dict[str, list[bool]]] | None = None,
    ) -> ConsolidationCycleResult:
        """Run a full consolidation cycle.

        Steps:
        1. Accumulate features for all entities
        2. Generate concept hypotheses
        3. Consolidate each hypothesis (predictive utility test)
        4. Register confirmed concepts
        5. Generate predictions for new concepts
        6. Detect heterogeneity in existing concepts
        7. Detect degradation in existing concepts

        Args:
            utility_tests: Pre-computed utility tests per hypothesis ID.
                If not provided, hypotheses are DEFERred.
            member_outcomes: concept_id → {entity_id → [correct, ...]}
                for heterogeneity detection.

        Returns:
            ConsolidationCycleResult with full metrics.
        """
        result = ConsolidationCycleResult()

        # Step 1: Accumulate features
        features = self._accumulator.accumulate_all()
        result.entities_processed = len(features)

        if not features:
            return result

        # Step 2: Generate hypotheses
        hypotheses = self._generator.generate(features)
        result.hypotheses_generated = len(hypotheses)

        # Collect existing concept members for redundancy check
        existing_members: dict[str, set[str]] = {}
        for concept in self._registry.all_concepts():
            members = self._registry.concept_members(concept.id)
            existing_members[concept.id] = set(members)

        # Step 3: Consolidate each hypothesis
        for hypothesis in hypotheses:
            utility = utility_tests.get(hypothesis.hypothesis_id) if utility_tests else None

            if utility is None:
                # Default: defer if no utility test provided
                utility = PredictiveUtilityTest()

            consolidation = self._consolidator.consolidate(
                hypothesis=hypothesis,
                utility_test=utility,
                existing_concept_members=existing_members or None,
            )

            if consolidation.decision == ConsolidationDecision.ACCEPT:
                concept_id = self._registry.register(
                    feature_prototype=hypothesis.feature_prototype,
                    member_ids=hypothesis.member_ids,
                    behavioral_regularities=hypothesis.behavioral_regularities,
                    domain=hypothesis.domain,
                    formation_source=hypothesis.formation_source,
                    formation_score=hypothesis.coherence_scores,
                    utility_delta=consolidation.utility_delta,
                )
                result.concepts_formed += 1
                result.formed_concept_ids.append(concept_id)

            elif consolidation.decision == ConsolidationDecision.REJECT:
                result.concepts_rejected += 1
            elif consolidation.decision == ConsolidationDecision.DEFER:
                result.concepts_deferred += 1
            elif consolidation.decision == ConsolidationDecision.MERGE:
                result.concepts_merged += 1

        # Step 4: Generate predictions for new concepts
        if result.formed_concept_ids:
            specs = self._bridge.generate_predictions()
            result.predictions_generated = len(specs)

        # Step 5: Detect heterogeneity in existing concepts
        if member_outcomes:
            for concept_id, outcomes in member_outcomes.items():
                ref_signal = self._consolidator.detect_heterogeneity(
                    concept_id=concept_id,
                    member_outcomes=outcomes,
                )
                if ref_signal:
                    result.refinement_signals.append(ref_signal)

        # Step 6: Detect degradation in existing concepts
        for concept in self._registry.all_concepts():
            deg_signal = self._consolidator.detect_degradation(
                concept_id=concept.id,
                confidence=concept.confidence,
                prediction_accuracy=concept.prediction_accuracy,
            )
            if deg_signal:
                result.degradation_signals.append(deg_signal)

        return result
