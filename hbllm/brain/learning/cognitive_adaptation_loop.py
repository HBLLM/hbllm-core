"""Cognitive Adaptation Loop — orchestrates the full A14 learning cycle.

The outer loop that ties all A14 components together:

    accumulate errors (fast/slow)
         ↓
    classify batch (ErrorClassifier)
         ↓
    gate decisions (AdaptationGate)
         ↓
    adapt models (AdaptationEngine)
         ↓
    register changes (PredictiveModelRegistry)
         ↓
    evaluate outcomes (PredictionEvaluationEngine)
         ↓
    record in EventChronicle

**Deterministic replay invariant:**

    same event journal → same classifier inputs → same gate decisions
    → same adaptations → same model state → same evaluation

Every model mutation is reconstructible from HCIR events.
No hidden mutation inside Python objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hbllm.brain.learning.adaptation_engine import AdaptationEngine, AdaptationRecord
from hbllm.brain.learning.adaptation_gate import (
    AdaptationGate,
    ErrorEvidenceAccumulator,
)
from hbllm.brain.learning.error_classifier import ErrorClassifier, ErrorContext
from hbllm.brain.learning.learning_signal_router import (
    LearningSignalRouter,
    RoutingAction,
    RoutingResult,
)
from hbllm.brain.learning.prediction_evaluation_engine import (
    EvaluationResult,
    PredictionEvaluationEngine,
)
from hbllm.brain.learning.predictive_model_registry import PredictiveModelRegistry
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Loop Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptationCycleResult:
    """Result of a single adaptation cycle."""

    errors_processed: int = 0
    adaptations_performed: int = 0
    rules_extracted: int = 0
    evaluations: list[EvaluationResult] = field(default_factory=list)
    routing_results: list[RoutingResult] = field(default_factory=list)
    adaptation_records: list[AdaptationRecord] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Adaptation Loop
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveAdaptationLoop:
    """Orchestrates the full A14 prediction-error-centered learning cycle.

    Combines all A14 components into a coherent feedback loop:

    1. ErrorClassifier — probabilistic error diagnosis
    2. ErrorEvidenceAccumulator — evidence accumulation
    3. AdaptationGate — sole authority over model mutation
    4. AdaptationEngine — executes authorized adaptations
    5. PredictiveModelRegistry — HCIR-native model tracking
    6. LearningSignalRouter — fast/slow routing
    7. PredictionEvaluationEngine — meta-learning

    Usage::

        loop = CognitiveAdaptationLoop(graph)

        # Fast path: process individual errors
        result = loop.process_error(
            error_id="err_001",
            model_id="model_phys_01",
            context=ErrorContext(error_magnitude=0.4, ...),
        )

        # Slow path: periodic batch review
        cycle = loop.run_batch_cycle()
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        min_evidence_count: int = 3,
        model_error_threshold: float = 0.6,
    ) -> None:
        self._graph = graph

        # Build the A14 component stack
        self._classifier = ErrorClassifier()
        self._accumulator = ErrorEvidenceAccumulator()
        self._gate = AdaptationGate(
            accumulator=self._accumulator,
            min_evidence_count=min_evidence_count,
            model_error_threshold=model_error_threshold,
        )
        self._engine = AdaptationEngine(graph)
        self._registry = PredictiveModelRegistry(graph)
        self._router = LearningSignalRouter(
            classifier=self._classifier,
            accumulator=self._accumulator,
            gate=self._gate,
        )
        self._evaluator = PredictionEvaluationEngine()

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def classifier(self) -> ErrorClassifier:
        return self._classifier

    @property
    def accumulator(self) -> ErrorEvidenceAccumulator:
        return self._accumulator

    @property
    def gate(self) -> AdaptationGate:
        return self._gate

    @property
    def engine(self) -> AdaptationEngine:
        return self._engine

    @property
    def registry(self) -> PredictiveModelRegistry:
        return self._registry

    @property
    def router(self) -> LearningSignalRouter:
        return self._router

    @property
    def evaluator(self) -> PredictionEvaluationEngine:
        return self._evaluator

    # ── Fast Path (per-error) ─────────────────────────────────────────

    def process_error(
        self,
        error_id: str,
        model_id: str,
        context: ErrorContext,
        error_magnitude: float = 0.0,
        domain: str = "",
        timestamp: float | None = None,
    ) -> RoutingResult:
        """Process a single prediction error through the full A14 pipeline.

        1. Route (classify + accumulate + gate)
        2. If ADAPT → execute adaptation
        3. Return routing result

        Args:
            error_id: ID of the PredictionErrorNode.
            model_id: ID of the PredictiveModelNode.
            context: Error context for classification.
            error_magnitude: Magnitude of the error.
            domain: Prediction domain.
            timestamp: Optional timestamp.

        Returns:
            RoutingResult indicating what action was taken.
        """
        result = self._router.route(
            error_id=error_id,
            model_id=model_id,
            context=context,
            error_magnitude=error_magnitude or context.error_magnitude,
            domain=domain or context.prediction_domain,
            timestamp=timestamp,
        )

        # If gate authorizes adaptation, execute it
        if result.action == RoutingAction.ADAPT:
            self._execute_adaptation(model_id, timestamp=timestamp)

        return result

    # ── Slow Path (batch) ─────────────────────────────────────────────

    def run_batch_cycle(
        self,
        timestamp: float | None = None,
    ) -> AdaptationCycleResult:
        """Run a periodic batch review of all accumulated evidence.

        1. Batch evaluate all tracked models
        2. Execute adaptations for models that pass the gate
        3. Evaluate whether adaptations helped

        Returns:
            AdaptationCycleResult with metrics.
        """
        cycle = AdaptationCycleResult()

        # Get all models ready for adaptation
        ready = self._router.batch_evaluate()
        cycle.routing_results = ready

        for routing_result in ready:
            if routing_result.action == RoutingAction.ADAPT:
                record = self._execute_adaptation(
                    routing_result.model_id,
                    timestamp=timestamp,
                )
                if record:
                    cycle.adaptation_records.append(record)
                    cycle.adaptations_performed += 1

        cycle.errors_processed = sum(
            ev.occurrences
            for mid in self._accumulator.tracked_models
            if (ev := self._accumulator.get_evidence(mid)) is not None
        )

        return cycle

    # ── Adaptation Execution ──────────────────────────────────────────

    def _execute_adaptation(
        self,
        model_id: str,
        timestamp: float | None = None,
    ) -> AdaptationRecord | None:
        """Execute an authorized adaptation for a model.

        1. Get model and evidence
        2. Record pre-adaptation accuracy
        3. Execute adaptation
        4. Record post-adaptation state
        5. Clear evidence
        """
        model = self._registry.get_model(model_id)
        if model is None:
            logger.warning("CognitiveAdaptationLoop: model %s not found", model_id)
            return None

        evidence = self._accumulator.get_evidence(model_id)
        if evidence is None:
            return None

        # Record pre-adaptation accuracy for evaluation
        accuracy_before = model.accuracy

        # Execute adaptation
        record = self._engine.adapt_parameters(
            model_node=model,
            evidence=evidence,
            timestamp=timestamp,
        )

        # Record adaptation in accumulator (for anti-oscillation tracking)
        self._accumulator.record_adaptation(model_id, timestamp=timestamp)

        # Clear evidence after adaptation
        self._accumulator.clear_evidence(model_id)

        logger.debug(
            "CognitiveAdaptationLoop: adapted model %s (accuracy %.2f → %.2f, lr=%.4f)",
            model_id,
            accuracy_before,
            model.accuracy,
            record.learning_rate_used,
        )

        return record

    # ── Rule Extraction (slow path) ───────────────────────────────────

    def extract_rule_from_evidence(
        self,
        model_id: str,
        condition: str,
        prediction: str,
        domain: str = "",
        timestamp: float | None = None,
    ) -> str | None:
        """Extract a LearnedRuleNode from accumulated evidence.

        Only callable when sufficient evidence has accumulated.

        Returns:
            The ID of the created LearnedRuleNode, or None.
        """
        evidence = self._accumulator.get_evidence(model_id)
        if evidence is None or evidence.occurrences < 3:
            return None

        rule = self._engine.extract_rule(
            condition=condition,
            prediction=prediction,
            evidence=evidence,
            domain=domain,
            timestamp=timestamp,
        )

        return rule.id

    # ── Evaluation Integration ────────────────────────────────────────

    def evaluate_adaptation(
        self,
        adaptation_id: str,
        model_id: str,
        accuracy_before: float,
        accuracy_after: float,
    ) -> EvaluationResult:
        """Evaluate whether an adaptation improved the model.

        Returns:
            EvaluationResult with outcome and cross-domain impact.
        """
        return self._evaluator.evaluate_from_accuracy(
            adaptation_id=adaptation_id,
            model_id=model_id,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
        )
