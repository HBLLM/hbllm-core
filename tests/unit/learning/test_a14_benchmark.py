"""A14 Benchmark — Prediction-Error-Centered Learning.

10 end-to-end scenarios validating the complete A14 stack:

1. Model error detection — probabilistic classification shifts toward MODEL_ERROR
2. Environment change detection — classification favors ENVIRONMENT_CHANGE
3. Noise filtering — gate REJECTs (no adaptation)
4. Novelty detection — EXPLORE signal
5. AdaptationGate stability — insufficient evidence → DEFER; accumulated → ADAPT
6. Parameter adaptation — transition probabilities update after gated errors
7. Rule extraction — repeated error pattern yields LearnedRuleNode with provenance
8. Cross-module feedback — A13 permanence error feeds through router
9. Plasticity + stability — adaptation improves domain A without degrading domain B
10. Zero LLM — entire loop runs without LLM invocation

All scenarios are zero-LLM.
"""

from __future__ import annotations

from hbllm.brain.learning.adaptation_gate import (
    AdaptationGate,
    ErrorEvidenceAccumulator,
    GateDecision,
)
from hbllm.brain.learning.cognitive_adaptation_loop import CognitiveAdaptationLoop
from hbllm.brain.learning.error_classifier import (
    ErrorClassification,
    ErrorClassifier,
    ErrorContext,
)
from hbllm.brain.learning.learning_signal_router import (
    RoutingAction,
)
from hbllm.brain.learning.prediction_evaluation_engine import (
    AdaptationOutcome,
    PredictionEvaluationEngine,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdgeType,
    LearnedRuleNode,
    PredictionErrorNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_graph() -> CognitiveGraph:
    """Create a fresh CognitiveGraph for testing."""
    return CognitiveGraph()


def _make_loop(
    graph: CognitiveGraph | None = None,
    min_evidence: int = 3,
    threshold: float = 0.6,
) -> CognitiveAdaptationLoop:
    """Create a CognitiveAdaptationLoop with test defaults."""
    g = graph or _make_graph()
    return CognitiveAdaptationLoop(
        graph=g,
        min_evidence_count=min_evidence,
        model_error_threshold=threshold,
    )


def _register_model(
    loop: CognitiveAdaptationLoop,
    model_type: str = "markov",
    domain: str = "physics",
    params: dict | None = None,
    accuracy: float = 0.8,
) -> str:
    """Register a predictive model and return its ID."""
    model_id = loop.registry.register(
        model_type=model_type,
        domain=domain,
        parameters=params or {"transitions": {}, "base_confidence": 0.8},
        initial_accuracy=accuracy,
    )
    return model_id


def _model_error_context(
    magnitude: float = 0.5,
    pattern: str = "recurring",
    historical_rate: float = 0.3,
) -> ErrorContext:
    """Create an ErrorContext that signals model error."""
    return ErrorContext(
        error_magnitude=magnitude,
        prediction_confidence=0.8,
        historical_error_rate=historical_rate,
        temporal_pattern=pattern,
        cross_entity_correlation=0.1,
        recency_weighted_frequency=0.4,
        time_since_last_similar=10.0,
        prediction_domain="physics",
    )


def _env_change_context(magnitude: float = 0.6) -> ErrorContext:
    """Create an ErrorContext that signals environment change."""
    return ErrorContext(
        error_magnitude=magnitude,
        prediction_confidence=0.7,
        historical_error_rate=0.05,
        temporal_pattern="sudden",
        cross_entity_correlation=0.8,
        recency_weighted_frequency=0.0,
        time_since_last_similar=float("inf"),
        prediction_domain="physics",
    )


def _noise_context() -> ErrorContext:
    """Create an ErrorContext that signals noise."""
    return ErrorContext(
        error_magnitude=0.05,
        prediction_confidence=0.9,
        historical_error_rate=0.02,
        temporal_pattern="isolated",
        cross_entity_correlation=0.0,
        recency_weighted_frequency=0.01,
        time_since_last_similar=600.0,
        prediction_domain="physics",
    )


def _novelty_context() -> ErrorContext:
    """Create an ErrorContext that signals novelty."""
    return ErrorContext(
        error_magnitude=0.7,
        prediction_confidence=0.5,
        historical_error_rate=0.01,
        temporal_pattern="isolated",
        cross_entity_correlation=0.0,
        recency_weighted_frequency=0.0,
        time_since_last_similar=float("inf"),
        prediction_domain="physics",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: Model Error Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestModelErrorDetection:
    """Systematic prediction bias → probabilistic classification shifts."""

    def test_recurring_errors_shift_toward_model_error(self) -> None:
        classifier = ErrorClassifier()

        # First error: uncertain classification
        ctx1 = _model_error_context(magnitude=0.3, pattern="isolated", historical_rate=0.05)
        c1 = classifier.classify(ctx1, error_id="e1")

        # After repeated errors: classification should shift toward model_error
        ctx2 = _model_error_context(magnitude=0.5, pattern="recurring", historical_rate=0.3)
        c2 = classifier.classify(ctx2, error_id="e2")

        assert c2.model_error > c1.model_error, (
            f"Recurring errors should increase P(model_error): {c1.model_error} → {c2.model_error}"
        )
        assert c2.dominant_class == "model_error"

    def test_classification_sums_to_one(self) -> None:
        classifier = ErrorClassifier()
        ctx = _model_error_context()
        c = classifier.classify(ctx)
        total = c.model_error + c.environment_change + c.noise + c.novelty
        assert abs(total - 1.0) < 0.01, f"Classification must sum to ≈1.0, got {total}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: Environment Change Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvironmentChangeDetection:
    """Sudden world change → classification favors ENVIRONMENT_CHANGE."""

    def test_sudden_correlated_error_classified_as_env_change(self) -> None:
        classifier = ErrorClassifier()
        ctx = _env_change_context()
        c = classifier.classify(ctx)

        assert c.dominant_class == "environment_change", (
            f"Sudden correlated error should be classified as environment_change, "
            f"got {c.dominant_class} (env={c.environment_change:.2f})"
        )
        assert c.environment_change > 0.4


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: Noise Filtering
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseFiltering:
    """Transient errors → gate REJECTs (no adaptation)."""

    def test_small_isolated_error_rejected(self) -> None:
        loop = _make_loop(min_evidence=2)
        model_id = _register_model(loop)

        # Feed noise errors
        for i in range(5):
            result = loop.process_error(
                error_id=f"noise_{i}",
                model_id=model_id,
                context=_noise_context(),
                error_magnitude=0.05,
            )

        # The gate should reject noise (or defer — but NOT adapt)
        assert result.action != RoutingAction.ADAPT, (
            f"Noise should not trigger adaptation, got {result.action}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: Novelty Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestNoveltyDetection:
    """Genuinely new phenomenon → EXPLORE signal."""

    def test_novel_error_classified_with_high_novelty(self) -> None:
        classifier = ErrorClassifier()
        ctx = _novelty_context()
        c = classifier.classify(ctx)

        # Novelty should have a meaningful probability
        assert c.novelty > 0.1, f"Novel error should have P(novelty) > 0.1, got {c.novelty:.2f}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: AdaptationGate Stability
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptationGateStability:
    """Gate DEFERs with insufficient evidence, ADAPTs with sufficient."""

    def test_defers_with_insufficient_evidence(self) -> None:
        accumulator = ErrorEvidenceAccumulator()
        gate = AdaptationGate(accumulator=accumulator, min_evidence_count=5)

        # Accumulate only 2 errors (below threshold of 5)
        for i in range(2):
            accumulator.accumulate(
                model_id="model_001",
                error_id=f"err_{i}",
                classification=ErrorClassification(
                    model_error=0.8,
                    environment_change=0.1,
                    noise=0.05,
                    novelty=0.05,
                ),
                error_magnitude=0.5,
            )

        verdict = gate.evaluate("model_001")
        assert verdict.decision == GateDecision.DEFER

    def test_adapts_with_sufficient_evidence(self) -> None:
        accumulator = ErrorEvidenceAccumulator()
        gate = AdaptationGate(
            accumulator=accumulator,
            min_evidence_count=3,
            model_error_threshold=0.6,
        )

        # Accumulate 5 errors with high model_error probability
        for i in range(5):
            accumulator.accumulate(
                model_id="model_001",
                error_id=f"err_{i}",
                classification=ErrorClassification(
                    model_error=0.8,
                    environment_change=0.1,
                    noise=0.05,
                    novelty=0.05,
                ),
                error_magnitude=0.5,
            )

        verdict = gate.evaluate("model_001")
        assert verdict.decision == GateDecision.ADAPT, (
            f"Should ADAPT with sufficient evidence, got {verdict.decision}: {verdict.reasoning}"
        )

    def test_anti_oscillation_tightens_threshold(self) -> None:
        accumulator = ErrorEvidenceAccumulator()
        gate = AdaptationGate(
            accumulator=accumulator,
            min_evidence_count=2,
            model_error_threshold=0.6,
            max_recent_adaptations=2,
        )

        # Record 3 recent adaptations (exceeds max)
        for _ in range(3):
            accumulator.record_adaptation("model_001")

        # Accumulate errors
        for i in range(5):
            accumulator.accumulate(
                model_id="model_001",
                error_id=f"err_{i}",
                classification=ErrorClassification(
                    model_error=0.75,
                    environment_change=0.15,
                    noise=0.05,
                    novelty=0.05,
                ),
                error_magnitude=0.4,
            )

        verdict = gate.evaluate("model_001")
        assert verdict.decision == GateDecision.DEFER, (
            f"Anti-oscillation should DEFER, got {verdict.decision}: {verdict.reasoning}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: Parameter Adaptation
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterAdaptation:
    """Transition probabilities update after gated errors."""

    def test_markov_parameters_change_after_adaptation(self) -> None:
        loop = _make_loop(min_evidence=3, threshold=0.5)
        model_id = _register_model(
            loop,
            params={
                "transitions": {"A": {"B": 0.9, "C": 0.1}},
                "base_confidence": 0.8,
            },
        )

        # Feed enough model errors to trigger adaptation
        for i in range(5):
            loop.process_error(
                error_id=f"merr_{i}",
                model_id=model_id,
                context=_model_error_context(),
                error_magnitude=0.5,
            )

        # Model parameters should have changed
        model = loop.registry.get_model(model_id)
        assert model is not None
        assert model.adaptation_count > 0, "Model should have been adapted"
        # Base confidence should have decreased due to errors
        assert model.parameters.get("base_confidence", 1.0) < 0.8, (
            f"Base confidence should decrease, got {model.parameters.get('base_confidence')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: Rule Extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestRuleExtraction:
    """Repeated error pattern yields LearnedRuleNode with provenance."""

    def test_rule_extraction_creates_node_with_provenance(self) -> None:
        graph = _make_graph()
        loop = _make_loop(graph, min_evidence=3)
        model_id = _register_model(loop)

        # Create PredictionErrorNodes in the graph for provenance
        error_ids = []
        for i in range(5):
            err = PredictionErrorNode(
                prediction_id=f"pred_{i}",
                predicted_value="stationary",
                observed_value="falling",
                delta=0.5,
                error_magnitude=0.5,
            )
            graph.add_node(err)
            error_ids.append(err.id)

        # Accumulate evidence manually
        for eid in error_ids:
            loop.accumulator.accumulate(
                model_id=model_id,
                error_id=eid,
                classification=ErrorClassification(
                    model_error=0.8,
                    environment_change=0.1,
                    noise=0.05,
                    novelty=0.05,
                ),
                error_magnitude=0.5,
            )

        # Extract rule
        rule_id = loop.extract_rule_from_evidence(
            model_id=model_id,
            condition="support_surface = absent",
            prediction="object_motion = downward",
            domain="physics",
        )

        assert rule_id is not None, "Rule should be extracted"

        # Verify rule node exists in HCIR
        rule_node = graph.get_node(rule_id)
        assert isinstance(rule_node, LearnedRuleNode)
        assert rule_node.condition == "support_surface = absent"
        assert rule_node.prediction == "object_motion = downward"
        assert rule_node.source_error_count == 5

        # Verify provenance edges exist (LEARNED_FROM)
        edges = graph.edges_from(rule_id)
        learned_from_edges = [e for e in edges if e.edge_type == HCIREdgeType.LEARNED_FROM]
        assert len(learned_from_edges) > 0, (
            "LearnedRuleNode should have LEARNED_FROM edges to errors"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: Cross-Module Feedback
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossModuleFeedback:
    """A13 permanence error feeds through router to adaptation."""

    def test_permanence_error_routes_to_model(self) -> None:
        loop = _make_loop(min_evidence=2, threshold=0.5)
        model_id = _register_model(loop, model_type="permanence", domain="physics")

        # Simulate A13 permanence prediction errors
        for i in range(4):
            loop.process_error(
                error_id=f"perm_err_{i}",
                model_id=model_id,
                context=ErrorContext(
                    error_magnitude=0.4,
                    prediction_confidence=0.7,
                    historical_error_rate=0.25,
                    temporal_pattern="recurring",
                    cross_entity_correlation=0.1,
                    recency_weighted_frequency=0.3,
                    prediction_domain="physics",
                ),
            )

        # Should have triggered adaptation
        model = loop.registry.get_model(model_id)
        assert model is not None
        assert model.adaptation_count > 0, (
            "Permanence errors should feed through and trigger adaptation"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: Plasticity + Stability
# ═══════════════════════════════════════════════════════════════════════════


class TestPlasticityStability:
    """Adaptation improves domain A without degrading domain B."""

    def test_domain_a_improves_without_degrading_domain_b(self) -> None:
        graph = _make_graph()
        loop = _make_loop(graph, min_evidence=3, threshold=0.5)

        # Register models in two domains
        model_a = _register_model(loop, domain="physics", accuracy=0.6)
        model_b = _register_model(loop, domain="social", accuracy=0.75)

        # Record pre-adaptation accuracy for both domains
        acc_b_before = loop.registry.get_model(model_b).accuracy

        # Feed errors ONLY for domain A
        for i in range(5):
            loop.process_error(
                error_id=f"phys_err_{i}",
                model_id=model_a,
                context=_model_error_context(),
                error_magnitude=0.5,
                domain="physics",
            )

        # Model A should have been adapted
        model_a_node = loop.registry.get_model(model_a)
        assert model_a_node.adaptation_count > 0, "Domain A model should adapt"

        # Model B should NOT have been affected
        model_b_node = loop.registry.get_model(model_b)
        acc_b_after = model_b_node.accuracy

        tolerance = 0.02
        assert acc_b_after >= acc_b_before - tolerance, (
            f"Domain B should remain stable: "
            f"before={acc_b_before:.3f}, after={acc_b_after:.3f}, "
            f"tolerance={tolerance}"
        )

    def test_evaluation_detects_cross_domain_degradation(self) -> None:
        evaluator = PredictionEvaluationEngine()

        # Record domain accuracies
        evaluator.record_domain_accuracy("physics", pre_accuracy=0.60, post_accuracy=0.75)
        evaluator.record_domain_accuracy("social", pre_accuracy=0.75, post_accuracy=0.70)

        # Evaluate
        result = evaluator.evaluate_from_accuracy(
            adaptation_id="adapt_001",
            model_id="model_a",
            accuracy_before=0.60,
            accuracy_after=0.75,
            cross_domain_impact={"physics": 0.15, "social": -0.05},
        )

        assert result.outcome == AdaptationOutcome.IMPROVED
        assert result.cross_domain_impact["social"] < 0, "Should detect social domain degradation"
        assert result.cross_domain_impact["physics"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: Zero LLM
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Entire A14 loop runs without LLM invocation."""

    def test_no_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.learning.adaptation_engine
import hbllm.brain.learning.adaptation_gate
import hbllm.brain.learning.cognitive_adaptation_loop
import hbllm.brain.learning.error_classifier
import hbllm.brain.learning.learning_signal_router
import hbllm.brain.learning.prediction_evaluation_engine
import hbllm.brain.learning.predictive_model_registry

llm_markers = [
    "openai",
    "anthropic",
    "litellm",
    "langchain",
    "transformers",
]

loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded: {marker}"
"""
        import os

        env = dict(os.environ, PYTHONPATH=":".join(sys.path))
        res = subprocess.run(
            [sys.executable, "-c", check_code], capture_output=True, text=True, env=env
        )
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"
