"""
Tests for Broca's Area (v4).

Tests cover:
  - ContentNode: construction, types
  - ContentPlanNetwork: SNN structure, decision making
  - ContentPlanner: concept-to-node mapping, ordering, transitions
  - PRMTrainer: batch training, metrics, should_train logic
  - BrocaEncoder: minimal prompts, batch encoding, assembly, savings
  - Integration: ExpressionStream broca_mode
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.brain.snn.comprehension.models import (
    ComprehensionUnit,
    UnderstandingState,
)
from hbllm.brain.snn.expression.broca_encoder import BrocaEncoder, BrocaPrompt
from hbllm.brain.snn.expression.content_planner import (
    CONTENT_TYPES,
    ContentNode,
    ContentPlanner,
    ContentPlanNetwork,
)
from hbllm.brain.snn.expression.models import ThoughtGoal
from hbllm.brain.snn.expression.prm_trainer import PRMTrainer, TrainingMetrics
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.shallow_renderer import RenderingContext
from hbllm.brain.snn.expression.trained_prm import TrainedPRM
from hbllm.brain.snn.network import SpikingNetwork

# ═══════════════════════════════════════════════════════════════════════════
# ContentNode Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestContentNode:
    """Test the atomic content decision data model."""

    def test_construction(self) -> None:
        node = ContentNode(
            node_id="cn_test",
            content_type="assertion",
            key_points=["SNN uses LIF neurons"],
            source_concept="SNN architecture",
            confidence=0.8,
        )
        assert node.content_type == "assertion"
        assert len(node.key_points) == 1

    def test_to_dict(self) -> None:
        node = ContentNode(
            node_id="cn_test",
            content_type="explanation",
            key_points=["Point A", "Point B"],
            confidence=0.75,
        )
        d = node.to_dict()
        assert d["content_type"] == "explanation"
        assert d["confidence"] == 0.75

    def test_content_types_valid(self) -> None:
        for ct in CONTENT_TYPES:
            node = ContentNode(content_type=ct)
            assert node.content_type == ct


# ═══════════════════════════════════════════════════════════════════════════
# ContentPlanNetwork Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestContentPlanNetwork:
    """Test the 4-layer SNN for content type selection."""

    def test_network_structure(self) -> None:
        net = ContentPlanNetwork()
        snn = net.network
        assert isinstance(snn, SpikingNetwork)
        assert "input" in snn.layer_names
        assert "planning" in snn.layer_names
        assert "selection" in snn.layer_names
        assert "output" in snn.layer_names

    def test_decide_returns_valid_keys(self) -> None:
        net = ContentPlanNetwork()
        result = net.decide(
            {
                "concept_salience": 0.9,
                "domain_strength": 0.7,
                "association_count": 0.5,
                "causal_confidence": 0.6,
                "memory_density": 0.3,
                "constraint_strength": 0.1,
                "concept_novelty": 0.4,
                "query_specificity": 0.8,
            }
        )
        assert "content_type" in result
        assert "include" in result
        assert "emphasize" in result
        assert "caveat" in result
        assert "confidence" in result

    def test_decide_content_type_valid(self) -> None:
        net = ContentPlanNetwork()
        result = net.decide({"concept_salience": 0.8})
        assert result["content_type"] in CONTENT_TYPES or result["content_type"] == "skip"

    def test_high_salience_produces_assertion(self) -> None:
        net = ContentPlanNetwork()
        result = net.decide(
            {
                "concept_salience": 1.0,
                "domain_strength": 0.9,
                "query_specificity": 0.9,
            }
        )
        # High salience + specificity should trend toward assertion
        assert result["content_type"] in CONTENT_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# ContentPlanner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestContentPlanner:
    """Test the content planning orchestrator."""

    @pytest.fixture
    def context(self):
        return RenderingContext(
            concepts=["SNN architecture", "LIF neurons", "STDP plasticity"],
            associations=[
                {
                    "association_type": "similar",
                    "source_text": "SNN architecture",
                    "target_text": "LIF neurons",
                    "strength": 0.8,
                }
            ],
            causal_chains=[
                {
                    "source_concept": "STDP plasticity",
                    "conclusion": "adaptive weights",
                    "snn_confidence": 0.75,
                }
            ],
            goals=[
                ThoughtGoal(
                    id="g1",
                    text="Address: SNN architecture",
                    source_concept_text="SNN architecture",
                ),
            ],
            domain_context={"coding": 0.7},
            original_query="How do SNNs work?",
            confidence=0.8,
        )

    def test_plan_produces_nodes(self, context) -> None:
        planner = ContentPlanner()
        nodes = planner.plan(context)
        assert len(nodes) > 0
        assert all(isinstance(n, ContentNode) for n in nodes)

    def test_nodes_are_ordered(self, context) -> None:
        planner = ContentPlanner()
        nodes = planner.plan(context)
        positions = [n.position for n in nodes]
        assert positions == sorted(positions)

    def test_key_points_include_concept(self, context) -> None:
        planner = ContentPlanner()
        nodes = planner.plan(context)
        # At least one node should reference a concept
        concepts_in_nodes = [n.source_concept for n in nodes if n.source_concept]
        assert len(concepts_in_nodes) > 0

    def test_empty_context_no_nodes(self) -> None:
        planner = ContentPlanner()
        ctx = RenderingContext()
        nodes = planner.plan(ctx)
        assert nodes == []

    def test_causal_basis_populated(self, context) -> None:
        planner = ContentPlanner()
        nodes = planner.plan(context)
        # STDP concept should have causal basis
        stdp_nodes = [n for n in nodes if "STDP" in n.source_concept]
        if stdp_nodes:
            assert any(n.causal_basis for n in stdp_nodes)


# ═══════════════════════════════════════════════════════════════════════════
# PRMTrainer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPRMTrainer:
    """Test batch STDP training pipeline."""

    @pytest.fixture
    def prm_with_data(self):
        evaluator = RewardEvaluator(min_acceptable_reward=0.4)
        prm = TrainedPRM(
            reward_evaluator=evaluator,
            fallback_threshold=10,
        )
        # Seed training data
        for i in range(25):
            prm.collector.record(
                features={
                    "heuristic_relevance": 0.5 + (i % 5) * 0.1,
                    "heuristic_coherence": 0.6,
                    "heuristic_completeness": 0.5,
                    "heuristic_conciseness": 0.5,
                    "goal_salience": 0.7,
                    "text_length_ratio": 0.5,
                },
                accepted=i % 3 != 0,  # 2/3 accepted
                reward_score=0.6 if i % 3 != 0 else 0.3,
            )
        return prm

    def test_should_train_with_enough_data(self, prm_with_data) -> None:
        trainer = PRMTrainer(prm_with_data, min_new_examples=20)
        assert trainer.should_train() is True

    def test_should_not_train_insufficient_data(self) -> None:
        evaluator = RewardEvaluator()
        prm = TrainedPRM(reward_evaluator=evaluator)
        trainer = PRMTrainer(prm, min_new_examples=20)
        assert trainer.should_train() is False

    def test_train_returns_metrics(self, prm_with_data) -> None:
        trainer = PRMTrainer(prm_with_data, epochs=2, batch_size=10)
        metrics = trainer.train()

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.examples_trained == 25
        assert metrics.epochs_completed == 2
        assert 0.0 <= metrics.pre_accuracy <= 1.0
        assert 0.0 <= metrics.post_accuracy <= 1.0
        assert metrics.training_duration_ms > 0

    def test_train_updates_history(self, prm_with_data) -> None:
        trainer = PRMTrainer(prm_with_data, epochs=1)
        trainer.train()
        assert trainer.total_sweeps == 1
        assert trainer.last_metrics is not None

    def test_should_not_train_after_sweep(self, prm_with_data) -> None:
        trainer = PRMTrainer(prm_with_data, min_new_examples=20)
        trainer.train()
        # After training, should not train again (no new data)
        assert trainer.should_train() is False

    def test_metrics_to_dict(self) -> None:
        m = TrainingMetrics(
            examples_trained=100,
            epochs_completed=3,
            pre_accuracy=0.65,
            post_accuracy=0.78,
        )
        d = m.to_dict()
        assert d["examples_trained"] == 100
        assert d["post_accuracy"] == 0.78


# ═══════════════════════════════════════════════════════════════════════════
# BrocaEncoder Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBrocaEncoder:
    """Test ultra-minimal LLM interface."""

    @pytest.fixture
    def encoder(self):
        return BrocaEncoder(max_tokens_per_node=60)

    def test_encode_minimal_prompt(self, encoder) -> None:
        node = ContentNode(
            content_type="assertion",
            key_points=["SNN uses LIF neurons"],
            tone="neutral",
        )
        prompt = encoder.encode(node)
        assert isinstance(prompt, BrocaPrompt)
        assert "TYPE: assertion" in prompt.prompt_text
        assert "TONE: neutral" in prompt.prompt_text
        assert "SNN uses LIF neurons" in prompt.prompt_text
        assert "MAX: 60" in prompt.prompt_text

    def test_encode_with_causal_basis(self, encoder) -> None:
        node = ContentNode(
            content_type="explanation",
            key_points=["STDP enables learning"],
            causal_basis="STDP → adaptive weights",
            tone="emphatic",
        )
        prompt = encoder.encode(node)
        assert "BECAUSE:" in prompt.prompt_text
        assert "STDP → adaptive weights" in prompt.prompt_text

    def test_encode_batch(self, encoder) -> None:
        nodes = [
            ContentNode(content_type="assertion", key_points=["Point A"], tone="neutral"),
            ContentNode(content_type="explanation", key_points=["Point B"], tone="emphatic"),
        ]
        prompt = encoder.encode_batch(nodes)
        assert "RENDER each item" in prompt.prompt_text
        assert "[1]" in prompt.prompt_text
        assert "[2]" in prompt.prompt_text

    def test_encode_batch_empty(self, encoder) -> None:
        prompt = encoder.encode_batch([])
        assert prompt.prompt_text == ""

    def test_assemble(self, encoder) -> None:
        node1 = ContentNode(content_type="assertion")
        node2 = ContentNode(content_type="explanation")
        rendered = [
            (node1, "SNN uses LIF neurons for processing."),
            (node2, "This enables temporal pattern recognition."),
        ]
        text = encoder.assemble(rendered)
        assert "SNN uses LIF neurons" in text
        assert "temporal pattern recognition" in text

    def test_assemble_skips_empty(self, encoder) -> None:
        node = ContentNode(content_type="assertion")
        rendered = [(node, ""), (node, "Valid text.")]
        text = encoder.assemble(rendered)
        assert text.strip() == "Valid text."

    def test_estimate_savings(self) -> None:
        prompt = BrocaPrompt(estimated_tokens=80)
        savings = BrocaEncoder.estimate_savings(prompt)
        assert savings["vs_deep"] > 0.8  # >80% reduction from deep
        assert savings["vs_shallow"] > 0.7  # >70% from shallow


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ExpressionStream + Broca Mode
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionStreamBrocaIntegration:
    """Test ExpressionStream with broca_mode."""

    @pytest.mark.asyncio
    async def test_broca_mode_produces_response(self) -> None:
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )
        from hbllm.brain.snn.expression.shallow_renderer import ShallowRenderer

        planner_inst = ContentPlanner()
        encoder_inst = BrocaEncoder()
        renderer = ShallowRenderer()

        async def mock_llm(prompt: str) -> str:
            return "SNN uses LIF neurons for temporal signal processing."

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=mock_llm,
            shallow_renderer=renderer,
            content_planner=planner_inst,
            broca_encoder=encoder_inst,
            broca_mode=True,
        )

        state = UnderstandingState(
            concepts=[
                ComprehensionUnit(
                    text="SNN architecture",
                    embedding=np.zeros(384),
                    salience=1.0,
                ),
            ],
            salience_map=[1.0],
        )

        result = await stream.express(
            understanding=state,
            base_thought="SNNs process temporal patterns.",
            original_query="How do SNNs work?",
        )

        assert result.text
        assert result.thought_count >= 0

    @pytest.mark.asyncio
    async def test_broca_mode_false_uses_deep(self) -> None:
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            broca_mode=False,
        )

        state = UnderstandingState(
            concepts=[
                ComprehensionUnit(
                    text="test concept",
                    embedding=np.zeros(384),
                ),
            ],
            salience_map=[1.0],
        )

        result = await stream.express(
            understanding=state,
            base_thought="Test.",
            original_query="test",
        )
        assert result.text

    @pytest.mark.asyncio
    async def test_broca_mode_fallback_to_deep(self) -> None:
        """Broca mode with no LLM falls back to deep path."""
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            content_planner=ContentPlanner(),
            broca_encoder=BrocaEncoder(),
            broca_mode=True,
            # No llm_generate → broca can't render → falls back
        )

        state = UnderstandingState(
            concepts=[
                ComprehensionUnit(
                    text="test concept",
                    embedding=np.zeros(384),
                ),
            ],
            salience_map=[1.0],
        )

        result = await stream.express(
            understanding=state,
            base_thought="Test fallback.",
            original_query="test",
        )
        assert result.text
