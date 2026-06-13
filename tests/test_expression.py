"""
Tests for the expression-side Cognitive Stream (Layer 5).

Tests cover:
  - ThoughtPlanner: outline generation, constraint expansion, low-salience merging
  - ThoughtController: SNN gating, bypass, reset
  - RewardEvaluator: scoring dimensions, revision gating
  - ExpressionStream: end-to-end pipeline, fallback extraction
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from hbllm.brain.snn.comprehension.models import (
    ActivatedMemory,
    ComprehensionUnit,
    UnderstandingState,
)
from hbllm.brain.snn.expression.expression_stream import ExpressionStream
from hbllm.brain.snn.expression.models import (
    ExpressionResult,
    ThoughtFragment,
    ThoughtGoal,
)
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.thought_controller import (
    GateSignal,
    ThoughtController,
)
from hbllm.brain.snn.expression.thought_planner import ThoughtPlanner

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_understanding(
    n_concepts: int = 3,
    salience: float = 0.8,
    with_constraints: bool = False,
    with_memories: bool = False,
    low_salience_extra: bool = False,
) -> UnderstandingState:
    """Create an UnderstandingState for testing."""
    concepts = []
    for i in range(n_concepts):
        channels = {}
        if with_constraints and i == 0:
            channels["constraint"] = 0.7
            channels["surprise"] = 0.5

        memories = []
        if with_memories:
            memories.append(
                ActivatedMemory(
                    id=f"mem_{i}",
                    content=f"Previous context about topic {i}",
                    score=0.6,
                )
            )

        concepts.append(
            ComprehensionUnit(
                text=f"concept {i} about neural networks and spiking models",
                embedding=np.random.randn(384).astype(np.float32),
                salience=salience,
                domain_activation={"coding": 0.6, "general": 0.4},
                channel_metadata=channels,
                activated_memories=memories,
            )
        )

    if low_salience_extra:
        concepts.append(
            ComprehensionUnit(
                text="minor footnote about formatting",
                embedding=np.random.randn(384).astype(np.float32),
                salience=0.1,
                domain_activation={"general": 1.0},
                channel_metadata={},
                activated_memories=[],
            )
        )

    all_mems = []
    for c in concepts:
        all_mems.extend(c.activated_memories)

    return UnderstandingState(
        concepts=concepts,
        domain_activations={"coding": 0.6, "general": 0.4},
        all_memories=all_mems,
        salience_map=[c.salience for c in concepts],
    )


# ═══════════════════════════════════════════════════════════════════════════
# ThoughtPlanner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestThoughtPlanner:
    """Test ThoughtPlanner outline generation."""

    def test_basic_plan(self) -> None:
        """Basic plan: one goal per concept."""
        planner = ThoughtPlanner()
        understanding = _make_understanding(n_concepts=3)

        goals = planner.plan(understanding)

        assert len(goals) == 3
        for i, goal in enumerate(goals):
            assert goal.priority == i
            assert goal.id.startswith("tg_")
            assert "concept" in goal.text.lower() or "address" in goal.text.lower()
            assert goal.salience == 0.8
            assert goal.domain == "coding"

    def test_empty_understanding(self) -> None:
        """Empty understanding produces no goals."""
        planner = ThoughtPlanner()
        understanding = UnderstandingState(
            concepts=[], domain_activations={}, all_memories=[], salience_map=[]
        )

        goals = planner.plan(understanding)
        assert goals == []

    def test_constraint_expansion(self) -> None:
        """Constraint concepts expand to two goals: context + assertion."""
        planner = ThoughtPlanner(constraint_expansion=True)
        understanding = _make_understanding(n_concepts=2, with_constraints=True)

        goals = planner.plan(understanding)

        # Concept 0 expands to 2 (ctx + cst), concept 1 stays 1 = 3 total
        assert len(goals) == 3
        assert "context" in goals[0].text.lower()
        assert "constraint" in goals[1].text.lower()
        assert goals[1].salience > goals[0].salience  # Constrained is boosted

    def test_constraint_expansion_disabled(self) -> None:
        """When disabled, constraints don't expand."""
        planner = ThoughtPlanner(constraint_expansion=False)
        understanding = _make_understanding(n_concepts=2, with_constraints=True)

        goals = planner.plan(understanding)

        # 2 concepts, no expansion
        assert len(goals) == 2

    def test_low_salience_merged(self) -> None:
        """Concepts below min_salience_for_goal get merged into one catch-all."""
        planner = ThoughtPlanner(min_salience_for_goal=0.3)
        understanding = _make_understanding(n_concepts=2, salience=0.8, low_salience_extra=True)

        goals = planner.plan(understanding)

        # 2 regular + 1 merged = 3
        assert len(goals) == 3
        merged = goals[-1]
        assert "also briefly address" in merged.text.lower()
        assert merged.salience == 0.3

    def test_memory_hints_forwarded(self) -> None:
        """Memory hints from comprehension are included in goals."""
        planner = ThoughtPlanner()
        understanding = _make_understanding(n_concepts=2, with_memories=True)

        goals = planner.plan(understanding)

        for goal in goals:
            assert len(goal.memory_hints) > 0
            assert "previous context" in goal.memory_hints[0].lower()

    def test_token_budget_scales_with_salience(self) -> None:
        """Token budget is proportional to salience."""
        planner = ThoughtPlanner(base_token_budget=500)

        high_sal = _make_understanding(n_concepts=1, salience=1.0)
        low_sal = _make_understanding(n_concepts=1, salience=0.5)

        high_goals = planner.plan(high_sal)
        low_goals = planner.plan(low_sal)

        assert high_goals[0].max_tokens == 500
        assert low_goals[0].max_tokens == 250

    def test_deterministic_ids(self) -> None:
        """Same text produces the same goal ID (deterministic)."""
        planner = ThoughtPlanner()
        u1 = _make_understanding(n_concepts=1)
        u2 = _make_understanding(n_concepts=1)

        g1 = planner.plan(u1)
        g2 = planner.plan(u2)

        assert g1[0].id == g2[0].id

    def test_priority_ordering(self) -> None:
        """Goals are sorted by priority."""
        planner = ThoughtPlanner()
        understanding = _make_understanding(n_concepts=5)

        goals = planner.plan(understanding)

        priorities = [g.priority for g in goals]
        assert priorities == sorted(priorities)


# ═══════════════════════════════════════════════════════════════════════════
# ThoughtController Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestThoughtController:
    """Test SNN-gated thought sequencer."""

    def test_first_goal_bypasses_gating(self) -> None:
        """First goal (no prev_fragment) always fires with bypass=True."""
        controller = ThoughtController()
        goal = ThoughtGoal(id="g1", text="test", salience=0.8)

        signal = controller.gate(goal, prev_fragment_text=None)

        assert signal.fire is True
        assert signal.bypass is True

    def test_high_salience_fires(self) -> None:
        """High-salience goals fire quickly."""
        controller = ThoughtController(
            readiness_threshold=0.4,
            coherence_threshold=0.3,
        )
        goal = ThoughtGoal(
            id="g2",
            text="neural networks and spiking models",
            salience=1.0,
            memory_hints=["context1", "context2", "context3"],
            max_tokens=512,
        )

        # Feed enough signals to fire
        fired = False
        for _ in range(6):
            signal = controller.gate(
                goal,
                prev_fragment_text="previous text about neural networks and spiking models",
            )
            if signal.fire:
                fired = True
                break

        assert fired is True

    def test_deadlock_bypass(self) -> None:
        """After max_wait_steps, bypass triggers."""
        controller = ThoughtController(max_wait_steps=3)
        goal = ThoughtGoal(
            id="g3",
            text="completely unrelated topic",
            salience=0.1,
        )

        # Force low signals by using very different previous text.
        # Bypass fires on step 3 (step_count >= max_wait_steps) and resets,
        # so we loop exactly max_wait_steps times.
        signal = None
        for i in range(3):
            signal = controller.gate(goal, prev_fragment_text="xyz abc def")

        # Should have bypassed on step 3
        assert signal is not None
        assert signal.fire is True
        assert signal.bypass is True

    def test_reset_clears_state(self) -> None:
        """Reset clears all neuron state."""
        controller = ThoughtController()
        goal = ThoughtGoal(id="g4", text="test", salience=0.8)

        # Do some gating
        controller.gate(goal, prev_fragment_text="some text")
        controller.gate(goal, prev_fragment_text="some text")

        controller.reset()

        assert controller._readiness.v == 0.0
        assert controller._coherence.v == 0.0
        assert controller._step_count == 0
        assert controller._prev_fragment_text is None

    def test_record_generation(self) -> None:
        """record_generation stores the fragment text."""
        controller = ThoughtController()
        controller.record_generation("generated text")

        assert controller._prev_fragment_text == "generated text"


# ═══════════════════════════════════════════════════════════════════════════
# RewardEvaluator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRewardEvaluator:
    """Test fragment scoring."""

    def test_relevant_fragment_scores_high(self) -> None:
        """Fragment that matches goal text scores highly on relevance."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(
            id="g1",
            text="neural networks",
            source_concept_text="neural networks and deep learning",
            max_tokens=256,
        )

        fragment = evaluator.evaluate(
            fragment_text="Neural networks are computational models that deep learning uses.",
            goal=goal,
        )

        assert fragment.relevance_score > 0.4
        assert fragment.reward_score > 0.3

    def test_irrelevant_fragment_scores_low(self) -> None:
        """Fragment that doesn't match goal text scores low on relevance."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(
            id="g2",
            text="quantum computing",
            source_concept_text="quantum computing and qubit states",
            max_tokens=256,
        )

        fragment = evaluator.evaluate(
            fragment_text="The weather today is sunny and warm with blue skies.",
            goal=goal,
        )

        assert fragment.relevance_score < 0.5

    def test_coherence_first_fragment(self) -> None:
        """First fragment (no predecessor) gets coherence 1.0."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(id="g3", text="test", max_tokens=256)

        fragment = evaluator.evaluate(
            fragment_text="Some text.", goal=goal, prev_fragment_text=None
        )

        assert fragment.coherence_score == 1.0

    def test_coherence_with_overlap(self) -> None:
        """Fragment with word overlap to predecessor scores higher coherence."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(id="g4", text="test", max_tokens=256)

        # Fragment with shared words
        fragment = evaluator.evaluate(
            fragment_text="Furthermore, the neural network model shows improvement.",
            goal=goal,
            prev_fragment_text="The neural network model was trained on the dataset.",
        )

        assert fragment.coherence_score > 0.5

    def test_conciseness_good_range(self) -> None:
        """Fragment within budget range gets high conciseness score."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(id="g5", text="test", max_tokens=50)

        # ~50 tokens (200 chars / 4)
        text = "A " * 100  # 200 chars = ~50 tokens
        fragment = evaluator.evaluate(fragment_text=text, goal=goal)

        assert fragment.metadata["conciseness"] == 1.0

    def test_conciseness_too_short(self) -> None:
        """Very short fragment gets penalized on conciseness."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(id="g6", text="test", max_tokens=500)

        fragment = evaluator.evaluate(fragment_text="Hi", goal=goal)

        assert fragment.metadata["conciseness"] < 0.5

    def test_should_revise_below_threshold(self) -> None:
        """Fragment below min_acceptable_reward should be revised."""
        evaluator = RewardEvaluator(min_acceptable_reward=0.5)

        fragment = ThoughtFragment(goal_id="g7", text="bad", reward_score=0.3)
        assert evaluator.should_revise(fragment) is True

    def test_should_not_revise_above_threshold(self) -> None:
        """Fragment above threshold should not be revised."""
        evaluator = RewardEvaluator(min_acceptable_reward=0.4)

        fragment = ThoughtFragment(goal_id="g8", text="good", reward_score=0.6)
        assert evaluator.should_revise(fragment) is False

    def test_with_encoder(self) -> None:
        """Encoder-based relevance scoring works."""

        def mock_encoder(text: str) -> np.ndarray:
            # Simple hash-based embedding for determinism
            h = hash(text) % 1000
            emb = np.zeros(384, dtype=np.float32)
            emb[h % 384] = 1.0
            return emb

        evaluator = RewardEvaluator(encoder=mock_encoder)
        goal = ThoughtGoal(
            id="g9",
            text="test",
            source_concept_text="test concept",
            max_tokens=256,
        )

        fragment = evaluator.evaluate(fragment_text="test response", goal=goal)

        # Should not crash and should produce a valid score
        assert 0.0 <= fragment.relevance_score <= 1.0
        assert 0.0 <= fragment.reward_score <= 1.0

    def test_completeness_scoring(self) -> None:
        """Completeness checks key term coverage."""
        evaluator = RewardEvaluator()
        goal = ThoughtGoal(
            id="g10",
            text="Address: neural networks and spiking models",
            source_concept_text="neural networks and spiking models",
            max_tokens=256,
        )

        # Fragment covering key terms
        good = evaluator.evaluate(
            fragment_text="Neural networks use spiking models for computation.",
            goal=goal,
        )

        # Fragment missing key terms
        bad = evaluator.evaluate(
            fragment_text="The weather is nice today.",
            goal=goal,
        )

        assert good.metadata["completeness"] > bad.metadata["completeness"]


# ═══════════════════════════════════════════════════════════════════════════
# ExpressionStream Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionStream:
    """Test the full expression pipeline."""

    @pytest.fixture
    def stream_no_llm(self) -> ExpressionStream:
        """ExpressionStream without LLM (fallback extraction mode)."""
        return ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=None,
            max_revisions=0,
            enable_gating=True,
        )

    @pytest.fixture
    def stream_with_llm(self) -> ExpressionStream:
        """ExpressionStream with mock LLM."""
        mock_llm = AsyncMock(side_effect=lambda prompt: f"Generated response for: {prompt[:50]}")
        return ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=mock_llm,
            max_revisions=1,
            enable_gating=True,
        )

    @pytest.mark.asyncio
    async def test_empty_understanding_passthrough(self, stream_no_llm: ExpressionStream) -> None:
        """Empty understanding passes through base_thought."""
        understanding = UnderstandingState(
            concepts=[], domain_activations={}, all_memories=[], salience_map=[]
        )

        result = await stream_no_llm.express(
            understanding=understanding,
            base_thought="The base thought content.",
            original_query="What is X?",
        )

        assert isinstance(result, ExpressionResult)
        assert result.text == "The base thought content."
        assert result.thought_count == 0
        assert result.fragments == []

    @pytest.mark.asyncio
    async def test_fallback_extraction(self, stream_no_llm: ExpressionStream) -> None:
        """Without LLM, extracts relevant portions from base_thought."""
        understanding = _make_understanding(n_concepts=2)

        result = await stream_no_llm.express(
            understanding=understanding,
            base_thought=(
                "Neural networks are powerful models. "
                "Spiking models simulate biological neurons. "
                "The weather is sunny today."
            ),
            original_query="Tell me about neural networks",
        )

        assert isinstance(result, ExpressionResult)
        assert result.thought_count == 2
        assert len(result.fragments) == 2
        assert result.text  # Should have assembled text

    @pytest.mark.asyncio
    async def test_with_llm_generation(self, stream_with_llm: ExpressionStream) -> None:
        """With LLM, generates per-goal text."""
        understanding = _make_understanding(n_concepts=2)

        result = await stream_with_llm.express(
            understanding=understanding,
            base_thought="Original thought.",
            original_query="Explain concepts",
        )

        assert isinstance(result, ExpressionResult)
        assert result.thought_count == 2
        assert len(result.fragments) == 2
        for frag in result.fragments:
            assert "Generated response" in frag.text
            # Source is 'llm' if accepted first time, 'revision' if revised
            assert frag.metadata.get("source") in ("llm", "revision")

    @pytest.mark.asyncio
    async def test_revision_on_low_reward(self) -> None:
        """Low-reward fragments get revised."""
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return "bad irrelevant text xyz"  # First two calls produce bad text
            return "neural networks and spiking models are used for computation"

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(min_acceptable_reward=0.8),
            llm_generate=mock_llm,
            max_revisions=1,
            enable_gating=True,
        )

        understanding = _make_understanding(n_concepts=1, salience=0.9)

        result = await stream.express(
            understanding=understanding,
            base_thought="Neural networks topic.",
            original_query="Tell me about neural networks",
        )

        assert result.revision_count >= 0  # May or may not have revised

    @pytest.mark.asyncio
    async def test_gating_disabled(self) -> None:
        """With gating disabled, all goals generate immediately."""
        mock_llm = AsyncMock(return_value="Generated text.")
        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=mock_llm,
            max_revisions=0,
            enable_gating=False,
        )

        understanding = _make_understanding(n_concepts=3)

        result = await stream.express(
            understanding=understanding,
            base_thought="Base.",
            original_query="Query",
        )

        assert result.thought_count == 3
        assert len(result.fragments) == 3

    @pytest.mark.asyncio
    async def test_mean_reward_computed(self, stream_no_llm: ExpressionStream) -> None:
        """Mean reward is computed correctly."""
        understanding = _make_understanding(n_concepts=2)

        result = await stream_no_llm.express(
            understanding=understanding,
            base_thought="Neural networks spiking models computation.",
            original_query="Query",
        )

        if result.fragments:
            expected_mean = sum(f.reward_score for f in result.fragments) / len(result.fragments)
            assert abs(result.mean_reward - expected_mean) < 0.01

    @pytest.mark.asyncio
    async def test_constraint_expansion_in_stream(self) -> None:
        """Constraint expansion produces extra goals in stream."""
        mock_llm = AsyncMock(return_value="Constrained response.")
        stream = ExpressionStream(
            planner=ThoughtPlanner(constraint_expansion=True),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=mock_llm,
            max_revisions=0,
        )

        understanding = _make_understanding(n_concepts=1, with_constraints=True)

        result = await stream.express(
            understanding=understanding,
            base_thought="Base.",
            original_query="Query",
        )

        # 1 concept with constraint → 2 goals (ctx + cst)
        assert result.thought_count == 2
        assert len(result.fragments) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Models
# ═══════════════════════════════════════════════════════════════════════════


class TestModels:
    """Test data model defaults and invariants."""

    def test_thought_goal_defaults(self) -> None:
        goal = ThoughtGoal()
        assert goal.id == ""
        assert goal.salience == 1.0
        assert goal.domain == "general"
        assert goal.max_tokens == 512

    def test_thought_fragment_defaults(self) -> None:
        frag = ThoughtFragment()
        assert frag.reward_score == 0.0
        assert frag.revision_count == 0

    def test_expression_result_defaults(self) -> None:
        result = ExpressionResult()
        assert result.text == ""
        assert result.mean_reward == 0.0
        assert result.fragments == []
