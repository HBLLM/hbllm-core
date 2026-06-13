"""
Tests for Shallow Renderer (v3 Step 5).

Tests cover:
  - RenderingContext: construction, properties, serialization
  - RenderPromptBuilder: full render, per-goal render, prompt structure
  - ShallowRenderer: context building, confidence gating, prompt generation
  - Integration: ExpressionStream shallow vs deep mode
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.brain.snn.comprehension.models import (
    ComprehensionUnit,
    UnderstandingState,
)
from hbllm.brain.snn.expression.models import ThoughtGoal
from hbllm.brain.snn.expression.shallow_renderer import (
    RenderingContext,
    RenderPromptBuilder,
    ShallowRenderer,
)


# ═══════════════════════════════════════════════════════════════════════════
# RenderingContext Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderingContext:
    """Test the pre-reasoned data payload."""

    def test_construction(self) -> None:
        ctx = RenderingContext(
            concepts=["SNN architecture", "LIF neurons"],
            original_query="How do SNNs work?",
            confidence=0.75,
        )
        assert ctx.concept_count == 2
        assert ctx.confidence == 0.75

    def test_primary_domain(self) -> None:
        ctx = RenderingContext(
            domain_context={"coding": 0.8, "general": 0.3},
        )
        assert ctx.primary_domain == "coding"

    def test_primary_domain_empty(self) -> None:
        ctx = RenderingContext()
        assert ctx.primary_domain == "general"

    def test_has_associations(self) -> None:
        ctx = RenderingContext(
            associations=[{"type": "similar", "source_text": "A", "target_text": "B"}],
        )
        assert ctx.has_associations is True

        empty = RenderingContext()
        assert empty.has_associations is False

    def test_has_causal_chains(self) -> None:
        ctx = RenderingContext(
            causal_chains=[{"source_concept": "X", "conclusion": "Y"}],
        )
        assert ctx.has_causal_chains is True

    def test_to_dict(self) -> None:
        ctx = RenderingContext(
            concepts=["A", "B"],
            confidence=0.85,
            domain_context={"coding": 0.9},
        )
        d = ctx.to_dict()
        assert d["concept_count"] == 2
        assert d["confidence"] == 0.85
        assert d["primary_domain"] == "coding"


# ═══════════════════════════════════════════════════════════════════════════
# RenderPromptBuilder Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderPromptBuilder:
    """Test prompt generation for shallow rendering."""

    @pytest.fixture
    def builder(self):
        return RenderPromptBuilder()

    @pytest.fixture
    def context(self):
        return RenderingContext(
            concepts=["SNN architecture", "LIF neurons", "STDP plasticity"],
            associations=[{
                "association_type": "similar",
                "source_text": "SNN architecture",
                "target_text": "LIF neurons",
                "strength": 0.8,
            }],
            causal_chains=[{
                "source_concept": "STDP plasticity",
                "conclusion": "adaptive weights",
                "snn_confidence": 0.75,
            }],
            memory_hints=["SNNs process temporal patterns"],
            domain_context={"coding": 0.7},
            original_query="How do SNNs work?",
            confidence=0.8,
        )

    def test_full_render_contains_conclusions(self, builder, context) -> None:
        prompt = builder.build_full_render(context)
        assert "RENDER" in prompt
        assert "SNN architecture" in prompt
        assert "LIF neurons" in prompt

    def test_full_render_no_reasoning_instructions(self, builder, context) -> None:
        prompt = builder.build_full_render(context)
        # Should NOT contain reasoning instructions
        assert "reason about" not in prompt.lower()
        assert "figure out" not in prompt.lower()
        # Should contain rendering instructions
        assert "Do NOT add new reasoning" in prompt

    def test_full_render_includes_associations(self, builder, context) -> None:
        prompt = builder.build_full_render(context)
        assert "RELATIONSHIPS" in prompt
        assert "similar" in prompt

    def test_full_render_includes_causal(self, builder, context) -> None:
        prompt = builder.build_full_render(context)
        assert "CAUSAL" in prompt
        assert "STDP plasticity" in prompt
        assert "adaptive weights" in prompt

    def test_per_goal_render(self, builder, context) -> None:
        goal = ThoughtGoal(
            id="test_1",
            text="Address: SNN architecture",
            source_concept_text="SNN architecture",
            max_tokens=200,
        )
        prompt = builder.build_per_goal_render(context, goal)
        assert "RENDER one section" in prompt
        assert "SNN architecture" in prompt
        assert "Do not reason" in prompt

    def test_per_goal_with_prev_text(self, builder, context) -> None:
        goal = ThoughtGoal(text="Test", source_concept_text="test", max_tokens=100)
        prompt = builder.build_per_goal_render(
            context, goal, prev_text="Previous section content here."
        )
        assert "Previous section ended with" in prompt

    def test_domain_tone_coding(self, builder) -> None:
        tone = builder._domain_tone("coding")
        assert "Technical" in tone

    def test_domain_tone_unknown(self, builder) -> None:
        tone = builder._domain_tone("unknown_domain")
        assert "Clear" in tone  # Falls back to general


# ═══════════════════════════════════════════════════════════════════════════
# ShallowRenderer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestShallowRenderer:
    """Test the orchestration layer."""

    @pytest.fixture
    def renderer(self):
        return ShallowRenderer(min_confidence=0.3)

    @pytest.fixture
    def understanding(self):
        return UnderstandingState(
            concepts=[
                ComprehensionUnit(
                    text="SNN architecture",
                    embedding=np.zeros(384),
                    salience=1.2,
                ),
                ComprehensionUnit(
                    text="LIF neurons",
                    embedding=np.zeros(384),
                    salience=0.8,
                ),
            ],
            domain_activations={"coding": 0.7, "general": 0.3},
            salience_map=[1.2, 0.8],
        )

    def test_build_context(self, renderer, understanding) -> None:
        goals = [ThoughtGoal(text="test", source_concept_text="SNN architecture")]
        ctx = renderer.build_context(
            understanding, goals, "How do SNNs work?", "Base thought"
        )
        assert isinstance(ctx, RenderingContext)
        assert ctx.concept_count == 2
        assert "coding" in ctx.domain_context

    def test_should_use_shallow_high_confidence(self, renderer) -> None:
        ctx = RenderingContext(
            concepts=["test concept"],
            confidence=0.8,
        )
        assert renderer.should_use_shallow(ctx) is True

    def test_should_not_use_shallow_low_confidence(self, renderer) -> None:
        ctx = RenderingContext(
            concepts=["test concept"],
            confidence=0.1,
        )
        assert renderer.should_use_shallow(ctx) is False

    def test_should_not_use_shallow_no_concepts(self, renderer) -> None:
        ctx = RenderingContext(confidence=0.8)
        assert renderer.should_use_shallow(ctx) is False

    def test_render_prompt_full(self, renderer) -> None:
        ctx = RenderingContext(
            concepts=["concept A"],
            original_query="test query",
            confidence=0.7,
        )
        prompt = renderer.render_prompt(ctx)
        assert "RENDER" in prompt
        assert renderer.render_count == 1

    def test_render_prompt_per_goal(self, renderer) -> None:
        ctx = RenderingContext(
            concepts=["concept A"],
            original_query="test query",
        )
        goal = ThoughtGoal(text="test goal", source_concept_text="concept A")
        prompt = renderer.render_prompt(ctx, goal=goal)
        assert "RENDER one section" in prompt

    def test_stats_tracking(self, renderer) -> None:
        assert renderer.render_count == 0
        assert renderer.fallback_count == 0

        # Trigger a render
        ctx_good = RenderingContext(concepts=["A"], confidence=0.8)
        renderer.render_prompt(ctx_good)
        assert renderer.render_count == 1

        # Trigger a fallback
        ctx_bad = RenderingContext(concepts=["B"], confidence=0.1)
        renderer.should_use_shallow(ctx_bad)
        assert renderer.fallback_count == 1
        assert renderer.shallow_rate == 0.5

    def test_reset_stats(self, renderer) -> None:
        ctx = RenderingContext(concepts=["A"], confidence=0.8)
        renderer.render_prompt(ctx)
        renderer.reset_stats()
        assert renderer.render_count == 0
        assert renderer.fallback_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ExpressionStream + ShallowRenderer
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionStreamShallowIntegration:
    """Test ExpressionStream with shallow rendering."""

    @pytest.mark.asyncio
    async def test_stream_with_shallow_mode(self) -> None:
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )

        renderer = ShallowRenderer(min_confidence=0.3)

        async def mock_llm(prompt: str) -> str:
            return "SNN uses LIF neurons for temporal processing."

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            llm_generate=mock_llm,
            shallow_renderer=renderer,
            shallow_mode=True,
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
            base_thought="SNNs process temporal patterns using spiking neurons.",
            original_query="How do SNNs work?",
        )

        assert result.text
        assert result.thought_count >= 0

    @pytest.mark.asyncio
    async def test_stream_shallow_mode_false(self) -> None:
        """When shallow_mode=False, deep path is used."""
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )

        renderer = ShallowRenderer()

        stream = ExpressionStream(
            planner=ThoughtPlanner(),
            controller=ThoughtController(),
            evaluator=RewardEvaluator(),
            shallow_renderer=renderer,
            shallow_mode=False,  # Explicitly off
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
            base_thought="Test response.",
            original_query="test",
        )

        assert result.text
        # Renderer should not have been used
        assert renderer.render_count == 0

    @pytest.mark.asyncio
    async def test_stream_without_renderer(self) -> None:
        """Without shallow_renderer, deep path always used."""
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
