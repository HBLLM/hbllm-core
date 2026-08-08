"""Tests for ModifierPipeline and concrete modifier implementations."""

from __future__ import annotations

import pytest

from hbllm.execution.plan import ExecutionPlan
from hbllm.execution.text.modifiers.grammar_modifier import GrammarModifier
from hbllm.execution.text.modifiers.modifier import ModifierPipeline
from hbllm.execution.text.modifiers.no_modifier import NoModifier
from hbllm.execution.text.modifiers.prompt_modifier import PromptModifier


class TestNoModifier:
    @pytest.mark.asyncio()
    async def test_pass_through(self) -> None:
        mod = NoModifier()
        plan = ExecutionPlan()

        assert mod.name == "none"
        assert mod.modifier_type == "none"
        assert await mod.is_available() is True
        assert mod.supports("any-model") is True

        # All lifecycle hooks are identity
        assert await mod.before_context(plan) is plan
        assert await mod.before_prompt("hello", plan) == "hello"
        assert await mod.after_generation("world", plan) == "world"
        assert await mod.after_validation("test", plan) == "test"


class TestPromptModifier:
    @pytest.mark.asyncio()
    async def test_style_injection(self) -> None:
        mod = PromptModifier(style="concise")
        plan = ExecutionPlan()

        result = await mod.before_prompt("What is Python?", plan)
        assert "[Style:" in result
        assert "concise" in result.lower() or "Respond concisely" in result
        assert "What is Python?" in result

    @pytest.mark.asyncio()
    async def test_custom_instruction(self) -> None:
        mod = PromptModifier(
            style="custom",
            custom_instruction="Respond only in haiku format",
        )
        plan = ExecutionPlan()

        result = await mod.before_prompt("What is life?", plan)
        assert "haiku" in result
        assert "What is life?" in result

    def test_name(self) -> None:
        mod = PromptModifier(style="formal")
        assert mod.name == "prompt-formal"
        assert mod.modifier_type == "prompt"

    @pytest.mark.asyncio()
    async def test_always_available(self) -> None:
        mod = PromptModifier(style="concise")
        assert await mod.is_available() is True
        assert mod.supports("any-model") is True


class TestGrammarModifier:
    @pytest.mark.asyncio()
    async def test_json_instruction_injection(self) -> None:
        mod = GrammarModifier(format_type="json")
        plan = ExecutionPlan()

        result = await mod.before_prompt("List colors", plan)
        assert "JSON" in result
        assert "List colors" in result

    @pytest.mark.asyncio()
    async def test_json_with_schema(self) -> None:
        schema = {"type": "object", "properties": {"color": {"type": "string"}}}
        mod = GrammarModifier(schema=schema, format_type="json")
        plan = ExecutionPlan()

        result = await mod.before_prompt("List colors", plan)
        assert "schema" in result.lower()
        assert "color" in result

    @pytest.mark.asyncio()
    async def test_valid_json_passes(self) -> None:
        mod = GrammarModifier(format_type="json")
        plan = ExecutionPlan()
        await mod.before_generation(plan)

        result = await mod.after_validation('{"key": "value"}', plan)
        assert result == '{"key": "value"}'

    @pytest.mark.asyncio()
    async def test_json_extraction_from_markdown(self) -> None:
        mod = GrammarModifier(format_type="json")
        plan = ExecutionPlan()
        await mod.before_generation(plan)

        text = 'Here is the result:\n```json\n{"color": "red"}\n```'
        result = await mod.after_validation(text, plan)
        assert result == '{"color": "red"}'

    @pytest.mark.asyncio()
    async def test_invalid_json_returns_original(self) -> None:
        mod = GrammarModifier(format_type="json")
        plan = ExecutionPlan()
        await mod.before_generation(plan)

        result = await mod.after_validation("not json at all", plan)
        assert result == "not json at all"

    def test_name(self) -> None:
        mod = GrammarModifier(format_type="json")
        assert mod.name == "grammar-json"
        assert mod.modifier_type == "grammar"


class TestModifierPipeline:
    @pytest.mark.asyncio()
    async def test_empty_pipeline(self) -> None:
        pipeline = ModifierPipeline()
        plan = ExecutionPlan()

        assert pipeline.size == 0
        result_plan = await pipeline.run_before_context(plan)
        assert result_plan is plan

    @pytest.mark.asyncio()
    async def test_single_modifier(self) -> None:
        pipeline = ModifierPipeline()
        pipeline.add(PromptModifier(style="formal"))

        plan = ExecutionPlan()
        result = await pipeline.run_before_prompt("Hello", plan)
        assert "[Style:" in result

    @pytest.mark.asyncio()
    async def test_composable_pipeline(self) -> None:
        """Multiple modifiers compose in pipeline order."""
        pipeline = ModifierPipeline()
        pipeline.add(PromptModifier(style="concise"))  # priority 50
        pipeline.add(GrammarModifier(format_type="json"))  # priority 75

        plan = ExecutionPlan()
        result = await pipeline.run_before_prompt("List items", plan)

        # Grammar (higher priority) runs first, then prompt
        assert "JSON" in result
        assert "List items" in result

    @pytest.mark.asyncio()
    async def test_priority_ordering(self) -> None:
        """Modifiers should be ordered by priority (highest first)."""
        pipeline = ModifierPipeline()

        prompt = PromptModifier(style="concise")  # priority 50
        grammar = GrammarModifier()  # priority 75

        pipeline.add(prompt)
        pipeline.add(grammar)

        assert pipeline.modifier_names[0] == "grammar-json"  # Higher priority first
        assert pipeline.modifier_names[1] == "prompt-concise"

    def test_remove(self) -> None:
        pipeline = ModifierPipeline()
        pipeline.add(NoModifier())
        pipeline.add(PromptModifier(style="formal"))

        assert pipeline.size == 2
        assert pipeline.remove("none") is True
        assert pipeline.size == 1
        assert pipeline.remove("nonexistent") is False

    @pytest.mark.asyncio()
    async def test_cleanup_handles_errors(self) -> None:
        """Cleanup should not raise even if a modifier cleanup fails."""
        pipeline = ModifierPipeline()
        pipeline.add(NoModifier())
        # NoModifier cleanup is a no-op, so this should work fine
        await pipeline.cleanup()

    @pytest.mark.asyncio()
    async def test_metrics(self) -> None:
        pipeline = ModifierPipeline()
        pipeline.add(NoModifier())
        pipeline.add(PromptModifier(style="formal"))

        metrics = pipeline.all_metrics()
        assert "none" in metrics
        assert "prompt-formal" in metrics
