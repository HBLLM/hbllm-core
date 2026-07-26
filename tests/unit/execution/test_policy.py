"""Tests for GenerationPolicy and PolicyRule evaluation."""

from __future__ import annotations

from hbllm.execution.policy import (
    GenerationPolicy,
    PolicyCondition,
    PolicyRule,
    SystemState,
)


class TestPolicyCondition:
    def test_empty_condition_always_true(self) -> None:
        condition = PolicyCondition()
        state = SystemState()
        assert condition.evaluate(state) is True

    def test_vram_check(self) -> None:
        condition = PolicyCondition(min_vram_gb=4.0)
        assert condition.evaluate(SystemState(available_vram_gb=8.0)) is True
        assert condition.evaluate(SystemState(available_vram_gb=2.0)) is False

    def test_battery_check(self) -> None:
        condition = PolicyCondition(battery_above=0.2)
        assert condition.evaluate(SystemState(battery_level=0.5)) is True
        assert condition.evaluate(SystemState(battery_level=0.1)) is False

    def test_provider_check(self) -> None:
        condition = PolicyCondition(provider="local")
        assert condition.evaluate(SystemState(active_provider="local")) is True
        assert condition.evaluate(SystemState(active_provider="openai")) is False

    def test_tenant_check(self) -> None:
        condition = PolicyCondition(tenant_id="alice")
        assert condition.evaluate(SystemState(), tenant_id="alice") is True
        assert condition.evaluate(SystemState(), tenant_id="bob") is False

    def test_concurrent_check(self) -> None:
        condition = PolicyCondition(max_concurrent=5)
        assert condition.evaluate(SystemState(concurrent_executions=3)) is True
        assert condition.evaluate(SystemState(concurrent_executions=5)) is False

    def test_combined_conditions(self) -> None:
        condition = PolicyCondition(min_vram_gb=4.0, battery_above=0.3)
        state = SystemState(available_vram_gb=8.0, battery_level=0.5)
        assert condition.evaluate(state) is True

        state_low_battery = SystemState(available_vram_gb=8.0, battery_level=0.1)
        assert condition.evaluate(state_low_battery) is False


class TestGenerationPolicy:
    def test_default_policy(self) -> None:
        policy = GenerationPolicy.default()
        state = SystemState()
        assert policy.resolve_modifiers(state) == []

    def test_single_rule_match(self) -> None:
        policy = GenerationPolicy(
            rules=[
                PolicyRule(
                    condition=PolicyCondition(min_vram_gb=4.0),
                    modifiers=["lora-medical"],
                    priority=10,
                ),
            ],
            default_modifiers=["none"],
        )
        state = SystemState(available_vram_gb=8.0)
        assert policy.resolve_modifiers(state) == ["lora-medical"]

    def test_rule_no_match_uses_default(self) -> None:
        policy = GenerationPolicy(
            rules=[
                PolicyRule(
                    condition=PolicyCondition(min_vram_gb=16.0),
                    modifiers=["lora-large"],
                    priority=10,
                ),
            ],
            default_modifiers=["prompt-basic"],
        )
        state = SystemState(available_vram_gb=4.0)
        assert policy.resolve_modifiers(state) == ["prompt-basic"]

    def test_priority_ordering(self) -> None:
        policy = GenerationPolicy(
            rules=[
                PolicyRule(
                    condition=PolicyCondition(),
                    modifiers=["low-priority"],
                    priority=1,
                ),
                PolicyRule(
                    condition=PolicyCondition(),
                    modifiers=["high-priority"],
                    priority=10,
                ),
            ],
        )
        state = SystemState()
        # Highest priority rule should win
        assert policy.resolve_modifiers(state) == ["high-priority"]

    def test_resolve_provider(self) -> None:
        policy = GenerationPolicy(
            rules=[
                PolicyRule(
                    condition=PolicyCondition(battery_above=0.5),
                    modifiers=[],
                    provider_preference="local",
                    priority=10,
                ),
                PolicyRule(
                    condition=PolicyCondition(),
                    modifiers=[],
                    provider_preference="openai",
                    priority=1,
                ),
            ],
        )
        # High battery → prefer local
        assert policy.resolve_provider(SystemState(battery_level=0.8)) == "local"
        # Low battery → fallback to openai
        assert policy.resolve_provider(SystemState(battery_level=0.3)) == "openai"
