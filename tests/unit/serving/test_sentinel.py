"""Tests for the HCIR Policy Governance Engine."""

from __future__ import annotations

from hbllm.hcir.kernel.governance.policies.policy_engine import (
    Policy,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyType,
)


def make_engine_with_rules() -> PolicyEngine:
    """Create a PolicyEngine with test rules."""
    engine = PolicyEngine(context_provider=None)
    engine.add_policy(
        Policy(
            name="no-door-night",
            type=PolicyType.DENY,
            pattern="door.*open|open.*door",
            action=PolicyAction.BLOCK,
            severity="critical",
            conditions=[PolicyCondition("time_hour", "gte", 21)],
        )
    )
    engine.add_policy(
        Policy(
            name="quiet-baby",
            type=PolicyType.DENY,
            pattern="loud|yell|shout",
            action=PolicyAction.BLOCK,
            severity="high",
            conditions=[PolicyCondition("baby_state", "eq", "sleeping")],
        )
    )
    engine.add_policy(
        Policy(
            name="no-finance-strangers",
            type=PolicyType.DENY,
            pattern="salary|bank|mortgage",
            action=PolicyAction.BLOCK,
            severity="high",
            conditions=[PolicyCondition("person_type", "neq", "family")],
        )
    )
    return engine


class TestPolicyEngineGovernance:
    def test_policy_engine_blocks_matching_content_and_context(self) -> None:
        """PolicyEngine should block content that violates active conditional rules."""
        engine = make_engine_with_rules()

        result = engine.evaluate(
            "Let me discuss your salary and bank details",
            context={"person_type": "stranger"},
        )
        assert not result.passed

        result = engine.evaluate(
            "Let me discuss your salary and bank details",
            context={"person_type": "family"},
        )
        assert result.passed

    def test_conditional_night_rule(self) -> None:
        engine = make_engine_with_rules()
        assert not engine.evaluate("door open", context={"time_hour": 22}).passed
        assert engine.evaluate("door open", context={"time_hour": 14}).passed
