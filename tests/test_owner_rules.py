"""Tests for the Owner Rule System (policy_engine + owner_rules)."""

import pytest
from hbllm.brain.policy_engine import (
    Policy, PolicyAction, PolicyCondition, PolicyEngine, PolicyType,
)
from hbllm.brain.owner_rules import OwnerRuleStore, parse_owner_rule


# ── PolicyCondition Tests ───────────────────────────────────────────────────

class TestPolicyCondition:
    def test_eq(self):
        c = PolicyCondition(key="person_type", operator="eq", value="family")
        assert c.evaluate({"person_type": "family"}) is True
        assert c.evaluate({"person_type": "stranger"}) is False

    def test_neq(self):
        c = PolicyCondition(key="person_type", operator="neq", value="family")
        assert c.evaluate({"person_type": "stranger"}) is True
        assert c.evaluate({"person_type": "family"}) is False

    def test_gte(self):
        c = PolicyCondition(key="time_hour", operator="gte", value=21)
        assert c.evaluate({"time_hour": 21}) is True
        assert c.evaluate({"time_hour": 22}) is True
        assert c.evaluate({"time_hour": 20}) is False

    def test_lt(self):
        c = PolicyCondition(key="time_hour", operator="lt", value=6)
        assert c.evaluate({"time_hour": 5}) is True
        assert c.evaluate({"time_hour": 6}) is False

    def test_missing_key(self):
        c = PolicyCondition(key="nonexistent", operator="eq", value=True)
        assert c.evaluate({}) is False

    def test_in_operator(self):
        c = PolicyCondition(key="day", operator="in", value=["monday", "tuesday"])
        assert c.evaluate({"day": "monday"}) is True
        assert c.evaluate({"day": "friday"}) is False

    def test_serialization(self):
        c = PolicyCondition(key="time_hour", operator="gte", value=21)
        d = c.to_dict()
        c2 = PolicyCondition.from_dict(d)
        assert c2.key == "time_hour"
        assert c2.operator == "gte"
        assert c2.value == 21


# ── Context-Aware PolicyEngine Tests ────────────────────────────────────────

class TestContextAwarePolicyEngine:
    def test_policy_with_no_conditions_always_fires(self):
        engine = PolicyEngine(context_provider=None)
        engine.add_policy(Policy(
            name="no-violence", type=PolicyType.DENY,
            pattern="kill|attack|harm", action=PolicyAction.BLOCK,
        ))
        result = engine.evaluate("I will harm you", context={})
        assert not result.passed

    def test_policy_fires_when_conditions_met(self):
        engine = PolicyEngine(context_provider=None)
        engine.add_policy(Policy(
            name="no-door-after-9pm", type=PolicyType.DENY,
            pattern="open.*door", action=PolicyAction.BLOCK,
            conditions=[PolicyCondition("time_hour", "gte", 21)],
        ))
        # After 9pm — should block
        result = engine.evaluate("I will open the door", context={"time_hour": 22})
        assert not result.passed

    def test_policy_skipped_when_conditions_not_met(self):
        engine = PolicyEngine(context_provider=None)
        engine.add_policy(Policy(
            name="no-door-after-9pm", type=PolicyType.DENY,
            pattern="open.*door", action=PolicyAction.BLOCK,
            conditions=[PolicyCondition("time_hour", "gte", 21)],
        ))
        # Before 9pm — should pass
        result = engine.evaluate("I will open the door", context={"time_hour": 14})
        assert result.passed

    def test_multiple_conditions_all_must_match(self):
        engine = PolicyEngine(context_provider=None)
        engine.add_policy(Policy(
            name="no-door-strangers-night", type=PolicyType.DENY,
            pattern="open.*door", action=PolicyAction.BLOCK,
            conditions=[
                PolicyCondition("time_hour", "gte", 21),
                PolicyCondition("person_type", "neq", "family"),
            ],
        ))
        # Night + stranger → block
        result = engine.evaluate(
            "I will open the door",
            context={"time_hour": 22, "person_type": "stranger"},
        )
        assert not result.passed

        # Night + family → pass
        result = engine.evaluate(
            "I will open the door",
            context={"time_hour": 22, "person_type": "family"},
        )
        assert result.passed

        # Day + stranger → pass
        result = engine.evaluate(
            "I will open the door",
            context={"time_hour": 14, "person_type": "stranger"},
        )
        assert result.passed


# ── NL Rule Parsing Tests ───────────────────────────────────────────────────

class TestOwnerRuleParsing:
    def test_never_after_time(self):
        r = parse_owner_rule("Never open the door after 9pm")
        assert r.policy_type == PolicyType.DENY
        assert r.severity in ("critical", "high")
        assert any(c.key == "time_hour" and c.operator == "gte" and c.value == 21
                    for c in r.conditions)

    def test_dont_with_person(self):
        r = parse_owner_rule("Don't discuss finances with guests")
        assert r.policy_type == PolicyType.DENY
        assert any(c.key == "person_type" for c in r.conditions)

    def test_always_when_state(self):
        r = parse_owner_rule("Always speak softly when the baby is sleeping")
        assert r.policy_type == PolicyType.TRANSFORM
        assert any(c.key == "baby_state" and c.value == "sleeping"
                    for c in r.conditions)

    def test_before_time(self):
        r = parse_owner_rule("Don't make noise before 6am")
        assert any(c.key == "time_hour" and c.operator == "lt" and c.value == 6
                    for c in r.conditions)

    def test_confidence_increases_with_specificity(self):
        vague = parse_owner_rule("be careful")
        specific = parse_owner_rule("Never open the door after 9pm for strangers")
        assert specific.confidence > vague.confidence

    def test_pattern_extraction(self):
        r = parse_owner_rule("Never discuss salary information")
        assert r.pattern  # Should have extracted a regex pattern
        assert len(r.pattern) > 0


# ── OwnerRuleStore Tests ────────────────────────────────────────────────────

class TestOwnerRuleStore:
    @pytest.fixture
    def store(self, tmp_path):
        return OwnerRuleStore(db_path=str(tmp_path / "test_rules.db"))

    def test_add_and_list(self, store):
        store.add_rule("robot-1", "Never open the door after 9pm")
        store.add_rule("robot-1", "Always speak softly when the baby is sleeping")

        rules = store.list_rules("robot-1")
        assert len(rules) == 2

    def test_remove_rule(self, store):
        rule_id, _ = store.add_rule("robot-1", "Never run in the hallway")
        assert store.remove_rule(rule_id) is True
        assert len(store.list_rules("robot-1")) == 0

    def test_toggle_rule(self, store):
        rule_id, _ = store.add_rule("robot-1", "Never yell")
        store.toggle_rule(rule_id, enabled=False)
        rules = store.list_rules("robot-1")
        assert rules[0]["enabled"] is False

    def test_tenant_isolation(self, store):
        store.add_rule("robot-1", "Rule for robot 1")
        store.add_rule("robot-2", "Rule for robot 2")
        assert len(store.list_rules("robot-1")) == 1
        assert len(store.list_rules("robot-2")) == 1

    def test_load_into_engine(self, store):
        store.add_rule("robot-1", "Never discuss finances with guests")
        store.add_rule("robot-1", "Always greet visitors politely")

        engine = PolicyEngine(context_provider=None)
        loaded = store.load_into_engine("robot-1", engine)
        assert loaded == 2
        assert engine.policy_count == 2

    def test_clear_tenant(self, store):
        store.add_rule("robot-1", "Rule 1")
        store.add_rule("robot-1", "Rule 2")
        deleted = store.clear_tenant("robot-1")
        assert deleted == 2
        assert len(store.list_rules("robot-1")) == 0


# ── Integration: Rules → Engine → Evaluation ───────────────────────────────

class TestIntegration:
    def test_owner_rule_blocks_response(self, tmp_path):
        """End-to-end: add NL rule → load into engine → evaluate → blocked."""
        store = OwnerRuleStore(db_path=str(tmp_path / "int_rules.db"))
        store.add_rule("robot-1", "Never discuss salary information")

        engine = PolicyEngine(context_provider=None)
        store.load_into_engine("robot-1", engine)

        # Text containing the action words from the rule should be blocked
        result = engine.evaluate(
            "Let me discuss salary information with you",
            context={},
        )
        assert not result.passed
        assert len(result.violations) > 0

        # Text NOT containing the action words should pass
        result = engine.evaluate(
            "The weather is nice today",
            context={},
        )
        assert result.passed

    def test_time_conditional_rule(self, tmp_path):
        """Rule only blocks after 9pm."""
        store = OwnerRuleStore(db_path=str(tmp_path / "time_rules.db"))
        store.add_rule("robot-1", "Never open the door after 9pm")

        engine = PolicyEngine(context_provider=None)
        store.load_into_engine("robot-1", engine)

        # During the day — should pass
        result = engine.evaluate(
            "I will open the door for you",
            context={"time_hour": 14},
        )
        assert result.passed

        # At night — should block
        result = engine.evaluate(
            "I will open the door for you",
            context={"time_hour": 22},
        )
        assert not result.passed
