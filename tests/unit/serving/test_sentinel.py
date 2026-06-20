"""Tests for the Agentic Governance Layer (SentinelNode + PlannerNode policy gate)."""

import pytest

from hbllm.brain.planner_node import PlannerNode
from hbllm.brain.policy_engine import (
    Policy,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyType,
)
from hbllm.brain.sentinel_node import SentinelNode
from hbllm.network.messages import Message, MessageType

# ── Helpers ────────────────────────────────────────────────────────────────


class FakeBus:
    """Minimal bus mock for testing nodes."""

    def __init__(self):
        self.published: list[tuple[str, Message]] = []
        self.subscriptions: dict[str, list] = {}

    async def subscribe(self, topic: str, handler):
        self.subscriptions.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, message: Message):
        self.published.append((topic, message))

    async def start(self):
        pass

    async def stop(self):
        pass


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


# ── SentinelNode Tests ────────────────────────────────────────────────────


class TestSentinelNode:
    @pytest.fixture
    def sentinel(self):
        engine = make_engine_with_rules()
        node = SentinelNode(
            node_id="test_sentinel",
            policy_engine=engine,
            poll_interval=999,  # Don't auto-poll in tests
        )
        node._bus = FakeBus()
        node._running = True
        return node

    @pytest.mark.asyncio
    async def test_context_update_triggers_violation(self, sentinel):
        """Door open at night should trigger a corrective action."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"time_hour": 22, "door_state": "open"},
        )
        await sentinel._on_context_update(msg)

        # Check context was merged
        assert sentinel.current_context["time_hour"] == 22
        assert sentinel.current_context["door_state"] == "open"

        # Check corrective action published
        published = sentinel.bus.published
        action_msgs = [(t, m) for t, m in published if t == "sentinel.action"]
        assert len(action_msgs) >= 1
        assert action_msgs[0][1].payload["action"] == "lock_door"

    @pytest.mark.asyncio
    async def test_no_violation_during_day(self, sentinel):
        """Door open during the day should not trigger."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"time_hour": 14, "door_state": "open"},
        )
        await sentinel._on_context_update(msg)

        action_msgs = [(t, m) for t, m in sentinel.bus.published if t == "sentinel.action"]
        assert len(action_msgs) == 0

    @pytest.mark.asyncio
    async def test_baby_sleeping_blocks_loud(self, sentinel):
        """'Loud' context with baby sleeping should trigger."""
        # First set baby sleeping
        msg1 = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"baby_state": "sleeping"},
        )
        await sentinel._on_context_update(msg1)

        # Publishes might include notification about the state
        sentinel.bus.published.clear()

        # Now simulate a loud action check by directly evaluating
        result = sentinel.policy_engine.evaluate(
            "I will shout loudly",
            context={"baby_state": "sleeping"},
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_alert_history_recorded(self, sentinel):
        """Violations should be recorded in alert history."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"time_hour": 23, "door_state": "open"},
        )
        await sentinel._on_context_update(msg)

        assert len(sentinel.alert_history) >= 1
        alert = sentinel.alert_history[0]
        assert alert.action_taken == "corrective"
        assert "no-door-night" in alert.rule_name

    @pytest.mark.asyncio
    async def test_no_duplicate_alerts(self, sentinel):
        """Same context should not trigger duplicate alerts."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"time_hour": 22, "door_state": "open"},
        )
        await sentinel._on_context_update(msg)
        count_1 = len(sentinel.alert_history)

        await sentinel._on_context_update(msg)
        count_2 = len(sentinel.alert_history)

        assert count_2 == count_1  # No duplicate

    @pytest.mark.asyncio
    async def test_status_query(self, sentinel):
        """Handle direct status queries."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="sentinel.query",
            payload={"action": "status"},
        )
        resp = await sentinel.handle_message(msg)
        assert resp is not None
        assert "context" in resp.payload
        assert "alert_count" in resp.payload

    def test_stats(self, sentinel):
        """Stats method should return monitoring info."""
        stats = sentinel.stats()
        assert "context_keys" in stats
        assert "alert_count" in stats

    @pytest.mark.asyncio
    async def test_context_to_text(self, sentinel):
        """Should generate readable text from context dict."""
        text = sentinel._context_to_text(
            {
                "time_hour": 22,
                "door_state": "open",
                "baby_state": "sleeping",
            }
        )
        assert "door" in text.lower()
        assert "baby" in text.lower()


# ── PlannerNode Policy Gate Tests ──────────────────────────────────────────


class TestPlannerPolicyGate:
    def test_planner_accepts_policy_engine(self):
        """PlannerNode should accept a policy_engine parameter."""
        engine = make_engine_with_rules()
        planner = PlannerNode(
            node_id="test_planner",
            policy_engine=engine,
        )
        assert planner.policy_engine is engine

    def test_planner_without_policy_engine(self):
        """PlannerNode should work without a policy engine."""
        planner = PlannerNode(node_id="test_planner")
        assert planner.policy_engine is None

    def test_policy_engine_blocks_thought_content(self):
        """PolicyEngine should block thoughts that violate rules."""
        engine = make_engine_with_rules()

        # This should be blocked (finance + stranger)
        result = engine.evaluate(
            "Let me discuss your salary and bank details",
            context={"person_type": "stranger"},
        )
        assert not result.passed

        # This should pass (finance + family)
        result = engine.evaluate(
            "Let me discuss your salary and bank details",
            context={"person_type": "family"},
        )
        assert result.passed


# ── Integration: Sentinel + PolicyEngine ────────────────────────────────────


class TestAgenticIntegration:
    @pytest.mark.asyncio
    async def test_full_sentinel_flow(self):
        """End-to-end: engine → sentinel → context change → corrective action."""
        engine = PolicyEngine(context_provider=None)
        engine.add_policy(
            Policy(
                name="lights-off-midnight",
                type=PolicyType.DENY,
                pattern="light.*on|lights.*on",
                action=PolicyAction.BLOCK,
                severity="high",
                conditions=[
                    PolicyCondition("time_hour", "gte", 0),
                    PolicyCondition("time_hour", "lt", 6),
                ],
            )
        )

        sentinel = SentinelNode(
            node_id="sentinel",
            policy_engine=engine,
            poll_interval=999,
        )
        sentinel._bus = FakeBus()
        sentinel._running = True

        # 3am with lights on
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"time_hour": 3, "light_state": "on"},
        )
        await sentinel._on_context_update(msg)

        # Should have published alert/action
        assert len(sentinel.bus.published) > 0
        output_msgs = [(t, m) for t, m in sentinel.bus.published if t == "sensory.output"]
        assert len(output_msgs) >= 1  # Owner was notified

    @pytest.mark.asyncio
    async def test_sentinel_safe_context_no_action(self):
        """Safe context should not trigger any actions."""
        engine = PolicyEngine(context_provider=None)
        engine.add_policy(
            Policy(
                name="no-door-night",
                type=PolicyType.DENY,
                pattern="door.*open",
                action=PolicyAction.BLOCK,
                conditions=[PolicyCondition("time_hour", "gte", 21)],
            )
        )

        sentinel = SentinelNode(
            node_id="sentinel",
            policy_engine=engine,
            poll_interval=999,
        )
        sentinel._bus = FakeBus()
        sentinel._running = True

        # 2pm, door closed — perfectly safe
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="context.update",
            payload={"time_hour": 14, "door_state": "closed"},
        )
        await sentinel._on_context_update(msg)

        action_msgs = [(t, m) for t, m in sentinel.bus.published if t == "sentinel.action"]
        assert len(action_msgs) == 0
