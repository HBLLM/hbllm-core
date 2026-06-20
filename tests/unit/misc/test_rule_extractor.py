"""
Tests for RuleExtractorNode — pattern mining from high-salience events.
"""

import asyncio

import pytest

from hbllm.brain.rule_extractor import (
    ExtractedRule,
    RuleExtractorNode,
    extract_rules_from_text,
)
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── Standalone rule extraction ───────────────────────────────────────────────


class TestExtractRulesFromText:
    def test_when_then_pattern(self):
        rules = extract_rules_from_text(
            "When the system detects high memory usage, then it should trigger garbage collection."
        )
        assert len(rules) >= 1
        condition, action, confidence = rules[0]
        assert "memory" in condition.lower()
        assert "garbage collection" in action.lower()
        assert confidence > 0.4

    def test_always_leads_to_pattern(self):
        rules = extract_rules_from_text(
            "High CPU temperature always leads to thermal throttling and reduced performance."
        )
        assert len(rules) >= 1
        assert any("leads to" in r[1].lower() or "thermal" in r[1].lower() for r in rules)

    def test_user_prefers_pattern(self):
        rules = extract_rules_from_text("User prefers dark mode interface when working at night.")
        assert len(rules) >= 1

    def test_requires_pattern(self):
        rules = extract_rules_from_text(
            "The authentication module requires valid JWT tokens for all protected endpoints."
        )
        assert len(rules) >= 1

    def test_short_phrases_filtered(self):
        rules = extract_rules_from_text("If x, then y.")
        assert len(rules) == 0  # Too short

    def test_no_rules_in_plain_text(self):
        rules = extract_rules_from_text("The weather is sunny today and birds are singing.")
        assert len(rules) == 0

    def test_multiple_rules(self):
        text = (
            "When the server restarts unexpectedly, then check the system logs immediately. "
            "The database connection pool usually leads to connection timeout errors. "
            "If response latency exceeds threshold, then scale up the service."
        )
        rules = extract_rules_from_text(text)
        assert len(rules) >= 2


# ── RuleExtractorNode integration ────────────────────────────────────────────


class TestRuleExtractorNode:
    @pytest.fixture
    async def extractor_system(self):
        bus = InProcessBus()
        await bus.start()
        node = RuleExtractorNode(
            node_id="rule_extractor",
            extraction_interval=60.0,  # Won't auto-trigger in test
            min_events_for_extraction=1,
        )
        await node.start(bus)
        yield bus, node
        await node.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_priority_event_buffered(self, extractor_system):
        bus, node = extractor_system

        # Publish a high-salience event
        await bus.publish(
            "system.salience",
            Message(
                type=MessageType.SALIENCE_SCORE,
                source_node_id="test",
                topic="system.salience",
                payload={
                    "is_priority": True,
                    "content": "When the cache expires, then the system should refresh data from the database.",
                    "score": 0.85,
                    "message_id": "msg_001",
                },
            ),
        )
        await asyncio.sleep(0.1)

        assert len(node._priority_buffer) == 1

    @pytest.mark.asyncio
    async def test_non_priority_ignored(self, extractor_system):
        bus, node = extractor_system

        await bus.publish(
            "system.salience",
            Message(
                type=MessageType.SALIENCE_SCORE,
                source_node_id="test",
                topic="system.salience",
                payload={"is_priority": False, "content": "Normal event", "score": 0.3},
            ),
        )
        await asyncio.sleep(0.1)

        assert len(node._priority_buffer) == 0

    @pytest.mark.asyncio
    async def test_manual_extraction(self, extractor_system):
        bus, node = extractor_system

        # Buffer a priority event with rule-like content
        node._priority_buffer.append(
            {
                "content": "When authentication fails repeatedly, then the account should be locked automatically.",
                "score": 0.9,
                "message_id": "msg_002",
                "timestamp": 1234567890,
            }
        )

        # Run extraction
        new_rules = await node._run_extraction()
        assert new_rules >= 1
        assert len(node.rules) >= 1

        # Check rule structure
        rule = node.rules[0]
        assert rule.condition
        assert rule.action
        assert 0.0 < rule.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_duplicate_rule_increments_occurrences(self, extractor_system):
        bus, node = extractor_system

        content = "When the server crashes unexpectedly, then restart the service immediately."

        for _ in range(3):
            node._priority_buffer.append(
                {
                    "content": content,
                    "score": 0.9,
                    "message_id": "msg_003",
                    "timestamp": 1234567890,
                }
            )
            await node._run_extraction()

        # Should have 1 rule with occurrences >= 2
        assert len(node._rules) >= 1
        rule = list(node._rules.values())[0]
        assert rule.occurrences >= 2

    @pytest.mark.asyncio
    async def test_publishes_new_rules_to_bus(self, extractor_system):
        bus, node = extractor_system

        published = []

        async def capture(msg):
            published.append(msg)

        await bus.subscribe("system.rules.new", capture)

        node._priority_buffer.append(
            {
                "content": "If the request queue depth exceeds the limit, then enable load shedding to protect the system.",
                "score": 0.85,
                "message_id": "msg_004",
                "timestamp": 1234567890,
            }
        )

        await node._run_extraction()
        await asyncio.sleep(0.1)

        assert len(published) >= 1
        assert "condition" in published[0].payload
        assert "action" in published[0].payload

    @pytest.mark.asyncio
    async def test_rule_to_dict(self, extractor_system):
        bus, node = extractor_system

        rule = ExtractedRule(
            rule_id="abc123",
            condition="server load exceeds 90%",
            action="scale horizontally",
            confidence=0.8,
            occurrences=3,
        )
        d = rule.to_dict()
        assert d["rule_id"] == "abc123"
        assert d["confidence"] == 0.8
        assert d["occurrences"] == 3
