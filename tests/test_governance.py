"""Tests for Security, Policy Engine, and Durable Bus."""

import asyncio
import os
import tempfile
import time

import pytest
import yaml

from hbllm.serving.security import (
    ApiKeyManager, ApiKey, RateLimiter, TokenBucket, InputSanitizer,
)
from hbllm.brain.policy_engine import (
    PolicyEngine, Policy, PolicyType, PolicyAction, PolicyResult,
)
from hbllm.network.durable_bus import DurableBus, MessageStatus
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


# ═══════════════════════════════════════════════════════════════════════════════
# Security Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiKeyManager:
    def test_add_and_validate(self):
        mgr = ApiKeyManager()
        mgr.add_key("sk-test-123", tenant_id="t1", name="test")
        key = mgr.validate("sk-test-123")
        assert key is not None
        assert key.tenant_id == "t1"

    def test_invalid_key(self):
        mgr = ApiKeyManager()
        mgr.add_key("sk-real", tenant_id="t1")
        assert mgr.validate("sk-wrong") is None

    def test_disabled(self):
        mgr = ApiKeyManager()
        mgr.enabled = False
        key = mgr.validate("anything")
        assert key is not None
        assert "*" in key.scopes

    def test_has_scope(self):
        mgr = ApiKeyManager()
        key = mgr.add_key("sk-1", "t1", scopes=["chat", "health"])
        assert mgr.has_scope(key, "chat")
        assert not mgr.has_scope(key, "admin")

    def test_hash_deterministic(self):
        h1 = ApiKeyManager.hash_key("test")
        h2 = ApiKeyManager.hash_key("test")
        assert h1 == h2

    def test_key_count(self):
        mgr = ApiKeyManager()
        assert mgr.key_count == 0
        mgr.add_key("k1", "t1")
        mgr.add_key("k2", "t2")
        assert mgr.key_count == 2


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=10)
        allowed, _ = rl.check("t1")
        assert allowed is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=2)
        rl.check("t1")
        rl.check("t1")
        allowed, retry = rl.check("t1")
        assert allowed is False
        assert retry > 0

    def test_tenant_isolation(self):
        rl = RateLimiter(burst_size=1)
        rl.check("t1")
        allowed, _ = rl.check("t2")  # different tenant
        assert allowed is True

    def test_disabled(self):
        rl = RateLimiter(burst_size=1)
        rl.enabled = False
        for _ in range(100):
            allowed, _ = rl.check("t1")
            assert allowed

    def test_reset(self):
        rl = RateLimiter(burst_size=1)
        rl.check("t1")
        rl.check("t1")
        rl.reset("t1")
        allowed, _ = rl.check("t1")
        assert allowed


class TestInputSanitizer:
    def test_normal_input(self):
        s = InputSanitizer()
        text, warnings = s.sanitize("Hello, how are you?")
        assert text == "Hello, how are you?"
        assert len(warnings) == 0

    def test_length_truncation(self):
        s = InputSanitizer(max_length=10)
        text, warnings = s.sanitize("a" * 100)
        assert len(text) <= 10
        assert any("truncated" in w for w in warnings)

    def test_html_stripping(self):
        s = InputSanitizer()
        text, warnings = s.sanitize("Hello <b>world</b>")
        assert "<b>" not in text
        assert "html_stripped" in warnings

    def test_script_injection_detection(self):
        s = InputSanitizer()
        _, warnings = s.sanitize('Hello <script>alert("xss")</script>')
        assert any("injection" in w for w in warnings)

    def test_empty_input(self):
        s = InputSanitizer()
        text, warnings = s.sanitize("")
        assert text == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyEngine:
    def test_deny_policy_blocks(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="no_bombs",
            type=PolicyType.DENY,
            action=PolicyAction.BLOCK,
            pattern="(?i)how to make a bomb",
        ))
        result = engine.evaluate("Here's how to make a bomb")
        assert not result.passed
        assert len(result.violations) > 0

    def test_deny_policy_allows_clean(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="no_bombs",
            type=PolicyType.DENY,
            action=PolicyAction.BLOCK,
            pattern="(?i)how to make a bomb",
        ))
        result = engine.evaluate("The weather is nice today")
        assert result.passed

    def test_deny_warn(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="legal_flag",
            type=PolicyType.DENY,
            action=PolicyAction.WARN,
            pattern="(?i)you should sue",
        ))
        result = engine.evaluate("You should sue them")
        assert result.passed  # Warn doesn't block
        assert len(result.warnings) > 0

    def test_require_policy_append(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="need_disclaimer",
            type=PolicyType.REQUIRE,
            action=PolicyAction.APPEND,
            pattern="(?i)disclaimer",
            content="⚠️ This is not legal advice.",
        ))
        result = engine.evaluate("You should do X")  # No disclaimer
        assert "⚠️" in result.modified_text

    def test_transform_append(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="always_footer",
            type=PolicyType.TRANSFORM,
            action=PolicyAction.APPEND,
            content="— Generated by HBLLM",
        ))
        result = engine.evaluate("Some response")
        assert "— Generated by HBLLM" in result.modified_text

    def test_transform_prepend(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="header",
            type=PolicyType.TRANSFORM,
            action=PolicyAction.PREPEND,
            content="🤖 AI Response:",
        ))
        result = engine.evaluate("Hello")
        assert result.modified_text.startswith("🤖 AI Response:")

    def test_scope_restricts_domain(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="demo_scope",
            type=PolicyType.SCOPE,
            action=PolicyAction.RESTRICT,
            domains=["general"],
            tenant_ids=["demo"],
        ))
        result = engine.evaluate("coding help", tenant_id="demo", domain="coding")
        assert not result.passed
        assert "coding" in result.violations[0]

    def test_scope_allows_permitted_domain(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="demo_scope",
            type=PolicyType.SCOPE,
            action=PolicyAction.RESTRICT,
            domains=["general"],
            tenant_ids=["demo"],
        ))
        result = engine.evaluate("hello", tenant_id="demo", domain="general")
        assert result.passed

    def test_tenant_filtering(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="t1_only",
            type=PolicyType.DENY,
            action=PolicyAction.BLOCK,
            pattern="secret",
            tenant_ids=["t1"],
        ))
        # Applies to t1
        r1 = engine.evaluate("This is secret", tenant_id="t1")
        assert not r1.passed
        # Doesn't apply to t2
        r2 = engine.evaluate("This is secret", tenant_id="t2")
        assert r2.passed

    def test_priority_ordering(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(name="low", type=PolicyType.TRANSFORM, action=PolicyAction.APPEND, content="LOW", priority=1))
        engine.add_policy(Policy(name="high", type=PolicyType.TRANSFORM, action=PolicyAction.APPEND, content="HIGH", priority=10))
        result = engine.evaluate("text")
        # High priority applied first
        assert result.applied_policies[0] == "high"

    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"policies": [
                {"name": "test_deny", "type": "deny", "action": "block", "pattern": "bad"},
            ]}, f)
            f.flush()
            engine = PolicyEngine()
            loaded = engine.load_from_yaml(f.name)
            assert loaded == 1
            assert engine.policy_count == 1
            os.unlink(f.name)

    def test_load_real_policies_yaml(self):
        engine = PolicyEngine()
        loaded = engine.load_from_yaml("config/policies.yaml")
        assert loaded > 0

    def test_list_policies(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(name="p1", type=PolicyType.DENY))
        policies = engine.list_policies()
        assert len(policies) == 1
        assert policies[0]["name"] == "p1"

    def test_remove_policy(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(name="p1", type=PolicyType.DENY))
        assert engine.remove_policy("p1")
        assert engine.policy_count == 0

    def test_disabled_policy_skipped(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            name="disabled",
            type=PolicyType.DENY,
            action=PolicyAction.BLOCK,
            pattern="bad",
            enabled=False,
        ))
        result = engine.evaluate("this is bad")
        assert result.passed  # Policy disabled


# ═══════════════════════════════════════════════════════════════════════════════
# Durable Bus Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDurableBus:
    @pytest.fixture
    async def durable(self, tmp_path):
        db_path = str(tmp_path / "test_messages.db")
        bus = DurableBus(inner=InProcessBus(), db_path=db_path, max_retries=2)
        await bus.start()
        yield bus
        await bus.stop()

    async def test_publish_and_deliver(self, durable):
        received = []
        async def handler(msg):
            received.append(msg)
        await durable.subscribe("test.topic", handler)
        msg = Message(type=MessageType.EVENT, source_node_id="s1", topic="test.topic", payload={"x": 1})
        await durable.publish("test.topic", msg)
        await asyncio.sleep(0.1)
        assert len(received) == 1

    async def test_stats(self, durable):
        msg = Message(type=MessageType.EVENT, source_node_id="s1", topic="t", payload={})
        await durable.publish("t", msg)
        stats = durable.stats()
        assert stats["inner_bus"] == "InProcessBus"
        assert stats["delivered"] >= 0

    async def test_dead_letter_count(self, durable):
        assert durable.dead_letter_count() == 0

    async def test_pending_count(self, durable):
        assert durable.pending_count() >= 0

    async def test_get_dead_letters(self, durable):
        letters = durable.get_dead_letters()
        assert isinstance(letters, list)

    async def test_deduplication(self, durable):
        received = []
        async def handler(msg):
            received.append(msg)
        await durable.subscribe("dedup", handler)
        msg = Message(id="same-id", type=MessageType.EVENT, source_node_id="s1", topic="dedup", payload={})
        await durable.publish("dedup", msg)
        await durable.publish("dedup", msg)  # duplicate
        await asyncio.sleep(0.1)
        assert len(received) == 1


class TestTokenBucket:
    def test_consume(self):
        bucket = TokenBucket(capacity=5, refill_rate=1.0, tokens=5)
        assert bucket.consume()
        assert bucket.consume()

    def test_exhaustion(self):
        bucket = TokenBucket(capacity=1, refill_rate=0.1, tokens=1)
        assert bucket.consume()
        assert not bucket.consume()

    def test_wait_time(self):
        bucket = TokenBucket(capacity=1, refill_rate=1.0, tokens=0)
        assert bucket.wait_time > 0


class TestPolicyResult:
    def test_to_dict(self):
        r = PolicyResult(passed=True, original_text="a", modified_text="b")
        d = r.to_dict()
        assert d["passed"] is True

    def test_is_truncated(self):
        r = PolicyResult(passed=True, original_text="a", modified_text="b")
        assert r.passed
