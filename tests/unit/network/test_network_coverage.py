"""
Network & Transport — Integration test coverage.

Covers uncovered lines in:
  - hbllm/network/degraded.py
  - hbllm/network/rate_limiter.py
  - hbllm/network/metrics.py
  - hbllm/network/serialization.py
  - hbllm/network/federation/cipher.py
  - hbllm/network/health.py
  - hbllm/network/registry.py
  - hbllm/network/plugin_manager.py
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

# ═══════════════════════════════════════════════════════════════════════
# network/serialization.py
# ═══════════════════════════════════════════════════════════════════════


class TestSerialization:
    def _make_message(self):
        from hbllm.network.messages import Message, MessageType

        return Message(
            type=MessageType.QUERY,
            source_node_id="test_node",
            topic="test.topic",
            payload={"key": "value", "count": 42},
        )

    def test_json_roundtrip(self):
        from hbllm.network.serialization import JsonSerializer

        s = JsonSerializer()
        msg = self._make_message()
        data = s.serialize(msg)
        assert isinstance(data, bytes)
        restored = s.deserialize(data)
        assert restored.topic == "test.topic"
        assert restored.payload["key"] == "value"

    def test_msgpack_roundtrip(self):
        from hbllm.network.serialization import MsgpackSerializer

        s = MsgpackSerializer()
        msg = self._make_message()
        data = s.serialize(msg)
        assert isinstance(data, bytes)
        restored = s.deserialize(data)
        assert restored.topic == "test.topic"

    def test_get_serializer_json(self):
        from hbllm.network.serialization import JsonSerializer, get_serializer

        s = get_serializer("json")
        assert isinstance(s, JsonSerializer)

    def test_get_serializer_msgpack(self):
        from hbllm.network.serialization import MsgpackSerializer, get_serializer

        s = get_serializer("msgpack")
        assert isinstance(s, MsgpackSerializer)

    def test_get_serializer_protobuf(self):
        from hbllm.network.serialization import ProtobufSerializer, get_serializer

        s = get_serializer("protobuf")
        assert isinstance(s, ProtobufSerializer)

    def test_get_serializer_default(self):
        from hbllm.network.serialization import JsonSerializer, get_serializer

        s = get_serializer("unknown_format")
        assert isinstance(s, JsonSerializer)


# ═══════════════════════════════════════════════════════════════════════
# network/federation/cipher.py
# ═══════════════════════════════════════════════════════════════════════


class TestEnvelopeCipher:
    def test_generate_keypair(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        cipher = EnvelopeCipher()
        assert cipher.public_key_hex is not None
        assert len(cipher.public_key_hex) > 0

    def test_sign_payload(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        cipher = EnvelopeCipher()
        payload = {"action": "test", "timestamp": time.time()}
        sig = cipher.sign_payload(payload)
        assert isinstance(sig, str) and len(sig) > 0

    def test_verify_envelope_valid(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        cipher = EnvelopeCipher()
        payload = {"action": "test", "timestamp": time.time()}
        sig = cipher.sign_payload(payload)
        assert cipher.verify_envelope(cipher.public_key_hex, payload, sig)

    def test_verify_envelope_expired(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        cipher = EnvelopeCipher()
        payload = {"action": "test", "timestamp": time.time() - 600}
        sig = cipher.sign_payload(payload)
        assert not cipher.verify_envelope(cipher.public_key_hex, payload, sig)

    def test_verify_envelope_bad_signature(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        cipher = EnvelopeCipher()
        payload = {"action": "test", "timestamp": time.time()}
        assert not cipher.verify_envelope(cipher.public_key_hex, payload, "badsig==")

    def test_pack_envelope(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        cipher = EnvelopeCipher()
        other = EnvelopeCipher()
        envelope = cipher.pack_envelope(other.public_key_hex, "test.topic", {"data": "hello"})
        assert "envelope" in envelope and "signature" in envelope
        assert envelope["envelope"]["topic"] == "test.topic"
        assert envelope["envelope"]["sender"] == cipher.public_key_hex

    def test_from_private_key_bytes(self):
        from hbllm.network.federation.cipher import EnvelopeCipher

        # Generate then recreate from same key bytes
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519

            key = ed25519.Ed25519PrivateKey.generate()
            raw = key.private_bytes_raw()
            cipher = EnvelopeCipher(private_key_bytes=raw)
            assert cipher.public_key_hex is not None
        except ImportError:
            pytest.skip("cryptography not installed")


# ═══════════════════════════════════════════════════════════════════════
# network/rate_limiter.py (bus-level interceptor)
# ═══════════════════════════════════════════════════════════════════════


class TestRateLimitInterceptor:
    @pytest.mark.asyncio
    async def test_system_tenant_bypasses(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=1)
        msg = Message(type=MessageType.EVENT, source_node_id="sys", tenant_id="system", topic="t")
        result = await interceptor.intercept(msg)
        assert result is msg

    @pytest.mark.asyncio
    async def test_empty_tenant_bypasses(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=1)
        msg = Message(type=MessageType.EVENT, source_node_id="sys", tenant_id="", topic="t")
        result = await interceptor.intercept(msg)
        assert result is msg

    @pytest.mark.asyncio
    async def test_audio_topic_bypasses(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=1)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="audio",
            tenant_id="t1",
            topic="sensory.audio.chunk",
        )
        result = await interceptor.intercept(msg)
        assert result is msg

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=60, burst_multiplier=2.0)
        msg = Message(type=MessageType.QUERY, source_node_id="n", tenant_id="t1", topic="q")
        result = await interceptor.intercept(msg)
        assert result is msg

    @pytest.mark.asyncio
    async def test_drops_when_exhausted(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=1, burst_multiplier=1.0)
        msg = Message(type=MessageType.QUERY, source_node_id="n", tenant_id="t1", topic="q")
        # First should pass (burst = 1)
        r1 = await interceptor.intercept(msg)
        assert r1 is msg
        # Second should be dropped (no tokens left)
        r2 = await interceptor.intercept(msg)
        assert r2 is None

    @pytest.mark.asyncio
    async def test_per_tenant_isolation(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=1, burst_multiplier=1.0)
        msg_a = Message(type=MessageType.QUERY, source_node_id="n", tenant_id="t1", topic="q")
        msg_b = Message(type=MessageType.QUERY, source_node_id="n", tenant_id="t2", topic="q")
        await interceptor.intercept(msg_a)  # consume t1 tokens
        result = await interceptor.intercept(msg_b)  # t2 should still have tokens
        assert result is msg_b

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        from hbllm.network.messages import Message, MessageType
        from hbllm.network.rate_limiter import RateLimitInterceptor

        interceptor = RateLimitInterceptor(target_rpm=60, max_tracked_tenants=2)
        for i in range(5):
            msg = Message(
                type=MessageType.QUERY, source_node_id="n", tenant_id=f"tenant_{i}", topic="q"
            )
            await interceptor.intercept(msg)
        # Only 2 tenants should remain tracked
        assert len(interceptor._locks) <= 2


# ═══════════════════════════════════════════════════════════════════════
# network/metrics.py
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsCollector:
    def test_singleton(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m1 = MetricsCollector.get_instance()
        m2 = MetricsCollector.get_instance()
        assert m1 is m2
        MetricsCollector.reset()

    def test_record_request(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.record_request("test.topic", tenant_id="t1", status="ok")
        MetricsCollector.reset()

    def test_record_message(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.record_message("test.topic", "event")
        MetricsCollector.reset()

    def test_record_error(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.record_error("n1", "timeout")
        MetricsCollector.reset()

    def test_observe_duration(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.observe_duration("workspace", 0.5)
        MetricsCollector.reset()

    def test_observe_node_latency(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.observe_node_latency("n1", 0.1)
        MetricsCollector.reset()

    def test_set_active_nodes(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.set_active_nodes(10)
        MetricsCollector.reset()

    def test_set_healthy_nodes(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.set_healthy_nodes(8)
        MetricsCollector.reset()

    def test_inc_dec_active_requests(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        m.inc_active_requests()
        m.dec_active_requests()
        MetricsCollector.reset()

    def test_backend_type(self):
        from hbllm.network.metrics import MetricsCollector

        MetricsCollector.reset()
        m = MetricsCollector.get_instance()
        assert m.backend in ("prometheus", "inmemory")
        MetricsCollector.reset()


# ═══════════════════════════════════════════════════════════════════════
# network/degraded.py
# ═══════════════════════════════════════════════════════════════════════


class TestSystemCapabilities:
    def test_fully_operational(self):
        from hbllm.network.degraded import SystemCapabilities

        caps = SystemCapabilities(
            available={"memory", "chat"}, degraded={}, offline={}, total_nodes=10, healthy_nodes=10
        )
        assert caps.is_fully_operational
        assert caps.operational_percentage == 100.0
        assert "✅" in caps.status_summary()

    def test_degraded_state(self):
        from hbllm.network.degraded import SystemCapabilities

        caps = SystemCapabilities(
            available={"chat"},
            degraded={"memory": "⚠️ Running on fallback store"},
            offline={},
            total_nodes=10,
            healthy_nodes=8,
        )
        assert not caps.is_fully_operational
        assert caps.operational_percentage == 100.0  # available+degraded = 2/2
        summary = caps.status_summary()
        assert "⚠️" in summary

    def test_offline_state(self):
        from hbllm.network.degraded import SystemCapabilities

        caps = SystemCapabilities(
            available={"chat"},
            degraded={},
            offline={"memory": "❌ Memory node crashed"},
            total_nodes=10,
            healthy_nodes=5,
        )
        assert not caps.is_fully_operational
        assert caps.operational_percentage == 50.0  # 1 available / 2 total
        summary = caps.status_summary()
        assert "❌" in summary

    def test_empty_capabilities(self):
        from hbllm.network.degraded import SystemCapabilities

        caps = SystemCapabilities(
            available=set(), degraded={}, offline={}, total_nodes=0, healthy_nodes=0
        )
        assert caps.is_fully_operational
        assert caps.operational_percentage == 100.0


class TestDegradedModeManager:
    @pytest.mark.asyncio
    async def test_register_and_get_capabilities(self):
        from hbllm.network.degraded import DegradedModeManager

        mock_registry = AsyncMock()
        mock_registry.get_all_health.return_value = {
            "n1": MagicMock(status=MagicMock(value="healthy")),
            "n2": MagicMock(status=MagicMock(value="degraded")),
        }
        mock_registry.get_available_capabilities.return_value = {"chat", "memory"}

        mock_fallback = AsyncMock()
        mock_fallback.get_system_status.return_value = {
            "chat": {"status": "healthy"},
            "memory": {"status": "degraded", "message": "⚠️ Using fallback"},
        }

        mgr = DegradedModeManager(mock_registry, mock_fallback)
        mgr.register_expected_capability("chat")
        mgr.register_expected_capability("memory")

        caps = await mgr.get_system_capabilities()
        assert "chat" in caps.available
        assert "memory" in caps.degraded

    @pytest.mark.asyncio
    async def test_get_response_disclaimer_healthy(self):
        from hbllm.network.degraded import DegradedModeManager

        mock_registry = AsyncMock()
        mock_fallback = AsyncMock()
        mock_fallback.resolve.return_value = MagicMock(is_fallback=False)

        mgr = DegradedModeManager(mock_registry, mock_fallback)
        disclaimer = await mgr.get_response_disclaimer("chat")
        assert disclaimer is None

    @pytest.mark.asyncio
    async def test_get_response_disclaimer_degraded(self):
        from hbllm.network.degraded import DegradedModeManager

        mock_registry = AsyncMock()
        mock_fallback = AsyncMock()
        result = MagicMock()
        result.is_fallback = True
        result.degraded_message = "⚠️ Using fallback"
        mock_fallback.resolve.return_value = result

        mgr = DegradedModeManager(mock_registry, mock_fallback)
        disclaimer = await mgr.get_response_disclaimer("chat")
        assert disclaimer is not None
        assert "⚠️" in disclaimer

    @pytest.mark.asyncio
    async def test_get_response_disclaimer_offline(self):
        from hbllm.network.degraded import DegradedModeManager

        mock_registry = AsyncMock()
        mock_fallback = AsyncMock()
        mock_fallback.resolve.return_value = None

        mgr = DegradedModeManager(mock_registry, mock_fallback)
        disclaimer = await mgr.get_response_disclaimer("memory")
        assert disclaimer is not None
        assert "❌" in disclaimer and "offline" in disclaimer
