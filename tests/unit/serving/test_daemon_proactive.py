"""Unit tests for CognitiveDaemon, ProactiveProcessor, and SSEChannel."""

import pytest

from hbllm.serving.daemon import CognitiveDaemon
from hbllm.serving.notifications import NotificationPriority
from hbllm.serving.proactive import ProactiveEvent, SSEChannel

# ── CognitiveDaemon Tests ────────────────────────────────────────────────────


class TestCognitiveDaemon:
    def test_init_defaults(self):
        daemon = CognitiveDaemon()
        assert daemon._brain is None
        assert daemon.host == "0.0.0.0"
        assert daemon.port == 8000
        assert daemon.serve_http is True
        assert daemon.provider == "openai/gpt-4o-mini"

    def test_custom_config(self):
        daemon = CognitiveDaemon(
            provider="anthropic/claude-sonnet-4-20250514",
            host="127.0.0.1",
            port=9999,
            serve_http=False,
            data_dir="/tmp/test_data",
        )
        assert daemon.port == 9999
        assert daemon.host == "127.0.0.1"
        assert daemon.serve_http is False
        assert daemon.data_dir == "/tmp/test_data"

    def test_local_mode(self):
        daemon = CognitiveDaemon(local=True, model_size="1.5b")
        assert daemon.local is True
        assert daemon.model_size == "1.5b"

    def test_snapshot_before_start(self):
        daemon = CognitiveDaemon()
        snap = daemon.snapshot()
        assert "uptime_s" in snap
        assert snap["brain_nodes"] == 0

    @pytest.mark.asyncio
    async def test_stop_before_start(self):
        daemon = CognitiveDaemon()
        # Should not raise even when nothing was started
        await daemon.stop()


# ── SSEChannel Tests ─────────────────────────────────────────────────────────


class TestSSEChannel:
    def test_init(self):
        channel = SSEChannel()
        assert len(channel._queues) == 0

    @pytest.mark.asyncio
    async def test_push_creates_queue(self):
        channel = SSEChannel()
        event = ProactiveEvent(
            tenant_id="tenant1",
            title="Test",
            body="Hello",
            priority=NotificationPriority.INFO,
        )
        result = await channel.push(event)
        assert result is True
        assert "tenant1" in channel._queues
        assert channel._queues["tenant1"].qsize() == 1

    @pytest.mark.asyncio
    async def test_push_multiple(self):
        channel = SSEChannel()
        for i in range(3):
            await channel.push(
                ProactiveEvent(
                    tenant_id="tenant1",
                    title=f"Test {i}",
                    body=f"Body {i}",
                    priority=NotificationPriority.INFO,
                ),
            )
        assert channel._queues["tenant1"].qsize() == 3

    @pytest.mark.asyncio
    async def test_push_different_tenants(self):
        channel = SSEChannel()
        for tenant in ["t1", "t2", "t3"]:
            await channel.push(
                ProactiveEvent(
                    tenant_id=tenant,
                    title="Test",
                    body="Body",
                    priority=NotificationPriority.INFO,
                ),
            )
        assert len(channel._queues) == 3

    def test_remove_tenant(self):
        channel = SSEChannel()
        channel.get_queue("tenant1")
        assert "tenant1" in channel._queues
        channel.remove_tenant("tenant1")
        assert "tenant1" not in channel._queues

    @pytest.mark.asyncio
    async def test_push_full_queue_drops(self):
        channel = SSEChannel(max_queue_size=2)
        for i in range(3):
            result = await channel.push(
                ProactiveEvent(
                    tenant_id="tenant1",
                    title=f"Test {i}",
                    body=f"Body {i}",
                    priority=NotificationPriority.INFO,
                ),
            )
        # Third push should fail (queue full)
        assert result is False


# ── ProactiveEvent Tests ─────────────────────────────────────────────────────


class TestProactiveEvent:
    def test_creation(self):
        e = ProactiveEvent(
            tenant_id="user1",
            title="Alert",
            body="Something happened",
            priority=NotificationPriority.HIGH,
        )
        assert e.title == "Alert"
        assert e.priority == NotificationPriority.HIGH

    def test_to_dict(self):
        e = ProactiveEvent(
            tenant_id="user1",
            title="Test",
            body="Body",
        )
        d = e.to_dict()
        assert d["tenant_id"] == "user1"
        assert d["title"] == "Test"
        assert "created_at" in d

    def test_priority_values(self):
        assert NotificationPriority.SUGGESTION.value == "suggestion"
        assert NotificationPriority.INFO.value == "info"
        assert NotificationPriority.HIGH.value == "high"
        assert NotificationPriority.CRITICAL.value == "critical"
