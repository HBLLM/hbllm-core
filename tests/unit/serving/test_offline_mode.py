"""Tests for OfflineManager — connectivity detection and graceful degradation."""

import pytest

from hbllm.serving.offline_mode import (
    ConnectivityState,
    OfflineManager,
)


@pytest.fixture
def manager():
    return OfflineManager()


@pytest.fixture
def manager_with_providers():
    return OfflineManager(
        health_endpoints={
            "openai": "https://api.openai.com/v1/models",
            "anthropic": "https://api.anthropic.com/v1/messages",
        },
        failure_threshold=2,
    )


# ── State Tests ──────────────────────────────────────────────────────────────


class TestState:
    def test_initial_state_online(self, manager):
        assert manager.state == ConnectivityState.ONLINE
        assert manager.is_online
        assert not manager.is_offline

    def test_no_providers_stays_online(self, manager):
        """Without configured providers, state is always ONLINE."""
        assert manager.is_online


# ── Provider Failure Reporting ───────────────────────────────────────────────


class TestFailureReporting:
    def test_report_failure_below_threshold(self, manager_with_providers):
        manager_with_providers.report_failure("openai", "timeout")
        # 1 failure < threshold of 2
        assert "openai" in manager_with_providers.get_healthy_providers()

    def test_report_failure_exceeds_threshold(self, manager_with_providers):
        manager_with_providers.report_failure("openai", "timeout")
        manager_with_providers.report_failure("openai", "timeout again")
        assert "openai" in manager_with_providers.get_unhealthy_providers()

    def test_report_success_resets(self, manager_with_providers):
        manager_with_providers.report_failure("openai", "err")
        manager_with_providers.report_failure("openai", "err")
        assert "openai" in manager_with_providers.get_unhealthy_providers()

        manager_with_providers.report_success("openai")
        assert "openai" in manager_with_providers.get_healthy_providers()

    def test_report_unknown_provider(self, manager_with_providers):
        # Should not raise
        manager_with_providers.report_failure("unknown_provider")
        manager_with_providers.report_success("unknown_provider")


# ── Request Queue Tests ──────────────────────────────────────────────────────


class TestRequestQueue:
    def test_queue_request(self, manager):
        manager.queue_request("r1", "openai", {"prompt": "hello"})
        assert manager.pending_count() == 1

    def test_queue_overflow(self):
        manager = OfflineManager()
        manager._max_queue = 3
        for i in range(5):
            manager.queue_request(f"r{i}", "openai", {"n": i})
        assert manager.pending_count() == 3

    @pytest.mark.asyncio
    async def test_flush_queue(self, manager):
        manager.queue_request("r1", "openai", {})
        manager.queue_request("r2", "openai", {})
        await manager._flush_queue()
        assert manager.pending_count() == 0


# ── Response Cache Tests ─────────────────────────────────────────────────────


class TestResponseCache:
    def test_cache_hit(self, manager):
        manager.cache_response("query:hello", {"answer": "world"})
        assert manager.get_cached("query:hello") == {"answer": "world"}

    def test_cache_miss(self, manager):
        assert manager.get_cached("nonexistent") is None

    def test_cache_lru_eviction(self):
        manager = OfflineManager(cache_size=2)
        manager.cache_response("a", 1)
        manager.cache_response("b", 2)
        manager.cache_response("c", 3)  # Should evict "a"
        assert manager.get_cached("a") is None
        assert manager.get_cached("b") == 2
        assert manager.get_cached("c") == 3

    def test_cache_access_refreshes_lru(self):
        manager = OfflineManager(cache_size=2)
        manager.cache_response("a", 1)
        manager.cache_response("b", 2)
        manager.get_cached("a")  # Refresh "a"
        manager.cache_response("c", 3)  # Should evict "b" (not "a")
        assert manager.get_cached("a") == 1
        assert manager.get_cached("b") is None


# ── Stats Tests ──────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_structure(self, manager_with_providers):
        stats = manager_with_providers.stats()
        assert stats["state"] == "online"
        assert "providers" in stats
        assert "openai" in stats["providers"]

    def test_stats_after_operations(self, manager):
        manager.queue_request("r1", "openai", {})
        manager.cache_response("k1", "v1")
        stats = manager.stats()
        assert stats["pending_queue"] == 1
        assert stats["cache_entries"] == 1


# ── Lifecycle Tests ──────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        await manager.start()
        assert manager._running
        await manager.stop()
        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_stop_with_providers(self, manager_with_providers):
        await manager_with_providers.start()
        assert manager_with_providers._running
        await manager_with_providers.stop()
        assert not manager_with_providers._running
