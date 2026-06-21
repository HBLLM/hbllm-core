"""Tests for OfflineManager — graceful network degradation."""

import time

import pytest

from hbllm.serving.offline_mode import (
    ConnectivityState,
    OfflineManager,
    QueuedRequest,
)


class TestConnectivityState:
    """Tests for state management."""

    def test_initial_state_online(self):
        om = OfflineManager()
        assert om.state == ConnectivityState.ONLINE
        assert om.is_online
        assert not om.is_offline

    def test_state_properties(self):
        om = OfflineManager()
        om._state = ConnectivityState.OFFLINE
        assert om.is_offline
        assert not om.is_online


class TestRequestQueue:
    """Tests for outbound request queuing."""

    @pytest.fixture
    def manager(self):
        return OfflineManager(queue_max_size=10)

    def test_enqueue_returns_id(self, manager):
        rid = manager.enqueue("push", {"msg": "hello"})
        assert rid.startswith("req_")
        assert manager.queue_size == 1

    def test_multiple_enqueue(self, manager):
        for i in range(5):
            manager.enqueue("push", {"msg": f"msg_{i}"})
        assert manager.queue_size == 5

    def test_queue_overflow_drops_oldest(self, manager):
        for i in range(12):
            manager.enqueue("push", {"msg": f"msg_{i}"})
        assert manager.queue_size <= 10

    def test_queue_summary_by_type(self, manager):
        manager.enqueue("push", {"a": 1})
        manager.enqueue("push", {"a": 2})
        manager.enqueue("webhook", {"b": 1})
        summary = manager.get_queue_summary()
        assert summary["push"] == 2
        assert summary["webhook"] == 1

    def test_expired_request(self):
        req = QueuedRequest(
            request_id="test",
            request_type="push",
            max_age_s=0.001,  # Expire immediately
        )
        time.sleep(0.01)
        assert req.is_expired


class TestResponseCache:
    """Tests for cloud response caching."""

    @pytest.fixture
    def manager(self):
        return OfflineManager(cache_max_size=5)

    def test_cache_store_and_retrieve(self, manager):
        manager.cache_response("query1", {"answer": "42"})
        cached = manager.get_cached("query1")
        assert cached == {"answer": "42"}

    def test_cache_miss(self, manager):
        cached = manager.get_cached("nonexistent")
        assert cached is None

    def test_cache_eviction(self, manager):
        for i in range(7):
            manager.cache_response(f"key_{i}", f"val_{i}")
        # Cache max is 5, oldest should be evicted
        assert manager.get_cached("key_0") is None
        assert manager.get_cached("key_6") == "val_6"

    def test_stale_cache_returns_none(self, manager):
        manager.cache_response("stale", "data", ttl_s=0.001)
        time.sleep(0.01)
        assert manager.get_cached("stale") is None

    def test_clear_cache(self, manager):
        manager.cache_response("k1", "v1")
        manager.clear_cache()
        assert manager.get_cached("k1") is None

    def test_cache_stats(self, manager):
        manager.cache_response("k1", "v1")
        manager.get_cached("k1")  # hit
        manager.get_cached("k2")  # miss
        s = manager.stats()
        assert s["cache_hits"] == 1
        assert s["cache_misses"] == 1


class TestOfflineManagerStats:
    """Tests for telemetry."""

    def test_stats_structure(self):
        om = OfflineManager()
        s = om.stats()
        assert "state" in s
        assert "queue_size" in s
        assert "cache_size" in s
        assert "total_checks" in s
        assert s["state"] == "online"
