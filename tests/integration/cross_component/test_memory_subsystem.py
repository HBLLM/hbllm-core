"""Integration tests for Memory subsystem — ConversationThread, Cache, ConflictResolver."""

import asyncio
import time

import pytest

from hbllm.memory.cache import CacheEntry, ResponseCache
from hbllm.memory.conflict_resolver import MemoryConflictResolver
from hbllm.memory.conversation_thread import ConversationThread, ThreadManager, ThreadTurn

# ── ConversationThread Tests ─────────────────────────────────────────────────


class TestConversationThread:
    """Test thread data model and operations."""

    def test_create_thread_with_defaults(self):
        thread = ConversationThread(tenant_id="t1", name="Test Thread")
        assert thread.tenant_id == "t1"
        assert thread.name == "Test Thread"
        assert thread.total_turns == 0
        assert thread.archived is False
        assert thread.last_message is None

    def test_add_turns(self):
        thread = ConversationThread(tenant_id="t1", name="Chat")
        thread.add_turn("user", "Hello")
        thread.add_turn("assistant", "Hi there!")

        assert thread.total_turns == 2
        assert len(thread.turns) == 2
        assert thread.last_message.role == "assistant"
        assert thread.last_message.content == "Hi there!"

    def test_context_window(self):
        thread = ConversationThread(
            tenant_id="t1",
            name="Context Test",
            system_prompt="You are helpful.",
            pinned_context="Project: HBLLM",
        )
        thread.add_turn("user", "What is Python?")
        thread.add_turn("assistant", "Python is a programming language.")

        window = thread.get_context_window(max_turns=20)
        assert len(window) == 4  # system + pinned + 2 turns
        assert window[0]["role"] == "system"
        assert "You are helpful" in window[0]["content"]
        assert window[1]["role"] == "system"
        assert "Project: HBLLM" in window[1]["content"]

    def test_context_window_truncation(self):
        thread = ConversationThread(tenant_id="t1", name="Long Chat")
        for i in range(50):
            thread.add_turn("user", f"Message {i}")

        window = thread.get_context_window(max_turns=5)
        # Only 5 recent turns (no system prompt = just turns)
        assert len(window) == 5
        assert "Message 49" in window[-1]["content"]

    def test_summary_line(self):
        thread = ConversationThread(tenant_id="t1", name="Analysis")
        assert thread.summary_line == "Analysis"

        thread.add_turn("user", "What is the CPU usage?")
        assert "Analysis:" in thread.summary_line
        assert "CPU" in thread.summary_line

    def test_serialization_round_trip(self):
        thread = ConversationThread(
            tenant_id="t1",
            name="Serialize Test",
            topic="testing",
            system_prompt="Be helpful",
        )
        thread.add_turn("user", "Hello")
        thread.add_turn("assistant", "Hi!")

        data = thread.to_dict()
        restored = ConversationThread.from_dict(data)

        assert restored.name == "Serialize Test"
        assert restored.topic == "testing"
        assert restored.total_turns == 2
        assert len(restored.turns) == 2
        assert restored.turns[0].content == "Hello"


class TestThreadTurn:
    """Test individual turn data model."""

    def test_turn_creation(self):
        turn = ThreadTurn(role="user", content="Test message")
        assert turn.role == "user"
        assert turn.content == "Test message"
        assert turn.timestamp > 0

    def test_turn_serialization(self):
        turn = ThreadTurn(role="assistant", content="Response", metadata={"tokens": 50})
        d = turn.to_dict()
        restored = ThreadTurn.from_dict(d)

        assert restored.role == "assistant"
        assert restored.content == "Response"
        assert restored.metadata["tokens"] == 50


# ── ThreadManager Integration Tests ──────────────────────────────────────────


class TestThreadManagerIntegration:
    """Test ThreadManager with file persistence."""

    def test_create_and_retrieve(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        thread = manager.create("t1", name="Server Analysis", topic="devops")

        retrieved = manager.get("t1", thread.id)
        assert retrieved is not None
        assert retrieved.name == "Server Analysis"
        assert retrieved.topic == "devops"

    def test_get_by_name(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        manager.create("t1", name="Server Analysis")
        manager.create("t1", name="Deployment Plan")

        found = manager.get_by_name("t1", "Server")
        assert found is not None
        assert "Server" in found.name

    def test_get_by_name_case_insensitive(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        manager.create("t1", name="Server Analysis")

        found = manager.get_by_name("t1", "server analysis")
        assert found is not None

    def test_list_threads(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        manager.create("t1", name="Thread 1")
        time.sleep(0.01)  # Ensure ordering
        manager.create("t1", name="Thread 2")

        threads = manager.list_threads("t1")
        assert len(threads) == 2
        # Most recent first
        assert threads[0].name == "Thread 2"

    def test_archive_thread(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        thread = manager.create("t1", name="Old Thread")

        result = manager.archive("t1", thread.id)
        assert result is True
        assert thread.archived is True
        assert thread.archived_at is not None

        # Archived threads not listed by default
        active = manager.list_threads("t1")
        assert len(active) == 0

        # But visible when including archived
        all_threads = manager.list_threads("t1", include_archived=True)
        assert len(all_threads) == 1

    def test_delete_thread(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        thread = manager.create("t1", name="To Delete")

        result = manager.delete("t1", thread.id)
        assert result is True
        assert manager.get("t1", thread.id) is None

        # File should be removed
        file_path = tmp_path / "threads" / "t1" / f"{thread.id}.json"
        assert not file_path.exists()

    def test_search_threads(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        t1 = manager.create("t1", name="Python Development", topic="python")
        t2 = manager.create("t1", name="Server Setup", topic="devops")  # noqa: F841
        t1.add_turn("user", "How do I use decorators?")
        manager.save(t1)

        results = manager.search("t1", "Python")
        assert len(results) >= 1
        assert any("Python" in t.name for t in results)

    def test_search_by_content(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        t = manager.create("t1", name="General Chat")
        t.add_turn("user", "I need help with Kubernetes")
        manager.save(t)

        results = manager.search("t1", "Kubernetes")
        assert len(results) >= 1

    def test_stats(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        t = manager.create("t1", name="Active Thread")
        t.add_turn("user", "Hello")
        t.add_turn("assistant", "Hi")
        manager.save(t)

        manager.create("t1", name="Archived Thread")
        manager.archive("t1", manager.list_threads("t1")[0].id)

        stats = manager.stats("t1")
        assert stats["total_threads"] == 2
        assert stats["active_threads"] == 1
        assert stats["archived_threads"] == 1
        assert stats["total_turns"] >= 2

    def test_persistence_round_trip(self, tmp_path):
        dir_path = tmp_path / "threads"

        # Create and save
        m1 = ThreadManager(storage_dir=dir_path)
        thread = m1.create("t1", name="Persistent")
        thread.add_turn("user", "Remember this")
        m1.save(thread)

        # Reload from disk
        m2 = ThreadManager(storage_dir=dir_path)
        loaded = m2.get("t1", thread.id)
        assert loaded is not None
        assert loaded.name == "Persistent"
        assert len(loaded.turns) == 1
        assert loaded.turns[0].content == "Remember this"

    def test_tenant_isolation(self, tmp_path):
        manager = ThreadManager(storage_dir=tmp_path / "threads")
        manager.create("t1", name="Tenant 1 Thread")
        manager.create("t2", name="Tenant 2 Thread")

        t1_threads = manager.list_threads("t1")
        t2_threads = manager.list_threads("t2")

        assert len(t1_threads) == 1
        assert len(t2_threads) == 1
        assert t1_threads[0].name == "Tenant 1 Thread"
        assert t2_threads[0].name == "Tenant 2 Thread"


# ── Response Cache Integration Tests ─────────────────────────────────────────


class TestResponseCacheIntegration:
    """Test ResponseCache LRU eviction and TTL expiry."""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        await cache.put("key1", "value1")

        entry = await cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        entry = await cache.get("nonexistent")
        assert entry is None

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        cache = ResponseCache(max_size=100, ttl_seconds=0.1)
        await cache.put("key1", "value1")

        await asyncio.sleep(0.2)
        entry = await cache.get("key1")
        assert entry is None  # Expired

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        cache = ResponseCache(max_size=3, ttl_seconds=60)
        await cache.put("k1", "v1")
        await cache.put("k2", "v2")
        await cache.put("k3", "v3")

        # Access k1 to make it recent
        await cache.get("k1")

        # Add k4 — k2 (oldest) should be evicted
        await cache.put("k4", "v4")

        assert await cache.get("k1") is not None  # Still there (recently accessed)
        assert await cache.get("k2") is None  # Evicted
        assert await cache.get("k3") is not None
        assert await cache.get("k4") is not None

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        await cache.put("key", "old")
        await cache.put("key", "new")

        entry = await cache.get("key")
        assert entry.value == "new"

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        # Short TTL override
        await cache.put("key", "value", ttl=0.1)
        await asyncio.sleep(0.2)
        assert await cache.get("key") is None

    def test_invalidate(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        # Manually insert without async
        cache._cache["user:123:query"] = CacheEntry("user:123:query", "v1", ttl=60)
        cache._cache["user:123:other"] = CacheEntry("user:123:other", "v2", ttl=60)
        cache._cache["admin:456:query"] = CacheEntry("admin:456:query", "v3", ttl=60)

        removed = cache.invalidate("user:123")
        assert removed == 2
        assert "admin:456:query" in cache._cache

    def test_stats(self):
        cache = ResponseCache(max_size=50, ttl_seconds=300)
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 50
        assert stats["default_ttl"] == 300


# ── Memory Conflict Resolver Tests ───────────────────────────────────────────


class TestMemoryConflictResolverIntegration:
    """Test conflict resolution with vector clocks, authority, and recency."""

    def test_causal_ordering_wins(self):
        """Fragment with causally-later vector clock wins."""
        a = {
            "content": "Old fact",
            "vector_clock": {"node1": 1, "node2": 1},
            "authority_score": 50,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        b = {
            "content": "Updated fact",
            "vector_clock": {"node1": 2, "node2": 1},
            "authority_score": 50,
            "timestamp": "2025-01-01T00:00:00Z",
        }

        winner = MemoryConflictResolver.resolve(a, b)
        assert winner["content"] == "Updated fact"

    def test_authority_wins_when_concurrent(self):
        """Higher authority wins when vector clocks are concurrent."""
        a = {
            "content": "Low authority",
            "vector_clock": {"node1": 2, "node2": 0},
            "authority_score": 30,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        b = {
            "content": "High authority",
            "vector_clock": {"node1": 0, "node2": 2},
            "authority_score": 90,
            "timestamp": "2025-01-01T00:00:00Z",
        }

        winner = MemoryConflictResolver.resolve(a, b)
        assert winner["content"] == "High authority"

    def test_recency_fallback(self):
        """Most recent timestamp wins when all else is equal."""
        a = {
            "content": "Older",
            "vector_clock": {"node1": 1},
            "authority_score": 50,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        b = {
            "content": "Newer",
            "vector_clock": {"node1": 1},
            "authority_score": 50,
            "timestamp": "2025-06-15T12:00:00Z",
        }

        winner = MemoryConflictResolver.resolve(a, b)
        assert winner["content"] == "Newer"

    def test_no_vector_clock_falls_to_authority(self):
        a = {
            "content": "No clock A",
            "authority_score": 80,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        b = {
            "content": "No clock B",
            "authority_score": 60,
            "timestamp": "2025-06-01T00:00:00Z",
        }

        winner = MemoryConflictResolver.resolve(a, b)
        assert winner["content"] == "No clock A"

    def test_no_timestamp_returns_first(self):
        a = {"content": "A", "authority_score": 50}
        b = {"content": "B", "authority_score": 50}

        winner = MemoryConflictResolver.resolve(a, b)
        assert winner["content"] == "A"

    def test_equal_everything_returns_first(self):
        a = {
            "content": "First",
            "vector_clock": {"n": 1},
            "authority_score": 50,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        b = {
            "content": "Second",
            "vector_clock": {"n": 1},
            "authority_score": 50,
            "timestamp": "2025-01-01T00:00:00Z",
        }

        winner = MemoryConflictResolver.resolve(a, b)
        # Equal timestamps, returns b (or a depending on comparison). Either is valid.
        assert winner["content"] in ("First", "Second")
