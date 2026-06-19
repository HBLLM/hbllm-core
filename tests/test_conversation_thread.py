"""Tests for ConversationThread — named resumable threads."""

import tempfile
from pathlib import Path


from hbllm.memory.conversation_thread import ConversationThread, ThreadManager, ThreadTurn


class TestThreadTurn:
    def test_defaults(self):
        turn = ThreadTurn(role="user", content="Hello")
        assert turn.timestamp > 0
        assert turn.metadata == {}

    def test_serialization(self):
        turn = ThreadTurn(role="assistant", content="Hi there")
        d = turn.to_dict()
        t2 = ThreadTurn.from_dict(d)
        assert t2.role == "assistant"
        assert t2.content == "Hi there"


class TestConversationThread:
    def test_create(self):
        thread = ConversationThread(tenant_id="t1", name="Test Thread")
        assert thread.id
        assert thread.name == "Test Thread"
        assert thread.total_turns == 0

    def test_add_turn(self):
        thread = ConversationThread(tenant_id="t1", name="Test")
        thread.add_turn("user", "Hello")
        thread.add_turn("assistant", "Hi!")
        assert thread.total_turns == 2
        assert thread.last_message.content == "Hi!"

    def test_context_window(self):
        thread = ConversationThread(tenant_id="t1", name="Test")
        thread.system_prompt = "You are helpful."
        thread.add_turn("user", "Q1")
        thread.add_turn("assistant", "A1")
        ctx = thread.get_context_window()
        assert ctx[0]["role"] == "system"
        assert len(ctx) == 3  # system + 2 turns

    def test_context_window_truncation(self):
        thread = ConversationThread(tenant_id="t1", name="Test")
        for i in range(30):
            thread.add_turn("user", f"Message {i}")
        ctx = thread.get_context_window(max_turns=5)
        assert len(ctx) == 5

    def test_pinned_context(self):
        thread = ConversationThread(tenant_id="t1", name="Test")
        thread.pinned_context = "This is about the server migration project."
        ctx = thread.get_context_window()
        assert any("server migration" in m["content"] for m in ctx)

    def test_serialization(self):
        thread = ConversationThread(tenant_id="t1", name="Test")
        thread.add_turn("user", "Hello")
        d = thread.to_dict()
        t2 = ConversationThread.from_dict(d)
        assert t2.name == "Test"
        assert t2.total_turns == 1
        assert len(t2.turns) == 1

    def test_summary_line(self):
        thread = ConversationThread(tenant_id="t1", name="Server Analysis")
        thread.add_turn("user", "Check the CPU metrics")
        assert "Server Analysis" in thread.summary_line
        assert "CPU" in thread.summary_line


class TestThreadManager:
    def test_create_and_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            thread = mgr.create("t1", name="Deploy v2")
            assert (Path(tmpdir) / "t1" / f"{thread.id}.json").exists()

    def test_get_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            mgr.create("t1", name="Server Analysis")
            found = mgr.get_by_name("t1", "server")
            assert found is not None
            assert found.name == "Server Analysis"

    def test_get_by_name_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            assert mgr.get_by_name("t1", "nonexistent") is None

    def test_list_threads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            mgr.create("t1", name="Thread 1")
            mgr.create("t1", name="Thread 2")
            threads = mgr.list_threads("t1")
            assert len(threads) == 2

    def test_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            thread = mgr.create("t1", name="Old Thread")
            mgr.archive("t1", thread.id)
            active = mgr.list_threads("t1", include_archived=False)
            assert len(active) == 0
            all_threads = mgr.list_threads("t1", include_archived=True)
            assert len(all_threads) == 1

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            thread = mgr.create("t1", name="Delete Me")
            mgr.delete("t1", thread.id)
            assert mgr.get("t1", thread.id) is None

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            mgr.create("t1", name="Server Analysis")
            mgr.create("t1", name="Deployment Plan")
            results = mgr.search("t1", "server")
            assert len(results) == 1
            assert results[0].name == "Server Analysis"

    def test_persistence_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            thread = mgr.create("t1", name="Persistent")
            thread.add_turn("user", "Hello")
            mgr.save(thread)

            mgr2 = ThreadManager(storage_dir=tmpdir)
            found = mgr2.get("t1", thread.id)
            assert found is not None
            assert len(found.turns) == 1

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ThreadManager(storage_dir=tmpdir)
            mgr.create("t1", name="Thread 1")
            stats = mgr.stats("t1")
            assert stats["total_threads"] == 1
            assert stats["active_threads"] == 1
