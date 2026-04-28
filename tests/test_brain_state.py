"""Tests for core BrainState persistence."""

import os
import tempfile

import pytest

from hbllm.persistence import BrainState


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_state.db")


@pytest.fixture
def state(tmp_db):
    s = BrainState(path=tmp_db)
    yield s
    s.close()


class TestKeyValueStore:
    def test_save_load(self, state):
        state.save("key1", {"name": "test", "value": 42})
        result = state.load("key1")
        assert result == {"name": "test", "value": 42}

    def test_load_default(self, state):
        result = state.load("missing", default="fallback")
        assert result == "fallback"

    def test_overwrite(self, state):
        state.save("k", "v1")
        state.save("k", "v2")
        assert state.load("k") == "v2"

    def test_delete(self, state):
        state.save("del_me", True)
        state.delete("del_me")
        assert state.load("del_me") is None


class TestMessages:
    def test_append_and_get(self, state):
        state.append_message("user", "Hello")
        state.append_message("assistant", "Hi there!")

        messages = state.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_message_limit(self, state):
        for i in range(10):
            state.append_message("user", f"msg {i}")

        messages = state.get_messages(limit=3)
        assert len(messages) == 3

    def test_clear_messages(self, state):
        state.append_message("user", "temp")
        state.clear_messages()
        assert len(state.get_messages()) == 0

    def test_message_metadata(self, state):
        state.append_message("user", "test", metadata={"tool": "search"})
        msg = state.get_messages()[0]
        assert msg["metadata"]["tool"] == "search"


class TestCheckpoints:
    def test_checkpoint_and_latest(self, state):
        state.checkpoint({"step": 1, "memory": 10})
        state.checkpoint({"step": 2, "memory": 20})

        latest = state.latest_checkpoint()
        assert latest is not None
        assert latest["data"]["step"] == 2

    def test_no_checkpoints(self, state):
        assert state.latest_checkpoint() is None

    def test_list_checkpoints(self, state):
        for i in range(5):
            state.checkpoint({"step": i})

        cps = state.list_checkpoints(limit=3)
        assert len(cps) == 3


class TestToolLogs:
    def test_log_and_retrieve(self, state):
        state.log_tool_call("web_search", "query=AI", "Results...", duration_ms=150.0)
        logs = state.get_tool_logs()
        assert len(logs) == 1
        assert logs[0]["tool"] == "web_search"
        assert logs[0]["duration_ms"] == 150.0

    def test_filter_by_tool(self, state):
        state.log_tool_call("search", "q1", "r1")
        state.log_tool_call("exec", "q2", "r2")
        state.log_tool_call("search", "q3", "r3")

        logs = state.get_tool_logs(tool_name="search")
        assert len(logs) == 2


class TestStats:
    def test_stats(self, state):
        state.save("k", "v")
        state.append_message("user", "hi")
        state.checkpoint({"x": 1})
        state.log_tool_call("t", "i", "o")

        s = state.stats
        assert s["kv_entries"] == 1
        assert s["messages"] == 1
        assert s["checkpoints"] == 1
        assert s["tool_logs"] == 1


class TestPersistence:
    def test_survives_reconnect(self, tmp_db):
        s1 = BrainState(path=tmp_db)
        s1.save("persistent", True)
        s1.append_message("user", "remember me")
        s1.close()

        s2 = BrainState(path=tmp_db)
        assert s2.load("persistent") is True
        assert len(s2.get_messages()) == 1
        s2.close()
