"""
Memory, Persistence & SNN Calibrator — Integration test coverage.
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════════
# persistence/state.py — BrainState (synchronous SQLite backend)
# ═══════════════════════════════════════════════════════════════════════


class TestBrainState:
    @pytest.fixture
    def state(self, tmp_path):
        from hbllm.persistence.state import BrainState

        return BrainState(path=str(tmp_path / "test_brain.db"))

    def test_save_and_load(self, state):
        state.save("test_key", {"data": "hello"}, tenant_id="t1")
        loaded = state.load("test_key", tenant_id="t1")
        assert loaded["data"] == "hello"

    def test_load_missing_returns_none(self, state):
        assert state.load("nonexistent", tenant_id="t1") is None

    def test_save_overwrite(self, state):
        state.save("key", {"v": 1}, tenant_id="t1")
        state.save("key", {"v": 2}, tenant_id="t1")
        assert state.load("key", tenant_id="t1")["v"] == 2

    def test_delete(self, state):
        state.save("key", {"v": 1}, tenant_id="t1")
        state.delete("key", tenant_id="t1")
        assert state.load("key", tenant_id="t1") is None

    def test_delete_nonexistent(self, state):
        state.delete("nonexistent", tenant_id="t1")  # should not raise

    def test_messages_crud(self, state):
        state.append_message("user", "Hello", tenant_id="t1")
        state.append_message("assistant", "Hi!", tenant_id="t1")
        msgs = state.get_messages(tenant_id="t1")
        assert len(msgs) == 2

    def test_messages_limit(self, state):
        for i in range(10):
            state.append_message("user", f"msg {i}", tenant_id="t1")
        msgs = state.get_messages(tenant_id="t1", limit=3)
        assert len(msgs) == 3

    def test_clear_messages(self, state):
        state.append_message("user", "Hello", tenant_id="t1")
        state.clear_messages(tenant_id="t1")
        msgs = state.get_messages(tenant_id="t1")
        assert len(msgs) == 0

    def test_checkpoint(self, state):
        state.checkpoint({"model": "v1"}, tenant_id="t1")
        latest = state.latest_checkpoint(tenant_id="t1")
        assert latest is not None

    def test_list_checkpoints(self, state):
        state.checkpoint({"v": 1}, tenant_id="t1")
        state.checkpoint({"v": 2}, tenant_id="t1")
        cps = state.list_checkpoints(tenant_id="t1")
        assert len(cps) >= 2

    def test_tool_logging(self, state):
        state.log_tool_call(
            tool_name="python_exec",
            input_data="print(1)",
            output="1",
            duration_ms=42.0,
            tenant_id="t1",
            user_id="",
            device_id="",
        )
        logs = state.get_tool_logs(tenant_id="t1")
        assert len(logs) >= 1
        assert logs[0]["tool"] == "python_exec"

    def test_tool_logs_filter_by_tool(self, state):
        state.log_tool_call("tool_a", "in", "out", 10, "t1")
        state.log_tool_call("tool_b", "in", "out", 20, "t1")
        logs = state.get_tool_logs(tenant_id="t1", tool_name="tool_a")
        assert len(logs) == 1

    def test_close(self, state):
        state.close()
        state.close()  # double close should not raise

    def test_tenant_isolation(self, state):
        state.save("key", {"v": "A"}, tenant_id="t1")
        state.save("key", {"v": "B"}, tenant_id="t2")
        assert state.load("key", tenant_id="t1")["v"] == "A"
        assert state.load("key", tenant_id="t2")["v"] == "B"


# ═══════════════════════════════════════════════════════════════════════
# memory/scope.py
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryScope:
    def test_scope_values(self):
        from hbllm.memory.scope import MemoryScope

        assert MemoryScope.WORKING is not None
        assert MemoryScope.EPISODIC is not None
        assert MemoryScope.SEMANTIC is not None
        assert MemoryScope.SENSITIVE is not None


# ═══════════════════════════════════════════════════════════════════════
# hcir/snn/comprehension/calibrator.py
# ═══════════════════════════════════════════════════════════════════════


class TestSNNCalibrator:
    def test_record_outcome(self, tmp_path):
        from hbllm.hcir.snn.comprehension.calibrator import SNNCalibrator

        cal = SNNCalibrator(data_dir=str(tmp_path))
        cal.record_outcome(
            domain="math",
            params_used={"threshold": 0.5},
            num_concepts=5,
            response_quality=0.8,
            memory_relevance=0.7,
        )

    def test_record_multiple_outcomes(self, tmp_path):
        from hbllm.hcir.snn.comprehension.calibrator import SNNCalibrator

        cal = SNNCalibrator(data_dir=str(tmp_path))
        for i in range(10):
            cal.record_outcome(
                domain="science",
                params_used={"threshold": 0.3 + i * 0.05},
                num_concepts=i + 1,
                response_quality=0.5 + i * 0.05,
                memory_relevance=0.6,
            )
