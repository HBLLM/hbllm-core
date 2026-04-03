"""Tests for ToolMemory — usage tracking, recommendations, discovery."""

import pytest

from hbllm.actions.tool_memory import ToolMemory, ToolUsageRecord


@pytest.fixture
def tool_memory(tmp_path):
    return ToolMemory(data_dir=str(tmp_path))


class TestRecording:
    def test_record_usage(self, tool_memory):
        tool_memory.record(ToolUsageRecord(
            tool_name="api_node", query_type="data_fetch",
            success=True, latency_ms=150.0, result_quality=0.9,
        ))
        stats = tool_memory.stats()
        assert stats["total_usages"] == 1

    def test_record_sequence(self, tool_memory):
        tool_memory.record_sequence(
            task_type="research", tools=["browser", "api", "logic"],
            success=True, total_ms=500.0,
        )
        stats = tool_memory.stats()
        assert stats["sequences"] == 1


class TestRecommendations:
    def test_recommend_tool_returns_ranked(self, tool_memory):
        # Add enough data to meet min threshold
        for i in range(5):
            tool_memory.record(ToolUsageRecord(
                tool_name="api_node", query_type="data_fetch",
                success=True, latency_ms=100.0, result_quality=0.9,
            ))
        for i in range(5):
            tool_memory.record(ToolUsageRecord(
                tool_name="browser_node", query_type="data_fetch",
                success=i > 1, latency_ms=300.0, result_quality=0.5,
            ))
        recs = tool_memory.recommend_tool("data_fetch")
        assert len(recs) >= 1
        assert recs[0]["tool"] == "api_node"  # higher success + quality

    def test_recommend_no_data(self, tool_memory):
        recs = tool_memory.recommend_tool("unknown_type")
        assert recs == []

    def test_recommend_sequence(self, tool_memory):
        for _ in range(3):
            tool_memory.record_sequence("coding", ["logic", "execution"], True, 200.0)
        seq = tool_memory.recommend_sequence("coding")
        assert seq == ["logic", "execution"]


class TestDiscovery:
    def test_discover_patterns(self, tool_memory):
        for i in range(10):
            tool_memory.record(ToolUsageRecord(
                tool_name="api_node", query_type="api_call",
                success=True, latency_ms=50.0, result_quality=0.85,
            ))
        patterns = tool_memory.discover_patterns()
        assert len(patterns) >= 1
        assert patterns[0]["best_tool"] == "api_node"
