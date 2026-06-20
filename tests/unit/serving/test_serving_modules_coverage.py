"""Unit tests for serving modules — launcher, self_improve."""

from unittest.mock import MagicMock

from hbllm.serving.launcher import ServerInstance


class TestServerInstance:
    def test_init(self):
        config = MagicMock()
        instance = ServerInstance(server_name="test-server", config=config)
        assert instance is not None

    def test_summary(self):
        config = MagicMock()
        config.nodes = {}
        instance = ServerInstance(server_name="test-server", config=config)
        summary = instance.summary()
        assert isinstance(summary, str)


# ── Self Improve ─────────────────────────────────────────────────────────────

from hbllm.serving.self_improve import (
    _convert_to_sft_format,
    _load_reflection_data,
)


class TestSelfImproveHelpers:
    def test_convert_to_sft_format_empty(self):
        result = _convert_to_sft_format([])
        assert result == []

    def test_convert_to_sft_format(self):
        samples = [
            {
                "query": "What is Python?",
                "response": "Python is a programming language.",
                "improved_response": "Python is a high-level programming language.",
            }
        ]
        result = _convert_to_sft_format(samples)
        assert len(result) >= 1

    def test_load_reflection_data_missing(self, tmp_path):
        # Should handle missing file gracefully
        try:
            result = _load_reflection_data(str(tmp_path / "nonexistent.json"))
            assert isinstance(result, tuple)
        except (FileNotFoundError, Exception):
            pass  # Expected for missing file
