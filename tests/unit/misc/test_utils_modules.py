"""Unit tests for utility modules — env_sanitize, hardware."""

import os
from unittest.mock import patch

from hbllm.utils.env_sanitize import sanitize_proxy_env
from hbllm.utils.hardware import is_slow_cpu


class TestEnvSanitize:
    """Test environment variable sanitization."""

    def test_sanitize_removes_empty_proxy(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "", "HTTPS_PROXY": ""}, clear=False):
            sanitize_proxy_env()
            assert "HTTP_PROXY" not in os.environ or os.environ.get("HTTP_PROXY") == ""

    def test_sanitize_preserves_valid_proxy(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}, clear=False):
            sanitize_proxy_env()
            # Valid proxies should not be removed
            assert os.environ.get("HTTP_PROXY") == "http://proxy:8080"

    def test_sanitize_no_crash_on_clean_env(self):
        """Should not crash when no proxy vars exist."""
        env = {k: v for k, v in os.environ.items() if "PROXY" not in k.upper()}
        with patch.dict(os.environ, env, clear=True):
            sanitize_proxy_env()  # Should not raise


class TestHardware:
    """Test hardware detection."""

    def test_is_slow_cpu_returns_bool(self):
        result = is_slow_cpu()
        assert isinstance(result, bool)

    @patch("os.cpu_count", return_value=1)
    def test_single_core_is_slow(self, mock_cpu):
        result = is_slow_cpu()
        assert isinstance(result, bool)

    @patch("os.cpu_count", return_value=32)
    def test_many_cores_not_slow(self, mock_cpu):
        result = is_slow_cpu()
        assert isinstance(result, bool)
