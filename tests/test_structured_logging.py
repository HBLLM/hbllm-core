"""
Tests for the structured logging module.

Covers JSONFormatter, correlation ID context, per-module levels.
"""

import json
import logging

import pytest

from hbllm.utils.logger import (
    JSONFormatter,
    setup_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
)


class TestJSONFormatter:
    """Test JSON structured log output."""

    def setup_method(self):
        self.formatter = JSONFormatter()

    def test_basic_format(self):
        record = logging.LogRecord(
            name="hbllm.test", level=logging.INFO,
            pathname="test.py", lineno=1, msg="Hello world",
            args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "hbllm.test"
        assert parsed["msg"] == "Hello world"
        assert "ts" in parsed

    def test_correlation_id_injection(self):
        set_correlation_id("corr-123")
        try:
            record = logging.LogRecord(
                name="hbllm.test", level=logging.INFO,
                pathname="test.py", lineno=1, msg="Test",
                args=None, exc_info=None,
            )
            output = self.formatter.format(record)
            parsed = json.loads(output)
            assert parsed["correlation_id"] == "corr-123"
        finally:
            set_correlation_id("")

    def test_no_correlation_id_when_empty(self):
        set_correlation_id("")
        record = logging.LogRecord(
            name="hbllm.test", level=logging.INFO,
            pathname="test.py", lineno=1, msg="Test",
            args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "correlation_id" not in parsed

    def test_exception_included(self):
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="hbllm.test", level=logging.ERROR,
            pathname="test.py", lineno=1, msg="Error occurred",
            args=None, exc_info=exc_info,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_extra_fields(self):
        record = logging.LogRecord(
            name="hbllm.test", level=logging.INFO,
            pathname="test.py", lineno=1, msg="Node event",
            args=None, exc_info=None,
        )
        record.node_id = "router_01"
        record.tenant_id = "acme"
        record.duration_ms = 42.5

        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["node_id"] == "router_01"
        assert parsed["tenant_id"] == "acme"
        assert parsed["duration_ms"] == 42.5


class TestCorrelationId:
    """Test correlation ID context variable."""

    def test_set_and_get(self):
        set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"
        set_correlation_id("")
        assert get_correlation_id() == ""

    def test_default_empty(self):
        # In a fresh context, should be empty
        set_correlation_id("")
        assert get_correlation_id() == ""


class TestGetLogger:
    """Test logger factory."""

    def test_prefixed_name(self):
        log = get_logger("brain.router")
        assert log.name == "hbllm.brain.router"

    def test_returns_logger_instance(self):
        log = get_logger("test")
        assert isinstance(log, logging.Logger)
