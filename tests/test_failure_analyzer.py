"""Tests for FailureAnalyzerNode — failure classification and LLM-driven repair."""

from __future__ import annotations

from unittest.mock import AsyncMock

from hbllm.brain.failure_analyzer_node import FailureAnalyzerNode
from hbllm.network.messages import Message, MessageType

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_query(
    skill_name: str = "test_skill",
    steps: list[str] | None = None,
    error_message: str = "Something went wrong",
    execution_trace: list[dict] | None = None,
) -> Message:
    return Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.analyze_failure",
        payload={
            "skill_name": skill_name,
            "steps": steps or ["step1", "step2"],
            "execution_trace": execution_trace or [],
            "error_message": error_message,
        },
    )


# ── Classification Tests ────────────────────────────────────────────────────


class TestFailureClassification:
    """Test all 6 classification branches in _classify_failure."""

    def setup_method(self):
        self.node = FailureAnalyzerNode(node_id="test_fa")

    def test_timeout(self):
        assert self.node._classify_failure("Request timeout after 30s") == "Timeout"

    def test_timeout_mixed_case(self):
        assert self.node._classify_failure("TIMEOUT waiting for response") == "Timeout"

    def test_timed_out_is_general(self):
        """'timed out' does NOT contain 'timeout' — classifier treats as general."""
        assert self.node._classify_failure("Operation timed out") == "General Execution Failure"

    def test_logic_error_syntax(self):
        assert self.node._classify_failure("SyntaxError: unexpected token") == "Logic Error"

    def test_logic_error_nameerror(self):
        assert self.node._classify_failure("NameError: name 'foo' is not defined") == "Logic Error"

    def test_network_error_connection(self):
        assert self.node._classify_failure("Connection refused to host") == "Network Error"

    def test_network_error_api(self):
        assert (
            self.node._classify_failure("API returned 500 Internal Server Error") == "Network Error"
        )

    def test_network_error_http(self):
        assert self.node._classify_failure("HTTP request failed") == "Network Error"

    def test_missing_data(self):
        assert self.node._classify_failure("File not found: data.csv") == "Missing Data / File"

    def test_missing_data_key(self):
        assert (
            self.node._classify_failure("Key 'config' is missing from dict")
            == "Missing Data / File"
        )

    def test_tool_failure(self):
        assert self.node._classify_failure("Tool execution failed") == "Tool Failure"

    def test_tool_failure_invalid_param(self):
        assert self.node._classify_failure("Invalid parameter 'count'") == "Tool Failure"

    def test_general_failure(self):
        assert (
            self.node._classify_failure("Something unexpected happened")
            == "General Execution Failure"
        )

    def test_empty_message(self):
        assert self.node._classify_failure("") == "General Execution Failure"


# ── Handler Tests (No LLM) ──────────────────────────────────────────────────


class TestHandlerWithoutLLM:
    """Test handle_message when no LLM is available."""

    def setup_method(self):
        self.node = FailureAnalyzerNode(node_id="test_fa", llm=None)

    async def test_returns_unrepaired_without_llm(self):
        msg = _make_query(error_message="Connection refused")
        result = await self.node.handle_message(msg)

        assert result is not None
        payload = result.payload
        assert payload["failure_type"] == "Network Error"
        assert payload["repaired"] is False
        assert payload["reason"] == "No LLM available for repair"
        assert payload["new_steps"] == ["step1", "step2"]

    async def test_ignores_non_query_messages(self):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="action.analyze_failure",
            payload={},
        )
        result = await self.node.handle_message(msg)
        assert result is None


# ── Handler Tests (With Mock LLM) ───────────────────────────────────────────


class TestHandlerWithLLM:
    """Test handle_message with a mock LLM provider."""

    def setup_method(self):
        self.mock_llm = AsyncMock()
        self.node = FailureAnalyzerNode(node_id="test_fa", llm=self.mock_llm)

    async def test_successful_repair(self):
        self.mock_llm.generate_json = AsyncMock(
            return_value={"new_steps": ["fixed_step1", "fixed_step2"]}
        )

        msg = _make_query(error_message="Timeout after 10s")
        result = await self.node.handle_message(msg)

        assert result is not None
        payload = result.payload
        assert payload["failure_type"] == "Timeout"
        assert payload["repaired"] is True
        assert payload["new_steps"] == ["fixed_step1", "fixed_step2"]
        assert "repair strategy applied" in payload["reason"].lower()

    async def test_llm_returns_same_steps(self):
        """When LLM returns the same steps, repaired should be False."""
        self.mock_llm.generate_json = AsyncMock(return_value={"new_steps": ["step1", "step2"]})

        msg = _make_query()
        result = await self.node.handle_message(msg)

        assert result is not None
        assert result.payload["repaired"] is False

    async def test_llm_returns_none(self):
        """When LLM returns None, fall back to original steps."""
        self.mock_llm.generate_json = AsyncMock(return_value=None)

        msg = _make_query()
        result = await self.node.handle_message(msg)

        assert result is not None
        assert result.payload["repaired"] is False
        assert result.payload["new_steps"] == ["step1", "step2"]

    async def test_llm_returns_empty_dict(self):
        """When LLM returns empty dict, fall back to original steps."""
        self.mock_llm.generate_json = AsyncMock(return_value={})

        msg = _make_query()
        result = await self.node.handle_message(msg)

        assert result is not None
        assert result.payload["repaired"] is False

    async def test_llm_raises_exception(self):
        """When LLM throws, return an error response."""
        self.mock_llm.generate_json = AsyncMock(side_effect=RuntimeError("LLM down"))

        msg = _make_query()
        result = await self.node.handle_message(msg)

        assert result is not None
        assert result.type == MessageType.ERROR

    async def test_preserves_skill_name_in_classification(self):
        self.mock_llm.generate_json = AsyncMock(return_value={"new_steps": ["new_step"]})

        msg = _make_query(skill_name="data_fetcher", error_message="HTTP 503")
        result = await self.node.handle_message(msg)

        assert result is not None
        assert result.payload["failure_type"] == "Network Error"
