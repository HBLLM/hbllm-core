"""Tests for GenerationNode — cognitive → execution translation boundary.

These tests are isolated from the full import chain (hbllm.network.messages
uses StrEnum which requires Python 3.11+). They test the pure translation
logic via direct imports of the data classes.
"""

from __future__ import annotations

import sys

import pytest

# Guard: Skip if hbllm.network can't import (Python <3.11 / StrEnum)
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="hbllm.network.messages requires Python 3.11+ (StrEnum)",
)


def _import_generation_node():
    """Lazy import to avoid collection-time ImportError."""
    from hbllm.brain.execution.generation_node import GenerationNode, RoutingResult

    return GenerationNode, RoutingResult


class TestRoutingResult:
    def test_defaults(self) -> None:
        _, RoutingResult = _import_generation_node()
        r = RoutingResult()
        assert r.domain == "general"
        assert r.language == "en"
        assert r.complexity == "medium"
        assert r.confidence == 0.8
        assert r.requires_planning is False

    def test_no_execution_metadata(self) -> None:
        """RoutingResult has NO execution concerns."""
        _, RoutingResult = _import_generation_node()
        r = RoutingResult()
        assert not hasattr(r, "provider")
        assert not hasattr(r, "model_id")
        assert not hasattr(r, "adapter")
        assert not hasattr(r, "lora")
        assert not hasattr(r, "modifier")

    def test_custom_values(self) -> None:
        _, RoutingResult = _import_generation_node()
        r = RoutingResult(
            domain="coding.python",
            language="fr",
            complexity="high",
            audience="technical",
            intent="code_generation",
        )
        assert r.domain == "coding.python"
        assert r.language == "fr"
        assert r.complexity == "high"


class TestGenerationNode:
    def test_build_routing_result_str(self) -> None:
        GenerationNode, _ = _import_generation_node()
        node = GenerationNode()
        result = node._build_routing_result("coding", {"language": "en"})
        assert result.domain == "coding"
        assert result.language == "en"

    def test_build_routing_result_dict(self) -> None:
        GenerationNode, _ = _import_generation_node()
        node = GenerationNode()
        weights = {"coding": 0.7, "math": 0.3}
        result = node._build_routing_result(weights, {})
        assert result.domain == "coding"  # Highest weight
        assert result.domain_weights == weights

    def test_build_routing_result_metadata(self) -> None:
        GenerationNode, _ = _import_generation_node()
        node = GenerationNode()
        metadata = {
            "language": "ja",
            "complexity": "high",
            "audience": "academic",
        }
        result = node._build_routing_result("general", metadata)
        assert result.language == "ja"
        assert result.complexity == "high"
        assert result.audience == "academic"

    def test_node_type(self) -> None:
        GenerationNode, _ = _import_generation_node()
        node = GenerationNode()
        assert node.node_id == "generation_node"
        assert "generation" in node.capabilities
