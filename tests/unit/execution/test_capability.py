"""Tests for RuntimeCapabilities and CapabilityResolver."""

from __future__ import annotations

import pytest

from hbllm.execution.capability import CapabilityResolver, RuntimeCapabilities


class TestRuntimeCapabilities:
    def test_defaults(self) -> None:
        caps = RuntimeCapabilities()
        assert caps.streaming is False
        assert caps.json_mode is False
        assert caps.max_context == 4096

    def test_satisfies_empty(self) -> None:
        caps = RuntimeCapabilities()
        assert caps.satisfies(()) is True

    def test_satisfies_available(self) -> None:
        caps = RuntimeCapabilities(streaming=True, json_mode=True)
        assert caps.satisfies(("streaming",)) is True
        assert caps.satisfies(("json_mode",)) is True
        assert caps.satisfies(("streaming", "json_mode")) is True

    def test_satisfies_unavailable(self) -> None:
        caps = RuntimeCapabilities(streaming=True, json_mode=False)
        assert caps.satisfies(("json_mode",)) is False
        assert caps.satisfies(("streaming", "json_mode")) is False

    def test_satisfies_unknown_capability(self) -> None:
        """Unknown capabilities are treated as satisfied (permissive)."""
        caps = RuntimeCapabilities()
        assert caps.satisfies(("unknown_feature",)) is True

    def test_frozen(self) -> None:
        caps = RuntimeCapabilities(streaming=True)
        with pytest.raises(AttributeError):
            caps.streaming = False  # type: ignore[misc]


class TestCapabilityResolver:
    def test_instantiation(self) -> None:
        resolver = CapabilityResolver()
        assert resolver._fallback_chains is not None

    def test_fallback_chains_defined(self) -> None:
        resolver = CapabilityResolver()
        assert "lora" in resolver._fallback_chains
        assert "grammar" in resolver._fallback_chains
