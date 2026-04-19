"""Tests for NatsBus distributed message bus."""

from __future__ import annotations

import pytest

from hbllm.network.messages import Message, MessageType


class TestNatsBusImport:
    def test_module_imports(self):
        from hbllm.network import nats_bus

        assert hasattr(nats_bus, "NatsBus")

    def test_raises_without_nats(self):
        from hbllm.network.nats_bus import _HAS_NATS, NatsBus

        if not _HAS_NATS:
            with pytest.raises(RuntimeError, match="nats-py"):
                NatsBus()

    def test_has_message_bus_methods(self):
        from hbllm.network.nats_bus import _HAS_NATS

        if not _HAS_NATS:
            pytest.skip("nats-py not installed")
        from hbllm.network.nats_bus import NatsBus

        bus = NatsBus()
        for method in ("publish", "subscribe", "request", "start", "stop", "add_interceptor"):
            assert hasattr(bus, method)

    def test_default_config(self):
        from hbllm.network.nats_bus import _HAS_NATS

        if not _HAS_NATS:
            pytest.skip("nats-py not installed")
        from hbllm.network.nats_bus import NatsBus

        bus = NatsBus()
        assert bus._prefix == "hbllm"
        assert not bus._running

    def test_custom_config(self):
        from hbllm.network.nats_bus import _HAS_NATS

        if not _HAS_NATS:
            pytest.skip("nats-py not installed")
        from hbllm.network.nats_bus import NatsBus

        bus = NatsBus(
            servers=["nats://node1:4222", "nats://node2:4222"],
            subject_prefix="myapp",
            use_jetstream=True,
        )
        assert bus._prefix == "myapp"
        assert bus._use_jetstream is True
        assert len(bus._servers) == 2
