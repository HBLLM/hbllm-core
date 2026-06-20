"""
Tests for the Adaptive Hybrid Network Architecture.

Covers: Transport abstractions, NodeState Engine, CapabilityRegistry,
ExecutionContext, RoutingIntelligenceLayer, and GossipSync.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from hbllm.network.discovery.gossip import GossipEntry, GossipMessage, GossipSync
from hbllm.network.discovery.registry import CapabilityRegistry
from hbllm.network.messages import Message, MessageType
from hbllm.network.node_state import (
    NodeRole,
    NodeStateEngine,
    NodeStateStatus,
    PeerInfo,
    TransportInfo,
)
from hbllm.network.routing.context import ExecutionContext
from hbllm.network.routing.ril import RoutingIntelligenceLayer
from hbllm.network.transports.base import TransportMetrics, TransportState
from hbllm.network.transports.inprocess import InProcessTransport

# ──────────────────────────────────────────────
# ExecutionContext tests
# ──────────────────────────────────────────────


class TestExecutionContext:
    def test_creation_defaults(self):
        ctx = ExecutionContext()
        assert ctx.context_id
        assert ctx.hop_count == 0
        assert not ctx.is_expired
        assert not ctx.has_exceeded_max_hops
        assert ctx.total_latency_ms == 0.0

    def test_add_hop(self):
        ctx = ExecutionContext()
        ctx.add_hop("ipc", "inprocess", "node_a", latency_ms=0.5)
        ctx.add_hop("ws", "websocket", "node_b", latency_ms=15.0)
        assert ctx.hop_count == 2
        assert ctx.total_latency_ms == pytest.approx(15.5)

    def test_loop_detection(self):
        ctx = ExecutionContext()
        ctx.add_hop("ipc", "inprocess", "node_a")
        assert ctx.visited_node("node_a")
        assert not ctx.visited_node("node_b")

    def test_max_hops(self):
        ctx = ExecutionContext(max_hops=2)
        ctx.add_hop("t1", "inprocess", "n1")
        assert not ctx.has_exceeded_max_hops
        ctx.add_hop("t2", "inprocess", "n2")
        assert ctx.has_exceeded_max_hops

    def test_deadline_expiry(self):
        ctx = ExecutionContext(deadline=time.monotonic() - 1.0)
        assert ctx.is_expired

    def test_no_deadline_not_expired(self):
        ctx = ExecutionContext(deadline=None)
        assert not ctx.is_expired


# ──────────────────────────────────────────────
# TransportMetrics tests
# ──────────────────────────────────────────────


class TestTransportMetrics:
    def test_record_latency(self):
        m = TransportMetrics()
        m.record_latency(10.0)
        m.record_latency(20.0)
        assert m.avg_latency_ms == pytest.approx(15.0)
        assert m.max_latency_ms == pytest.approx(20.0)
        assert m.min_latency_ms == pytest.approx(10.0)

    def test_record_send_receive(self):
        m = TransportMetrics()
        m.record_send(100)
        m.record_send(200)
        m.record_receive(50)
        assert m.messages_sent == 2
        assert m.bytes_sent == 300
        assert m.messages_received == 1
        assert m.bytes_received == 50

    def test_error_rate(self):
        m = TransportMetrics()
        m.record_send(10)
        m.record_send(10)
        m.record_error()
        assert m.error_rate == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_error_rate_no_messages(self):
        m = TransportMetrics()
        assert m.error_rate == 0.0

    def test_record_drop(self):
        m = TransportMetrics()
        m.record_drop()
        m.record_drop()
        assert m.messages_dropped == 2


# ──────────────────────────────────────────────
# InProcessTransport tests
# ──────────────────────────────────────────────


class TestInProcessTransport:
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        t = InProcessTransport()
        assert t.state == TransportState.DISCONNECTED
        await t.start()
        assert t.state == TransportState.CONNECTED
        await t.stop()
        assert t.state == TransportState.STOPPED

    @pytest.mark.asyncio
    async def test_pub_sub(self):
        t = InProcessTransport()
        await t.start()

        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await t.subscribe("test.topic", handler)
        assert t.has_subscribers("test.topic")

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test.topic",
            payload={"data": "hello"},
        )
        await t.send("test.topic", msg)
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0].payload["data"] == "hello"
        await t.stop()

    @pytest.mark.asyncio
    async def test_request_response(self):
        t = InProcessTransport()
        await t.start()

        async def echo(msg: Message) -> Message:
            return msg.create_response({"echoed": True})

        await t.subscribe("echo", echo)
        msg = Message(type=MessageType.QUERY, source_node_id="test", topic="echo")
        resp = await t.send_request("echo", msg, timeout=2.0)
        assert resp.payload["echoed"] is True
        await t.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        t = InProcessTransport()
        await t.start()
        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        sub = await t.subscribe("topic", handler)
        await t.unsubscribe(sub)
        assert not t.has_subscribers("topic")

        msg = Message(type=MessageType.EVENT, source_node_id="t", topic="topic")
        await t.send("topic", msg)
        await asyncio.sleep(0.1)
        assert len(received) == 0
        await t.stop()

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        t = InProcessTransport()
        await t.start()

        async def handler(msg: Message) -> None:
            pass

        await t.subscribe("m", handler)
        msg = Message(type=MessageType.EVENT, source_node_id="t", topic="m")
        await t.send("m", msg)
        await asyncio.sleep(0.1)

        metrics = t.get_metrics()
        assert metrics.messages_sent >= 1
        await t.stop()


# ──────────────────────────────────────────────
# NodeStateEngine tests
# ──────────────────────────────────────────────


class TestNodeStateEngine:
    def test_creation(self):
        e = NodeStateEngine("server", role=NodeRole.COORDINATOR, authority_score=90)
        assert e.node_id == "server"
        assert e.role == NodeRole.COORDINATOR
        assert e.status == NodeStateStatus.STARTING
        assert e.authority_score == 90

    def test_load_score(self):
        e = NodeStateEngine("n")
        e.update_load(cpu_load=0.5, memory_load=0.5, task_queue_depth=50)
        # 0.5*0.4 + 0.5*0.3 + 0.5*0.3 = 0.2 + 0.15 + 0.15 = 0.5
        assert e.load_score == pytest.approx(0.5)
        assert not e.is_overloaded

    def test_overloaded(self):
        e = NodeStateEngine("n")
        e.update_load(cpu_load=1.0, memory_load=1.0, task_queue_depth=100)
        assert e.is_overloaded
        assert e.status == NodeStateStatus.OVERLOADED

    def test_auto_healthy_recovery(self):
        e = NodeStateEngine("n")
        e.update_load(cpu_load=1.0, memory_load=1.0, task_queue_depth=100)
        assert e.status == NodeStateStatus.OVERLOADED
        e.update_load(cpu_load=0.1, memory_load=0.1, task_queue_depth=0)
        assert e.status == NodeStateStatus.HEALTHY

    def test_role_shift(self):
        e = NodeStateEngine("n", role=NodeRole.EDGE)
        e.set_role(NodeRole.RELAY, reason="partition")
        assert e.role == NodeRole.RELAY
        assert len(e._role_history) == 1
        assert e._role_history[0]["from"] == "edge"
        assert e._role_history[0]["to"] == "relay"

    def test_role_shift_noop(self):
        e = NodeStateEngine("n", role=NodeRole.EDGE)
        e.set_role(NodeRole.EDGE, reason="noop")
        assert len(e._role_history) == 0

    def test_peer_management(self):
        e = NodeStateEngine("server")
        e.register_peer(PeerInfo(node_id="mobile", capabilities=["gps", "camera"]))
        e.register_peer(PeerInfo(node_id="car", capabilities=["obd2", "gps"]))
        assert len(e.reachable_peers) == 2
        assert len(e.find_peer_by_capability("gps")) == 2
        assert len(e.find_peer_by_capability("camera")) == 1

    def test_peer_unreachable(self):
        e = NodeStateEngine("server")
        e.register_peer(PeerInfo(node_id="mobile", capabilities=["gps"]))
        e.mark_peer_unreachable("mobile")
        assert len(e.reachable_peers) == 0
        assert len(e.find_peer_by_capability("gps")) == 0

    def test_peer_reachable_again(self):
        e = NodeStateEngine("server")
        e.register_peer(PeerInfo(node_id="mobile"))
        e.mark_peer_unreachable("mobile")
        e.mark_peer_reachable("mobile", latency_ms=10.0)
        assert len(e.reachable_peers) == 1

    def test_snapshot(self):
        e = NodeStateEngine("server", role=NodeRole.COORDINATOR, device_tier="server")
        e.set_status(NodeStateStatus.HEALTHY)
        e.register_peer(PeerInfo(node_id="p1"))
        snap = e.snapshot()
        assert snap.node_id == "server"
        assert snap.role == NodeRole.COORDINATOR
        assert snap.reachable_peer_count == 1
        assert snap.uptime_seconds > 0

    def test_transport_tracking(self):
        e = NodeStateEngine("n")
        e.update_transport(
            TransportInfo(transport_id="ipc", transport_type="inprocess", state="connected")
        )
        e.update_transport(
            TransportInfo(transport_id="ws", transport_type="websocket", state="connected")
        )
        snap = e.snapshot()
        assert len(snap.active_transports) == 2
        e.remove_transport("ws")
        snap2 = e.snapshot()
        assert len(snap2.active_transports) == 1


# ──────────────────────────────────────────────
# CapabilityRegistry tests
# ──────────────────────────────────────────────


class TestCapabilityRegistry:
    def test_register_and_find(self):
        reg = CapabilityRegistry()
        reg.register("node_a", ["llm", "search"], is_local=True)
        results = reg.find_by_capability("llm")
        assert len(results) == 1
        assert results[0].node_id == "node_a"
        assert results[0].is_local

    def test_deregister(self):
        reg = CapabilityRegistry()
        reg.register("node_a", ["llm"])
        reg.deregister("node_a")
        assert reg.find_by_capability("llm") == []
        assert reg.node_count == 0

    def test_re_register_replaces(self):
        reg = CapabilityRegistry()
        reg.register("node_a", ["llm", "search"])
        reg.register("node_a", ["llm"])  # Re-register with fewer caps
        assert reg.get_node_capabilities("node_a") == ["llm"]
        assert reg.find_by_capability("search") == []

    def test_find_best_prefers_local(self):
        reg = CapabilityRegistry()
        reg.register("remote", ["llm"], latency_ms=5.0, is_local=False)
        reg.register("local", ["llm"], latency_ms=0.1, is_local=True)
        best = reg.find_best_for_capability("llm")
        assert best is not None
        assert best.node_id == "local"

    def test_find_best_lowest_latency(self):
        reg = CapabilityRegistry()
        reg.register("slow", ["gps"], latency_ms=100.0)
        reg.register("fast", ["gps"], latency_ms=10.0)
        best = reg.find_best_for_capability("gps")
        assert best is not None
        assert best.node_id == "fast"

    def test_find_best_nonexistent(self):
        reg = CapabilityRegistry()
        assert reg.find_best_for_capability("nonexistent") is None

    def test_update_node_health(self):
        reg = CapabilityRegistry()
        reg.register("node_a", ["llm"], latency_ms=10.0)
        reg.update_node_health("node_a", latency_ms=5.0, load=0.8)
        entries = reg.find_by_capability("llm")
        assert entries[0].latency_ms == 5.0
        assert entries[0].load == 0.8

    def test_unreachable_filtered(self):
        reg = CapabilityRegistry()
        reg.register("node_a", ["llm"])
        reg.update_node_health("node_a", is_reachable=False)
        assert reg.find_by_capability("llm", reachable_only=True) == []
        assert len(reg.find_by_capability("llm", reachable_only=False)) == 1

    def test_get_all_capabilities(self):
        reg = CapabilityRegistry()
        reg.register("a", ["llm", "search"])
        reg.register("b", ["gps"])
        caps = reg.get_all_capabilities()
        assert set(caps) == {"llm", "search", "gps"}

    def test_network_summary(self):
        reg = CapabilityRegistry()
        reg.register("a", ["llm"])
        reg.register("b", ["gps", "camera"])
        summary = reg.get_network_summary()
        assert summary["total_nodes"] == 2
        assert summary["total_capabilities"] == 3

    def test_refresh_node(self):
        reg = CapabilityRegistry(default_ttl=0.1)
        reg.register("a", ["llm"])
        reg.refresh_node("a")
        entries = reg.find_by_capability("llm")
        assert len(entries) == 1  # Still alive after refresh

    def test_ttl_pruning(self):
        reg = CapabilityRegistry(default_ttl=0.0)  # Instant expiry
        reg.register("stale", ["llm"])
        # Force prune via query
        results = reg.find_by_capability("llm")
        assert len(results) == 0
        assert reg.node_count == 0


# ──────────────────────────────────────────────
# GossipSync tests
# ──────────────────────────────────────────────


class TestGossipSync:
    def test_publish_local_state(self):
        engine = NodeStateEngine("server", role=NodeRole.COORDINATOR)
        engine.set_status(NodeStateStatus.HEALTHY)
        registry = CapabilityRegistry()
        registry.register("server", ["llm", "search"], is_local=True)

        gossip = GossipSync(node_id="server")
        gossip.set_node_state(engine)
        gossip.set_capability_registry(registry)
        gossip._publish_local_state()

        assert len(gossip._state) >= 5  # role, status, device_tier, load, authority + caps
        digest = gossip.get_digest()
        assert len(digest) >= 5

    @pytest.mark.asyncio
    async def test_receive_merges_newer(self):
        gossip = GossipSync(node_id="server")
        msg = GossipMessage(
            source_node="mobile",
            entries=[
                GossipEntry(node_id="mobile", key="role", value="edge", version=1),
                GossipEntry(node_id="mobile", key="load", value=0.3, version=1),
            ],
            ttl=2,
        )
        await gossip.receive(msg)
        assert gossip.entries_merged == 2
        assert gossip.messages_received == 1

    @pytest.mark.asyncio
    async def test_receive_rejects_older(self):
        gossip = GossipSync(node_id="server")
        # Insert version 5
        gossip._state[("mobile", "role")] = GossipEntry(
            node_id="mobile", key="role", value="relay", version=5
        )
        # Send version 2 — should be rejected
        msg = GossipMessage(
            source_node="mobile",
            entries=[GossipEntry(node_id="mobile", key="role", value="edge", version=2)],
            ttl=1,
        )
        await gossip.receive(msg)
        assert gossip.entries_rejected == 1
        assert gossip._state[("mobile", "role")].value == "relay"

    @pytest.mark.asyncio
    async def test_dedup_seen_messages(self):
        gossip = GossipSync(node_id="server")
        msg = GossipMessage(
            source_node="mobile",
            entries=[GossipEntry(node_id="mobile", key="x", value=1, version=1)],
        )
        await gossip.receive(msg)
        await gossip.receive(msg)  # Duplicate
        assert gossip.messages_received == 1

    @pytest.mark.asyncio
    async def test_gossip_applies_to_registry(self):
        registry = CapabilityRegistry()
        engine = NodeStateEngine("server")

        gossip = GossipSync(node_id="server")
        gossip.set_capability_registry(registry)
        gossip.set_node_state(engine)

        msg = GossipMessage(
            source_node="car",
            entries=[
                GossipEntry(
                    node_id="car",
                    key="capabilities",
                    value=["obd2", "dashcam"],
                    version=1,
                    originator="car",
                ),
                GossipEntry(node_id="car", key="role", value="edge", version=1, originator="car"),
                GossipEntry(
                    node_id="car", key="device_tier", value="edge", version=1, originator="car"
                ),
            ],
        )
        await gossip.receive(msg)

        assert "obd2" in registry.get_all_capabilities()
        assert "dashcam" in registry.get_all_capabilities()
        assert len(engine.reachable_peers) == 1

    def test_get_state_for_node(self):
        gossip = GossipSync(node_id="server")
        gossip._state[("mobile", "role")] = GossipEntry(
            node_id="mobile", key="role", value="edge", version=1
        )
        gossip._state[("mobile", "load")] = GossipEntry(
            node_id="mobile", key="load", value=0.5, version=1
        )
        state = gossip.get_state_for_node("mobile")
        assert state["role"] == "edge"
        assert state["load"] == 0.5

    def test_stats(self):
        gossip = GossipSync(node_id="server")
        stats = gossip.get_stats()
        assert stats["node_id"] == "server"
        assert stats["messages_sent"] == 0


# ──────────────────────────────────────────────
# RIL tests
# ──────────────────────────────────────────────


class TestRoutingIntelligenceLayer:
    @pytest.mark.asyncio
    async def test_register_transport(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        t = InProcessTransport(transport_id="ipc")
        ril.register_transport(t)
        assert ril.get_transport("ipc") is t

    @pytest.mark.asyncio
    async def test_deregister_transport(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        t = InProcessTransport(transport_id="ipc")
        ril.register_transport(t)
        ril.deregister_transport("ipc")
        assert ril.get_transport("ipc") is None

    @pytest.mark.asyncio
    async def test_publish_routes_to_transport(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        t = InProcessTransport(transport_id="ipc")
        ril.register_transport(t)
        await ril.start()

        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await t.subscribe("test", handler)

        msg = Message(type=MessageType.EVENT, source_node_id="s", topic="test")
        await ril.publish("test", msg)
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert "_execution_context" in received[0].payload
        await ril.stop()

    @pytest.mark.asyncio
    async def test_publish_when_stopped(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        msg = Message(type=MessageType.EVENT, source_node_id="s", topic="t")
        # Should not raise
        await ril.publish("t", msg)

    @pytest.mark.asyncio
    async def test_request_when_stopped(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        msg = Message(type=MessageType.QUERY, source_node_id="s", topic="t")
        with pytest.raises(RuntimeError, match="not running"):
            await ril.request("t", msg)

    @pytest.mark.asyncio
    async def test_subscribe_across_transports(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        t = InProcessTransport(transport_id="ipc")
        ril.register_transport(t)
        await ril.start()

        async def handler(msg: Message) -> None:
            pass

        sub = await ril.subscribe("topic", handler)
        assert sub is not None
        assert ril.has_subscribers("topic")
        await ril.stop()

    @pytest.mark.asyncio
    async def test_load_penalty_applied(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        t = InProcessTransport(transport_id="ipc")
        ril.register_transport(t)

        # Without NodeState
        score_no_load = ril._score_transport(t, "test")

        # With overloaded NodeState
        engine = NodeStateEngine("server")
        engine.update_load(cpu_load=1.0, memory_load=1.0, task_queue_depth=100)
        ril.set_node_state(engine)

        score_with_load = ril._score_transport(t, "test")
        assert score_with_load < score_no_load  # Load penalty applied

    @pytest.mark.asyncio
    async def test_interceptor_blocks_message(self):
        ril = RoutingIntelligenceLayer(node_id="server")
        t = InProcessTransport(transport_id="ipc")
        ril.register_transport(t)
        await ril.start()

        received = []

        async def handler(msg: Message) -> None:
            received.append(msg)

        await t.subscribe("test", handler)

        async def block_all(msg: Message) -> Message | None:
            return None  # Block

        ril.add_interceptor(block_all)

        msg = Message(type=MessageType.EVENT, source_node_id="s", topic="test")
        await ril.publish("test", msg)
        await asyncio.sleep(0.1)
        assert len(received) == 0
        await ril.stop()
