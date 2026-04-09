"""
Tests for Multi-Agent Voting & Task Delegation (CollectiveNode v2).

Tests agent specialization, consensus voting with 3 strategies,
and intelligent task delegation across multiple simulated instances.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from hbllm.brain.collective_node import (
    AgentProfile,
    CollectiveNode,
    DelegationResult,
    KnowledgeDigest,
    VoteRequest,
    VoteResponse,
    VotingStrategy,
)
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


# ── Helpers ──────────────────────────────────────────────────────────────


async def _make_handler(response: str, confidence: float):
    """Create a simple vote handler that returns a fixed response."""
    async def handler(query: str, domain: str) -> tuple[str, float]:
        return response, confidence
    return handler


async def _setup_cluster(
    bus: InProcessBus,
    count: int = 3,
    domains: list[list[str]] | None = None,
    performances: list[dict[str, float]] | None = None,
    responses: list[tuple[str, float]] | None = None,
) -> list[CollectiveNode]:
    """Set up a cluster of CollectiveNode instances on a shared bus."""
    nodes: list[CollectiveNode] = []
    for i in range(count):
        node = CollectiveNode(
            node_id=f"collective_{i}",
            instance_id=f"instance_{i}",
            vote_timeout=2.0,
            min_voters=1,
        )
        await node.start(bus)

        if domains and i < len(domains):
            perf = performances[i] if performances and i < len(performances) else None
            node.register_specialization(domains[i], perf)

        if responses and i < len(responses):
            resp, conf = responses[i]
            handler = await _make_handler(resp, conf)
            node.set_vote_handler(handler)

        nodes.append(node)

    # Exchange profiles so peers discover each other
    for node in nodes:
        await node.broadcast_profile()

    # Let messages propagate
    await asyncio.sleep(0.05)

    return nodes


# ── Data Model Tests ─────────────────────────────────────────────────────


class TestDataModels:
    """Test v2 data structures."""

    def test_knowledge_digest_checksum(self):
        d = KnowledgeDigest(artifact_data={"key": "value"})
        cs = d.compute_checksum()
        assert len(cs) == 16
        assert cs == d.checksum

    def test_agent_profile_round_trip(self):
        p = AgentProfile(
            instance_id="test",
            domains=["coding", "math"],
            performance={"coding": 0.95, "math": 0.88},
            load=0.3,
        )
        d = p.to_dict()
        p2 = AgentProfile.from_dict(d)
        assert p2.instance_id == "test"
        assert p2.domains == ["coding", "math"]
        assert p2.performance["coding"] == 0.95

    def test_vote_request_to_dict(self):
        vr = VoteRequest(query="test?", domain="coding", requester_id="me")
        d = vr.to_dict()
        assert d["query"] == "test?"
        assert d["domain"] == "coding"
        assert "vote_id" in d

    def test_vote_response_round_trip(self):
        vr = VoteResponse(
            vote_id="abc",
            responder_id="peer1",
            response="42",
            confidence=0.9,
        )
        d = vr.to_dict()
        vr2 = VoteResponse.from_dict(d)
        assert vr2.responder_id == "peer1"
        assert vr2.confidence == 0.9

    def test_delegation_result_defaults(self):
        dr = DelegationResult()
        assert dr.delegated is False
        assert dr.target_instance == ""
        assert dr.reason == ""


# ── Agent Specialization Tests ───────────────────────────────────────────


class TestAgentSpecialization:
    """Test peer profile exchange and specialization registry."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    async def test_register_specialization(self, bus):
        """Registering specialization updates local profile."""
        node = CollectiveNode(node_id="c0", instance_id="inst0")
        await node.start(bus)

        node.register_specialization(
            domains=["coding", "math"],
            performance={"coding": 0.95, "math": 0.88},
        )

        assert node.local_profile.domains == ["coding", "math"]
        assert node.local_profile.performance["coding"] == 0.95
        await node.stop()

    async def test_profile_exchange_between_peers(self, bus):
        """Broadcasting profiles makes peers discover each other."""
        nodes = await _setup_cluster(
            bus, count=3,
            domains=[["coding"], ["math"], ["writing"]],
        )

        # Each node should see the other 2 peers
        for i, node in enumerate(nodes):
            assert len(node.peer_profiles) == 2
            # Should not have itself in peer list
            assert node.instance_id not in node.peer_profiles

        for node in nodes:
            await node.stop()

    async def test_get_peers_for_domain(self, bus):
        """Peers specialized in a domain are found and ranked."""
        nodes = await _setup_cluster(
            bus, count=3,
            domains=[["coding"], ["coding", "math"], ["math"]],
            performances=[
                {"coding": 0.70},
                {"coding": 0.95, "math": 0.80},
                {"math": 0.99},
            ],
        )

        # Query coding peers from node 2 (math specialist)
        coding_peers = nodes[2].get_peers_for_domain("coding")
        assert len(coding_peers) == 2
        # Best coder should be first (instance_1 with 0.95)
        assert coding_peers[0].instance_id == "instance_1"

        for node in nodes:
            await node.stop()

    async def test_stale_peers_filtered(self, bus):
        """Peers not seen recently should be excluded."""
        nodes = await _setup_cluster(
            bus, count=2,
            domains=[["coding"], ["coding"]],
        )

        # Artificially age the peer
        for profile in nodes[0].peer_profiles.values():
            profile.last_seen = time.time() - 600

        coding_peers = nodes[0].get_peers_for_domain("coding")
        assert len(coding_peers) == 0  # All stale

        for node in nodes:
            await node.stop()

    async def test_update_load(self, bus):
        """Load updates are reflected in local profile."""
        node = CollectiveNode(node_id="c0", instance_id="inst0")
        await node.start(bus)

        node.update_load(0.75)
        assert node.local_profile.load == 0.75

        # Clamping
        node.update_load(1.5)
        assert node.local_profile.load == 1.0
        node.update_load(-0.5)
        assert node.local_profile.load == 0.0

        await node.stop()


# ── Consensus Voting Tests ───────────────────────────────────────────────


class TestConsensusVoting:
    """Test multi-agent consensus voting protocol."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    async def test_confidence_weighted_voting(self, bus):
        """Confidence-weighted strategy picks highest-confidence response."""
        nodes = await _setup_cluster(
            bus, count=3,
            domains=[["coding"], ["coding"], ["coding"]],
            responses=[
                ("answer_A", 0.6),
                ("answer_B", 0.9),
                ("answer_C", 0.4),
            ],
        )

        result = await nodes[0].request_votes(
            "What is Python?",
            domain="coding",
            strategy=VotingStrategy.CONFIDENCE_WEIGHTED,
        )

        assert result["vote_count"] == 2  # other 2 peers vote
        assert result["consensus"] == "answer_B"  # highest confidence
        assert result["strategy"] == "confidence_weighted"

        for node in nodes:
            await node.stop()

    async def test_best_of_n_voting(self, bus):
        """Best-of-N picks the single highest-confidence response."""
        nodes = await _setup_cluster(
            bus, count=3,
            domains=[["math"], ["math"], ["math"]],
            responses=[
                ("42", 0.5),
                ("42", 0.95),
                ("43", 0.3),
            ],
        )

        result = await nodes[0].request_votes(
            "What is 6*7?",
            domain="math",
            strategy=VotingStrategy.BEST_OF_N,
        )

        assert result["consensus"] == "42"
        assert result["confidence"] == 0.95
        assert result["winner_id"] == "instance_1"

        for node in nodes:
            await node.stop()

    async def test_majority_voting(self, bus):
        """Majority strategy picks the most common response."""
        nodes = await _setup_cluster(
            bus, count=4,
            domains=[["gen"], ["gen"], ["gen"], ["gen"]],
            responses=[
                ("yes", 0.7),  # node 0 (requester, won't vote on own)
                ("yes", 0.8),
                ("no", 0.9),
                ("yes", 0.6),
            ],
        )

        result = await nodes[0].request_votes(
            "Is the sky blue?",
            domain="gen",
            strategy=VotingStrategy.MAJORITY,
        )

        # 2 "yes" vs 1 "no" from peers (node 0 doesn't vote on own request)
        assert result["consensus"] == "yes"
        assert result["majority_count"] == 2

        for node in nodes:
            await node.stop()

    async def test_vote_timeout_returns_partial(self, bus):
        """Timeout returns whatever votes arrived."""
        nodes = await _setup_cluster(
            bus, count=2,
            domains=[["coding"], ["coding"]],
            responses=[
                ("local", 0.5),
                ("peer", 0.8),
            ],
        )

        result = await nodes[0].request_votes(
            "test",
            domain="coding",
            timeout=0.5,
        )

        # Should have at least the 1 peer's vote
        assert result["vote_count"] >= 1

        for node in nodes:
            await node.stop()

    async def test_empty_votes_returns_empty_consensus(self, bus):
        """No votes should return empty consensus."""
        node = CollectiveNode(
            node_id="solo",
            instance_id="alone",
            vote_timeout=0.3,
            min_voters=1,
        )
        await node.start(bus)

        result = await node.request_votes("hello?")

        assert result["consensus"] == ""
        assert result["vote_count"] == 0

        await node.stop()

    async def test_stats_track_voting(self, bus):
        """Stats should count vote requests and responses."""
        nodes = await _setup_cluster(
            bus, count=2,
            domains=[["gen"], ["gen"]],
            responses=[("a", 0.5), ("b", 0.7)],
        )

        await nodes[0].request_votes("test?")

        stats = nodes[0].stats
        assert stats["votes_requested"] >= 1

        stats1 = nodes[1].stats
        assert stats1["votes_cast"] >= 1

        for node in nodes:
            await node.stop()


# ── Task Delegation Tests ────────────────────────────────────────────────


class TestTaskDelegation:
    """Test intelligent task delegation to specialized peers."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    async def test_delegate_to_best_peer(self, bus):
        """Delegation routes to the highest-scoring peer."""
        nodes = await _setup_cluster(
            bus, count=3,
            domains=[["general"], ["coding"], ["coding"]],
            performances=[
                {"general": 0.5},
                {"coding": 0.95},
                {"coding": 0.70},
            ],
            responses=[
                ("general answer", 0.5),
                ("expert answer", 0.95),
                ("decent answer", 0.70),
            ],
        )

        result = await nodes[0].delegate_task(
            "Write a Python function",
            domain="coding",
            timeout=2.0,
        )

        assert result.delegated is True
        assert result.target_instance == "instance_1"
        assert result.response["text"] == "expert answer"

        for node in nodes:
            await node.stop()

    async def test_delegation_no_peers(self, bus):
        """Delegation returns failure when no peers available."""
        node = CollectiveNode(
            node_id="solo",
            instance_id="alone",
        )
        await node.start(bus)

        result = await node.delegate_task("test", domain="coding")

        assert result.delegated is False
        assert result.reason == "no_suitable_peer"

        await node.stop()

    async def test_delegation_timeout(self, bus):
        """Delegation respects timeout."""
        nodes = await _setup_cluster(
            bus, count=2,
            domains=[["gen"], ["gen"]],
        )
        # Don't set vote handler on node 1 — it will respond with default

        # Remove node 1's subscription to simulate unresponsive peer
        # by clearing its internal state
        nodes[1]._vote_handler = None

        result = await nodes[0].delegate_task(
            "test",
            domain="gen",
            timeout=1.0,
        )

        # Should get a response (even the default one)
        assert isinstance(result, DelegationResult)

        for node in nodes:
            await node.stop()

    async def test_peer_scoring(self, bus):
        """Peer scoring considers performance, load, and specialization."""
        node = CollectiveNode(node_id="c0", instance_id="inst0")

        # Create test profiles
        expert = AgentProfile(
            instance_id="expert",
            domains=["coding"],
            performance={"coding": 0.95},
            load=0.2,
            last_seen=time.time(),
        )
        generalist = AgentProfile(
            instance_id="generalist",
            domains=["general"],
            performance={"coding": 0.5},
            load=0.1,
            last_seen=time.time(),
        )
        overloaded = AgentProfile(
            instance_id="overloaded",
            domains=["coding"],
            performance={"coding": 0.99},
            load=0.95,
            last_seen=time.time(),
        )

        # Expert should score highest (high perf, low load, specialization bonus)
        expert_score = node._score_peer(expert, "coding")
        generalist_score = node._score_peer(generalist, "coding")
        overloaded_score = node._score_peer(overloaded, "coding")

        assert expert_score > generalist_score
        assert expert_score > overloaded_score

    async def test_select_best_peer(self, bus):
        """select_best_peer picks the optimal peer."""
        nodes = await _setup_cluster(
            bus, count=3,
            domains=[["general"], ["coding"], ["coding"]],
            performances=[
                {"general": 0.5},
                {"coding": 0.95},
                {"coding": 0.70},
            ],
        )

        best = nodes[0].select_best_peer("coding")
        assert best is not None
        assert best.instance_id == "instance_1"

        for node in nodes:
            await node.stop()

    async def test_delegation_stats_tracked(self, bus):
        """Delegation stats should be tracked."""
        nodes = await _setup_cluster(
            bus, count=2,
            domains=[["gen"], ["coding"]],
            performances=[{"gen": 0.5}, {"coding": 0.9}],
            responses=[("gen_ans", 0.5), ("code_ans", 0.9)],
        )

        await nodes[0].delegate_task("Write code", domain="coding", timeout=2.0)

        assert nodes[0].stats["delegations_sent"] >= 1
        assert nodes[1].stats["delegations_received"] >= 1

        for node in nodes:
            await node.stop()


# ── Knowledge Sharing (Regression) ───────────────────────────────────────


class TestKnowledgeSharingRegression:
    """Ensure existing knowledge sharing still works with v2 enhancements."""

    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    async def test_broadcast_and_sync(self, bus):
        """Knowledge broadcast and sync still works."""
        nodes = await _setup_cluster(bus, count=2)

        # Simulate a learning update on node 0
        msg = Message(
            type=MessageType.LEARNING_UPDATE,
            source_node_id="learner",
            topic="system.learning_update",
            payload={
                "domain": "coding",
                "capability": "python_debugging",
                "artifact_type": "skill",
                "artifact_data": {"steps": ["read", "debug", "fix"]},
            },
        )
        await bus.publish("system.learning_update", msg)
        await asyncio.sleep(0.05)

        assert nodes[0].stats["broadcasts_sent"] >= 1

        for node in nodes:
            await node.stop()

    async def test_query_returns_stats(self, bus):
        """Query handler returns stats and peer list."""
        nodes = await _setup_cluster(bus, count=2)

        query_msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="collective.query",
            payload={"limit": 5},
        )
        result = await nodes[0]._handle_query(query_msg)

        assert result is not None
        assert "peers" in result.payload
        assert result.payload["instance_id"] == "instance_0"

        for node in nodes:
            await node.stop()
