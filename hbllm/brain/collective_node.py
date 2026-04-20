"""
Collective Intelligence Network — cross-instance knowledge sharing,
multi-agent consensus voting, and task delegation.

Allows multiple HBLLM instances to share learned knowledge, reach
consensus on complex queries via voting protocols, and delegate tasks
to specialized peers based on domain expertise and load.

Communication happens over the RedisBus (for cross-process) or
InProcessBus (for testing).

v2 enhancements:
- Agent specialization profiles with peer discovery
- Multi-agent consensus voting (majority, confidence-weighted, best-of-n)
- Intelligent task delegation with expertise × load scoring
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Data Structures ─────────────────────────────────────────────────────


@dataclass
class KnowledgeDigest:
    """A shareable knowledge artifact from one instance."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_instance_id: str = ""
    domain: str = ""
    capability: str = ""
    artifact_type: str = ""  # "lora_weights", "skill", "semantic_fact", "identity_update"
    artifact_data: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    timestamp: float = field(default_factory=time.time)

    def compute_checksum(self) -> str:
        """Compute a content-based checksum for deduplication."""
        content = json.dumps(self.artifact_data, sort_keys=True)
        self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.checksum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentProfile:
    """Specialization profile for a peer instance."""

    instance_id: str = ""
    domains: list[str] = field(default_factory=list)
    performance: dict[str, float] = field(default_factory=dict)  # domain → success_rate
    load: float = 0.0  # 0.0 (idle) to 1.0 (maxed)
    capabilities: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProfile:
        return cls(
            instance_id=data.get("instance_id", ""),
            domains=data.get("domains", []),
            performance=data.get("performance", {}),
            load=data.get("load", 0.0),
            capabilities=data.get("capabilities", []),
            last_seen=data.get("last_seen", time.time()),
        )


class VotingStrategy(str, Enum):
    """Strategies for multi-agent consensus."""

    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BEST_OF_N = "best_of_n"


@dataclass
class VoteRequest:
    """A request for peer votes on a query."""

    vote_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str = ""
    domain: str = ""
    strategy: str = "confidence_weighted"
    requester_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VoteResponse:
    """A peer's vote/response to a voting request."""

    vote_id: str = ""
    responder_id: str = ""
    response: str = ""
    confidence: float = 0.0
    domain: str = ""
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoteResponse:
        return cls(
            vote_id=data.get("vote_id", ""),
            responder_id=data.get("responder_id", ""),
            response=data.get("response", ""),
            confidence=data.get("confidence", 0.0),
            domain=data.get("domain", ""),
            reasoning=data.get("reasoning", ""),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class DelegationResult:
    """Result of a task delegation attempt."""

    delegated: bool = False
    target_instance: str = ""
    score: float = 0.0
    reason: str = ""
    response: dict[str, Any] = field(default_factory=dict)


# ── CollectiveNode ──────────────────────────────────────────────────────


class CollectiveNode(Node):
    """
    Service node for cross-instance knowledge sharing, consensus
    voting, and task delegation.

    Subscribes to:
        system.learning_update — local learning completions
        collective.sync — incoming knowledge from peer instances
        collective.query — query the received knowledge log
        collective.profile — peer specialization announcements
        collective.vote.request — incoming vote requests from peers
        collective.vote.response — incoming vote responses from peers
        collective.delegate — incoming delegated tasks from peers

    Publishes:
        collective.broadcast — outgoing knowledge digests to peers
        collective.profile — own specialization profile
        collective.vote.request — outgoing vote requests
        collective.vote.response — outgoing vote responses
        collective.delegate — outgoing task delegations
        collective.delegate.response — delegation results
    """

    def __init__(
        self,
        node_id: str,
        instance_id: str = "",
        max_received: int = 500,
        vote_timeout: float = 5.0,
        min_voters: int = 2,
        default_strategy: VotingStrategy = VotingStrategy.CONFIDENCE_WEIGHTED,
        peer_stale_seconds: float = 300.0,
        skill_registry: Any = None,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=[
                "collective_intelligence",
                "knowledge_sharing",
                "consensus_voting",
                "task_delegation",
            ],
        )
        self.instance_id = instance_id or uuid.uuid4().hex[:8]
        self.max_received = max_received

        # Knowledge tracking
        self.broadcast_log: list[KnowledgeDigest] = []
        self.received_log: list[KnowledgeDigest] = []
        self.seen_checksums: set[str] = set()

        # v2: Agent specialization
        self.local_profile = AgentProfile(
            instance_id=self.instance_id,
        )
        self.peer_profiles: dict[str, AgentProfile] = {}
        self.peer_stale_seconds = peer_stale_seconds

        # v2: Consensus voting
        self.vote_timeout = vote_timeout
        self.min_voters = min_voters
        self.default_strategy = default_strategy
        self._pending_votes: dict[str, list[VoteResponse]] = {}
        self._vote_events: dict[str, asyncio.Event] = {}
        self._vote_handler: Any = None  # callable for local vote generation
        self.skill_registry = skill_registry

        # Stats
        self._stats = {
            "broadcasts_sent": 0,
            "digests_received": 0,
            "digests_integrated": 0,
            "duplicates_filtered": 0,
            "votes_requested": 0,
            "votes_cast": 0,
            "votes_received": 0,
            "delegations_sent": 0,
            "delegations_received": 0,
            "delegations_completed": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Return collective intelligence statistics."""
        return dict(self._stats)

    async def on_start(self) -> None:
        logger.info(
            "Starting CollectiveNode (instance=%s, voting=%s)",
            self.instance_id,
            self.default_strategy.value,
        )
        # Knowledge sharing
        await self.bus.subscribe("system.learning_update", self._handle_learning_update)
        await self.bus.subscribe("collective.sync", self._handle_sync)
        await self.bus.subscribe("collective.sync_skills", self._handle_sync_skills)
        await self.bus.subscribe("collective.query", self._handle_query)

        # v2: Specialization
        await self.bus.subscribe("collective.profile", self._handle_profile)

        # v2: Consensus voting
        await self.bus.subscribe("collective.vote.request", self._handle_vote_request)
        await self.bus.subscribe("collective.vote.response", self._handle_vote_response)

        # v2: Task delegation
        await self.bus.subscribe("collective.delegate", self._handle_delegation)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping CollectiveNode — sent=%d, received=%d, votes_requested=%d, delegations=%d",
            self._stats["broadcasts_sent"],
            self._stats["digests_received"],
            self._stats["votes_requested"],
            self._stats["delegations_sent"],
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Knowledge Sharing (existing) ─────────────────────────────────

    async def _handle_learning_update(self, message: Message) -> Message | None:
        """
        A local learning event completed — broadcast it to peers.
        Expected payload:
            domain: str
            capability: str
            artifact_type: str
            artifact_data: dict
        """
        payload = message.payload

        digest = KnowledgeDigest(
            source_instance_id=self.instance_id,
            domain=payload.get("domain", "general"),
            capability=payload.get("capability", ""),
            artifact_type=payload.get("artifact_type", "skill"),
            artifact_data=payload.get("artifact_data", {}),
        )
        digest.compute_checksum()

        # Avoid re-broadcasting our own digests
        if digest.checksum in self.seen_checksums:
            return None

        self.seen_checksums.add(digest.checksum)
        self.broadcast_log.append(digest)
        self._stats["broadcasts_sent"] += 1

        # Broadcast to peers
        await self.publish(
            "collective.broadcast",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="collective.broadcast",
                payload=digest.to_dict(),
            ),
        )

        logger.info(
            "Broadcast knowledge digest: domain=%s capability=%s checksum=%s",
            digest.domain,
            digest.capability,
            digest.checksum,
        )
        return None

    async def _handle_sync_skills(self, message: Message) -> Message | None:
        """Promote highly-confident local skills to the global tier and broadcast to peers."""
        if not self.skill_registry:
            return None

        # Fetch highly confident skills that are NOT already global
        tenant_id = message.payload.get("tenant_id", "default")
        # We find top 20 skills
        local_skills = self.skill_registry.list_skills(limit=100, tenant_id=tenant_id)

        promoted_count = 0
        for skill in local_skills:
            # If skill is highly confident, locally isolated, we promote it
            conf = skill.confidence_score
            is_global = (skill.tenant_id == "global")

            if conf > 0.95 and not is_global:
                # 1. Promote locally
                self.skill_registry.promote_skill(skill.skill_id)

                # 2. Re-fetch or manually update the skill tenant_id for the payload
                skill.tenant_id = "global"

                # 3. Create generic digest
                digest = KnowledgeDigest(
                    source_instance_id=self.instance_id,
                    domain=skill.category,
                    capability=skill.name,
                    artifact_type="skill",
                    artifact_data={
                        "steps": skill.steps,
                        "description": skill.description
                    },
                )
                digest.compute_checksum()

                if digest.checksum not in self.seen_checksums:
                    self.seen_checksums.add(digest.checksum)
                    self.broadcast_log.append(digest)
                    self._stats["broadcasts_sent"] += 1

                    await self.publish(
                        "collective.broadcast",
                        Message(
                            type=MessageType.EVENT,
                            source_node_id=self.node_id,
                            topic="collective.broadcast",
                            payload=digest.to_dict(),
                        ),
                    )
                promoted_count += 1

        if promoted_count > 0:
            logger.info("Promoted and broadcast %d elite skills for tenant %s", promoted_count, tenant_id)

        return None

    async def _handle_sync(self, message: Message) -> Message | None:
        """
        Receive a knowledge digest from a peer instance.
        Deduplicates by checksum, stores, and integrates into the local brain.
        """
        payload = message.payload

        checksum = payload.get("checksum", "")
        source = payload.get("source_instance_id", "")

        # Skip our own broadcasts
        if source == self.instance_id:
            return None

        # Deduplicate
        if checksum in self.seen_checksums:
            self._stats["duplicates_filtered"] += 1
            return None

        digest = KnowledgeDigest(
            id=payload.get("id", uuid.uuid4().hex[:12]),
            source_instance_id=source,
            domain=payload.get("domain", ""),
            capability=payload.get("capability", ""),
            artifact_type=payload.get("artifact_type", ""),
            artifact_data=payload.get("artifact_data", {}),
            checksum=checksum,
            timestamp=payload.get("timestamp", time.time()),
        )

        self.seen_checksums.add(checksum)
        self.received_log.append(digest)
        self._stats["digests_received"] += 1

        # Trim old entries
        if len(self.received_log) > self.max_received:
            removed = self.received_log.pop(0)
            self.seen_checksums.discard(removed.checksum)

        logger.info(
            "Received knowledge from instance %s: domain=%s capability=%s",
            source,
            digest.domain,
            digest.capability,
        )

        # ── Integrate the digest into the local brain ──
        await self._integrate_digest(digest)

        return None

    async def _integrate_digest(self, digest: KnowledgeDigest) -> None:
        """
        Route a received digest to the appropriate local subsystem for
        hot-loading into the running brain.
        """
        artifact_type = digest.artifact_type
        data = digest.artifact_data

        try:
            if artifact_type == "lora_weights":
                await self.publish(
                    "system.spawn",
                    Message(
                        type=MessageType.SPAWN_REQUEST,
                        source_node_id=self.node_id,
                        topic="system.spawn",
                        payload={
                            "topic": digest.domain,
                            "trigger_query": (f"Integrated from peer {digest.source_instance_id}"),
                            "confidence_score": 0.0,
                            "adapter_path": data.get("adapter_path", ""),
                            "from_collective": True,
                        },
                    ),
                )

            elif artifact_type == "skill":
                await self.publish(
                    "memory.skill.store",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="memory.skill.store",
                        payload={
                            "name": digest.capability or digest.domain,
                            "steps": data.get("steps", []),
                            "domain": digest.domain,
                            "from_collective": True,
                        },
                    ),
                )

            elif artifact_type == "semantic_fact":
                await self.publish(
                    "memory.store",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="memory.store",
                        payload={
                            "text": data.get("text", ""),
                            "domain": digest.domain,
                            "metadata": {
                                "source": f"collective:{digest.source_instance_id}",
                            },
                            "from_collective": True,
                        },
                    ),
                )

            elif artifact_type == "identity_update":
                await self.publish(
                    "identity.update",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="identity.update",
                        payload=data,
                    ),
                )
            else:
                logger.debug(
                    "Unknown artifact type '%s' from peer %s — stored but not integrated",
                    artifact_type,
                    digest.source_instance_id,
                )
                return

            self._stats["digests_integrated"] += 1
            logger.info(
                "Integrated %s digest from peer %s (domain=%s)",
                artifact_type,
                digest.source_instance_id,
                digest.domain,
            )
        except Exception as e:
            logger.warning(
                "Failed to integrate digest %s from peer %s: %s",
                digest.id,
                digest.source_instance_id,
                e,
            )

    async def _handle_query(self, message: Message) -> Message | None:
        """Return collective intelligence stats and recent digests."""
        payload = message.payload
        limit = int(payload.get("limit", 10))

        recent_received = [d.to_dict() for d in self.received_log[-limit:]]
        recent_broadcast = [d.to_dict() for d in self.broadcast_log[-limit:]]

        return message.create_response(
            {
                "instance_id": self.instance_id,
                "stats": self._stats,
                "recent_received": recent_received,
                "recent_broadcast": recent_broadcast,
                "peers": {pid: p.to_dict() for pid, p in self.peer_profiles.items()},
            }
        )

    # ── Agent Specialization ─────────────────────────────────────────

    def register_specialization(
        self,
        domains: list[str],
        performance: dict[str, float] | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        """Declare this instance's domain expertise."""
        self.local_profile.domains = domains
        if performance:
            self.local_profile.performance = performance
        if capabilities:
            self.local_profile.capabilities = capabilities
        logger.info("Registered specialization: domains=%s", domains)

    async def broadcast_profile(self) -> None:
        """Broadcast this instance's specialization profile to peers."""
        self.local_profile.last_seen = time.time()
        await self.publish(
            "collective.profile",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="collective.profile",
                payload=self.local_profile.to_dict(),
            ),
        )

    def update_load(self, load: float) -> None:
        """Update this instance's current load (0.0–1.0)."""
        self.local_profile.load = max(0.0, min(1.0, load))

    async def _handle_profile(self, message: Message) -> Message | None:
        """Receive a peer's specialization profile."""
        payload = message.payload
        peer_id = payload.get("instance_id", "")

        # Skip our own profile
        if peer_id == self.instance_id:
            return None

        profile = AgentProfile.from_dict(payload)
        profile.last_seen = time.time()
        self.peer_profiles[peer_id] = profile

        logger.debug(
            "Updated peer profile: %s domains=%s load=%.2f",
            peer_id,
            profile.domains,
            profile.load,
        )
        return None

    def get_peers_for_domain(self, domain: str) -> list[AgentProfile]:
        """Get peers specialized in a given domain, sorted by performance."""
        now = time.time()
        candidates = []
        for profile in self.peer_profiles.values():
            # Skip stale peers
            if now - profile.last_seen > self.peer_stale_seconds:
                continue
            if domain in profile.domains:
                candidates.append(profile)

        # Sort by domain performance (descending), then load (ascending)
        candidates.sort(
            key=lambda p: (
                -p.performance.get(domain, 0.5),
                p.load,
            )
        )
        return candidates

    # ── Consensus Voting ─────────────────────────────────────────────

    def set_vote_handler(self, handler: Any) -> None:
        """
        Set the callable used to generate local vote responses.

        handler signature: async def handler(query: str, domain: str)
            -> tuple[str, float]  # (response_text, confidence)
        """
        self._vote_handler = handler

    async def request_votes(
        self,
        query: str,
        domain: str = "general",
        strategy: VotingStrategy | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Request votes from all peers on a query.

        Returns a dict with:
            consensus: str — the winning response
            confidence: float — aggregate confidence
            vote_count: int — number of votes received
            strategy: str — strategy used
            votes: list[dict] — all individual votes
        """
        vote_strategy = strategy or self.default_strategy
        vote_timeout = timeout or self.vote_timeout

        vote_req = VoteRequest(
            query=query,
            domain=domain,
            strategy=vote_strategy.value,
            requester_id=self.instance_id,
        )

        # Prepare collection
        self._pending_votes[vote_req.vote_id] = []
        self._vote_events[vote_req.vote_id] = asyncio.Event()

        # Broadcast vote request
        await self.publish(
            "collective.vote.request",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="collective.vote.request",
                payload=vote_req.to_dict(),
            ),
        )
        self._stats["votes_requested"] += 1

        # Wait for responses with timeout
        try:
            await asyncio.wait_for(
                self._wait_for_votes(vote_req.vote_id),
                timeout=vote_timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(
                "Vote collection timed out after %.1fs (got %d votes)",
                vote_timeout,
                len(self._pending_votes.get(vote_req.vote_id, [])),
            )

        # Tally
        votes = self._pending_votes.pop(vote_req.vote_id, [])
        self._vote_events.pop(vote_req.vote_id, None)

        return self._tally_votes(votes, vote_strategy)

    async def _wait_for_votes(self, vote_id: str) -> None:
        """Wait until we have enough votes or all peers have responded."""
        peer_count = len(self.peer_profiles)
        target = max(self.min_voters, peer_count)

        while len(self._pending_votes.get(vote_id, [])) < target:
            event = self._vote_events.get(vote_id)
            if event is None:
                return
            event.clear()
            await event.wait()

    async def _handle_vote_request(self, message: Message) -> Message | None:
        """Receive a vote request from a peer — generate and send response."""
        payload = message.payload
        requester = payload.get("requester_id", "")

        # Don't vote on our own requests
        if requester == self.instance_id:
            return None

        vote_id = payload.get("vote_id", "")
        query = payload.get("query", "")
        domain = payload.get("domain", "general")

        # Generate a response
        response_text = ""
        confidence = 0.0

        if self._vote_handler:
            try:
                response_text, confidence = await self._vote_handler(query, domain)
            except Exception as e:
                logger.warning("Vote handler failed: %s", e)
                response_text = ""
                confidence = 0.0

        if not response_text:
            # Default fallback: acknowledge but no substantive vote
            response_text = f"[{self.instance_id}] No response available"
            confidence = 0.0

        vote_resp = VoteResponse(
            vote_id=vote_id,
            responder_id=self.instance_id,
            response=response_text,
            confidence=confidence,
            domain=domain,
        )

        await self.publish(
            "collective.vote.response",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="collective.vote.response",
                payload=vote_resp.to_dict(),
            ),
        )
        self._stats["votes_cast"] += 1

        return None

    async def _handle_vote_response(self, message: Message) -> Message | None:
        """Receive a vote response from a peer."""
        payload = message.payload
        responder = payload.get("responder_id", "")

        # Skip our own responses
        if responder == self.instance_id:
            return None

        vote_id = payload.get("vote_id", "")
        if vote_id not in self._pending_votes:
            return None

        vote = VoteResponse.from_dict(payload)
        self._pending_votes[vote_id].append(vote)
        self._stats["votes_received"] += 1

        # Signal the waiter
        event = self._vote_events.get(vote_id)
        if event:
            event.set()

        return None

    def _tally_votes(
        self,
        votes: list[VoteResponse],
        strategy: VotingStrategy,
    ) -> dict[str, Any]:
        """
        Aggregate votes using the specified strategy.

        Returns consensus result dict.
        """
        if not votes:
            return {
                "consensus": "",
                "confidence": 0.0,
                "vote_count": 0,
                "strategy": strategy.value,
                "votes": [],
            }

        vote_dicts = [v.to_dict() for v in votes]

        if strategy == VotingStrategy.BEST_OF_N:
            # Simply pick the highest-confidence response
            best = max(votes, key=lambda v: v.confidence)
            return {
                "consensus": best.response,
                "confidence": best.confidence,
                "vote_count": len(votes),
                "strategy": strategy.value,
                "votes": vote_dicts,
                "winner_id": best.responder_id,
            }

        elif strategy == VotingStrategy.MAJORITY:
            # Group by response text, pick most common
            response_counts: dict[str, list[VoteResponse]] = {}
            for v in votes:
                key = v.response.strip().lower()
                if key not in response_counts:
                    response_counts[key] = []
                response_counts[key].append(v)

            # Find the group with the most votes
            best_group = max(response_counts.values(), key=lambda g: len(g))
            # Use the highest-confidence response from the winning group
            best_vote = max(best_group, key=lambda v: v.confidence)
            avg_confidence = sum(v.confidence for v in best_group) / len(best_group)

            return {
                "consensus": best_vote.response,
                "confidence": avg_confidence,
                "vote_count": len(votes),
                "majority_count": len(best_group),
                "strategy": strategy.value,
                "votes": vote_dicts,
            }

        else:  # CONFIDENCE_WEIGHTED (default)
            # Weight responses by confidence and pick the highest
            total_confidence = sum(v.confidence for v in votes)
            if total_confidence == 0:
                best = votes[0]
            else:
                best = max(votes, key=lambda v: v.confidence)

            # Aggregate confidence: weighted average
            avg_confidence = total_confidence / len(votes) if votes else 0.0

            return {
                "consensus": best.response,
                "confidence": avg_confidence,
                "vote_count": len(votes),
                "strategy": strategy.value,
                "votes": vote_dicts,
                "winner_id": best.responder_id,
            }

    # ── Task Delegation ──────────────────────────────────────────────

    def _score_peer(self, profile: AgentProfile, domain: str) -> float:
        """
        Score a peer for task delegation.
        Score = domain_performance × (1 - load) × availability_bonus
        """
        perf = profile.performance.get(domain, 0.3)
        load_factor = 1.0 - profile.load
        # Bonus for peers that actively specialize in this domain
        specialization_bonus = 1.2 if domain in profile.domains else 1.0

        return perf * load_factor * specialization_bonus

    def select_best_peer(self, domain: str) -> AgentProfile | None:
        """Select the best peer for a domain based on scoring."""
        candidates = self.get_peers_for_domain(domain)

        if not candidates:
            # Try all non-stale peers as fallback
            now = time.time()
            candidates = [
                p
                for p in self.peer_profiles.values()
                if now - p.last_seen <= self.peer_stale_seconds
            ]

        if not candidates:
            return None

        return max(candidates, key=lambda p: self._score_peer(p, domain))

    async def delegate_task(
        self,
        query: str,
        domain: str = "general",
        timeout: float = 10.0,
    ) -> DelegationResult:
        """
        Delegate a task to the best available peer.

        Returns a DelegationResult indicating success/failure and
        the peer's response.
        """
        best_peer = self.select_best_peer(domain)

        if best_peer is None:
            return DelegationResult(
                delegated=False,
                reason="no_suitable_peer",
            )

        delegation_id = uuid.uuid4().hex[:12]

        # Set up response collection
        self._pending_votes[delegation_id] = []
        self._vote_events[delegation_id] = asyncio.Event()

        # Send delegation request
        await self.publish(
            "collective.delegate",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="collective.delegate",
                payload={
                    "delegation_id": delegation_id,
                    "query": query,
                    "domain": domain,
                    "requester_id": self.instance_id,
                    "target_instance": best_peer.instance_id,
                },
            ),
        )
        self._stats["delegations_sent"] += 1

        # Wait for response
        try:
            await asyncio.wait_for(
                self._vote_events[delegation_id].wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._pending_votes.pop(delegation_id, None)
            self._vote_events.pop(delegation_id, None)
            return DelegationResult(
                delegated=False,
                target_instance=best_peer.instance_id,
                reason="timeout",
            )

        responses = self._pending_votes.pop(delegation_id, [])
        self._vote_events.pop(delegation_id, None)

        if responses:
            resp = responses[0]
            return DelegationResult(
                delegated=True,
                target_instance=resp.responder_id,
                score=resp.confidence,
                response={
                    "text": resp.response,
                    "confidence": resp.confidence,
                    "domain": resp.domain,
                },
            )

        return DelegationResult(
            delegated=False,
            target_instance=best_peer.instance_id,
            reason="no_response",
        )

    async def _handle_delegation(self, message: Message) -> Message | None:
        """Receive a delegated task from a peer."""
        payload = message.payload
        requester = payload.get("requester_id", "")
        target = payload.get("target_instance", "")
        delegation_id = payload.get("delegation_id", "")

        # Only handle if we're the target (or it's a broadcast)
        if target and target != self.instance_id:
            return None

        # Skip our own delegations
        if requester == self.instance_id:
            return None

        self._stats["delegations_received"] += 1

        query = payload.get("query", "")
        domain = payload.get("domain", "general")

        # Process via vote handler (same as voting)
        response_text = ""
        confidence = 0.0

        if self._vote_handler:
            try:
                response_text, confidence = await self._vote_handler(query, domain)
            except Exception as e:
                logger.warning("Delegation handler failed: %s", e)

        if not response_text:
            response_text = f"[{self.instance_id}] Delegation processed"
            confidence = 0.1

        # Send response back using vote response mechanism
        vote_resp = VoteResponse(
            vote_id=delegation_id,
            responder_id=self.instance_id,
            response=response_text,
            confidence=confidence,
            domain=domain,
        )

        await self.publish(
            "collective.vote.response",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="collective.vote.response",
                payload=vote_resp.to_dict(),
            ),
        )
        self._stats["delegations_completed"] += 1

        return None
