"""
Web Research Node — autonomous knowledge acquisition from the internet.

Detects knowledge gaps during reasoning, searches the web via BrowserNode,
classifies findings into 3 tiers (Information / Task Knowledge / Core
Knowledge), verifies source credibility, and routes to the appropriate
storage tier.

Tier Model:
  T1 (Information)    — ephemeral, workspace-only, never stored
  T2 (Task Knowledge) — session-scoped episodic memory, sleep decides
  T3 (Core Knowledge) — permanent in KnowledgeBase + KnowledgeGraph
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.brain.reasoning.source_verifier import SourceCredibility, SourceVerifier
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class ResearchTier(str, Enum):
    INFORMATION = "information"
    TASK_KNOWLEDGE = "task_knowledge"
    CORE_KNOWLEDGE = "core_knowledge"


@dataclass
class ResearchFinding:
    """A verified web research finding ready for ingestion."""

    content: str
    url: str
    title: str
    domain: str
    tier: ResearchTier
    trust_score: float
    credibility: SourceCredibility | None = None
    session_id: str = ""
    ingested_at: float = field(default_factory=time.time)
    ttl_days: int | None = None  # None = permanent


# ── Tier Classification Heuristics ───────────────────────────────────────────

# Patterns that suggest ephemeral information (T1)
_T1_PATTERNS = {
    "current",
    "right now",
    "today",
    "latest",
    "live",
    "price of",
    "weather",
    "score",
    "stock",
    "exchange rate",
    "what time",
    "status of",
}

# Patterns that suggest task-scoped knowledge (T2)
_T2_PATTERNS = {
    "how to fix",
    "how to solve",
    "error",
    "bug",
    "issue",
    "example of",
    "tutorial",
    "docs for",
    "api for",
    "version",
    "install",
    "configure",
    "setup",
    "migration",
    "workaround",
    "deprecated",
}

# Patterns that suggest core knowledge (T3)
_T3_PATTERNS = {
    "how does",
    "what is",
    "explain",
    "concept",
    "theory",
    "architecture",
    "design pattern",
    "best practice",
    "algorithm",
    "protocol",
    "fundamentals",
    "principle",
    "difference between",
    "compare",
}


def classify_tier(query: str, llm: Any = None) -> ResearchTier:
    """
    Classify a query into a research tier.

    Uses LLM when available, falls back to keyword heuristics.
    """
    query_lower = query.lower()

    # Heuristic scoring
    t1_score = sum(1 for p in _T1_PATTERNS if p in query_lower)
    t2_score = sum(1 for p in _T2_PATTERNS if p in query_lower)
    t3_score = sum(1 for p in _T3_PATTERNS if p in query_lower)

    if t1_score > t2_score and t1_score > t3_score:
        return ResearchTier.INFORMATION
    elif t2_score > t3_score:
        return ResearchTier.TASK_KNOWLEDGE
    elif t3_score > 0:
        return ResearchTier.CORE_KNOWLEDGE

    # Default: task knowledge (safe middle ground)
    return ResearchTier.TASK_KNOWLEDGE


class WebResearchNode(Node):
    """
    Autonomous web research and knowledge acquisition node.

    Subscribes to low-confidence workspace events and error fallbacks,
    searches the web, verifies sources, classifies findings, and routes
    them to the appropriate storage tier.
    """

    def __init__(
        self,
        node_id: str,
        llm: Any = None,
        max_searches_per_hour: int = 10,
        min_trust_score: float = 0.6,
        knowledge_ttl_days: int = 30,
        search_max_results: int = 3,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["web_research", "knowledge_acquisition"],
        )
        self.llm = llm
        self.max_searches_per_hour = max_searches_per_hour
        self.min_trust_score = min_trust_score
        self.knowledge_ttl_days = knowledge_ttl_days
        self.search_max_results = search_max_results

        self.verifier = SourceVerifier(min_trust_score=min_trust_score)

        # Rate limiting: tenant_id -> list of timestamps
        self._search_timestamps: dict[str, list[float]] = defaultdict(list)
        # Cooldown: topic -> last_search_time (avoid re-searching same topic)
        self._topic_cooldown: dict[str, float] = {}
        self._topic_cooldown_seconds = 300.0  # 5 min

        # Telemetry
        self._total_searches = 0
        self._total_findings = 0
        self._findings_by_tier: dict[str, int] = defaultdict(int)
        self._rejected_sources = 0

    async def on_start(self) -> None:
        logger.info(
            "Starting WebResearchNode (max=%d/hr, min_trust=%.2f)",
            self.max_searches_per_hour,
            self.min_trust_score,
        )
        await self.bus.subscribe("workspace.thought", self._handle_low_confidence_thought)
        await self.bus.subscribe("workspace.fallback", self._handle_fallback)
        await self.bus.subscribe("system.research.request", self._handle_research_request)
        await self.bus.subscribe("system.research.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping WebResearchNode — searches=%d findings=%d rejected=%d",
            self._total_searches,
            self._total_findings,
            self._rejected_sources,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Gap Detection ────────────────────────────────────────────────

    async def _handle_low_confidence_thought(self, message: Message) -> Message | None:
        """Detect knowledge gaps from low-confidence workspace thoughts."""
        payload = message.payload
        confidence = payload.get("confidence", 1.0)
        thought_type = payload.get("type", "")

        # Skip meta-thoughts and high-confidence responses
        if thought_type in ("critique", "simulation_result", "web_research"):
            return None
        if confidence >= 0.4:
            return None

        query = str(payload.get("content", ""))[:200]
        if not query:
            return None

        logger.info(
            "[WebResearch] Low-confidence gap detected (%.2f): '%s...'",
            confidence,
            query[:60],
        )
        await self._execute_research(
            query=query,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            correlation_id=message.correlation_id,
        )
        return None

    async def _handle_fallback(self, message: Message) -> Message | None:
        """Trigger research on workspace error fallbacks."""
        query = message.payload.get("query", "")
        if not query:
            return None

        logger.info("[WebResearch] Fallback-triggered research: '%s...'", query[:60])
        await self._execute_research(
            query=query,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            correlation_id=message.correlation_id,
        )
        return None

    async def _handle_research_request(self, message: Message) -> Message | None:
        """Manual research trigger via system.research.request."""
        query = message.payload.get("query", "")
        if not query:
            if message.type == MessageType.QUERY:
                return message.create_error("Missing 'query' in payload")
            return None

        tier_hint = message.payload.get("tier")
        logger.info("[WebResearch] Manual research request: '%s...'", query[:60])

        findings = await self._execute_research(
            query=query,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            correlation_id=message.correlation_id,
            tier_override=tier_hint,
        )

        if message.type == MessageType.QUERY:
            return message.create_response(
                {
                    "status": "completed",
                    "findings_count": len(findings),
                    "findings": [
                        {
                            "title": f.title,
                            "tier": f.tier.value,
                            "trust_score": f.trust_score,
                            "url": f.url,
                        }
                        for f in findings
                    ],
                }
            )
        return None

    async def _handle_query(self, message: Message) -> Message | None:
        """Return research stats."""
        return message.create_response(self.stats())

    # ── Research Pipeline ────────────────────────────────────────────

    async def _execute_research(
        self,
        query: str,
        tenant_id: str = "",
        session_id: str = "",
        correlation_id: str | None = None,
        tier_override: str | None = None,
    ) -> list[ResearchFinding]:
        """Full pipeline: search → verify → classify → ingest."""
        # Rate check
        if not self._check_rate_limit(tenant_id):
            logger.warning("[WebResearch] Rate limit hit for tenant %s", tenant_id)
            return []

        # Cooldown check
        if not self._check_cooldown(query):
            logger.debug("[WebResearch] Cooldown active for query: '%s...'", query[:40])
            return []

        # 1. Search via BrowserNode
        results = await self._web_search(query)
        if not results:
            return []

        self._total_searches += 1
        self._topic_cooldown[query[:100]] = time.time()

        # 2. Verify sources
        credibility_reports = self.verifier.verify_sources(results)

        # 3. Classify tier
        tier = ResearchTier(tier_override) if tier_override else classify_tier(query, self.llm)

        # 4. Process trusted results
        findings: list[ResearchFinding] = []
        for i, cred in enumerate(credibility_reports):
            if not cred.is_trusted:
                self._rejected_sources += 1
                logger.debug(
                    "[WebResearch] Rejected untrusted source: %s (%.2f)",
                    cred.domain,
                    cred.trust_score,
                )
                continue

            result = results[i]
            finding = ResearchFinding(
                content=result.get("page_content", result.get("search_snippet", "")),
                url=cred.url,
                title=result.get("title", ""),
                domain=cred.domain,
                tier=tier,
                trust_score=cred.trust_score,
                credibility=cred,
                session_id=session_id,
                ttl_days=self.knowledge_ttl_days if tier == ResearchTier.CORE_KNOWLEDGE else None,
            )

            await self._ingest_finding(finding, tenant_id, session_id, correlation_id)
            findings.append(finding)
            self._total_findings += 1
            self._findings_by_tier[tier.value] += 1

        # 5. Publish research report
        await self._publish_report(query, findings, credibility_reports)

        logger.info(
            "[WebResearch] Research complete: query='%s...' tier=%s findings=%d rejected=%d",
            query[:40],
            tier.value,
            len(findings),
            len(credibility_reports) - len(findings),
        )
        return findings

    async def _web_search(self, query: str) -> list[dict[str, Any]]:
        """Send search query to BrowserNode via bus."""
        search_msg = Message(
            type=MessageType.QUERY,
            source_node_id=self.node_id,
            topic="task.execute.search",
            payload={
                "query": query,
                "max_results": self.search_max_results,
            },
        )
        try:
            resp = await self.bus.request("task.execute.search", search_msg, timeout=30.0)
            return resp.payload.get("results", [])
        except Exception as e:
            logger.warning("[WebResearch] Web search failed: %s", e)
            return []

    # ── Tier-Based Ingestion ─────────────────────────────────────────

    async def _ingest_finding(
        self,
        finding: ResearchFinding,
        tenant_id: str,
        session_id: str,
        correlation_id: str | None,
    ) -> None:
        """Route finding to correct storage based on tier."""
        # Truncate for workspace thought
        summary = finding.content[:500] if finding.content else finding.title

        if finding.tier == ResearchTier.INFORMATION:
            # T1: Inject into workspace only — never stored
            await self.bus.publish(
                "workspace.thought",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    topic="workspace.thought",
                    payload={
                        "type": "web_research",
                        "confidence": finding.trust_score,
                        "content": f"[Web: {finding.title}] {summary}",
                        "source_url": finding.url,
                        "tier": "information",
                    },
                    correlation_id=correlation_id,
                ),
            )

        elif finding.tier == ResearchTier.TASK_KNOWLEDGE:
            # T2: Store in episodic memory (session-scoped)
            await self.bus.publish(
                "memory.store",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    topic="memory.store",
                    payload={
                        "content": finding.content[:2000],
                        "session_id": session_id,
                        "role": "system",
                        "domain": "web_research",
                        "source_url": finding.url,
                        "source_title": finding.title,
                        "tier": "task_knowledge",
                        "trust_score": finding.trust_score,
                        "expires_after_session": True,
                    },
                ),
            )
            # Also publish to workspace for immediate use
            await self.bus.publish(
                "workspace.thought",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    topic="workspace.thought",
                    payload={
                        "type": "web_research",
                        "confidence": finding.trust_score,
                        "content": f"[Web: {finding.title}] {summary}",
                        "source_url": finding.url,
                        "tier": "task_knowledge",
                    },
                    correlation_id=correlation_id,
                ),
            )

        elif finding.tier == ResearchTier.CORE_KNOWLEDGE:
            # T3: Full ingestion — KnowledgeBase + KnowledgeGraph
            await self.bus.publish(
                "knowledge.ingest",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    topic="knowledge.ingest",
                    payload={
                        "content": finding.content[:5000],
                        "url": finding.url,
                        "title": finding.title,
                        "domain": finding.domain,
                        "trust_score": finding.trust_score,
                        "tier": "core_knowledge",
                        "ttl_days": finding.ttl_days,
                        "ingested_at": finding.ingested_at,
                        "source_type": "web",
                    },
                ),
            )
            # Also publish to workspace
            await self.bus.publish(
                "workspace.thought",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    topic="workspace.thought",
                    payload={
                        "type": "web_research",
                        "confidence": finding.trust_score,
                        "content": f"[Web: {finding.title}] {summary}",
                        "source_url": finding.url,
                        "tier": "core_knowledge",
                    },
                    correlation_id=correlation_id,
                ),
            )

    # ── Rate Limiting ────────────────────────────────────────────────

    def _check_rate_limit(self, tenant_id: str) -> bool:
        """Check per-tenant hourly rate limit."""
        now = time.time()
        hour_ago = now - 3600.0
        timestamps = self._search_timestamps[tenant_id]
        # Prune old timestamps
        self._search_timestamps[tenant_id] = [t for t in timestamps if t > hour_ago]
        if len(self._search_timestamps[tenant_id]) >= self.max_searches_per_hour:
            return False
        self._search_timestamps[tenant_id].append(now)
        return True

    def _check_cooldown(self, query: str) -> bool:
        """Check topic cooldown to avoid re-searching."""
        key = query[:100]
        last = self._topic_cooldown.get(key, 0)
        return (time.time() - last) > self._topic_cooldown_seconds

    # ── Telemetry ────────────────────────────────────────────────────

    async def _publish_report(
        self,
        query: str,
        findings: list[ResearchFinding],
        credibility: list[SourceCredibility],
    ) -> None:
        """Publish research report for observability."""
        await self.bus.publish(
            "system.research.report",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.research.report",
                payload={
                    "query": query[:200],
                    "findings_count": len(findings),
                    "rejected_count": len(credibility) - len(findings),
                    "tiers": sorted(list({f.tier.value for f in findings})),
                    "sources": [c.to_dict() for c in credibility],
                },
            ),
        )

    def stats(self) -> dict[str, Any]:
        """Return research telemetry."""
        return {
            "total_searches": self._total_searches,
            "total_findings": self._total_findings,
            "rejected_sources": self._rejected_sources,
            "findings_by_tier": dict(self._findings_by_tier),
        }
