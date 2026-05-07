"""Tests for WebResearchNode and SourceVerifier — autonomous web knowledge acquisition."""

import asyncio
import time

import pytest

from hbllm.brain.source_verifier import SourceCredibility, SourceVerifier
from hbllm.brain.web_research_node import (
    ResearchTier,
    WebResearchNode,
    classify_tier,
)
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── SourceVerifier Tests ─────────────────────────────────────────────────────


class TestSourceVerifier:
    """Test the multi-factor credibility scoring engine."""

    def setup_method(self):
        self.verifier = SourceVerifier(min_trust_score=0.6)

    def test_tier1_domain_scores_high(self):
        """Wikipedia and official docs should score 0.9."""
        score, tier = self.verifier.get_domain_reputation("https://en.wikipedia.org/wiki/Python")
        assert score == 0.9
        assert tier == "tier_1_authoritative"

    def test_tier2_domain_scores_medium(self):
        """StackOverflow should score 0.7."""
        score, tier = self.verifier.get_domain_reputation(
            "https://stackoverflow.com/questions/12345"
        )
        assert score == 0.7
        assert tier == "tier_2_community"

    def test_tier3_domain_scores_low(self):
        """Medium blogs should score 0.4."""
        score, tier = self.verifier.get_domain_reputation("https://medium.com/@user/some-article")
        assert score == 0.4
        assert tier == "tier_3_blog"

    def test_unknown_domain_scores_lowest(self):
        """Unknown domains should score 0.3."""
        score, tier = self.verifier.get_domain_reputation("https://random-blog-2024.xyz/article")
        assert score == 0.3
        assert tier == "tier_unknown"

    def test_subdomain_matching(self):
        """Subdomains of tier-1 domains should match."""
        score, _tier = self.verifier.get_domain_reputation(
            "https://docs.python.org/3/library/asyncio.html"
        )
        assert score == 0.9

    def test_corroboration_multiple_sources(self):
        """Same fact from 2+ sources should boost corroboration score."""
        claim = "Python is a high-level programming language with dynamic typing"
        results = [
            {
                "url": "https://wikipedia.org/wiki/Python",
                "page_content": "Python is a high-level programming language with dynamic typing",
            },
            {
                "url": "https://python.org",
                "page_content": "Python is a high-level general-purpose programming language with dynamic typing",
            },
            {
                "url": "https://stackoverflow.com",
                "page_content": "Python uses dynamic typing and is high-level",
            },
        ]
        score = self.verifier.compute_corroboration(
            claim, results, exclude_url="https://wikipedia.org/wiki/Python"
        )
        assert score >= 0.5

    def test_corroboration_no_other_sources(self):
        """Single source should get 0 corroboration."""
        results = [{"url": "https://example.com", "page_content": "some content"}]
        score = self.verifier.compute_corroboration(
            "some content", results, exclude_url="https://example.com"
        )
        assert score == 0.0

    def test_recency_current_year(self):
        """Content mentioning current year should score 1.0."""
        current_year = str(time.localtime().tm_year)
        score = self.verifier.compute_recency(f"Updated in {current_year} with latest features")
        assert score == 1.0

    def test_recency_old_content(self):
        """Content from 2015 should score low."""
        score = self.verifier.compute_recency("Last updated in 2015.")
        assert score <= 0.4

    def test_verify_source_composite(self):
        """Full source verification should produce a valid SourceCredibility."""
        result = {
            "url": "https://docs.python.org/3/tutorial/index.html",
            "title": "Python Tutorial",
            "page_content": "The Python tutorial in 2026",
        }
        cred = self.verifier.verify_source(result, [result])
        assert isinstance(cred, SourceCredibility)
        assert cred.domain_reputation == 0.9
        assert cred.trust_score > 0.5
        assert cred.is_trusted

    def test_verify_sources_sorted(self):
        """verify_sources should return results sorted by trust_score desc."""
        results = [
            {"url": "https://random.xyz", "page_content": "stuff"},
            {"url": "https://docs.python.org/3", "page_content": "Python docs 2026"},
        ]
        assessments = self.verifier.verify_sources(results)
        assert assessments[0].domain_reputation > assessments[1].domain_reputation


# ── Tier Classification Tests ────────────────────────────────────────────────


class TestTierClassification:
    """Test the heuristic tier classification."""

    def test_information_tier(self):
        """Time-sensitive queries should classify as T1."""
        assert classify_tier("what is the current price of Bitcoin") == ResearchTier.INFORMATION
        assert classify_tier("weather right now in London") == ResearchTier.INFORMATION

    def test_task_knowledge_tier(self):
        """'How to fix' queries should classify as T2."""
        assert classify_tier("how to fix React hydration error") == ResearchTier.TASK_KNOWLEDGE
        assert classify_tier("tutorial for FastAPI setup") == ResearchTier.TASK_KNOWLEDGE

    def test_core_knowledge_tier(self):
        """'How does X work' queries should classify as T3."""
        assert classify_tier("how does OAuth2 authentication work") == ResearchTier.CORE_KNOWLEDGE
        assert (
            classify_tier("explain the concept of dependency injection")
            == ResearchTier.CORE_KNOWLEDGE
        )

    def test_default_tier(self):
        """Ambiguous queries should default to T2 (task knowledge)."""
        assert classify_tier("something random") == ResearchTier.TASK_KNOWLEDGE


# ── WebResearchNode Integration Tests ────────────────────────────────────────


@pytest.fixture
async def research_env():
    """Set up a bus with WebResearchNode and mock BrowserNode."""
    bus = InProcessBus()
    await bus.start()

    # Mock BrowserNode: responds to task.execute.search
    async def mock_browser(msg: Message) -> Message | None:
        return msg.create_response(
            {
                "results": [
                    {
                        "title": "Python Tutorial",
                        "url": "https://docs.python.org/3/tutorial/",
                        "search_snippet": "The Python tutorial covers basics in 2026",
                        "page_content": "Python is a powerful programming language. "
                        "Updated 2026. It supports multiple paradigms.",
                    },
                    {
                        "title": "Python - Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "search_snippet": "Python is a high-level language",
                        "page_content": "Python is a high-level, general-purpose programming "
                        "language. Created by Guido van Rossum. Updated 2026.",
                    },
                ],
            }
        )

    await bus.subscribe("task.execute.search", mock_browser)

    node = WebResearchNode(
        node_id="research_test",
        max_searches_per_hour=10,
        min_trust_score=0.4,  # Low threshold for testing
    )
    await node.start(bus)

    yield bus, node

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_manual_research_request(research_env):
    """Manual research request should return findings."""
    bus, node = research_env

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.research.request",
        payload={"query": "how does Python work"},
    )

    response = await node._handle_research_request(request)
    assert response is not None
    assert response.payload["status"] == "completed"
    assert response.payload["findings_count"] >= 1


@pytest.mark.asyncio
async def test_t1_not_stored_in_memory(research_env):
    """Information tier should publish workspace thought but NOT memory.store."""
    bus, node = research_env

    memory_events = []
    workspace_events = []
    await bus.subscribe("memory.store", lambda msg: memory_events.append(msg))
    await bus.subscribe("workspace.thought", lambda msg: workspace_events.append(msg))

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.research.request",
        payload={"query": "current stock price of AAPL", "tier": "information"},
    )

    await node._handle_research_request(request)
    await asyncio.sleep(0.2)

    # T1 should NOT store in memory
    assert len(memory_events) == 0
    # But SHOULD publish workspace thoughts
    assert len(workspace_events) >= 1
    for evt in workspace_events:
        assert evt.payload["tier"] == "information"


@pytest.mark.asyncio
async def test_t2_stored_in_episodic(research_env):
    """Task Knowledge should store in episodic memory with session scope."""
    bus, node = research_env

    memory_events = []
    await bus.subscribe("memory.store", lambda msg: memory_events.append(msg))

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.research.request",
        payload={"query": "how to fix React error", "tier": "task_knowledge"},
        session_id="session_123",
    )

    await node._handle_research_request(request)
    await asyncio.sleep(0.2)

    assert len(memory_events) >= 1
    for evt in memory_events:
        assert evt.payload["tier"] == "task_knowledge"
        assert evt.payload["expires_after_session"] is True


@pytest.mark.asyncio
async def test_t3_stored_in_knowledge_base(research_env):
    """Core Knowledge should publish to knowledge.ingest."""
    bus, node = research_env

    kb_events = []
    await bus.subscribe("knowledge.ingest", lambda msg: kb_events.append(msg))

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.research.request",
        payload={"query": "explain how OAuth2 works", "tier": "core_knowledge"},
    )

    await node._handle_research_request(request)
    await asyncio.sleep(0.2)

    assert len(kb_events) >= 1
    for evt in kb_events:
        assert evt.payload["tier"] == "core_knowledge"
        assert evt.payload["source_type"] == "web"
        assert evt.payload["ttl_days"] == 30


@pytest.mark.asyncio
async def test_rate_limiting(research_env):
    """Should respect per-tenant hourly rate limits."""
    _bus, node = research_env
    node.max_searches_per_hour = 2

    # First 2 should work
    findings1 = await node._execute_research(query="query 1", tenant_id="tenant_a")
    findings2 = await node._execute_research(query="query 2", tenant_id="tenant_a")
    # Third should be rate-limited
    findings3 = await node._execute_research(query="query 3", tenant_id="tenant_a")

    assert len(findings1) > 0
    assert len(findings2) > 0
    assert len(findings3) == 0  # Rate limited


@pytest.mark.asyncio
async def test_cooldown(research_env):
    """Same query within cooldown should be skipped."""
    _bus, node = research_env
    node._topic_cooldown_seconds = 999.0  # Long cooldown

    # First should work
    findings1 = await node._execute_research(query="same query")
    # Same query should be blocked by cooldown
    findings2 = await node._execute_research(query="same query")

    assert len(findings1) > 0
    assert len(findings2) == 0  # Cooldown


@pytest.mark.asyncio
async def test_gap_detection_low_confidence(research_env):
    """Low-confidence workspace thoughts should trigger research."""
    bus, node = research_env

    reports = []
    await bus.subscribe("system.research.report", lambda msg: reports.append(msg))

    thought = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="workspace.thought",
        payload={
            "type": "answer",
            "confidence": 0.2,
            "content": "I'm not sure about quantum computing principles",
        },
    )
    await node._handle_low_confidence_thought(thought)
    await asyncio.sleep(0.3)

    assert len(reports) >= 1


@pytest.mark.asyncio
async def test_stats(research_env):
    """Stats should reflect research activity."""
    _bus, node = research_env

    await node._execute_research(query="test query")
    stats = node.stats()

    assert stats["total_searches"] >= 1
    assert stats["total_findings"] >= 1


# ── KnowledgeBase Web Methods Tests ──────────────────────────────────────────


class TestKnowledgeBaseWeb:
    """Test web content ingestion and staleness."""

    def test_ingest_web_content(self):
        import tempfile

        from hbllm.knowledge.knowledge_base import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(data_dir=tmpdir)
            doc_id = kb.ingest_web_content(
                content="OAuth2 is an authorization framework.",
                url="https://oauth.net/2/",
                title="OAuth 2.0",
                trust_score=0.9,
                domain="oauth.net",
                tier="core_knowledge",
            )
            assert doc_id != ""

            # Should be searchable
            results = kb.search("OAuth2 authorization")
            assert len(results) >= 1

    def test_mark_obsolete(self):
        import tempfile

        from hbllm.knowledge.knowledge_base import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(data_dir=tmpdir)
            doc_id = kb.ingest_web_content(
                content="Some old fact.",
                url="https://example.com",
                title="Old Fact",
                trust_score=0.7,
                domain="example.com",
            )
            result = kb.mark_obsolete(doc_id, reason="Stale after 30 days")
            assert result is True

    def test_get_stale_entries(self):
        import tempfile

        from hbllm.knowledge.knowledge_base import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(data_dir=tmpdir)
            doc_id = kb.ingest_web_content(
                content="Will become stale.",
                url="https://example.com",
                title="Stale Test",
                trust_score=0.7,
                domain="example.com",
            )
            # Manually backdate the ingested_at timestamp
            docs = getattr(kb.memory, "documents", None) or getattr(kb.memory, "_docs", {})
            if doc_id in docs:
                docs[doc_id]["metadata"]["ingested_at"] = time.time() - 40 * 86400
            stale = kb.get_stale_entries(ttl_days=30)
            assert len(stale) >= 1
            assert stale[0]["age_days"] > 30
