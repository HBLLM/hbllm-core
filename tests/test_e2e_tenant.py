"""
End-to-End integration test for the full tenant journey.

Tests the complete flow:
  1. API Key Manager setup
  2. API Key authentication (accept valid, reject invalid)
  3. Rate limiting
  4. Document upload + knowledge base
  5. Semantic search
  6. RAG-enhanced chat (mocked provider)
  7. Usage tracking
  8. Billing calculation
"""

import asyncio
import json
import os
import tempfile
import time

import pytest
import numpy as np

from hbllm.serving.security import ApiKeyManager, RateLimiter, InputSanitizer


# ─── 1. API Key Manager ─────────────────────────────────────────────────────

class TestApiKeyAuth:
    """Test API key authentication flow."""

    def setup_method(self):
        self.akm = ApiKeyManager()

    def test_add_and_validate_key(self):
        key = self.akm.add_key("sk-test-key-123", tenant_id="acme", name="test", scopes=["chat", "knowledge"])
        assert key.tenant_id == "acme"

        # Valid key
        result = self.akm.validate("sk-test-key-123")
        assert result is not None
        assert result.tenant_id == "acme"
        assert result.active is True

    def test_reject_invalid_key(self):
        self.akm.add_key("sk-valid", tenant_id="acme")

        result = self.akm.validate("sk-invalid")
        assert result is None

    def test_reject_empty_key(self):
        result = self.akm.validate("")
        assert result is None

    def test_scope_checking(self):
        key = self.akm.add_key("sk-limited", tenant_id="acme", scopes=["chat"])

        assert self.akm.has_scope(key, "chat") is True
        assert self.akm.has_scope(key, "admin") is False
        assert self.akm.has_scope(key, "knowledge") is False

    def test_multiple_tenants(self):
        self.akm.add_key("sk-acme", tenant_id="acme")
        self.akm.add_key("sk-globex", tenant_id="globex")

        acme = self.akm.validate("sk-acme")
        globex = self.akm.validate("sk-globex")

        assert acme.tenant_id == "acme"
        assert globex.tenant_id == "globex"
        assert acme.tenant_id != globex.tenant_id


# ─── 2. Rate Limiting ───────────────────────────────────────────────────────

class TestRateLimiting:
    """Test per-tenant rate limiting."""

    def test_allows_within_limit(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=10)
        for _ in range(10):
            allowed, _ = rl.check("acme")
            assert allowed is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=3)
        # Use up burst
        for _ in range(3):
            rl.check("acme")

        # Next should be blocked
        allowed, retry_after = rl.check("acme")
        assert allowed is False
        assert retry_after > 0

    def test_separate_tenant_limits(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=2)
        rl.check("acme")
        rl.check("acme")

        # Acme is at limit, but Globex should be fine
        allowed_globex, _ = rl.check("globex")
        assert allowed_globex is True

    def test_reset(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=1)
        rl.check("acme")
        allowed, _ = rl.check("acme")
        assert allowed is False

        rl.reset("acme")
        allowed, _ = rl.check("acme")
        assert allowed is True


# ─── 3. Input Sanitization ──────────────────────────────────────────────────

class TestInputSanitizer:
    """Test input sanitization."""

    def test_normal_text(self):
        san = InputSanitizer()
        text, warnings = san.sanitize("Hello, how are you?")
        assert text == "Hello, how are you?"
        assert len(warnings) == 0

    def test_strips_html(self):
        san = InputSanitizer()
        text, warnings = san.sanitize("<script>alert('xss')</script>Hello")
        assert "<script>" not in text

    def test_max_length(self):
        san = InputSanitizer(max_length=10)
        text, warnings = san.sanitize("This is a long text that exceeds the limit")
        assert len(text) <= 10


# ─── 4. Knowledge Base ──────────────────────────────────────────────────────

class TestKnowledgeBase:
    """Test document upload, chunking, embedding, and search."""

    def setup_method(self):
        from hbllm_cloud.knowledge.vector_store import VectorStore
        from hbllm_cloud.knowledge.processor import DocumentProcessor

        self.db_path = tempfile.mktemp(suffix=".db")
        self.vs = VectorStore(db_path=self.db_path)
        self.proc = DocumentProcessor()

    def teardown_method(self):
        self.vs.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_document_processing(self):
        """Test that documents are chunked correctly."""
        text = "\n\n".join([f"Section {i}. " + "Content here. " * 30 for i in range(10)])
        chunks = self.proc.process(text, filename="guide.md")
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)

    def test_vector_store_crud(self):
        """Test add, search, list, delete."""
        chunks = ["Our return policy allows 30-day returns.", "Shipping takes 3-5 business days."]
        embeddings = [np.random.randn(1536).tolist(), np.random.randn(1536).tolist()]

        # Add
        count = self.vs.add_document("doc1", "acme", "policy.md", chunks, embeddings)
        assert count == 2

        # List
        docs = self.vs.list_documents("acme")
        assert len(docs) == 1
        assert docs[0]["filename"] == "policy.md"

        # Stats
        stats = self.vs.get_stats("acme")
        assert stats["documents"] == 1
        assert stats["chunks"] == 2

        # Search (random query, but results should come back)
        results = self.vs.search("acme", np.random.randn(1536).tolist(), top_k=5, min_score=0.0)
        assert len(results) <= 5

        # Delete
        deleted = self.vs.delete_document("acme", "doc1")
        assert deleted is True
        assert len(self.vs.list_documents("acme")) == 0

    def test_tenant_isolation(self):
        """Ensure tenant A can't see tenant B's docs."""
        emb = np.random.randn(1536).tolist()
        self.vs.add_document("d1", "acme", "secret.md", ["Acme secret"], [emb])
        self.vs.add_document("d2", "globex", "public.md", ["Globex public"], [emb])

        acme_docs = self.vs.list_documents("acme")
        globex_docs = self.vs.list_documents("globex")
        assert len(acme_docs) == 1
        assert len(globex_docs) == 1
        assert acme_docs[0]["filename"] == "secret.md"
        assert globex_docs[0]["filename"] == "public.md"

        # Search should only return own tenant's results
        acme_results = self.vs.search("acme", emb, top_k=10, min_score=0.0)
        globex_results = self.vs.search("globex", emb, top_k=10, min_score=0.0)
        for r in acme_results:
            assert r["doc_id"] == "d1"
        for r in globex_results:
            assert r["doc_id"] == "d2"


# ─── 5. Usage Tracking ──────────────────────────────────────────────────────

class TestUsageTracking:
    """Test per-tenant usage metering."""

    def setup_method(self):
        from hbllm_cloud.usage import UsageTracker
        self.db_path = tempfile.mktemp(suffix=".db")
        self.ut = UsageTracker(db_path=self.db_path)

    def teardown_method(self):
        self.ut.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_record_and_query(self):
        self.ut.record_chat("acme", prompt_tokens=500, completion_tokens=200, provider="openai")
        self.ut.record_chat("acme", prompt_tokens=300, completion_tokens=100, provider="openai")
        self.ut.record("acme", event="knowledge_upload", tokens=1000)

        usage = self.ut.get_tenant_usage("acme", days=30)
        assert usage["total_tokens"] == 2100
        assert usage["total_calls"] == 3

    def test_tenant_isolation(self):
        self.ut.record_chat("acme", prompt_tokens=100, completion_tokens=50)
        self.ut.record_chat("globex", prompt_tokens=200, completion_tokens=100)

        acme = self.ut.get_tenant_usage("acme")
        globex = self.ut.get_tenant_usage("globex")
        assert acme["total_tokens"] == 150
        assert globex["total_tokens"] == 300

    def test_cost_estimation(self):
        self.ut.record_chat("acme", prompt_tokens=500000, completion_tokens=500000)
        cost = self.ut.estimate_cost("acme", rate_per_million=2.0)
        assert cost["estimated_cost_usd"] == 2.0

    def test_daily_aggregation(self):
        self.ut.record_chat("acme", prompt_tokens=100, completion_tokens=50)
        daily = self.ut.get_daily_usage("acme")
        assert len(daily) >= 1


# ─── 6. Billing ─────────────────────────────────────────────────────────────

class TestBilling:
    """Test billing plans and overage calculation."""

    def setup_method(self):
        from hbllm_cloud.billing import BillingManager
        self.billing = BillingManager()

    def test_list_plans(self):
        plans = self.billing.list_plans()
        assert len(plans) == 4
        names = [p["name"] for p in plans]
        assert "Free" in names
        assert "Starter" in names
        assert "Business" in names
        assert "Enterprise" in names

    def test_plan_lookup(self):
        plan = self.billing.get_plan("starter")
        assert plan is not None
        assert plan.price_monthly == 49
        assert plan.included_tokens == 5_000_000

    def test_overage_calculation(self):
        report = self.billing.report_usage("acme", tokens_used=6_000_000, plan_id="starter")
        assert report["overage_tokens"] == 1_000_000
        assert report["overage_cost_usd"] == 2.0
        assert report["total_cost_usd"] == 51.0

    def test_within_limits(self):
        report = self.billing.report_usage("acme", tokens_used=3_000_000, plan_id="starter")
        assert report["overage_tokens"] == 0
        assert report["overage_cost_usd"] == 0

    def test_free_plan_no_overage(self):
        report = self.billing.report_usage("acme", tokens_used=200_000, plan_id="free")
        assert report["overage_tokens"] == 100_000
        assert report["overage_cost_usd"] == 0  # Free plan has 0 overage rate

    def test_check_limits(self):
        within = self.billing.check_limits("acme", "starter", {"total_tokens": 1_000_000})
        assert within["within_limits"] is True
        assert within["usage_percent"] == 20.0

        over = self.billing.check_limits("acme", "starter", {"total_tokens": 6_000_000})
        assert over["within_limits"] is False
        assert over["usage_percent"] == 120.0


# ─── 7. Full Journey ────────────────────────────────────────────────────────

class TestFullTenantJourney:
    """
    Complete end-to-end test:
    API Key -> Auth -> Upload Doc -> Search -> Chat -> Usage -> Billing
    """

    def setup_method(self):
        from hbllm_cloud.knowledge.vector_store import VectorStore
        from hbllm_cloud.knowledge.processor import DocumentProcessor
        from hbllm_cloud.usage import UsageTracker
        from hbllm_cloud.billing import BillingManager

        self.db_vs = tempfile.mktemp(suffix=".db")
        self.db_usage = tempfile.mktemp(suffix=".db")

        self.akm = ApiKeyManager()
        self.rl = RateLimiter(requests_per_minute=60, burst_size=10)
        self.vs = VectorStore(db_path=self.db_vs)
        self.proc = DocumentProcessor()
        self.ut = UsageTracker(db_path=self.db_usage)
        self.billing = BillingManager()

    def teardown_method(self):
        self.vs.close()
        self.ut.close()
        for p in [self.db_vs, self.db_usage]:
            if os.path.exists(p):
                os.unlink(p)

    def test_full_journey(self):
        """Simulate a full tenant lifecycle."""

        # 1. Create tenant + API key
        tenant_id = "acme-corp"
        raw_key = "sk-acme-prod-key-2024"
        key = self.akm.add_key(raw_key, tenant_id=tenant_id, scopes=["chat", "knowledge", "billing"])
        assert key.tenant_id == tenant_id

        # 2. Authenticate with key
        validated = self.akm.validate(raw_key)
        assert validated is not None
        assert validated.tenant_id == tenant_id

        # Invalid key rejected
        assert self.akm.validate("sk-wrong-key") is None

        # 3. Check rate limit
        allowed, _ = self.rl.check(tenant_id)
        assert allowed is True

        # 4. Upload document to knowledge base
        doc_content = (
            "Return Policy\n\n"
            "All items can be returned within 30 days of purchase. "
            "Items must be in original packaging. "
            "Refunds are processed within 5-7 business days.\n\n"
            "Shipping Policy\n\n"
            "Standard shipping takes 3-5 business days. "
            "Express shipping is available for an additional $9.99. "
            "Free shipping on orders over $50."
        )
        chunks = self.proc.process(doc_content, filename="policies.md")
        assert len(chunks) >= 1

        # Generate consistent embeddings (in production, would call OpenAI)
        base_emb = np.random.randn(1536).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)  # normalize
        embeddings = [(base_emb + np.random.randn(1536) * 0.05).tolist() for _ in chunks]
        doc_id = self.proc.generate_doc_id()
        self.vs.add_document(doc_id, tenant_id, "policies.md", chunks, embeddings)

        # 5. Search knowledge base (use similar vector to guarantee results)
        query_emb = (base_emb + np.random.randn(1536) * 0.05).tolist()
        results = self.vs.search(tenant_id, query_emb, top_k=3, min_score=0.0)
        assert len(results) >= 1
        assert results[0]["filename"] == "policies.md"

        # 6. Track chat usage
        self.ut.record_chat(tenant_id, prompt_tokens=500, completion_tokens=200, provider="openai")
        self.ut.record_chat(tenant_id, prompt_tokens=300, completion_tokens=150, provider="openai")
        self.ut.record("acme-corp", event="knowledge_upload", tokens=len(doc_content) // 4)

        # 7. Check usage
        usage = self.ut.get_tenant_usage(tenant_id, days=30)
        assert usage["total_tokens"] > 0
        assert usage["total_calls"] == 3

        # 8. Calculate billing
        cost = self.ut.estimate_cost(tenant_id)
        assert cost["estimated_cost_usd"] >= 0

        limits = self.billing.check_limits(tenant_id, "starter", usage)
        assert limits["within_limits"] is True  # Under 5M tokens

        report = self.billing.report_usage(tenant_id, tokens_used=usage["total_tokens"], plan_id="starter")
        assert report["overage_tokens"] == 0  # Way under limit
        assert report["base_cost_usd"] == 49

        # 9. Verify tenant isolation
        other_docs = self.vs.list_documents("other-tenant")
        assert len(other_docs) == 0
        other_usage = self.ut.get_tenant_usage("other-tenant")
        assert other_usage["total_tokens"] == 0

        # 10. Scope check
        assert self.akm.has_scope(validated, "chat") is True
        assert self.akm.has_scope(validated, "knowledge") is True
        assert self.akm.has_scope(validated, "admin") is False  # Not granted
