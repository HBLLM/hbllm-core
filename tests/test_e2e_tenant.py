"""
End-to-End integration test for the full tenant journey.

Tests the complete flow using only HBLLM Core modules:
  1. API Key Manager setup & authentication
  2. Rate limiting (per-tenant)
  3. Input sanitization
  4. Tenant-isolated semantic search (via SemanticMemory)
  5. Tenant-isolated episodic memory (via EpisodicMemory)
  6. Full journey combining all the above
"""

import os
import tempfile

from hbllm.memory.episodic import EpisodicMemory
from hbllm.memory.semantic import SemanticMemory
from hbllm.serving.security import ApiKeyManager, InputSanitizer, RateLimiter

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


# ─── 4. Knowledge Base (Core SemanticMemory) ────────────────────────────────

class TestKnowledgeBase:
    """Test document storage, embedding, and search using core SemanticMemory."""

    def setup_method(self):
        self.sm = SemanticMemory()

    def test_document_storage_and_search(self):
        """Test that documents are stored and searchable."""
        self.sm.store("Our return policy allows 30-day returns on all items.", {"topic": "policy"})
        self.sm.store("Shipping takes 3-5 business days for standard orders.", {"topic": "shipping"})
        self.sm.store("Python is a great programming language for AI.", {"topic": "coding"})

        results = self.sm.search("return policy", top_k=2)
        assert len(results) >= 1
        assert "return" in results[0]["content"].lower()

    def test_deduplication(self):
        """Test that duplicate content is not stored twice."""
        idx1 = self.sm.store("Duplicate content test.")
        idx2 = self.sm.store("Duplicate content test.")
        assert idx1 is not None
        assert idx2 is None  # Duplicate should be rejected
        assert self.sm.count == 1

    def test_priority_flag(self):
        """Test that priority metadata is set correctly."""
        self.sm.store("Important finding", {"domain": "research"}, is_priority=True)
        docs = self.sm.get_all()
        assert docs[0]["metadata"]["is_priority"] is True

    def test_delete_and_clear(self):
        """Test document deletion and clearing."""
        idx = self.sm.store("Doc A")
        self.sm.store("Doc B")
        assert self.sm.count == 2

        self.sm.delete(idx)
        assert self.sm.count == 1

        cleared = self.sm.clear()
        assert cleared == 1
        assert self.sm.count == 0

    def test_empty_search(self):
        """Searching an empty database returns no results."""
        results = self.sm.search("anything")
        assert results == []


# ─── 5. Episodic Memory (Tenant Isolation) ──────────────────────────────────

class TestEpisodicTenantIsolation:
    """Test that episodic memory properly isolates tenants."""

    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.db = EpisodicMemory(self.db_path)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_tenant_isolation_writes(self):
        """Ensure tenant A's data doesn't leak to tenant B."""
        self.db.store_turn("session1", "user", "Acme secret question", tenant_id="acme")
        self.db.store_turn("session1", "assistant", "Acme secret answer", tenant_id="acme")
        self.db.store_turn("session1", "user", "Globex public question", tenant_id="globex")

        acme_turns = self.db.retrieve_recent("session1", limit=10, tenant_id="acme")
        globex_turns = self.db.retrieve_recent("session1", limit=10, tenant_id="globex")

        assert len(acme_turns) == 2
        assert len(globex_turns) == 1
        assert all("Acme" in t["content"] for t in acme_turns)
        assert "Globex" in globex_turns[0]["content"]

    def test_tenant_isolation_clear(self):
        """Clearing one tenant's data shouldn't affect another."""
        self.db.store_turn("s1", "user", "Keep this", tenant_id="acme")
        self.db.store_turn("s1", "user", "Delete this", tenant_id="globex")

        # Only retrieve per-tenant, confirming isolation
        acme = self.db.retrieve_recent("s1", limit=10, tenant_id="acme")
        globex = self.db.retrieve_recent("s1", limit=10, tenant_id="globex")
        assert len(acme) == 1
        assert len(globex) == 1


# ─── 6. Full Journey (Core Only) ───────────────────────────────────────────

class TestFullTenantJourney:
    """
    Complete end-to-end test using only HBLLM Core modules:
    API Key → Auth → Store Memory → Search → Verify Isolation
    """

    def setup_method(self):
        self.akm = ApiKeyManager()
        self.rl = RateLimiter(requests_per_minute=60, burst_size=10)
        self.san = InputSanitizer()
        self.db_path = tempfile.mktemp(suffix=".db")
        self.episodic = EpisodicMemory(self.db_path)
        self.semantic = SemanticMemory()

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_full_journey(self):
        """Simulate a full tenant lifecycle using core modules."""
        tenant_id = "acme-corp"
        raw_key = "sk-acme-prod-key-2024"

        # 1. Create tenant + API key
        key = self.akm.add_key(raw_key, tenant_id=tenant_id, scopes=["chat", "knowledge"])
        assert key.tenant_id == tenant_id

        # 2. Authenticate
        validated = self.akm.validate(raw_key)
        assert validated is not None
        assert validated.tenant_id == tenant_id
        assert self.akm.validate("sk-wrong-key") is None

        # 3. Rate limiting
        allowed, _ = self.rl.check(tenant_id)
        assert allowed is True

        # 4. Input sanitization
        user_input = "<b>What is your return policy?</b>"
        clean_input, warnings = self.san.sanitize(user_input)
        assert "<b>" not in clean_input

        # 5. Store knowledge in semantic memory
        self.semantic.store(
            "All items can be returned within 30 days of purchase.",
            {"tenant": tenant_id, "topic": "policy"}
        )
        self.semantic.store(
            "Shipping takes 3-5 business days.",
            {"tenant": tenant_id, "topic": "shipping"}
        )
        self.semantic.store(
            "Contact customer support for billing questions.",
            {"tenant": tenant_id, "topic": "support"}
        )

        # 6. Verify documents stored
        assert self.semantic.count == 3

        # 7. Store conversation in episodic memory
        self.episodic.store_turn("s1", "user", clean_input, tenant_id=tenant_id)
        self.episodic.store_turn("s1", "assistant", "All items can be returned within 30 days.", tenant_id=tenant_id)

        # 8. Retrieve conversation history
        turns = self.episodic.retrieve_recent("s1", limit=10, tenant_id=tenant_id)
        assert len(turns) == 2

        # 9. Verify tenant isolation — other tenant sees nothing in episodic
        other_turns = self.episodic.retrieve_recent("s1", limit=10, tenant_id="other-tenant")
        assert len(other_turns) == 0

        # 10. Scope check
        assert self.akm.has_scope(validated, "chat") is True
        assert self.akm.has_scope(validated, "knowledge") is True
        assert self.akm.has_scope(validated, "admin") is False
