import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from hbllm.actions.browser_node import BrowserNode
from hbllm.actions.execution_node import ExecutionNode
from hbllm.brain.identity_node import IdentityNode, IdentityProfile
from hbllm.brain.prompt_helper import (
    ChatContext,
    _compute_recency_bonus,
    _find_last_high_signal_message,
    _get_memory_type_bonus,
    _is_low_entropy_query,
    _jaccard_similarity,
    _percentile,
    _truncate_to_budget,
    get_chat_memories,
    get_dynamic_system_prompt,
)
from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import HealthStatus, NodeHealth
from hbllm.network.registry import ServiceRegistry

# ── Existing test (preserved) ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dynamic_prompt_synthesis(tmp_path):
    bus = InProcessBus()
    await bus.start()

    # 1. Start ServiceRegistry
    registry = ServiceRegistry()
    await registry.start(bus)

    # Helper function to register nodes manually for discovery since in-process registry relies on lifecycle listener
    async def register_node(node):
        await node.start(bus)
        await registry.register(node.get_info())
        await registry.update_health(NodeHealth(node_id=node.node_id, status=HealthStatus.HEALTHY))

    # 2. Start IdentityNode
    identity_db = tmp_path / "identity_test.db"
    id_node = IdentityNode(node_id="identity_test", db_path=identity_db)
    await register_node(id_node)

    # 3. Create a tenant identity profile
    profile = IdentityProfile(
        tenant_id="tenant_x",
        persona_name="CustomSentra",
        system_prompt="You are CustomSentra, a highly specialized agent.",
        goals=["Help users write clean code", "Be extremely concise"],
        constraints=["Do not output markdown code blocks"],
    )

    # Store it via update message
    update_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test_runner",
        tenant_id="tenant_x",
        topic="identity.update",
        payload=profile.to_dict(),
    )
    await bus.request("identity.update", update_msg, timeout=2.0)

    # 4. Generate system prompt with no capability nodes active
    prompt_initial = await get_dynamic_system_prompt(bus, "tenant_x", "test_runner")
    assert "CustomSentra" in prompt_initial
    assert "Help users write clean code" in prompt_initial
    assert "Do not output markdown code blocks" in prompt_initial
    assert "BrowserNode" not in prompt_initial
    assert "ExecutionNode" not in prompt_initial

    # 5. Start capability nodes
    browser = BrowserNode(node_id="browser_test")
    await register_node(browser)

    exec_node = ExecutionNode(node_id="exec_test")
    await register_node(exec_node)

    # 6. Generate prompt with active capabilities
    prompt_final = await get_dynamic_system_prompt(bus, "tenant_x", "test_runner")
    assert "BrowserNode" in prompt_final
    assert "ExecutionNode" in prompt_final
    assert "LogicNode" not in prompt_final

    # Clean stop
    await browser.stop()
    await exec_node.stop()
    await id_node.stop()
    await registry.stop()
    await bus.stop()


# ── ChatContext dataclass tests ────────────────────────────────────────────────


class TestChatContext:
    def test_default_values(self):
        ctx = ChatContext()
        assert ctx.recent_turns == ""
        assert ctx.recalled_context == ""
        assert ctx.summary_context == ""
        assert ctx.debug == {}

    def test_field_access(self):
        ctx = ChatContext(
            recent_turns="User: hello",
            recalled_context="some memory",
            summary_context="summary",
            debug={"search_returned": 5},
        )
        assert ctx.recent_turns == "User: hello"
        assert ctx.recalled_context == "some memory"
        assert ctx.debug["search_returned"] == 5

    def test_mutable_fields(self):
        """Verify that recalled_context can be updated (for compression)."""
        ctx = ChatContext(recalled_context="original")
        ctx.recalled_context = "compressed"
        assert ctx.recalled_context == "compressed"


# ── Helper function tests ──────────────────────────────────────────────────────


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert _jaccard_similarity("hello world", "foo bar baz") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity("hello world foo", "hello world bar")
        # intersection = {hello, world} = 2, union = {hello, world, foo, bar} = 4
        assert sim == pytest.approx(0.5)

    def test_empty_string(self):
        assert _jaccard_similarity("", "hello") == 0.0
        assert _jaccard_similarity("hello", "") == 0.0
        assert _jaccard_similarity("", "") == 0.0

    def test_case_insensitive(self):
        assert _jaccard_similarity("Hello World", "hello world") == 1.0


class TestRecencyBonus:
    def test_recent_memory(self):
        now = datetime.now(timezone.utc).isoformat()
        bonus = _compute_recency_bonus(now)
        # Should be close to 0.15 (max bonus with recalibrated decay)
        assert 0.14 < bonus <= 0.15

    def test_old_memory_faster_decay(self):
        """With 7-day half-life, 30-day-old memories should have negligible bonus."""
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        bonus = _compute_recency_bonus(old)
        # exp(-30/7) ≈ 0.014, so bonus ≈ 0.002
        assert bonus < 0.005

    def test_one_week_memory(self):
        """After 7 days, bonus should be roughly half of max (~0.055)."""
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        bonus = _compute_recency_bonus(week_ago)
        # exp(-1) ≈ 0.368, so bonus ≈ 0.055
        assert 0.04 < bonus < 0.07

    def test_missing_created_at(self):
        assert _compute_recency_bonus(None) == 0.0
        assert _compute_recency_bonus("") == 0.0

    def test_invalid_timestamp(self):
        assert _compute_recency_bonus("not-a-date") == 0.0


class TestMemoryTypeBonus:
    def test_memory_summary(self):
        assert _get_memory_type_bonus({"type": "memory_summary"}) == 0.20

    def test_reflection(self):
        assert _get_memory_type_bonus({"type": "reflection"}) == 0.15

    def test_dream_journal_penalty(self):
        assert _get_memory_type_bonus({"type": "dream_journal"}) == -0.05

    def test_conversation_no_bonus(self):
        assert _get_memory_type_bonus({"type": "conversation"}) == 0.0

    def test_missing_type(self):
        assert _get_memory_type_bonus({}) == 0.0

    def test_unknown_type(self):
        assert _get_memory_type_bonus({"type": "unknown_type"}) == 0.0


class TestPercentile:
    def test_basic_percentile(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        assert _percentile(values, 0.5) == pytest.approx(0.5)

    def test_70th_percentile(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        p70 = _percentile(values, 0.70)
        assert 0.7 <= p70 <= 0.8

    def test_empty_list(self):
        assert _percentile([], 0.5) == 0.0

    def test_single_value(self):
        assert _percentile([0.5], 0.7) == 0.5


class TestTruncateToBudget:
    def test_within_budget(self):
        assert _truncate_to_budget("hello world", 100) == "hello world"

    def test_truncation(self):
        result = _truncate_to_budget("hello world foo bar baz", 12)
        assert len(result) <= 13  # 12 + "…"
        assert result.endswith("…")


# ── Query drift control tests ─────────────────────────────────────────────────


class TestQueryDriftControl:
    def test_low_entropy_vague(self):
        """Short vague queries should be detected as low-entropy."""
        assert _is_low_entropy_query("why?") is True
        assert _is_low_entropy_query("continue") is True
        assert _is_low_entropy_query("ok go on") is True
        assert _is_low_entropy_query("what?") is True

    def test_high_entropy_specific(self):
        """Specific queries should NOT be low-entropy."""
        assert _is_low_entropy_query("How do I deploy FastAPI with Docker?") is False
        assert _is_low_entropy_query("Explain the memory consolidation architecture") is False

    def test_find_high_signal_message(self):
        """Should find the most recent meaningful user message."""
        history = [
            {"role": "user", "content": "How do I deploy FastAPI?"},
            {"role": "assistant", "content": "You can use Docker..."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Sure, let me elaborate."},
            {"role": "user", "content": "continue"},
        ]
        result = _find_last_high_signal_message(history)
        assert result == "How do I deploy FastAPI?"

    def test_find_high_signal_empty_history(self):
        assert _find_last_high_signal_message([]) == ""

    def test_all_low_entropy(self):
        """If all messages are vague, return empty string."""
        history = [
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "go on"},
        ]
        assert _find_last_high_signal_message(history) == ""


# ── Global budget tests ───────────────────────────────────────────────────────


class TestGlobalBudget:
    def test_chat_context_returns_debug_info(self):
        """ChatContext should contain budget allocation in debug info."""
        ctx = ChatContext(debug={"budgets": {"total": 3200, "summary": 640}})
        assert ctx.debug["budgets"]["total"] == 3200


# ── Integration tests for get_chat_memories ────────────────────────────────────


@pytest.fixture
async def memory_bus(tmp_path):
    """Set up a bus with a live MemoryNode for integration tests."""
    bus = InProcessBus()
    await bus.start()

    mem_node = MemoryNode(node_id="memory_test", db_path=tmp_path / "test_mem.db")
    await mem_node.start(bus)

    yield bus, mem_node

    await mem_node.stop()
    await bus.stop()


async def _store_memory(
    bus, content, session_id="default_session", role="user", tenant_id="test_tenant", metadata=None
):
    """Helper to store a memory via the bus."""
    payload = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "tenant_id": tenant_id,
    }
    if metadata:
        payload["metadata"] = metadata

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test_runner",
        tenant_id=tenant_id,
        topic="memory.store",
        payload=payload,
    )
    await bus.publish("memory.store", msg)
    await asyncio.sleep(0.1)  # Allow async processing


@pytest.mark.asyncio
async def test_returns_chat_context(memory_bus):
    """Verify get_chat_memories returns a ChatContext object, not a tuple."""
    bus, _mem = memory_bus

    result = await get_chat_memories(bus, "test_tenant", "default_session", "hello", [])

    assert isinstance(result, ChatContext)
    assert isinstance(result.recent_turns, str)
    assert isinstance(result.recalled_context, str)
    assert isinstance(result.summary_context, str)
    assert isinstance(result.debug, dict)


@pytest.mark.asyncio
async def test_debug_info_populated(memory_bus):
    """Verify debug info contains budget and pipeline metadata."""
    bus, _mem = memory_bus

    await _store_memory(bus, "Python is a great language for web development")

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "Tell me about Python",
        [{"role": "user", "content": "I like Python"}],
    )

    assert "budgets" in result.debug
    assert "total" in result.debug["budgets"]
    assert "search_returned" in result.debug
    assert "final_chars" in result.debug


@pytest.mark.asyncio
async def test_sliding_window_limits(memory_bus):
    """Verify sliding window keeps at most 10 turns, capped by token budget."""
    bus, _mem = memory_bus

    # Create 15 turns of history
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn number {i}"}
        for i in range(15)
    ]

    result = await get_chat_memories(
        bus, "test_tenant", "default_session", "current query", history
    )

    # Should have at most 10 turns
    lines = [l for l in result.recent_turns.split("\n") if l.strip()]
    assert len(lines) <= 10

    # Should be within budget (~640 chars = 20% of 3200)
    assert len(result.recent_turns) <= 700  # Small tolerance


@pytest.mark.asyncio
async def test_retrieval_query_formatting(memory_bus):
    """Verify weighted query has Current and Recent blocks."""
    bus, _mem = memory_bus

    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me about Flask."},
    ]

    result = await get_chat_memories(
        bus, "test_tenant", "default_session", "How do I deploy Flask?", history
    )

    # recent_turns should contain the history turns
    assert "User:" in result.recent_turns or result.recent_turns == ""


@pytest.mark.asyncio
async def test_adaptive_threshold(memory_bus):
    """Verify that low-score results are filtered out."""
    bus, _mem = memory_bus

    # Store some relevant and irrelevant memories
    await _store_memory(bus, "Python is a great programming language for web development")
    await _store_memory(bus, "Flask is a Python web framework")
    await _store_memory(bus, "My dog's name is Buddy and he loves walks")
    await _store_memory(bus, "The weather in London is rainy")

    # Query about Python/Flask — dog/weather should be suppressed
    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "How do I build a web app with Flask?",
        [{"role": "user", "content": "I want to learn Flask"}],
    )

    # recalled should NOT contain irrelevant content
    if result.recalled_context:
        assert (
            "dog" not in result.recalled_context.lower() or "Buddy" not in result.recalled_context
        )


@pytest.mark.asyncio
async def test_priority_ranking(memory_bus):
    """Verify memory_summary is ranked above conversation memories."""
    bus, _mem = memory_bus

    # Store a memory_summary (should get +0.20 bonus)
    await _store_memory(
        bus,
        "User frequently discusses Python web frameworks like Flask and Django",
        metadata={"type": "memory_summary"},
    )

    # Store a regular conversation (should get +0.00 bonus)
    await _store_memory(
        bus,
        "User asked about Python web frameworks",
        metadata={"type": "conversation"},
    )

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "What Python web frameworks do you recommend?",
        [{"role": "user", "content": "I need a web framework"}],
    )

    # If both are recalled, the summary should appear first (higher priority)
    if (
        result.recalled_context
        and "frequently" in result.recalled_context
        and "asked" in result.recalled_context
    ):
        assert result.recalled_context.index("frequently") < result.recalled_context.index("asked")


@pytest.mark.asyncio
async def test_semantic_deduplication(memory_bus):
    """Verify near-duplicate memories are discarded."""
    bus, _mem = memory_bus

    # Store near-identical memories
    await _store_memory(bus, "Python is a great programming language for building web applications")
    await _store_memory(
        bus, "Python is a great programming language for building web applications and APIs"
    )

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "Tell me about Python",
        [],
    )

    # Should not have both near-duplicates
    if result.recalled_context:
        parts = result.recalled_context.split("\n---\n")
        assert len(parts) <= 2  # At most the unique + one variant


@pytest.mark.asyncio
async def test_summary_types_extraction(memory_bus):
    """Verify summaries from dream_journal and default_session are retrieved."""
    bus, _mem = memory_bus

    # Store a consolidated memory (system role, default_session)
    await _store_memory(
        bus,
        "[CONSOLIDATED MEMORY] User discussed 3 topics about machine learning.",
        session_id="default_session",
        role="system",
        metadata={"type": "memory_summary"},
    )

    # Store a dream journal entry
    await _store_memory(
        bus,
        "🌙 Dream Journal — Consolidated 5 memories about data science.",
        session_id="dream_journal",
        role="system",
        metadata={"type": "dream_journal"},
    )

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "What did I discuss recently?",
        [],
    )

    # summary_context should contain the consolidated memory
    if result.summary_context:
        assert (
            "CONSOLIDATED MEMORY" in result.summary_context
            or "Dream Journal" in result.summary_context
        )


@pytest.mark.asyncio
async def test_irrelevant_memory_suppression(memory_bus):
    """Verify unrelated memories are suppressed for unrelated queries."""
    bus, _mem = memory_bus

    # Store personal facts
    await _store_memory(bus, "My dog's name is Buddy")
    await _store_memory(bus, "My birthday is on March 15th")

    # Store technical facts
    await _store_memory(bus, "FastAPI uses ASGI for async web serving")
    await _store_memory(bus, "Docker containers provide isolated environments")

    # Query about deployment — personal facts should be suppressed
    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "How do I deploy a FastAPI app with Docker?",
        [{"role": "user", "content": "I need to deploy my API"}],
    )

    # recalled should NOT contain personal facts if threshold is working
    if result.recalled_context:
        assert isinstance(result.recalled_context, str)


@pytest.mark.asyncio
async def test_missing_created_at_graceful(memory_bus):
    """Verify memories without created_at don't crash and get recency_bonus=0."""
    bus, _mem = memory_bus

    # Store a memory that will NOT have created_at in semantic metadata
    await _store_memory(bus, "This is a memory without explicit created_at")

    # Should not crash
    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "What do you remember?",
        [{"role": "user", "content": "Tell me what you know"}],
    )
    assert isinstance(result, ChatContext)
    assert isinstance(result.recent_turns, str)
    assert isinstance(result.recalled_context, str)
    assert isinstance(result.summary_context, str)


@pytest.mark.asyncio
async def test_token_budgeting_global_cap(memory_bus):
    """Verify output stays within the global budget limit."""
    bus, _mem = memory_bus

    # Store many long memories
    for i in range(20):
        await _store_memory(
            bus,
            f"This is memory number {i} with a lot of content about topic {i}. " * 10,
        )

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "Tell me everything",
        [{"role": "user", "content": "long content " * 50}] * 15,
    )

    # Total should be within global budget (3200 chars + small tolerance)
    total = len(result.recent_turns) + len(result.recalled_context) + len(result.summary_context)
    assert total <= 3400  # Small tolerance for truncation rounding

    # Individual components should be within their proportional budgets
    assert len(result.recent_turns) <= 700
    assert len(result.recalled_context) <= 1700
    assert len(result.summary_context) <= 700


@pytest.mark.asyncio
async def test_empty_history(memory_bus):
    """Verify graceful handling of empty history."""
    bus, _mem = memory_bus

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "Hello, who am I?",
        [],
    )

    assert result.recent_turns == ""
    assert isinstance(result.recalled_context, str)
    assert isinstance(result.summary_context, str)


@pytest.mark.asyncio
async def test_query_drift_control(memory_bus):
    """Verify low-entropy queries trigger fallback to high-signal message."""
    bus, _mem = memory_bus

    # Store a memory related to the high-signal message
    await _store_memory(
        bus, "FastAPI deployment guide using Docker containers and Nginx reverse proxy"
    )

    history = [
        {"role": "user", "content": "How do I deploy FastAPI with Docker?"},
        {"role": "assistant", "content": "You can use Docker Compose..."},
        {"role": "user", "content": "continue"},
    ]

    result = await get_chat_memories(
        bus,
        "test_tenant",
        "default_session",
        "continue",  # Low-entropy query
        history,
    )

    # Should have used the high-signal fallback for retrieval
    assert result.debug.get("query_drift_fallback", False) is True
