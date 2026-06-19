"""Tests for SemanticMemory — vector search with TF-IDF fallback."""

import pytest

from hbllm.memory.interface import MemoryType
from hbllm.memory.memory_node import MemoryNode
from hbllm.memory.semantic import SemanticMemory, _TfIdfEmbedder
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── TF-IDF Embedder Tests ───────────────────────────────────────────────────


def test_tfidf_tokenize():
    tokens = _TfIdfEmbedder._tokenize("Hello, World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "," not in tokens


def test_tfidf_encode():
    emb = _TfIdfEmbedder()
    emb.fit_token("the cat sat on the mat")
    emb.fit_token("the dog sat on the log")
    vectors = emb.encode(["cat mat", "dog log"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


# ── SemanticMemory Tests (TF-IDF fallback mode) ─────────────────────────────


@pytest.fixture
def sem_mem():
    """SemanticMemory in TF-IDF mode (no sentence-transformers needed)."""
    mem = SemanticMemory()
    mem._use_tfidf = True  # Force TF-IDF mode for testing
    return mem


def test_store_and_count(sem_mem):
    sem_mem.store("Python is great")
    sem_mem.store("JavaScript is popular")
    assert sem_mem.count == 2


def test_store_empty_string(sem_mem):
    """Empty strings should be rejected."""
    idx = sem_mem.store("")
    assert idx is None
    assert sem_mem.count == 0

    idx2 = sem_mem.store("   ")
    assert idx2 is None


def test_search_basic(sem_mem):
    sem_mem.store("Python is a programming language", {"lang": "python"})
    sem_mem.store("The weather is sunny today", {"topic": "weather"})
    sem_mem.store("Java is also a programming language", {"lang": "java"})

    results = sem_mem.search("programming language")
    assert len(results) >= 1
    # Programming-related docs should score higher than weather
    assert any("programming" in r["content"].lower() for r in results)


def test_search_empty_query(sem_mem):
    sem_mem.store("Some content")
    assert sem_mem.search("") == []
    assert sem_mem.search("   ") == []


def test_search_empty_db(sem_mem):
    assert sem_mem.search("anything") == []


def test_delete(sem_mem):
    idx1 = sem_mem.store("First doc")
    sem_mem.store("Second doc")
    assert sem_mem.count == 2

    deleted = sem_mem.delete(idx1)
    assert deleted is True
    assert sem_mem.count == 1
    assert sem_mem.get_all()[0]["content"] == "Second doc"


def test_delete_invalid_index(sem_mem):
    sem_mem.store("Only doc")
    assert sem_mem.delete("nonexistent-uuid") is False
    assert sem_mem.count == 1


def test_clear(sem_mem):
    sem_mem.store("Doc 1")
    sem_mem.store("Doc 2")
    sem_mem.store("Doc 3")

    count = sem_mem.clear()
    assert count == 3
    assert sem_mem.count == 0
    assert sem_mem.vectors is None


def test_get_all(sem_mem):
    sem_mem.store("Alpha", {"idx": 0})
    sem_mem.store("Beta", {"idx": 1})

    docs = sem_mem.get_all()
    assert len(docs) == 2
    assert docs[0]["content"] == "Alpha"
    assert docs[1]["metadata"]["idx"] == 1


def test_metadata_preserved(sem_mem):
    sem_mem.store("Test content", {"key": "value", "number": 42})
    results = sem_mem.search("Test content")
    assert len(results) >= 1
    assert results[0]["metadata"]["key"] == "value"


def test_search_returns_scores(sem_mem):
    sem_mem.store("Machine learning is AI")
    sem_mem.store("Cooking pasta")

    results = sem_mem.search("machine learning")
    assert len(results) >= 1
    assert "score" in results[0]
    assert results[0]["score"] > 0


# ── Enhanced Features Tests (Fuzzy Dedup, Usefulness, Consolidation) ─────────


def test_fuzzy_deduplication(sem_mem):
    """Verify that near-duplicate documents within the same tenant return None."""
    doc_id1 = sem_mem.store("The quick brown fox jumps over the lazy dog", tenant_id="tenant_a")
    assert doc_id1 is not None

    # Exact duplicate should return None
    doc_id2 = sem_mem.store("The quick brown fox jumps over the lazy dog", tenant_id="tenant_a")
    assert doc_id2 is None

    # Fuzzy duplicate (high similarity) in same tenant should return None
    doc_id3 = sem_mem.store("The quick brown fox jumped over the lazy dog", tenant_id="tenant_a")
    assert doc_id3 is None
    assert sem_mem.count == 1

    # Same content but in a different tenant should NOT deduplicate (security boundary)
    doc_id_other_tenant = sem_mem.store(
        "The quick brown fox jumps over the lazy dog", tenant_id="tenant_b"
    )
    assert doc_id_other_tenant is not None
    assert doc_id_other_tenant != doc_id1
    assert sem_mem.count == 2


def test_safe_save_disabled(sem_mem):
    """Verify safe_save=False skips fuzzy deduplication."""
    sem_mem.safe_save = False
    doc_id1 = sem_mem.store("Python is an awesome programming language", tenant_id="t1")
    doc_id2 = sem_mem.store("Python is a truly awesome programming language", tenant_id="t1")
    assert doc_id1 != doc_id2
    assert sem_mem.count == 2


def test_usefulness_feedback_ranking(sem_mem):
    """Verify that usefulness feedback boosts search scores."""
    # Store two different docs
    doc_id_py = sem_mem.store("Python is a popular language", tenant_id="t1")
    doc_id_js = sem_mem.store("JavaScript is a popular language", tenant_id="t1")

    # Initial search
    res1 = sem_mem.search("popular language", tenant_id="t1")
    score_py_1 = next(r["score"] for r in res1 if r["id"] == doc_id_py)
    score_js_1 = next(r["score"] for r in res1 if r["id"] == doc_id_js)

    # Apply positive feedback to JavaScript doc
    new_usefulness = sem_mem.feedback(doc_id_js, useful=True)
    assert new_usefulness == 1

    # Search again
    res2 = sem_mem.search("popular language", tenant_id="t1")
    score_py_2 = next(r["score"] for r in res2 if r["id"] == doc_id_py)
    score_js_2 = next(r["score"] for r in res2 if r["id"] == doc_id_js)

    # Score of Python should remain unchanged
    assert score_py_1 == pytest.approx(score_py_2)
    # Score of JavaScript should have increased due to usefulness boost
    assert score_js_2 > score_js_1


def test_consolidation(sem_mem):
    """Verify consolidation merges similar documents within a tenant and blends tags."""
    sem_mem.safe_save = False  # Disable safe save to store duplicates first
    _ = sem_mem.store(
        "Software engineering involves writing clean code",
        metadata={"tags": ["dev"]},
        tenant_id="t1",
    )
    doc_id2 = sem_mem.store(
        "Software engineering includes writing clean code",
        metadata={"tags": ["coding"]},
        tenant_id="t1",
    )
    # Also add one doc in different tenant
    sem_mem.store(
        "Software engineering involves writing clean code",
        metadata={"tags": ["other"]},
        tenant_id="t2",
    )

    assert sem_mem.count == 3

    # Consolidate tenant t1
    res = sem_mem.consolidate(tenant_id="t1", threshold=0.6)
    assert res["removed"] == 1
    assert sem_mem.count == 2

    # Check merged tags
    keeper = sem_mem.documents[doc_id2]
    tags = keeper["metadata"]["tags"]
    assert "dev" in tags
    assert "coding" in tags
    # Tenant t2 doc should not be removed
    assert len([d for d in sem_mem.documents.values() if d["metadata"]["tenant_id"] == "t2"]) == 1


@pytest.mark.asyncio
async def test_memory_node_handlers(tmp_path):
    """Verify MemoryNode handles feedback and consolidate messages over the bus."""
    bus = InProcessBus()
    await bus.start()

    node = MemoryNode(node_id="mem_node_test", db_path=tmp_path / "test.db")
    # Force TF-IDF mode
    node.semantic_db._use_tfidf = True
    await node.start(bus)

    # Store a document first
    _ = await node.store(
        memory_type=MemoryType.SEMANTIC,
        data="Cognitive architectures are complex systems",
        tenant_id="t1",
    )
    all_docs = node.semantic_db.get_all()
    assert len(all_docs) == 1
    stored_doc_id = all_docs[0]["id"]

    # 1. Test feedback handler
    feedback_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        topic="memory.feedback",
        payload={"note_id": stored_doc_id, "useful": True},
    )
    resp = await bus.request("memory.feedback", feedback_msg)
    assert resp is not None
    assert resp.payload["status"] == "updated"
    assert resp.payload["usefulness"] == 1

    # 2. Test consolidate handler
    consolidate_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        tenant_id="t1",
        topic="memory.consolidate",
        payload={"threshold": 0.8},
    )
    resp_c = await bus.request("memory.consolidate", consolidate_msg)
    assert resp_c is not None
    assert "removed" in resp_c.payload

    await node.stop()
    await bus.stop()
