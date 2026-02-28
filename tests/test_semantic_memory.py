"""Tests for SemanticMemory — vector search with TF-IDF fallback."""

import pytest
from hbllm.memory.semantic import SemanticMemory, _TfIdfEmbedder


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
    assert idx == -1
    assert sem_mem.count == 0

    idx2 = sem_mem.store("   ")
    assert idx2 == -1


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
    sem_mem.store("First doc")
    sem_mem.store("Second doc")
    assert sem_mem.count == 2

    deleted = sem_mem.delete(0)
    assert deleted is True
    assert sem_mem.count == 1
    assert sem_mem.documents[0]["content"] == "Second doc"


def test_delete_invalid_index(sem_mem):
    sem_mem.store("Only doc")
    assert sem_mem.delete(-1) is False
    assert sem_mem.delete(5) is False
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
