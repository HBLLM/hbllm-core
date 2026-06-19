from hbllm.memory.conflict_resolver import MemoryConflictResolver


def test_resolve_causal_ordering():
    # A is strictly after B
    frag_a = {
        "content": "New info",
        "vector_clock": {"node1": 2, "node2": 1},
        "authority_score": 50,
        "timestamp": "2024-01-01T12:01:00Z",
    }
    frag_b = {
        "content": "Old info",
        "vector_clock": {"node1": 1, "node2": 1},
        "authority_score": 50,
        "timestamp": "2024-01-01T12:00:00Z",
    }

    winner = MemoryConflictResolver.resolve(frag_a, frag_b)
    assert winner["content"] == "New info"


def test_resolve_authority_win():
    # Concurrent clocks, but A has higher authority
    frag_a = {
        "content": "Trusted info",
        "vector_clock": {"node1": 2, "node2": 1},
        "authority_score": 90,
        "timestamp": "2024-01-01T12:00:00Z",
    }
    frag_b = {
        "content": "Less trusted info",
        "vector_clock": {"node1": 1, "node2": 2},
        "authority_score": 50,
        "timestamp": "2024-01-01T12:00:00Z",
    }

    winner = MemoryConflictResolver.resolve(frag_a, frag_b)
    assert winner["content"] == "Trusted info"


def test_resolve_recency_fallback():
    # Equal authority and concurrent/missing clocks, use timestamp
    frag_a = {"content": "Later info", "authority_score": 50, "timestamp": "2024-01-01T12:05:00Z"}
    frag_b = {"content": "Earlier info", "authority_score": 50, "timestamp": "2024-01-01T12:00:00Z"}

    winner = MemoryConflictResolver.resolve(frag_a, frag_b)
    assert winner["content"] == "Later info"
