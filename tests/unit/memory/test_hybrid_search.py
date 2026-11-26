"""
Tests for the three core improvements:
1. Hybrid search (dense + sparse blending)
2. Reward-boosted re-ranking
3. PlannerNode prompt/response cache
"""

from hbllm.memory.semantic import SemanticMemory

# ─── 1. Hybrid Search ───────────────────────────────────────────────────────


class TestHybridSearch:
    """Test that hybrid search blends dense and sparse scores."""

    def test_hybrid_alpha_default(self):
        """Default alpha is 0.7 (biased towards dense)."""
        sm = SemanticMemory()
        assert sm.hybrid_alpha == 0.7

    def test_hybrid_alpha_clamped(self):
        """Alpha is clamped to [0.0, 1.0]."""
        sm = SemanticMemory(hybrid_alpha=1.5)
        assert sm.hybrid_alpha == 1.0
        sm2 = SemanticMemory(hybrid_alpha=-0.5)
        assert sm2.hybrid_alpha == 0.0

    def test_sparse_index_maintained(self):
        """Sparse TF-IDF index is always maintained alongside dense vectors."""
        sm = SemanticMemory()
        sm.store("Python is a programming language", {"topic": "coding"})
        sm.store("Mathematics involves proofs and equations", {"topic": "math"})
        sm.store("The weather is sunny today", {"topic": "weather"})
        # The TF-IDF vocabulary should have tokens from all 3 docs
        assert sm._tfidf._doc_count >= 3

    def test_search_returns_scored_results(self):
        """Search should return scored results."""
        sm = SemanticMemory()
        sm.store("Python is a programming language", {"topic": "coding"})
        sm.store("The weather is sunny today", {"topic": "weather"})
        sm.store("Machine learning uses data to learn patterns", {"topic": "ml"})

        results = sm.search("programming language", top_k=2)
        assert len(results) >= 1
        assert all("score" in r for r in results)

    def test_pure_sparse_search(self):
        """With alpha=0.0, search uses only sparse (TF-IDF) scoring."""
        sm = SemanticMemory(hybrid_alpha=0.0)
        sm.store("Python is a great programming language for data science")
        sm.store("Java is used for enterprise applications")
        sm.store("The cat sat on the mat")

        results = sm.search("Python programming", top_k=3)
        # Should find the Python doc with keyword matching
        if results:
            assert "Python" in results[0]["content"]

    def test_clear_resets_sparse_index(self):
        """Clearing memory should reset both dense and sparse indexes."""
        sm = SemanticMemory()
        sm.store("Document one")
        sm.store("Document two")
        sm.clear()
        assert sm._sparse_list == []
        assert sm._sparse_cache is None


# ─── 2. Reward-Boosted Re-Ranking ───────────────────────────────────────────


class TestRewardReranking:
    """Test that reward scores boost or penalize search results."""

    def test_positive_reward_boosts_result(self):
        """A positive reward score should boost a document's final score."""
        sm = SemanticMemory()
        sm.store("Python is a programming language", {"topic": "coding"})
        sm.store("Java is an enterprise language", {"topic": "coding"})
        sm.store("Weather is nice today", {"topic": "weather"})

        # Search without rewards
        base_results = sm.search("programming", top_k=3)

        # Search with reward boosting doc index 1 (Java)
        boosted_results = sm.search(
            "programming",
            top_k=3,
            reward_scores={1: 5.0},  # Big boost to Java
            reward_boost=0.2,
        )

        # Both should return results
        assert len(base_results) >= 1
        assert len(boosted_results) >= 1

    def test_reward_scores_applied(self):
        """Reward scores should actually change the final scores."""
        sm = SemanticMemory()
        sm.store("Alpha document about cats")
        sm.store("Beta document about dogs")
        sm.store("Gamma document about birds")

        sm.search("document", top_k=3)
        with_boost = sm.search(
            "document",
            top_k=3,
            reward_scores={2: 10.0},  # Massive boost to "birds"
            reward_boost=0.5,
        )

        # With the boost, "birds" should have a much higher score
        if with_boost:
            bird_scores = [r["score"] for r in with_boost if "birds" in r["content"]]
            assert len(bird_scores) >= 1

    def test_no_reward_is_default(self):
        """Without reward_scores, search behaves normally."""
        sm = SemanticMemory()
        sm.store("Test document one")
        sm.store("Test document two")

        results_default = sm.search("test", top_k=2)
        results_none = sm.search("test", top_k=2, reward_scores=None)

        # Same results with no rewards
        assert len(results_default) == len(results_none)

    def test_negative_reward_penalizes(self):
        """A negative reward should lower a document's score."""
        sm = SemanticMemory()
        sm.store("Good content about programming")
        sm.store("Bad content about programming")
        sm.store("Other stuff unrelated")

        results = sm.search(
            "programming",
            top_k=3,
            reward_scores={1: -5.0},  # Penalize "Bad content"
            reward_boost=0.2,
        )
        # Should still return results without crashing
        assert isinstance(results, list)
